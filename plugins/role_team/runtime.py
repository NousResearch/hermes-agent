"""Persistent and delegated role invocation runtime for the role-team plugin."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
import uuid
from concurrent.futures import CancelledError
from pathlib import Path
from typing import Any, Callable, Dict, Optional, cast

from agent.runtime_cwd import scoped_session_cwd
from tools.async_delegation import dispatch_async_delegation, get_durable_delegation
from tools.terminal_tool import clear_task_env_overrides, register_task_env_overrides

from .catalog import RoleCatalog, RoleDefinition
from .store import ActiveRoleInvocation, PlanStore

logger = logging.getLogger(__name__)

_CANCELLATION_SIGNALS = (
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
    InterruptedError,
    CancelledError,
    asyncio.CancelledError,
)


class PersistenceFailure(RuntimeError):
    pass


class ActivationFailure(RuntimeError):
    pass


def role_session_id(plan_id: str, role_slug: str) -> str:
    digest = hashlib.sha256(f"{plan_id}\0{role_slug}".encode("utf-8")).hexdigest()[:16]
    return f"role-{role_slug[:40]}-{digest}"


def _decode_model_config(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {}
    raw = row.get("model_config")
    if isinstance(raw, dict):
        return dict(raw)
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _summary_from_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("summary", "final_response", "content", "result"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return json.dumps(result, ensure_ascii=False, default=str)


class RoleTeamRuntime:
    def __init__(
        self,
        *,
        parent_agent: Any,
        workspace_root: Path | str,
        catalog: Optional[RoleCatalog] = None,
        role_runner: Optional[Callable[..., Any]] = None,
        fault_hook: Optional[Callable[[str], None]] = None,
    ):
        self.parent_agent = parent_agent
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        if not self.workspace_root.is_dir():
            raise ValueError(f"workspace root does not exist: {self.workspace_root}")
        self.catalog = catalog or RoleCatalog.default()
        self.role_runner = role_runner
        self.fault_hook = fault_hook

    @property
    def session_db(self):
        return getattr(self.parent_agent, "_session_db", None) if self.parent_agent else None

    def _after_write(self, stage: str) -> None:
        if self.fault_hook is None:
            return
        try:
            self.fault_hook(stage)
        except _CANCELLATION_SIGNALS:
            raise
        except BaseException as exc:
            raise PersistenceFailure(f"persistence fault {stage}: {exc}") from exc

    @staticmethod
    def _persist(stage: str, operation: Callable[[], Any]) -> Any:
        try:
            return operation()
        except _CANCELLATION_SIGNALS:
            raise
        except PersistenceFailure:
            raise
        except BaseException as exc:
            raise PersistenceFailure(f"persistence fault {stage}: {exc}") from exc

    def _resolve_workdir(self, value: Optional[str]) -> Path:
        candidate = Path(value).expanduser() if value else self.workspace_root
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        resolved = candidate.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"role workdir is not a directory: {resolved}")
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError("role workdir must be inside the workspace root") from exc
        return resolved

    def _validate(
        self,
        *,
        role: str,
        plan_id: str,
        summary: str,
        execution_mode: str,
        workdir: Optional[str],
    ) -> tuple[RoleDefinition, PlanStore, Path]:
        if self.parent_agent is None:
            raise ValueError("invoke_role requires an active parent agent")
        if not str(summary or "").strip():
            raise ValueError("summary is required")
        definition = self.catalog.resolve(role)
        if execution_mode not in definition.allowed_execution_modes:
            raise ValueError(
                f"role {definition.title} does not allow execution mode {execution_mode}"
            )
        if execution_mode == "persistent_role_instance" and self.session_db is None:
            raise ValueError("persistent role invocation requires the parent's SessionDB")
        canonical_workdir = self._resolve_workdir(workdir)
        if execution_mode == "delegated_subagent":
            if canonical_workdir != self.workspace_root:
                raise ValueError(
                    "authoritative delegated_subagent execution cannot guarantee a custom "
                    "workdir; use persistent_role_instance or the workspace root"
                )
            parent_task_id = getattr(self.parent_agent, "_current_task_id", None)
            if parent_task_id:
                from tools.terminal_tool import get_session_cwd

                live_cwd = get_session_cwd(parent_task_id)
                if live_cwd:
                    try:
                        live_path = Path(live_cwd).expanduser().resolve(strict=True)
                    except OSError as exc:
                        raise ValueError("parent's live workdir is unavailable") from exc
                    if live_path != canonical_workdir:
                        raise ValueError(
                            "authoritative delegation would inherit the parent's live workdir, "
                            "which differs from the claimed workspace root"
                        )
        return definition, PlanStore(self.workspace_root, plan_id), canonical_workdir

    @staticmethod
    def _packet(definition: RoleDefinition, plan_id: str, summary: str, workdir: Path) -> str:
        return (
            f"# Role packet: {definition.title}\n\n"
            f"Plan: `{plan_id}`\n\n"
            f"Canonical workdir: `{workdir}`\n\n"
            f"## Assignment\n\n{summary.strip()}\n\n"
            f"## Role contract\n\n{definition.prompt}\n"
        )

    def invoke(
        self,
        *,
        role: str,
        plan_id: str,
        summary: str,
        execution_mode: str = "persistent_role_instance",
        workdir: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            definition, store, canonical_workdir = self._validate(
                role=role,
                plan_id=plan_id,
                summary=summary,
                execution_mode=execution_mode,
                workdir=workdir,
            )
        except Exception as exc:
            return {"status": "rejected", "error": str(exc)}

        invocation_id = f"run-{uuid.uuid4().hex}"
        persistent_id = (
            role_session_id(plan_id, definition.slug)
            if execution_mode == "persistent_role_instance"
            else None
        )
        base_record = {
            "invocation_id": invocation_id,
            "role": definition.title,
            "role_slug": definition.slug,
            "execution_mode": execution_mode,
            "status": "preparing",
            "persistent_session_id": persistent_id,
            "workdir": str(canonical_workdir),
            "summary": summary.strip(),
            "delegation_id": None,
            "delivery_state": "pending",
            "end_reason": None,
        }
        try:
            store.reserve_invocation(base_record)
        except ActiveRoleInvocation as exc:
            return {"status": "rejected", "error": str(exc)}
        except Exception as exc:
            return {"status": "error", "error": f"could not reserve role invocation: {exc}"}

        packet = self._packet(definition, plan_id, summary, canonical_workdir)
        try:
            packet_path = store.write_artifact(
                definition.slug, invocation_id, "packet", packet
            )
            self._after_write("after_packet_write")
            store.transition(invocation_id, packet_path=packet_path, status="queued")
        except _CANCELLATION_SIGNALS:
            self._safe_block(
                store,
                invocation_id,
                "role invocation cancelled during packet publication",
                activated=False,
                status="cancelled",
                end_reason="role_cancelled",
            )
            raise
        except BaseException as exc:
            self._safe_block(store, invocation_id, str(exc), activated=False)
            return {"status": "error", "invocation_id": invocation_id, "error": str(exc)}

        if execution_mode == "delegated_subagent":
            return self._invoke_delegated(
                store, definition, invocation_id, summary, packet, canonical_workdir
            )
        return self._dispatch_persistent(
            store,
            definition,
            invocation_id,
            plan_id,
            summary,
            packet,
            canonical_workdir,
            persistent_id or "",
        )

    def _invoke_delegated(
        self,
        store: PlanStore,
        definition: RoleDefinition,
        invocation_id: str,
        summary: str,
        packet: str,
        workdir: Path,
    ) -> Dict[str, Any]:
        dispatch = getattr(self.parent_agent, "_dispatch_delegate_task", None)
        if not callable(dispatch):
            error = "active parent agent does not expose authoritative delegation"
            self._safe_block(store, invocation_id, error, activated=False)
            return {"status": "rejected", "invocation_id": invocation_id, "error": error}
        try:
            raw = dispatch(
                {
                    "goal": summary.strip(),
                    "context": packet,
                    "role": "leaf",
                    "background": True,
                    "toolsets": list(getattr(self.parent_agent, "enabled_toolsets", []) or []),
                }
            )
            result = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(result, dict):
                raise RuntimeError("delegation returned a non-object result")
            delegation_id = str(result.get("delegation_id") or "").strip()
            if result.get("status") in {"completed", "success"} and not delegation_id:
                rendered = _summary_from_result(result)
                output_path = store.write_artifact(
                    definition.slug, invocation_id, "output", rendered
                )
                store.transition(
                    invocation_id,
                    status="completed",
                    end_reason="delegated_completed",
                    delivery_state="inline",
                    output_path=output_path,
                    summary=rendered,
                )
                return {
                    "status": "completed",
                    "invocation_id": invocation_id,
                    "summary": rendered,
                    "result": result,
                    "delivery_state": "inline",
                }
            if result.get("status") != "dispatched" or not delegation_id:
                raise RuntimeError(result.get("error") or "delegation did not return a real handle")
            store.transition(
                invocation_id,
                status="delegated",
                delegation_id=delegation_id,
                delivery_state="pending",
                workdir=str(workdir),
            )
            return {
                "status": "dispatched",
                "invocation_id": invocation_id,
                "delegation_id": delegation_id,
                "delivery_state": "pending",
            }
        except Exception as exc:
            self._safe_block(store, invocation_id, str(exc), activated=False)
            return {"status": "error", "invocation_id": invocation_id, "error": str(exc)}

    def _dispatch_persistent(
        self,
        store: PlanStore,
        definition: RoleDefinition,
        invocation_id: str,
        plan_id: str,
        summary: str,
        packet: str,
        workdir: Path,
        persistent_id: str,
    ) -> Dict[str, Any]:
        async_ok, session_key, origin_ui_session_id = self._delivery_context()
        if not async_ok:
            store.transition(
                invocation_id,
                status="queued",
                delivery_state="inline",
            )
            inline = self._run_persistent(
                store=store,
                definition=definition,
                invocation_id=invocation_id,
                plan_id=plan_id,
                summary=summary,
                packet=packet,
                workdir=workdir,
                persistent_id=persistent_id,
                delegation_id="",
                agent_holder={},
            )
            inline["delivery_state"] = "inline"
            return inline

        ready = threading.Event()
        holder: Dict[str, Any] = {"delegation_id": None, "abort": None, "agent": None}

        def run():
            if not ready.wait(30):
                return {"status": "error", "summary": "role dispatch publication timed out"}
            if holder["abort"]:
                return {"status": "error", "summary": str(holder["abort"])}
            return self._run_persistent(
                store=store,
                definition=definition,
                invocation_id=invocation_id,
                plan_id=plan_id,
                summary=summary,
                packet=packet,
                workdir=workdir,
                persistent_id=persistent_id,
                delegation_id=str(holder["delegation_id"]),
                agent_holder=holder,
            )

        def interrupt():
            agent = holder.get("agent")
            if agent is not None and hasattr(agent, "interrupt"):
                agent.interrupt("role invocation interrupted")

        try:
            dispatched = dispatch_async_delegation(
                goal=summary.strip(),
                context=packet,
                toolsets=list(getattr(self.parent_agent, "enabled_toolsets", []) or []),
                role="leaf",
                model=str(getattr(self.parent_agent, "model", "") or "") or None,
                session_key=session_key,
                parent_session_id=str(getattr(self.parent_agent, "session_id", "") or "") or None,
                origin_ui_session_id=origin_ui_session_id,
                runner=run,
                interrupt_fn=interrupt,
            )
            if dispatched.get("status") != "dispatched" or not dispatched.get("delegation_id"):
                raise RuntimeError(dispatched.get("error") or "background dispatch was rejected")
            holder["delegation_id"] = str(dispatched["delegation_id"])
            store.transition(
                invocation_id,
                status="queued",
                delegation_id=holder["delegation_id"],
                delivery_state="pending",
            )
            ready.set()
            return {
                "status": "dispatched",
                "invocation_id": invocation_id,
                "delegation_id": holder["delegation_id"],
                "persistent_session_id": persistent_id,
                "delivery_state": "pending",
            }
        except Exception as exc:
            holder["abort"] = str(exc)
            ready.set()
            self._safe_block(store, invocation_id, str(exc), activated=False)
            return {"status": "error", "invocation_id": invocation_id, "error": str(exc)}

    def _delivery_context(self) -> tuple[bool, str, str]:
        """Resolve capability and completion ownership via current session APIs."""
        try:
            from gateway.session_context import (
                async_delivery_supported,
                get_session_env,
            )

            async_ok = async_delivery_supported()
            source = get_session_env("HERMES_SESSION_SOURCE", "")
            origin_ui_session_id = get_session_env("HERMES_UI_SESSION_ID", "")
        except Exception:
            async_ok = False
            source = ""
            origin_ui_session_id = ""

        try:
            from tools.approval import get_current_session_key

            session_key = get_current_session_key(default="")
        except Exception:
            session_key = ""

        parent_session_id = str(getattr(self.parent_agent, "session_id", "") or "")
        if source == "tui" and parent_session_id:
            session_key = parent_session_id
        if not session_key and parent_session_id:
            session_key = parent_session_id
        return async_ok, session_key, origin_ui_session_id

    def _activate_session(
        self,
        persistent_id: str,
        definition: RoleDefinition,
        invocation_id: str,
        plan_id: str,
        workdir: Path,
        delegation_id: str,
    ) -> Dict[str, Any]:
        now = time.time()

        def activate(current):
            config = _decode_model_config(current)
            metadata = dict(config.get("role_metadata") or {})
            metadata.pop("retired_at", None)
            metadata.pop("retire_reason", None)
            metadata.update(
                {
                    "role": definition.title,
                    "role_slug": definition.slug,
                    "plan_id": plan_id,
                    "status": "active",
                    "active_invocation_id": invocation_id,
                    "delegation_id": delegation_id,
                    "activated_at": now,
                    "workdir": str(workdir),
                }
            )
            config["role_metadata"] = metadata
            return {
                "model_config": config,
                "cwd": str(workdir),
                "ended_at": None,
                "end_reason": None,
            }

        return self.session_db.mutate_session(
            persistent_id,
            activate,
            create={
                "source": "role-team",
                "model": str(getattr(self.parent_agent, "model", "") or ""),
                "parent_session_id": str(getattr(self.parent_agent, "session_id", "") or "")
                or None,
                "cwd": str(workdir),
            },
        )

    def _finalize_session(
        self,
        persistent_id: str,
        invocation_id: str,
        status: str,
        end_reason: str,
    ) -> Dict[str, Any]:
        now = time.time()

        def finalize(current):
            if current is None:
                raise KeyError(f"persistent role session {persistent_id!r} is missing")
            config = _decode_model_config(current)
            metadata = dict(config.get("role_metadata") or {})
            # A late sibling may not retire a different active invocation.
            active_id = metadata.get("active_invocation_id")
            if active_id not in (None, invocation_id):
                raise RuntimeError(
                    f"session is owned by active invocation {active_id}; refusing stale finalization"
                )
            metadata.update(
                {
                    "status": status,
                    "active_invocation_id": None,
                    "retired_at": now,
                    "retire_reason": end_reason,
                }
            )
            config["role_metadata"] = metadata
            return {
                "model_config": config,
                "ended_at": now,
                "end_reason": end_reason,
            }

        return self.session_db.mutate_session(persistent_id, finalize)

    def _run_persistent(
        self,
        *,
        store: PlanStore,
        definition: RoleDefinition,
        invocation_id: str,
        plan_id: str,
        summary: str,
        packet: str,
        workdir: Path,
        persistent_id: str,
        delegation_id: str,
        agent_holder: Dict[str, Any],
    ) -> Dict[str, Any]:
        activated = False
        try:
            try:
                self._activate_session(
                    persistent_id,
                    definition,
                    invocation_id,
                    plan_id,
                    workdir,
                    delegation_id,
                )
            except _CANCELLATION_SIGNALS:
                raise
            except BaseException as exc:
                raise ActivationFailure(f"role session activation failed: {exc}") from exc
            activated = True
            self._after_write("after_session_activate")
            self._persist(
                "running_plan_publication",
                lambda: store.transition(
                    invocation_id, status="running", started_at=time.time()
                ),
            )

            register_task_env_overrides(persistent_id, {"cwd": str(workdir)})
            with scoped_session_cwd(str(workdir)):
                if self.role_runner is not None:
                    result = self.role_runner(
                        packet=packet,
                        summary=summary,
                        role=definition.title,
                        role_slug=definition.slug,
                        role_session_id=persistent_id,
                        parent_agent=self.parent_agent,
                    )
                else:
                    result = self._run_role_agent(
                        definition, summary, persistent_id, agent_holder
                    )
            if isinstance(result, dict) and result.get("interrupted") is True:
                interrupted_summary = _summary_from_result(result)
                self._safe_block(
                    store,
                    invocation_id,
                    interrupted_summary,
                    activated=True,
                    persistent_id=persistent_id,
                    status="cancelled",
                    end_reason="role_cancelled",
                )
                return {
                    "status": "cancelled",
                    "summary": interrupted_summary,
                    "invocation_id": invocation_id,
                    "persistent_session_id": persistent_id,
                }
            output = _summary_from_result(result)
            output_path = self._persist(
                "output_artifact",
                lambda: store.write_artifact(
                    definition.slug, invocation_id, "output", output
                ),
            )
            self._after_write("after_output_write")

            self._persist(
                "session_finalization",
                lambda: self._finalize_session(
                    persistent_id, invocation_id, "completed", "role_completed"
                ),
            )
            self._after_write("after_session_finalize")
            self._persist(
                "terminal_plan_publication",
                lambda: store.transition(
                    invocation_id,
                    status="completed",
                    end_reason="role_completed",
                    completed_at=time.time(),
                    output_path=output_path,
                    summary=output,
                ),
            )
            self._after_write("after_plan_finalize")
            return {
                "status": "completed",
                "summary": output,
                "invocation_id": invocation_id,
                "persistent_session_id": persistent_id,
            }
        except BaseException as exc:
            cancelled = isinstance(exc, _CANCELLATION_SIGNALS)
            persistence_fault = isinstance(exc, PersistenceFailure)
            activation_fault = isinstance(exc, ActivationFailure)
            status = "cancelled" if cancelled else "blocked"
            end_reason = (
                "role_cancelled"
                if cancelled
                else "persistence_failure"
                if persistence_fault
                else "activation_failed"
                if activation_fault
                else "role_failed"
            )
            self._safe_block(
                store,
                invocation_id,
                str(exc),
                activated=activated,
                persistent_id=persistent_id,
                status=status,
                end_reason=end_reason,
            )
            if cancelled:
                raise
            return {
                "status": "error",
                "summary": str(exc),
                "invocation_id": invocation_id,
                "persistent_session_id": persistent_id,
            }
        finally:
            clear_task_env_overrides(persistent_id)
            agent_holder["agent"] = None

    def _safe_block(
        self,
        store: PlanStore,
        invocation_id: str,
        error: str,
        *,
        activated: bool,
        persistent_id: Optional[str] = None,
        status: str = "blocked",
        end_reason: str = "persistence_failure",
    ) -> None:
        if activated and persistent_id and self.session_db is not None:
            try:
                self._finalize_session(
                    persistent_id, invocation_id, status, end_reason
                )
            except BaseException:
                logger.exception("could not reconcile persistent role session %s", persistent_id)
        try:
            store.transition(
                invocation_id,
                status=status,
                end_reason=end_reason,
                completed_at=time.time(),
                error=str(error),
            )
        except BaseException:
            logger.exception("could not reconcile role bundle invocation %s", invocation_id)

    def _run_role_agent(
        self,
        definition: RoleDefinition,
        summary: str,
        persistent_id: str,
        agent_holder: Dict[str, Any],
    ) -> Any:
        from run_agent import AIAgent

        parent = self.parent_agent
        # AIAgent's legacy annotations declare several ``None`` defaults as
        # plain ``str``. Cast only the constructor boundary until those core
        # annotations are corrected; runtime values mirror the parent agent.
        agent_factory = cast(Any, AIAgent)
        agent = agent_factory(
            base_url=getattr(parent, "base_url", None),
            api_key=getattr(parent, "api_key", None),
            provider=getattr(parent, "provider", None),
            api_mode=getattr(parent, "api_mode", None),
            model=getattr(parent, "model", ""),
            max_iterations=getattr(parent, "max_iterations", 90),
            tool_delay=getattr(parent, "tool_delay", 1.0),
            enabled_toolsets=list(getattr(parent, "enabled_toolsets", []) or []),
            disabled_toolsets=list(getattr(parent, "disabled_toolsets", []) or []),
            quiet_mode=True,
            ephemeral_system_prompt=definition.prompt,
            session_id=persistent_id,
            platform=getattr(parent, "platform", None),
            session_db=self.session_db,
            parent_session_id=getattr(parent, "session_id", None),
            credential_pool=getattr(parent, "_credential_pool", None),
            reasoning_config=getattr(parent, "reasoning_config", None),
            service_tier=getattr(parent, "service_tier", None),
            request_overrides=getattr(parent, "request_overrides", None),
            max_tokens=getattr(parent, "max_tokens", None),
            fallback_model=getattr(parent, "fallback_model", None),
            pass_session_id=True,
        )
        agent_holder["agent"] = agent
        history = self.session_db.get_messages_as_conversation(persistent_id)
        try:
            return agent.run_conversation(
                summary,
                conversation_history=history,
                task_id=persistent_id,
            )
        finally:
            try:
                agent.close()
            except Exception:
                logger.debug("role agent close failed", exc_info=True)

    def status(self, plan_id: str) -> Dict[str, Any]:
        store = PlanStore(self.workspace_root, plan_id)
        state = store.snapshot()
        for invocation_id, item in list(state["invocations"].items()):
            delegation_id = item.get("delegation_id")
            if not delegation_id:
                continue
            durable = get_durable_delegation(str(delegation_id))
            if not durable:
                continue
            delivery_state = str(durable.get("delivery_state") or "pending")
            patch: Dict[str, Any] = {"delivery_state": delivery_state}
            durable_state = str(durable.get("state") or "running")
            if (
                item.get("status") in {"preparing", "queued", "running", "delegated"}
                and durable_state not in {"running", "queued", "finalizing"}
            ):
                result = durable.get("result")
                if result is None and durable.get("result_json"):
                    try:
                        result = json.loads(durable["result_json"])
                    except (TypeError, json.JSONDecodeError):
                        result = None
                result = result if isinstance(result, dict) else {}
                result_status = str(result.get("status") or durable_state)
                completed = durable_state == "completed" and result_status in {
                    "completed", "success"
                }
                cancelled = durable_state == "cancelled" or result_status == "cancelled"
                terminal_status = (
                    "completed" if completed else "cancelled" if cancelled else "blocked"
                )
                end_reason = (
                    "delegated_completed"
                    if item.get("execution_mode") == "delegated_subagent" and completed
                    else "delegation_cancelled"
                    if item.get("execution_mode") == "delegated_subagent" and cancelled
                    else "delegation_failed"
                    if item.get("execution_mode") == "delegated_subagent"
                    else "role_completed"
                    if completed
                    else "role_cancelled"
                    if cancelled
                    else "role_failed"
                )
                summary = _summary_from_result(result) if result else str(
                    durable.get("state") or "delegation ended"
                )
                patch.update(
                    status=terminal_status,
                    end_reason=end_reason,
                    completed_at=durable.get("completed_at") or time.time(),
                    summary=summary,
                )
                persistent_id = item.get("persistent_session_id")
                if persistent_id and self.session_db is not None:
                    try:
                        self._finalize_session(
                            str(persistent_id), invocation_id, terminal_status, end_reason
                        )
                    except Exception:
                        logger.exception(
                            "could not reconcile terminal persistent role session %s",
                            persistent_id,
                        )
                        patch.update(
                            status="blocked",
                            end_reason="persistence_failure",
                        )
            try:
                store.transition(invocation_id, **patch)
            except Exception:
                logger.debug("role status publication failed", exc_info=True)
        return store.snapshot()
