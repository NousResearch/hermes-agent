"""Direct-control / background-job helpers for ``GatewayRunner``.

These methods power QQ NapCat oral admin shortcuts, durable background jobs,
and auto-background long-running work. They were extracted so the merge with
upstream main can keep the modular service modules wired without bloating
``gateway/run.py`` further.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from gateway.background_jobs import (
    BackgroundJobStore,
    background_job_chat_key,
    background_job_scope_key,
    launch_background_worker,
    stop_background_worker,
)
from gateway.config import Platform
from gateway.direct_control_router import DirectControlRouter
from gateway.direct_shortcut_runtime_service import get_direct_control_router
from gateway.direct_shortcut_trace_runtime_service import (
    build_direct_shortcut_runtime_summary as shared_build_direct_shortcut_runtime_summary,
)
from gateway.group_archive_runtime_service import (
    build_group_archive_runtime_summary as shared_build_group_archive_runtime_summary,
)
from gateway.group_monitoring_runtime_service import (
    build_group_monitoring_summary as shared_build_group_monitoring_summary,
)
from gateway.platforms.base import EphemeralReply, MessageEvent
from gateway.runtime_status_service import (
    _safe_float,
    render_status_command as shared_render_status_command,
)
from gateway.session import SessionSource
from gateway.whatsapp_identity import (
    expand_whatsapp_aliases as _expand_whatsapp_auth_aliases,
    normalize_whatsapp_identifier as _normalize_whatsapp_identifier,
)

logger = logging.getLogger("gateway.run")


def _coerce_id_list(value: Any) -> list[str]:
    """Normalize env/config user-id lists into stripped unique strings."""
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw_items = [str(item).strip() for item in value]
    else:
        raw_items = [str(value).strip()]
    return [item for item in raw_items if item]


class GatewayDirectOpsMixin:
    """Admin identity, direct shortcuts, and background-job helpers."""

    def _configured_admin_user_ids(self, platform: Optional[Platform]) -> list[str]:
        """Return configured administrator user IDs for a platform."""
        admin_ids: list[str] = []
        seen: set[str] = set()

        def _add(values) -> None:
            for value in values:
                normalized = str(value or "").strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                admin_ids.append(normalized)

        _add(_coerce_id_list(os.getenv("GATEWAY_ADMIN_USERS")))

        if platform:
            _add(_coerce_id_list(os.getenv(f"{platform.value.upper()}_ADMIN_USERS")))
            platform_cfg = (
                self.config.platforms.get(platform) if getattr(self, "config", None) else None
            )
            extra = getattr(platform_cfg, "extra", None) if platform_cfg else None
            if isinstance(extra, dict):
                _add(_coerce_id_list(extra.get("admin_users")))

        return admin_ids

    def _source_identity_candidates(self, source: SessionSource) -> set[str]:
        """Return normalized sender identifiers for auth/admin matching."""
        candidates: set[str] = set()
        for raw in (source.user_id, getattr(source, "user_id_alt", None)):
            value = str(raw or "").strip()
            if not value:
                continue
            candidates.add(value)
            if "@" in value:
                candidates.add(value.split("@", 1)[0])

        if source.platform == Platform.WHATSAPP:
            expanded: set[str] = set()
            for candidate in list(candidates):
                expanded.update(_expand_whatsapp_auth_aliases(candidate))
                normalized = _normalize_whatsapp_identifier(candidate)
                if normalized:
                    expanded.add(normalized)
            candidates.update(expanded)

        return candidates

    def _is_admin_user(self, source: SessionSource) -> bool:
        """Return True when the source user is explicitly configured as an admin."""
        admin_ids = set(self._configured_admin_user_ids(source.platform))
        if not admin_ids:
            return False
        return bool(self._source_identity_candidates(source) & admin_ids)

    def _admin_only_message(self, source: SessionSource | None, action: str) -> Optional[str]:
        """Return an admin-only rejection message when the user is not an admin."""
        if source is None:
            return None
        admin_ids = self._configured_admin_user_ids(source.platform)
        if not admin_ids or self._is_admin_user(source):
            return None

        if source.platform == Platform.QQ_NAPCAT:
            ids_text = "、".join(admin_ids)
            return f"这事得董事长拍板。当前只有 QQ {ids_text} 能授权这类操作。"

        noun = "user ID" if len(admin_ids) == 1 else "user IDs"
        ids_text = ", ".join(admin_ids)
        return f"Only administrator {noun} {ids_text} can {action}."

    def _get_auto_background_work(self, platform: Optional[Platform] = None) -> bool:
        """Return whether obvious long-running work should detach to background."""
        config = getattr(self, "config", None)
        if config and hasattr(config, "get_auto_background_work"):
            return bool(config.get_auto_background_work(platform))
        return False

    def _get_busy_input_mode(self, platform: Optional[Platform] = None) -> str:
        """Return how follow-up text should behave while the agent is active."""
        allowed = {"interrupt", "queue", "smart", "steer"}

        def _normalize(value: Any, default: str = "interrupt") -> str:
            if isinstance(value, str):
                mode = value.strip().lower()
                if mode in allowed:
                    return mode
            return default

        config = getattr(self, "config", None)
        # Resolve via the concrete class so MagicMock configs (unit tests)
        # cannot shadow runner._busy_input_mode with auto-created attrs.
        getter = getattr(type(config), "get_busy_input_mode", None) if config is not None else None
        if callable(getter):
            try:
                mode = getter(config, platform)
                if isinstance(mode, str) and mode.strip().lower() in allowed:
                    return mode.strip().lower()
            except Exception:
                pass
        return _normalize(getattr(self, "_busy_input_mode", None), "interrupt")

    def _busy_followup_force_queue_reason(
        self,
        session_key: str,
        running_agent: Any,
    ) -> str:
        """Return a reason when the active run must not be interrupted."""
        try:
            from tools.approval import has_blocking_approval

            if has_blocking_approval(session_key):
                return "approval_pending"
        except Exception:
            pass
        try:
            store = getattr(self, "_background_job_store", None)
            if store is not None and hasattr(store, "has_pending_approval_requests"):
                if store.has_pending_approval_requests(session_key):
                    return "approval_pending"
        except Exception:
            pass

        if running_agent is None:
            return ""
        # Avoid importing the run module sentinel here — treat unknown agents
        # with no activity summary as interruptible.
        if not hasattr(running_agent, "get_activity_summary"):
            return ""

        try:
            activity = running_agent.get_activity_summary()
        except Exception:
            return ""
        if not isinstance(activity, dict):
            return ""

        current_tool = str(activity.get("current_tool") or "").strip()
        if current_tool in {"qq_group_moderation"}:
            return f"critical_tool:{current_tool}"
        return ""

    def _get_direct_control_router(self) -> DirectControlRouter:
        return get_direct_control_router(self, router_cls=DirectControlRouter)

    @staticmethod
    def _looks_like_background_status_query(message_text: str) -> bool:
        from gateway.qq_intents import _looks_like_qq_background_status_query

        return _looks_like_qq_background_status_query(message_text)

    @staticmethod
    def _looks_like_runtime_status_query(message_text: str) -> bool:
        from gateway.qq_intents import (
            _looks_like_qq_runtime_short_query,
            _looks_like_qq_runtime_status_query,
        )

        return _looks_like_qq_runtime_status_query(message_text) or _looks_like_qq_runtime_short_query(
            message_text
        )

    @staticmethod
    def _looks_like_joined_group_list_query(message_text: str) -> bool:
        from gateway.qq_intents import _looks_like_qq_joined_group_list_query

        return _looks_like_qq_joined_group_list_query(message_text)

    @staticmethod
    def _looks_like_group_runtime_status_query(message_text: str) -> bool:
        from gateway.group_control_intents import looks_like_group_runtime_status_query

        return looks_like_group_runtime_status_query(message_text)

    @staticmethod
    def _looks_like_qq_group_listen_disable_request(message_text: str) -> bool:
        from gateway.group_control_intents import looks_like_group_listen_disable_request

        return looks_like_group_listen_disable_request(message_text)

    @staticmethod
    def _looks_like_qq_group_listen_enable_request(message_text: str) -> bool:
        from gateway.group_control_intents import looks_like_group_listen_enable_request

        return looks_like_group_listen_enable_request(message_text)

    @staticmethod
    def _format_intel_worker_status_label(status: str) -> str:
        return {
            "awaiting_group_approval": "等待入群通过",
            "active_collecting": "正在潜伏采集",
            "paused": "已暂停",
            "stopped": "已停止",
            "failed": "任务失联",
            "rejected": "已拒绝",
        }.get(str(status or "").strip().lower(), str(status or "").strip() or "unknown")

    def _ensure_background_job_state(self) -> None:
        """Initialize background-job runtime state for tests and older runner state."""
        if not hasattr(self, "_background_tasks"):
            self._background_tasks = set()
        if not hasattr(self, "_background_job_store") or self._background_job_store is None:
            self._background_job_store = BackgroundJobStore()

    def _get_background_job_store(self) -> BackgroundJobStore:
        self._ensure_background_job_state()
        return self._background_job_store

    def _background_job_chat_key(self, source: SessionSource) -> str:
        """Return a stable chat-scoped key for managed background jobs."""
        return background_job_chat_key(source)

    def _background_job_scope_key(
        self,
        source: SessionSource,
        *,
        session_key: str = "",
    ) -> str:
        """Return the session-scoped key used to isolate background jobs."""
        resolved = str(session_key or "").strip()
        if not resolved:
            try:
                resolved = str(self._session_key_for_source(source) or "").strip()
            except Exception:
                resolved = ""
        return background_job_scope_key(source, session_key=resolved)

    def _background_jobs_for_source(
        self,
        source: SessionSource,
        *,
        active_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return managed background jobs associated with the source chat."""
        chat_key = self._background_job_chat_key(source)
        scope_key = self._background_job_scope_key(source)
        try:
            durable_jobs = self._get_background_job_store().list_jobs(
                chat_key=chat_key,
                scope_key=scope_key,
                active_only=active_only,
            )
        except Exception:
            durable_jobs = []
        return sorted(
            durable_jobs,
            key=lambda item: _safe_float(item.get("created_at"), 0.0),
        )

    def _launch_background_worker(self, task_id: str) -> Dict[str, Any]:
        """Launch an external worker for a durable background job."""
        return launch_background_worker(task_id=task_id)

    def _stop_background_worker(self, job: Dict[str, Any]) -> bool:
        """Stop an external worker for a durable background job."""
        return stop_background_worker(job)

    def _format_background_job_age(self, job: Dict[str, Any]) -> str:
        """Return a short human-readable elapsed time for a background job."""
        started_at = _safe_float(job.get("started_at"), 0.0)
        created_at = _safe_float(job.get("created_at"), time.time())
        finished_at = _safe_float(job.get("finished_at"), time.time())
        anchor = (
            finished_at
            if job.get("status") in {"completed", "failed", "cancelled"}
            else time.time()
        )
        base = started_at or created_at
        elapsed = max(0, int(anchor - base))
        if elapsed >= 3600:
            return f"{elapsed // 3600}h{(elapsed % 3600) // 60:02d}m"
        if elapsed >= 60:
            return f"{elapsed // 60}m{elapsed % 60:02d}s"
        return f"{elapsed}s"

    @staticmethod
    def _runtime_session_metadata(session_key: str) -> dict[str, str]:
        parts = str(session_key or "").split(":")
        return {
            "platform": parts[2] if len(parts) > 2 else "",
            "chat_type": parts[3] if len(parts) > 3 else "",
            "chat_id": parts[4] if len(parts) > 4 else "",
        }

    def _build_runtime_model_summary(self) -> dict[str, Any]:
        from gateway.run import (
            _AGENT_PENDING_SENTINEL,
            _resolve_gateway_model,
            _resolve_runtime_agent_kwargs,
        )

        configured_model = str(_resolve_gateway_model() or "").strip()
        configured_base_url = ""
        try:
            configured_runtime = _resolve_runtime_agent_kwargs() or {}
            configured_provider = str(configured_runtime.get("provider") or "").strip()
            configured_base_url = str(configured_runtime.get("base_url") or "").strip()
        except Exception:
            configured_provider = ""
        active_model = str(getattr(self, "_effective_model", None) or configured_model).strip()
        active_provider = str(
            getattr(self, "_effective_provider", None) or configured_provider
        ).strip()
        fallback_pinned = False

        candidate_agents: list[Any] = []
        for agent_ref in getattr(self, "_running_agents", {}).values():
            if agent_ref not in (None, _AGENT_PENDING_SENTINEL):
                candidate_agents.append(agent_ref)
        for cached in getattr(self, "_agent_cache", {}).values():
            agent_ref = cached[0] if isinstance(cached, tuple) and cached else cached
            if agent_ref in (None, _AGENT_PENDING_SENTINEL):
                continue
            if agent_ref not in candidate_agents:
                candidate_agents.append(agent_ref)

        for agent_ref in candidate_agents:
            raw_model = getattr(agent_ref, "model", "")
            raw_provider = getattr(agent_ref, "provider", "")
            model = raw_model.strip() if isinstance(raw_model, str) else ""
            provider = raw_provider.strip() if isinstance(raw_provider, str) else ""
            if model and (not active_model or model != configured_model):
                active_model = model
            if provider and (model == active_model or not active_provider):
                active_provider = provider
            has_pinned_fallback = getattr(agent_ref, "_has_pinned_fallback", None)
            if callable(has_pinned_fallback):
                try:
                    if has_pinned_fallback():
                        fallback_pinned = True
                        if model:
                            active_model = model
                        if provider:
                            active_provider = provider
                except Exception:
                    logger.debug("Could not evaluate fallback pin state", exc_info=True)

        fallback_active = bool(
            active_model and configured_model and active_model != configured_model
        )
        degraded_runtime_count = 0
        degraded_runtimes: list[dict[str, Any]] = []
        primary_degraded = False
        primary_degraded_reason = ""
        primary_degraded_cooldown_seconds = 0
        try:
            from run_agent import get_provider_health_snapshot, _runtime_targets_match

            degraded_snapshot = get_provider_health_snapshot(limit=5)
            degraded_runtime_count = int(degraded_snapshot.get("count") or 0)
            degraded_runtimes = list(degraded_snapshot.get("runtimes") or [])
            for runtime in degraded_runtimes:
                if _runtime_targets_match(
                    configured_provider,
                    configured_model,
                    configured_base_url,
                    runtime.get("provider"),
                    runtime.get("model"),
                    runtime.get("base_url"),
                ):
                    primary_degraded = True
                    primary_degraded_reason = str(runtime.get("reason") or "").strip()
                    primary_degraded_cooldown_seconds = int(
                        max(0.0, float(runtime.get("cooldown_seconds") or 0.0))
                    )
                    break
        except Exception:
            logger.debug("Could not load provider health snapshot", exc_info=True)
        return {
            "configured_model": configured_model,
            "configured_provider": configured_provider,
            "configured_base_url": configured_base_url,
            "active_model": active_model or configured_model,
            "active_provider": active_provider,
            "fallback_active": fallback_active,
            "fallback_pinned": fallback_pinned,
            "primary_degraded": primary_degraded,
            "primary_degraded_reason": primary_degraded_reason,
            "primary_degraded_cooldown_seconds": primary_degraded_cooldown_seconds,
            "degraded_runtime_count": degraded_runtime_count,
            "degraded_runtimes": degraded_runtimes,
        }

    def _build_runtime_approval_summary(self) -> dict[str, Any]:
        store = self._get_background_job_store()
        try:
            pending_count = store.count_all_pending_approval_requests()
        except Exception:
            pending_count = 0

        live_sessions: set[str] = set(
            str(session_key or "").strip()
            for session_key in getattr(self, "_pending_approvals", {})
            if str(session_key or "").strip()
        )
        pending_sessions = set(live_sessions)
        live_sessions.update(
            str(session_key or "").strip()
            for session_key in getattr(self, "_running_agents", {})
            if str(session_key or "").strip()
        )
        for session_key in pending_sessions:
            try:
                already_counted = store.has_pending_approval_requests(session_key)
            except Exception:
                already_counted = False
            if not already_counted:
                pending_count += 1
        try:
            from tools.approval import has_blocking_approval

            for session_key in live_sessions:
                if not has_blocking_approval(session_key):
                    continue
                try:
                    already_counted = store.has_pending_approval_requests(session_key)
                except Exception:
                    already_counted = False
                if not already_counted:
                    pending_count += 1
        except Exception:
            pass

        return {"pending_count": int(max(pending_count, 0))}

    def _build_runtime_group_monitoring_summary(self) -> dict[str, Any]:
        # Prefer gateway.run re-exports so tests can patch them there.
        from gateway import run as gateway_run

        specs_fn = getattr(
            gateway_run,
            "build_group_monitoring_runtime_platform_specs",
            None,
        )
        if not callable(specs_fn):
            from gateway.group_runtime_platform_specs import (
                build_group_monitoring_runtime_platform_specs as specs_fn,
            )
        return shared_build_group_monitoring_summary(platform_specs=specs_fn())

    def _build_runtime_group_archive_summary(self) -> dict[str, Any]:
        from gateway import run as gateway_run

        specs_fn = getattr(
            gateway_run,
            "build_group_archive_runtime_platform_specs",
            None,
        )
        if not callable(specs_fn):
            from gateway.group_runtime_platform_specs import (
                build_group_archive_runtime_platform_specs as specs_fn,
            )
        return shared_build_group_archive_runtime_summary(platform_specs=specs_fn())

    def _build_runtime_direct_shortcut_summary(self) -> dict[str, Any]:
        return shared_build_direct_shortcut_runtime_summary(self)

    def _resolve_background_job_for_stop(
        self,
        source: SessionSource,
        raw_task_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Resolve a background job for /stop within the current chat."""
        active_jobs = self._background_jobs_for_source(source, active_only=True)
        if not active_jobs:
            return None

        task_id = str(raw_task_id or "").strip()
        if task_id:
            for job in active_jobs:
                if job.get("task_id") == task_id or str(job.get("task_id", "")).startswith(
                    task_id
                ):
                    return job
            return None

        if len(active_jobs) == 1:
            return active_jobs[0]
        return {"ambiguous": True, "jobs": active_jobs}

    async def _handle_status_command(self, event: MessageEvent) -> str:
        """Handle /status: upstream cockpit fields + QQ runtime extras."""
        import time as _time

        from gateway.run import _AGENT_PENDING_SENTINEL
        from gateway.runtime_status_service import _runtime_foreground_line
        from gateway.slash_commands import GatewaySlashCommandsMixin

        # Prefer the mainline slash-command status for tokens/model/context/
        # Matrix scope so tests and cockpit UX stay aligned with upstream.
        base = await GatewaySlashCommandsMixin._handle_status_command(self, event)

        # Inject live foreground activity detail (tool/iteration) that the
        # mainline cockpit omits but QQ ops status expects.
        try:
            if hasattr(self, "async_session_store"):
                session_entry = await self.async_session_store.get_or_create_session(
                    event.source
                )
            else:
                session_entry = self.session_store.get_or_create_session(event.source)
            session_key = session_entry.session_key
            foreground = _runtime_foreground_line(
                self,
                session_key=session_key,
                pending_sentinel=_AGENT_PENDING_SENTINEL,
                now_ts=_time.time(),
            )
            if foreground and "**Foreground:**" not in base:
                # Place after the Agent Running line when present.
                marker = "**Agent Running:**"
                if marker in base:
                    lines = base.splitlines()
                    out: list[str] = []
                    inserted = False
                    for line in lines:
                        out.append(line)
                        if not inserted and line.startswith(marker):
                            out.append(f"**Foreground:** {foreground}")
                            inserted = True
                    base = "\n".join(out)
                else:
                    base = f"{base.rstrip()}\n**Foreground:** {foreground}"
        except Exception:
            logger.debug("status foreground injection failed", exc_info=True)

        extras = await shared_render_status_command(
            self,
            event,
            pending_sentinel=_AGENT_PENDING_SENTINEL,
            extras_only=True,
        )
        if extras:
            return f"{base.rstrip()}\n{extras}"
        return base

    async def _handle_stop_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /stop for foreground agents, thread siblings, and background jobs."""
        from gateway.run import _AGENT_PENDING_SENTINEL, _INTERRUPT_REASON_STOP
        from agent.i18n import t

        source = event.source
        # Prefer async session store when available (mainline path).
        if hasattr(self, "async_session_store"):
            session_entry = await self.async_session_store.get_or_create_session(source)
        else:
            session_entry = self.session_store.get_or_create_session(source)
        session_key = session_entry.session_key

        agent = self._running_agents.get(session_key)
        if agent is _AGENT_PENDING_SENTINEL:
            if hasattr(self, "_interrupt_and_clear_session"):
                await self._interrupt_and_clear_session(
                    session_key,
                    source,
                    interrupt_reason=_INTERRUPT_REASON_STOP,
                    invalidation_reason="stop_command_pending",
                )
            elif session_key in self._running_agents:
                del self._running_agents[session_key]
            logger.info("STOP (pending) for session %s — sentinel cleared", session_key)
            return EphemeralReply(t("gateway.stop.stopped_pending"))
        if agent:
            if hasattr(self, "_interrupt_and_clear_session"):
                await self._interrupt_and_clear_session(
                    session_key,
                    source,
                    interrupt_reason=_INTERRUPT_REASON_STOP,
                    invalidation_reason="stop_command_handler",
                )
            elif session_key in self._running_agents:
                try:
                    agent.interrupt("Stop requested")
                except Exception:
                    pass
                del self._running_agents[session_key]
            return EphemeralReply(t("gateway.stop.stopped"))

        # No run under the caller's own session key. In per-user threads,
        # another participant's run lives under a sibling key — authorized
        # users should still be able to /stop it (#bernard-thread-stop).
        sibling_keys_fn = getattr(self, "_sibling_thread_run_keys", None)
        sibling_keys = (
            sibling_keys_fn(source, session_key) if callable(sibling_keys_fn) else []
        )
        is_authorized = getattr(self, "_is_user_authorized", None)
        if sibling_keys and callable(is_authorized) and is_authorized(source):
            for sibling_key in sibling_keys:
                if hasattr(self, "_interrupt_and_clear_session"):
                    await self._interrupt_and_clear_session(
                        sibling_key,
                        source,
                        interrupt_reason=_INTERRUPT_REASON_STOP,
                        invalidation_reason="stop_command_thread_sibling",
                    )
            logger.info(
                "STOP (thread sibling) by %s — interrupted %d run(s) in thread: %s",
                session_key,
                len(sibling_keys),
                ", ".join(sibling_keys),
            )
            return EphemeralReply(t("gateway.stop.stopped"))

        raw_job_id = event.get_command_args().strip()
        job = self._resolve_background_job_for_stop(source, raw_job_id)
        if isinstance(job, dict) and job.get("ambiguous"):
            active_ids = ", ".join(f"`{item['task_id']}`" for item in job.get("jobs", []))
            return f"Multiple background jobs are running here. Use `/stop <task_id>`: {active_ids}"
        if job:
            task_id = str(job.get("task_id") or "")
            try:
                self._stop_background_worker(job)
                job = (
                    self._get_background_job_store().mark_job_cancelled(
                        task_id,
                        reason="stop requested",
                    )
                    or job
                )
            except Exception as exc:
                logger.warning("Failed to stop external background job %s: %s", task_id, exc)
                job = self._get_background_job_store().mark_job_cancelling(task_id) or job
            return f"⏹️ Requested stop for background job `{task_id}`."

        # No running agent/job. Best-effort clear of stuck platform status
        # indicators (e.g. Slack assistant.threads.setStatus) so /stop always
        # dismisses a phantom "is thinking..." (#32295).
        adapter = getattr(self, "adapters", {}).get(source.platform)
        if adapter and hasattr(adapter, "_stop_typing_with_metadata"):
            try:
                thread_meta_fn = getattr(self, "_thread_metadata_for_source", None)
                reply_anchor_fn = getattr(self, "_reply_anchor_for_event", None)
                metadata = None
                if callable(thread_meta_fn):
                    reply_anchor = (
                        reply_anchor_fn(event) if callable(reply_anchor_fn) else None
                    )
                    metadata = thread_meta_fn(source, reply_anchor)
                await adapter._stop_typing_with_metadata(source.chat_id, metadata)
            except Exception:
                logger.debug(
                    "Failed to clear typing on /stop with no active agent",
                    exc_info=True,
                )

        return t("gateway.stop.no_active")

    def _maybe_auto_background_turn(
        self,
        *,
        event: Any,
        source: SessionSource,
        context: Any,
        session_key: str,
        history: list[dict[str, Any]] | None,
        context_prompt: str,
        session_id: str = "",
        logger: Any = None,
    ) -> str | None:
        """Detach long-running work to a durable background job when enabled."""
        from gateway.auto_background_runtime_service import (
            format_auto_background_ack,
            resolve_auto_background_dispatch,
        )
        from gateway.background_job_start_service import start_background_job
        from gateway.employee_routes import get_employee_routes
        from gateway.shared_group_history_runtime_service import prepare_history_for_agent

        history_for_agent = prepare_history_for_agent(
            list(history or []),
            shared_session_kind=getattr(context, "shared_session_kind", None),
            session_id=str(session_id or ""),
            logger=logger,
            visible_limit=20,
        )

        background_message_text = str(getattr(event, "text", "") or "")
        background_dispatch = resolve_auto_background_dispatch(
            event,
            background_message_text,
            auto_background_work_enabled=self._get_auto_background_work(
                getattr(source, "platform", None)
            ),
            employee_routes=get_employee_routes(
                self.config,
                platform=getattr(source, "platform", Platform.QQ_NAPCAT),
            ),
            conversation_history=list(history_for_agent or []),
        )
        if not background_dispatch:
            return None

        task_id = start_background_job(
            store=self._get_background_job_store(),
            launch_worker=self._launch_background_worker,
            prompt=background_message_text,
            source=source,
            conversation_history=list(history_for_agent or []),
            context_prompt=context_prompt,
            session_key=session_key,
            job_kind="auto",
            worker_name=str(background_dispatch.get("worker_name") or ""),
            preloaded_skills=list(background_dispatch.get("preloaded_skills") or []),
            admin_user_ids=list(getattr(context, "admin_user_ids", None) or []),
            is_admin_user=getattr(context, "is_admin_user", None),
            logger=logger,
        )
        return format_auto_background_ack(
            background_message_text,
            task_id,
            worker_name=str(background_dispatch.get("worker_name") or ""),
        )
