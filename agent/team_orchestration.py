"""Parent-managed Kanban execution over the existing worker and messaging owners."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid
from typing import Any, Iterable, Mapping, Optional

from agent.subagent_lifecycle import (
    SubagentLaunchRequest,
    SubagentLifecycleError,
    SubagentLifecycleService,
)


TEAM_TOOL_NAME = "kanban_team"
TEAM_CONTRACT_VERSION = "kanban-team-v1"
_UNKNOWN = "Unknown or unavailable team reference."
_TERMINAL_WORKER = {"SUCCEEDED", "FAILED", "INTERRUPTED", "CANCELLED"}
_REVIEW_EVIDENCE_MAX_CHARS = 4_000
_MONITORS: dict[tuple[str, str, int], threading.Event] = {}
_MONITOR_LOCK = threading.RLock()


def _text(value: Any, name: str = "value", *, required: bool = False) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise ValueError(f"{name} is required")
    return result


def _task_id(reference: Any) -> str:
    raw = _text(reference, "task_ref", required=True)
    prefix, sep, value = raw.partition(":")
    if prefix != "task" or not sep or not value:
        raise PermissionError(_UNKNOWN)
    return value


def _typed_targets(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("targets must be a nonempty list of typed references")
    targets = [_text(item, "target", required=True) for item in value]
    if len(targets) != len(set(targets)):
        raise ValueError("targets must not contain duplicates")
    return targets


def validate_team_execution_admission(
    agent: Any, admission: Mapping[str, Any], worker_id: str, run_id: str,
) -> Mapping[str, Any]:
    """Recheck the exact Kanban claim/attachment before a held worker lease."""
    service = TeamOrchestrationService(agent)
    service._require(TEAM_TOOL_NAME, "delegate_task", "kanban_heartbeat")
    scope = service._scope()
    owner = service._owner()
    task_id = _text(admission.get("task_id"), required=True)
    kanban_run_id = int(admission.get("kanban_run_id") or 0)
    claim_lock = _text(admission.get("claim_lock"), required=True)
    reference = admission.get("reference")
    if not isinstance(reference, Mapping):
        raise PermissionError(_UNKNOWN)
    attached_worker, attached_run = service._attachment_ids(reference)
    if attached_worker != worker_id or attached_run != run_id:
        raise PermissionError(_UNKNOWN)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect(board=scope.kanban_board)
    try:
        task = service._authorize_task(kb.get_task(conn, task_id))
        current = kb.get_execution_attachment(conn, task_id, kanban_run_id)
        if (
            task.status != "running" or int(task.current_run_id or 0) != kanban_run_id
            or task.claim_lock != claim_lock or int(task.claim_expires or 0) <= int(time.time())
            or current != dict(reference)
            or reference.get("admission_hash") != service._admission(
                scope, task_id, kanban_run_id, _text(reference.get("role"), required=True)
            )[0]
            or owner != task.session_id
        ):
            raise PermissionError("Team execution admission is stale or mismatched")
        return dict(reference)
    finally:
        conn.close()


class TeamOrchestrationService:
    """Thin coordinator; Kanban, WorkerStore, Bot Mode and rooms keep authority."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.lifecycle = SubagentLifecycleService(lambda: self.agent)

    def _tools(self) -> frozenset[str]:
        exact = getattr(self.agent, "_worker_effective_tool_names", None)
        if exact is None:
            exact = getattr(self.agent, "_executable_tool_names", ())
        return frozenset(exact or ())

    def _require(self, *names: str) -> None:
        missing = [name for name in names if name not in self._tools()]
        if missing:
            raise PermissionError(
                "Team action is unavailable under the current executable tool policy."
            )

    def _guard_action(self, action: str, args: Mapping[str, Any]) -> None:
        """Enforce canonical service authority even for direct/styled dispatch."""
        from tools.kanban_tools import _require_orchestrator_tool
        try:
            _require_orchestrator_tool(TEAM_TOOL_NAME)
        except Exception as exc:
            raise PermissionError(str(exc)) from exc
        required = {
            "create": ("kanban_create",),
            "start": ("delegate_task", "kanban_heartbeat"),
            "guide": (),
            "submit_review": ("kanban_request_review",),
            "accept": ("kanban_complete",),
            "request_changes": ("kanban_request_changes", "delegate_task", "kanban_heartbeat"),
            "cancel": ("delegate_task", "kanban_block"),
            "workflow_save": ("kanban_create",),
            "workflow_list": ("kanban_list",),
            "workflow_inspect": ("kanban_show",),
            "workflow_invoke": ("kanban_create", "delegate_task", "kanban_heartbeat"),
            "workflow_pause": ("kanban_block",),
            "workflow_resume": ("delegate_task", "kanban_heartbeat"),
            "workflow_cancel": ("delegate_task", "kanban_block"),
        }[action]
        self._require(TEAM_TOOL_NAME, *required)
        self._scope()
        self._owner()
        if action == "guide":
            for target in _typed_targets(args.get("targets")):
                kind = target.partition(":")[0]
                if kind == "task":
                    self._require("delegate_task")
                elif kind == "bot":
                    from tools.bot_mode_dm import (
                        MESSAGE_AGENT_TOOL_NAME,
                        message_agent_authorized,
                    )
                    injected = any(
                        isinstance(item, Mapping)
                        and item.get("function", {}).get("name") == MESSAGE_AGENT_TOOL_NAME
                        for item in (getattr(self.agent, "tools", None) or ())
                    )
                    if (
                        not message_agent_authorized(self.agent) or not injected
                        or MESSAGE_AGENT_TOOL_NAME not in getattr(self.agent, "valid_tool_names", set())
                    ):
                        raise PermissionError("Bot guidance is unavailable in this session")
                elif kind != "room":
                    raise PermissionError(_UNKNOWN)

    def _scope(self):
        from agent.shared_discovery import SharedDiscoveryScope, build_local_discovery_scope
        if getattr(self.agent, "_worker_id", None):
            raise PermissionError("Delegated workers cannot control parent-managed Kanban execution.")
        scope = getattr(self.agent, "_shared_discovery_scope", None)
        if not isinstance(scope, SharedDiscoveryScope):
            scope = build_local_discovery_scope()
            self.agent._shared_discovery_scope = scope
        from hermes_constants import get_hermes_home
        from hermes_cli.kanban_db import get_current_board, kanban_db_path
        current_board = str(get_current_board())
        current_path = str(kanban_db_path(board=current_board).resolve())
        if (
            str(Path(get_hermes_home()).resolve()) != scope.profile_home
            or current_board != scope.kanban_board
            or current_path != scope.kanban_db_path
        ):
            raise PermissionError("The session's Kanban scope is stale.")
        return scope

    @contextmanager
    def _board(self):
        scope = self._scope()
        from hermes_cli import kanban_db_connect as kbc
        conn = kbc.connect(board=scope.kanban_board)
        try:
            yield scope, conn
        finally:
            conn.close()

    def _owner(self) -> str:
        from agent.subagent_lifecycle import _owner_session_id_of
        owner = _owner_session_id_of(self.agent)
        if not owner:
            raise PermissionError("Team execution requires a stable parent session.")
        return owner

    def _authorize_task(self, task: Any) -> Any:
        if (
            task is None or task.execution_mode != "parent"
            or not task.session_id or task.session_id != self._owner()
        ):
            raise PermissionError(_UNKNOWN)
        return task

    @staticmethod
    def _digest(parts: Iterable[Any]) -> str:
        encoded = json.dumps(list(parts), ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def _admission(self, scope: Any, task_id: str, kanban_run_id: int, role: str) -> tuple[str, str, str]:
        digest = self._digest(
            (TEAM_CONTRACT_VERSION, self._owner(), scope.kanban_db_path, task_id, int(kanban_run_id), role)
        )
        return digest, f"worker-team-{digest[:32]}", f"team-{digest}"

    @staticmethod
    def _attachment_ids(reference: Mapping[str, Any]) -> tuple[str, str]:
        worker_ref = _text(reference.get("worker_ref"), "worker_ref", required=True)
        run_ref = _text(reference.get("run_ref"), "run_ref", required=True)
        if not worker_ref.startswith("worker:") or not run_ref.startswith("run:"):
            raise PermissionError(_UNKNOWN)
        return worker_ref.partition(":")[2], run_ref.partition(":")[2]

    def _worker_status(self, reference: Mapping[str, Any]) -> Mapping[str, Any]:
        worker_id, run_id = self._attachment_ids(reference)
        return self.lifecycle.control(
            "wait", worker_id=worker_id, run_id=run_id, timeout_seconds=0,
        )

    def _review_evidence(self, reference: Mapping[str, Any]) -> Mapping[str, Any]:
        """Collect a bounded parent-authorized receipt, never a sibling transcript."""
        from agent.redact import redact_sensitive_text
        worker_id, run_id = self._attachment_ids(reference)
        inspected = self.lifecycle.control("inspect", worker_id=worker_id, run_id=run_id)
        run = inspected.get("run") if isinstance(inspected, Mapping) else None
        run = run if isinstance(run, Mapping) else {}
        result = run.get("result") if isinstance(run.get("result"), Mapping) else {}
        raw_summary = str(result.get("summary") or "")
        summary = redact_sensitive_text(raw_summary, force=True)[:_REVIEW_EVIDENCE_MAX_CHARS]
        termination = result.get("termination") if isinstance(result.get("termination"), Mapping) else {}
        termination = {
            key: str(termination.get(key))[:256]
            for key in ("status", "reason") if termination.get(key) is not None
        }
        return {
            "version": "team-review-evidence-v1",
            "status": str(run.get("status") or "UNKNOWN"),
            "summary": summary,
            "summary_truncated": len(raw_summary) > _REVIEW_EVIDENCE_MAX_CHARS,
            "result_hash": str(result.get("result_hash") or "")[:256] or None,
            "termination": termination,
            "available": bool(summary or result.get("result_hash") or termination),
        }

    def _claim(self, conn: Any, task: Any) -> tuple[Any, str, str]:
        from hermes_cli import kanban_db as kb
        task = self._authorize_task(task)
        source = str(task.status)
        if source not in {"ready", "review"}:
            raise ValueError("Parent-managed task must be ready or in review before start")
        prior = kb.latest_run(conn, task.id)
        lock_digest = self._digest(
            (TEAM_CONTRACT_VERSION, self._owner(), task.id, source, prior.id if prior else 0)
        )
        claim_lock = f"team:{lock_digest[:32]}"
        claim = kb.claim_review_task if source == "review" else kb.claim_task
        claimed = claim(
            conn, task.id, claimer=claim_lock, expected_execution_mode="parent",
            expected_workflow_invocation_id=task.workflow_invocation_id,
        )
        if claimed is None:
            raise RuntimeError("Parent execution claim was lost or the task changed")
        return claimed, claim_lock, "reviewer" if source == "review" else "implementer"

    def _execution_plan(self, conn: Any, task: Any, role: str) -> tuple[str, str, Optional[tuple[str, str]]]:
        """Recover durable review/correction intent before worker admission."""
        from hermes_cli import kanban_db as kb
        if role == "reviewer":
            intent = kb.pending_team_intent(conn, task.id, kind="review")
            if not intent:
                raise RuntimeError("Durable review assignment is unavailable")
            if _text(intent.get("reviewer"), required=True) != _text(task.assignee, required=True):
                raise PermissionError("Review assignment changed after the durable handoff")
            evidence = intent.get("implementation") or {}
            receipt = intent.get("evidence") if isinstance(intent.get("evidence"), Mapping) else {}
            evidence_summary = _text(receipt.get("summary")) or "No implementation summary was available."
            goal = (
                f"Review task:{task.id}. Submitted handoff: {_text(intent.get('summary'))}. "
                f"Parent-authorized implementation evidence: status={_text(receipt.get('status'))}; "
                f"summary={evidence_summary} "
                f"Provenance references: {_text(evidence.get('worker_ref'))}, "
                f"{_text(evidence.get('run_ref'))}. These references are provenance only; "
                "do not inspect or control the implementation worker. Report concrete "
                "acceptance evidence or request changes."
            )
            context = (
                "Reviewer assignment for the submitted implementation handoff. Use the "
                "bounded parent-provided evidence and task artifacts; sibling worker access is not granted."
            )
            return goal, context, None
        correction = kb.pending_team_intent(conn, task.id, kind="correction")
        if correction:
            implementation = correction.get("implementation") or {}
            worker_id, previous_run_id = self._attachment_ids(implementation)
            return (
                _text(correction.get("message"), required=True),
                f"Retained correction for task:{task.id}; preserve the prior worker conversation.",
                (worker_id, previous_run_id),
            )
        goal = "\n\n".join(part for part in (_text(task.title), _text(task.body)) if part)
        return goal, f"Parent-managed Kanban task reference: task:{task.id}", None

    def _admit_attach_schedule(
        self, scope: Any, conn: Any, task: Any, claim_lock: str, role: str, *,
        previous: Optional[tuple[str, str]] = None, message: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Mapping[str, Any]:
        from hermes_cli import kanban_db as kb
        run_id = int(task.current_run_id or 0)
        if not run_id:
            raise RuntimeError("Claimed task has no active Kanban run")
        admission_hash, deterministic_worker, request_id = self._admission(
            scope, task.id, run_id, role,
        )
        worker_id = previous[0] if previous else deterministic_worker
        previous_run_id = previous[1] if previous else None
        goal = _text(message) or "\n\n".join(
            part for part in (_text(task.title), _text(task.body)) if part
        )
        request = SubagentLaunchRequest(
            goal=goal,
            context=_text(context) or f"Parent-managed Kanban task reference: task:{task.id}",
            profile=task.assignee,
            role="leaf",
            parent_session_id=self._owner(),
        )
        worker, run = self.lifecycle.admit_team_execution(
            request,
            worker_id=worker_id,
            request_id=request_id,
            previous_run_id=previous_run_id,
            admission_ref=admission_hash,
        )
        reference = {
            "version": TEAM_CONTRACT_VERSION,
            "worker_ref": f"worker:{worker['worker_id']}",
            "run_ref": f"run:{run['run_id']}",
            "admission_hash": admission_hash,
            "role": role,
            "profile": worker.get("profile"),
        }
        kb.attach_execution_reference(
            conn, task.id, run_id, claim_lock=claim_lock, reference=reference,
            owner_session_id=self._owner(),
        )
        # Attachment validation is the final authorization check before the
        # worker lease may be claimed.
        current = kb.get_task(conn, task.id)
        if (
            current is None or current.current_run_id != run_id
            or current.claim_lock != claim_lock or current.execution_mode != "parent"
        ):
            raise PermissionError("Parent execution claim changed before scheduling")
        admission = {
            "task_id": task.id, "kanban_run_id": run_id, "claim_lock": claim_lock,
            "reference": reference,
        }
        scheduled = self.lifecycle.schedule_team_execution(
            worker["worker_id"], run["run_id"], admission=admission,
        )
        self._start_monitor(scope, task.id, run_id, claim_lock, worker["worker_id"], run["run_id"])
        return {
            "task_ref": f"task:{task.id}",
            "kanban_run_id": run_id,
            **reference,
            "worker_status": scheduled.get("status"),
        }

    def _start_monitor(
        self, scope: Any, task_id: str, kanban_run_id: int, claim_lock: str,
        worker_id: str, worker_run_id: str,
    ) -> None:
        key = (self._owner(), task_id, int(kanban_run_id))
        with _MONITOR_LOCK:
            if key in _MONITORS:
                return
            stop = threading.Event()
            _MONITORS[key] = stop

        def monitor() -> None:
            try:
                while not stop.wait(15.0):
                    if not {TEAM_TOOL_NAME, "kanban_heartbeat"}.issubset(self._tools()):
                        return
                    status = self.lifecycle.control(
                        "wait", worker_id=worker_id, run_id=worker_run_id, timeout_seconds=0,
                    )
                    if status.get("status") in _TERMINAL_WORKER:
                        return
                    from hermes_cli import kanban_db as kb
                    from hermes_cli import kanban_db_connect as kbc
                    from hermes_cli import kanban_db_dispatch as kbd
                    if self._scope() != scope:
                        return
                    conn = kbc.connect(board=scope.kanban_board)
                    try:
                        if not kb.heartbeat_claim(conn, task_id, claimer=claim_lock):
                            return
                        if not kbd.heartbeat_worker(
                            conn, task_id, expected_run_id=kanban_run_id,
                        ):
                            return
                    finally:
                        conn.close()
            except Exception:
                return
            finally:
                with _MONITOR_LOCK:
                    _MONITORS.pop(key, None)

        threading.Thread(target=monitor, name=f"hermes-team-{task_id[:12]}", daemon=True).start()

    def _create(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        self._require(TEAM_TOOL_NAME, "kanban_create")
        title = _text(args.get("title"), "title", required=True)
        profile = _text(args.get("profile"), "profile", required=True)
        parent_refs = args.get("parent_refs") or []
        if not isinstance(parent_refs, list):
            raise ValueError("parent_refs must be a list")
        parents = [_task_id(item) for item in parent_refs]
        with self._board() as (scope, conn):
            from hermes_cli import kanban_db as kb
            for parent in parents:
                self._authorize_task(kb.get_task(conn, parent))
            key = _text(args.get("idempotency_key"))
            durable_key = self._digest((self._owner(), key)) if key else None
            task_id = kb.create_task(
                conn,
                title=title,
                body=_text(args.get("body")) or None,
                assignee=profile,
                created_by=scope.profile_name,
                parents=parents,
                idempotency_key=durable_key,
                initial_status="running",
                session_id=self._owner(),
                board=scope.kanban_board,
                execution_mode="parent",
            )
            task = kb.get_task(conn, task_id)
            return {
                "task_ref": f"task:{task_id}",
                "status": task.status if task else "unknown",
                "execution_mode": "parent",
                "parent_refs": [f"task:{item}" for item in parents],
            }

    def _start(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        task_id = _task_id(args.get("task_ref"))
        with self._board() as (scope, conn):
            from hermes_cli import kanban_db as kb
            task = self._authorize_task(kb.get_task(conn, task_id))
            requested_profile = _text(args.get("profile"))
            if requested_profile and requested_profile != task.assignee:
                raise PermissionError("Requested profile does not match the task assignment")
            if task.status == "running" and task.current_run_id:
                attachment = kb.get_execution_attachment(conn, task_id, task.current_run_id)
                if attachment is not None:
                    worker_id, run_id = self._attachment_ids(attachment)
                    admission = {
                        "task_id": task.id, "kanban_run_id": int(task.current_run_id),
                        "claim_lock": task.claim_lock, "reference": attachment,
                    }
                    status = self.lifecycle.schedule_team_execution(
                        worker_id, run_id, admission=admission,
                    )
                    self._start_monitor(
                        scope, task_id, task.current_run_id, task.claim_lock,
                        worker_id, run_id,
                    )
                    return {
                        "task_ref": f"task:{task_id}",
                        "kanban_run_id": task.current_run_id,
                        **attachment,
                        "worker_status": status.get("status"),
                        "recovered": True,
                    }
                role = "reviewer" if kb.run_claim_source(conn, task_id, task.current_run_id) == "review" else "implementer"
                if role == "implementer" and kb.pending_team_intent(conn, task_id, kind="correction"):
                    role = "correction"
                goal, context, previous = self._execution_plan(conn, task, role)
                return self._admit_attach_schedule(
                    scope, conn, task, task.claim_lock, role,
                    previous=previous, message=goal, context=context,
                )
            claimed, claim_lock, role = self._claim(conn, task)
            if role == "implementer" and kb.pending_team_intent(conn, task_id, kind="correction"):
                role = "correction"
            goal, context, previous = self._execution_plan(conn, claimed, role)
            return self._admit_attach_schedule(
                scope, conn, claimed, claim_lock, role,
                previous=previous, message=goal, context=context,
            )

    def _current_attachment(self, conn: Any, task_id: str) -> tuple[Any, int, Mapping[str, Any]]:
        from hermes_cli import kanban_db as kb
        task = kb.get_task(conn, task_id)
        if (
            task is None or task.execution_mode != "parent" or task.status != "running"
            or not task.current_run_id
        ):
            raise PermissionError(_UNKNOWN)
        self._authorize_task(task)
        reference = kb.get_execution_attachment(conn, task_id, task.current_run_id)
        if reference is None:
            raise PermissionError(_UNKNOWN)
        return task, int(task.current_run_id), reference

    def _guide_one(self, target: str, message: str, key: str) -> Mapping[str, Any]:
        kind, sep, _value = target.partition(":")
        if not sep:
            raise PermissionError(_UNKNOWN)
        if kind == "task":
            self._require(TEAM_TOOL_NAME, "delegate_task")
            with self._board() as (_scope, conn):
                task, _kanban_run, attachment = self._current_attachment(conn, _task_id(target))
                worker_id, run_id = self._attachment_ids(attachment)
                status = self.lifecycle.control(
                    "wait", worker_id=worker_id, run_id=run_id, timeout_seconds=0,
                )
                if status.get("status") != "RUNNING":
                    raise RuntimeError("Worker guidance requires the exact attached run to be RUNNING")
                result = self.lifecycle.control("message", worker_id=worker_id, run_id=run_id, message=message)
                return {"target": target, **result, "task_status": task.status}
        if kind == "bot":
            from tools.bot_mode_dm import message_agent_authorized, message_agent_tool
            if not message_agent_authorized(self.agent):
                raise PermissionError("Bot guidance is unavailable in this session")
            raw = message_agent_tool(target=target.partition(":")[2], message=message, agent=self.agent)
            result = json.loads(raw)
            if result.get("error") and "status" not in result:
                result["status"] = "not_delivered"
            return {"target": target, **result}
        if kind == "room":
            scope = self._scope()
            provider = scope.room_provider
            if provider is None:
                raise PermissionError(_UNKNOWN)
            event_id = f"team-{self._digest((self._owner(), target, key, message))}"
            result = provider.send(
                self.agent, target, event_id=event_id,
                payload={"text": message, "thread_id": key},
            )
            return {"target": target, "delivery_id": event_id, **dict(result)}
        raise PermissionError(_UNKNOWN)

    def _guide(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        message = _text(args.get("message"), "message", required=True)
        targets = _typed_targets(args.get("targets"))
        key = _text(args.get("idempotency_key")) or uuid.uuid4().hex
        outcomes = []
        for target in targets:
            try:
                outcomes.append({"status": "accepted", **self._guide_one(target, message, key)})
            except Exception as exc:
                from tui_gateway.session_discovery import RoomDeliveryUncertain
                if isinstance(exc, RoomDeliveryUncertain):
                    outcomes.append({
                        "target": target, "status": "delivery_uncertain",
                        "delivery_id": exc.event_id,
                        "reconciliation_ref": exc.reconciliation_ref,
                        "error": str(exc),
                    })
                    continue
                outcomes.append({"target": target, "error": str(exc), "status": "not_delivered"})
        return {"outcomes": outcomes, "idempotency_key": key}

    def _submit_review(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        task_id = _task_id(args.get("task_ref"))
        summary = _text(args.get("summary"), "summary", required=True)
        reviewer = _text(args.get("reviewer"), "reviewer", required=True)
        with self._board() as (_scope, conn):
            from hermes_cli import kanban_db as kb
            task, kanban_run, attachment = self._current_attachment(conn, task_id)
            if task.workflow_invocation_id:
                from hermes_cli import kanban_db_workflows as workflows
                policy = workflows.correction_policy(
                    conn, task_id, owner_session_id=self._owner(),
                )
                if policy and policy.get("reviewer") and policy["reviewer"] != reviewer:
                    raise PermissionError("Reviewer does not match the immutable workflow step")
            evidence = self._review_evidence(attachment)
            if evidence.get("status") != "SUCCEEDED":
                raise RuntimeError("Worker success is required before requesting review")
            kb.record_team_intent(
                conn, task_id, kanban_run, kind="review", owner_session_id=self._owner(),
                payload={
                    "summary": summary, "reviewer": reviewer,
                    "implementation": dict(attachment), "evidence": dict(evidence),
                },
            )
            ok, reason = kb.request_review(
                conn, task_id, summary=summary, reviewer=reviewer,
                expected_run_id=kanban_run, with_reason=True,
            )
            if not ok:
                raise RuntimeError(reason or "Review transition failed")
            return {"task_ref": f"task:{task_id}", "status": "review", "submitted_run_id": kanban_run}

    def _accept(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        task_id = _task_id(args.get("task_ref"))
        with self._board() as (_scope, conn):
            from hermes_cli import kanban_db as kb
            task, kanban_run, attachment = self._current_attachment(conn, task_id)
            if kb.run_claim_source(conn, task_id, kanban_run) != "review":
                raise RuntimeError("Only the current claimed review run may accept the task")
            evidence = self._worker_status(attachment)
            if evidence.get("status") != "SUCCEEDED":
                raise RuntimeError("Reviewer success is required before acceptance")
            if task.workflow_invocation_id:
                from hermes_cli import kanban_db_workflows as workflows
                workflows.record_acceptance_evidence(
                    conn, task_id, owner_session_id=self._owner(),
                    kanban_run_id=kanban_run,
                    worker_status=str(evidence.get("status")),
                )
            if not kb.complete_task(
                conn, task_id, summary=_text(args.get("summary")) or "Review accepted",
                expected_run_id=kanban_run,
            ):
                raise RuntimeError("Acceptance lost the exact review run fence")
            workflow_completed = False
            if task.workflow_invocation_id:
                from hermes_cli import kanban_db_workflows as workflows
                workflow_completed = workflows.finalize_completed(
                    conn, task.workflow_invocation_id, owner_session_id=self._owner(),
                )
            return {
                "task_ref": f"task:{task_id}", "status": "done",
                "accepted_run_id": kanban_run, "workflow_completed": workflow_completed,
            }

    def _request_changes(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        task_id = _task_id(args.get("task_ref"))
        reason = _text(args.get("message"), "message", required=True)
        with self._board() as (scope, conn):
            from hermes_cli import kanban_db as kb
            task, review_run, reviewer_attachment = self._current_attachment(conn, task_id)
            if kb.run_claim_source(conn, task_id, review_run) != "review":
                raise RuntimeError("Changes require the current claimed review run")
            if self._worker_status(reviewer_attachment).get("status") != "SUCCEEDED":
                raise RuntimeError("Reviewer success is required before requesting changes")
            if task.workflow_invocation_id:
                from hermes_cli import kanban_db_workflows as workflows
                policy = workflows.correction_policy(
                    conn, task_id, owner_session_id=self._owner(),
                )
                if policy is not None and not policy["allowed"]:
                    raise RuntimeError(
                        "Workflow correction limit reached "
                        f"({policy['used']}/{policy['limit']})"
                    )
            implementation = kb.latest_execution_attachment(
                conn, task_id, roles=("implementer", "correction"),
            )
            if implementation is None:
                raise RuntimeError("Original implementation execution is unavailable")
            _implementation_kanban_run, implementation_ref = implementation
            worker_id, previous_run_id = self._attachment_ids(implementation_ref)
            kb.record_team_intent(
                conn, task_id, review_run, kind="correction", owner_session_id=self._owner(),
                payload={
                    "message": reason, "implementation": dict(implementation_ref),
                    "reviewer": dict(reviewer_attachment), "review_run_id": review_run,
                },
            )
            ok, detail = kb.request_changes(
                conn, task_id, reason=reason, expected_run_id=review_run,
            )
            if not ok:
                raise RuntimeError(detail or "Request-changes transition failed")
            task = kb.get_task(conn, task_id)
            if task is None or task.status != "ready":
                return {"task_ref": f"task:{task_id}", "status": task.status if task else "unknown"}
            claimed, claim_lock, _role = self._claim(conn, task)
            goal, context, previous = self._execution_plan(conn, claimed, "correction")
            result = self._admit_attach_schedule(
                scope, conn, claimed, claim_lock, "correction",
                previous=previous or (worker_id, previous_run_id), message=goal, context=context,
            )
            return {**result, "changes_requested_from_run_id": review_run}

    def _cancel(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        task_id = _task_id(args.get("task_ref"))
        timeout = min(60.0, max(0.0, float(args.get("timeout_seconds") or 0)))
        with self._board() as (_scope, conn):
            from hermes_cli import kanban_db as kb
            _task, kanban_run, attachment = self._current_attachment(conn, task_id)
            worker_id, run_id = self._attachment_ids(attachment)
            result = self.lifecycle.control("interrupt", worker_id=worker_id, run_id=run_id)
            status = self.lifecycle.control(
                "wait", worker_id=worker_id, run_id=run_id, timeout_seconds=timeout,
            )
            if status.get("status") not in {"CANCELLED", "INTERRUPTED"}:
                return {
                    "task_ref": f"task:{task_id}", "kanban_run_id": kanban_run,
                    "worker_status": status.get("status"), "interrupt_requested": result.get("interrupt_requested"),
                    "task_disposition": "pending_terminal_worker_evidence",
                }
            if not kb.block_task(
                conn, task_id, reason="Parent-managed execution cancelled after terminal worker evidence.",
                expected_run_id=kanban_run,
            ):
                raise RuntimeError("Cancellation lost the exact Kanban run fence")
            kb.record_execution_cancelled(
                conn, task_id, kanban_run, worker_status=str(status.get("status")),
            )
            return {
                "task_ref": f"task:{task_id}", "status": "blocked",
                "kanban_run_id": kanban_run, "worker_status": status.get("status"),
            }

    def dispatch(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        action = _text(arguments.get("action"), "action", required=True).lower()
        handlers = {
            "create": self._create,
            "start": self._start,
            "guide": self._guide,
            "submit_review": self._submit_review,
            "accept": self._accept,
            "request_changes": self._request_changes,
            "cancel": self._cancel,
        }
        handler = handlers.get(action)
        from agent.workflow_orchestration import WORKFLOW_ACTIONS
        if handler is None and action not in WORKFLOW_ACTIONS:
            return {"error": f"Unsupported team action '{action}'."}
        try:
            self._guard_action(action, arguments)
            if action in WORKFLOW_ACTIONS:
                from agent.workflow_orchestration import WorkflowOrchestrationService
                payload = WorkflowOrchestrationService(self).dispatch(action, arguments)
            else:
                payload = handler(arguments)
            return {
                **payload,
                "team_service": TEAM_CONTRACT_VERSION,
                "action": action,
            }
        except (PermissionError, RuntimeError, SubagentLifecycleError, ValueError) as exc:
            return {
                "error": str(exc),
                "team_service": TEAM_CONTRACT_VERSION,
                "action": action,
            }
