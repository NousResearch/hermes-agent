"""Saved-workflow control over the existing Kanban and team services."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping


WORKFLOW_SERVICE_VERSION = "workflow-orchestration-v1"
WORKFLOW_ACTIONS = frozenset({
    "workflow_save",
    "workflow_list",
    "workflow_inspect",
    "workflow_invoke",
    "workflow_pause",
    "workflow_resume",
    "workflow_cancel",
})
_TERMINAL_WORKER = frozenset({"SUCCEEDED", "FAILED", "INTERRUPTED", "CANCELLED"})
_UNKNOWN = "Unknown or unavailable workflow reference."


def _text(value: Any, name: str, *, required: bool = False) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise ValueError(f"{name} is required")
    return result


def _version(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("expected_version must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError("expected_version must be a positive integer") from None
    if parsed < 1:
        raise ValueError("expected_version must be a positive integer")
    return parsed


class WorkflowOrchestrationService:
    """Thin workflow adapter; execution stays in ``TeamOrchestrationService``."""

    def __init__(self, team_service: Any) -> None:
        self.team = team_service

    @contextmanager
    def _readonly_board(self):
        """Open existing state without creating a board or running migrations."""
        from hermes_cli import kanban_db_connect as kbc

        scope = self.team._scope()
        owner = self.team._owner()
        conn = kbc.connect_existing_readonly(db_path=Path(scope.kanban_db_path))
        try:
            yield owner, conn
        finally:
            if conn is not None:
                conn.close()

    def _save(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        from hermes_cli import kanban_db_workflows as workflows

        definition = args.get("definition")
        if not isinstance(definition, Mapping):
            raise ValueError("definition must be an object")
        template_id = None
        if args.get("template_ref") is not None:
            template_id, _version_number = workflows.parse_template_ref(args.get("template_ref"))
        with self.team._board() as (scope, conn):
            return workflows.save_template(
                conn,
                definition,
                created_by=scope.profile_name,
                template_id=template_id,
            )

    def _list(self, _args: Mapping[str, Any]) -> Mapping[str, Any]:
        from hermes_cli import kanban_db_workflows as workflows

        with self._readonly_board() as (owner, conn):
            if conn is None or not workflows.workflow_schema_present(conn):
                return {"templates": [], "invocations": [], "schema_present": False}
            return {**workflows.list_visible(conn, owner_session_id=owner), "schema_present": True}

    def _inspect(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        from hermes_cli import kanban_db_workflows as workflows

        template_ref = _text(args.get("template_ref"), "template_ref")
        workflow_ref = _text(args.get("workflow_ref"), "workflow_ref")
        if bool(template_ref) == bool(workflow_ref):
            raise ValueError("provide exactly one of template_ref or workflow_ref")
        with self._readonly_board() as (owner, conn):
            if conn is None or not workflows.workflow_schema_present(conn):
                raise PermissionError(_UNKNOWN)
            if template_ref:
                template_id, version = workflows.parse_template_ref(template_ref)
                return workflows.template_detail(conn, template_id, version)
            invocation_id = workflows.parse_invocation_ref(workflow_ref)
            return workflows.invocation_detail(conn, invocation_id, owner_session_id=owner)

    def _advance(self, workflow_ref: str) -> Mapping[str, Any]:
        """Recover held runs and start currently claimable steps once each."""
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_workflows as workflows

        invocation_id = workflows.parse_invocation_ref(workflow_ref)
        owner = self.team._owner()
        with self.team._board() as (_scope, conn):
            detail = workflows.invocation_detail(conn, invocation_id, owner_session_id=owner)
            if detail["control_state"] != "active":
                return {"workflow": detail, "outcomes": []}
            rows = conn.execute(
                "SELECT id,status,current_run_id FROM tasks "
                "WHERE workflow_invocation_id=? AND current_step_key!='__coordinator__' "
                "ORDER BY created_at,id",
                (invocation_id,),
            ).fetchall()
            candidates: list[tuple[str, str, int, Mapping[str, Any] | None]] = []
            for row in rows:
                status = str(row["status"])
                if status not in {"ready", "review", "running"}:
                    continue
                run_id = int(row["current_run_id"] or 0)
                attachment = kb.get_execution_attachment(conn, row["id"], run_id) if run_id else None
                candidates.append((str(row["id"]), status, run_id, attachment))

        outcomes: list[dict[str, Any]] = []
        for task_id, status, _run_id, attachment in candidates:
            if status == "running" and attachment is not None:
                worker_id, run_id = self.team._attachment_ids(attachment)
                observed = self.team.lifecycle.control(
                    "wait", worker_id=worker_id, run_id=run_id, timeout_seconds=0,
                )
                worker_status = str(observed.get("status") or "UNKNOWN")
                if worker_status != "PENDING":
                    outcomes.append({
                        "task_ref": f"task:{task_id}",
                        "status": "observed",
                        "worker_status": worker_status,
                        **dict(attachment),
                    })
                    continue
            result = self.team.dispatch({"action": "start", "task_ref": f"task:{task_id}"})
            outcomes.append(dict(result))

        with self.team._board() as (_scope, conn):
            workflows.finalize_completed(conn, invocation_id, owner_session_id=owner)
            detail = workflows.invocation_detail(conn, invocation_id, owner_session_id=owner)
        return {"workflow": detail, "outcomes": outcomes}

    def _invoke(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        from hermes_cli import kanban_db_workflows as workflows

        template_ref = _text(args.get("template_ref"), "template_ref", required=True)
        admission_key = _text(args.get("admission_key"), "admission_key", required=True)
        input_payload = {} if args.get("input") is None else args.get("input")
        if not isinstance(input_payload, Mapping):
            raise ValueError("input must be an object")
        with self.team._board() as (scope, conn):
            admitted = workflows.invoke_workflow(
                conn,
                template_ref,
                owner_session_id=self.team._owner(),
                admission_key=admission_key,
                input_payload=input_payload,
                created_by=scope.profile_name,
                board=scope.kanban_board,
            )
        return {**admitted, "advancement": self._advance(admitted["workflow_ref"])}

    def _pause(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        from hermes_cli import kanban_db_workflows as workflows

        invocation_id = workflows.parse_invocation_ref(args.get("workflow_ref"))
        with self.team._board() as (_scope, conn):
            return workflows.set_control(
                conn,
                invocation_id,
                owner_session_id=self.team._owner(),
                expected_version=_version(args.get("expected_version")),
                action="pause",
            )

    def _resume(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        from hermes_cli import kanban_db_workflows as workflows

        workflow_ref = _text(args.get("workflow_ref"), "workflow_ref", required=True)
        invocation_id = workflows.parse_invocation_ref(workflow_ref)
        expected = _version(args.get("expected_version"))
        with self.team._board() as (_scope, conn):
            current = workflows.invocation_detail(
                conn, invocation_id, owner_session_id=self.team._owner(),
            )
            if current["control_state"] == "paused":
                workflows.set_control(
                    conn,
                    invocation_id,
                    owner_session_id=self.team._owner(),
                    expected_version=expected,
                    action="resume",
                )
            elif current["control_state"] != "active" or current["control_version"] != expected:
                raise RuntimeError("Workflow control version or state changed")
        return self._advance(workflow_ref)

    def _cancel(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        from hermes_cli import kanban_db_workflows as workflows

        workflow_ref = _text(args.get("workflow_ref"), "workflow_ref", required=True)
        invocation_id = workflows.parse_invocation_ref(workflow_ref)
        expected = _version(args.get("expected_version"))
        timeout = min(60.0, max(0.0, float(args.get("timeout_seconds") or 0)))
        owner = self.team._owner()
        with self.team._board() as (_scope, conn):
            workflows.begin_cancellation(
                conn,
                invocation_id,
                owner_session_id=owner,
                expected_version=expected,
            )
            targets = workflows.cancellation_targets(
                conn, invocation_id, owner_session_id=owner,
            )

        outcomes: list[dict[str, Any]] = []
        for target in targets:
            worker_id, run_id = self.team._attachment_ids(target)
            outcome = dict(target)
            interrupt_attempted = False
            try:
                observed = self.team.lifecycle.control(
                    "wait", worker_id=worker_id, run_id=run_id, timeout_seconds=0,
                )
                worker_status = str(observed.get("status") or "UNKNOWN")
                if worker_status not in _TERMINAL_WORKER:
                    with self.team._board() as (_scope, conn):
                        interrupt_attempted = workflows.begin_cancel_interrupt(
                            conn,
                            invocation_id,
                            owner_session_id=owner,
                            task_id=target["task_ref"].partition(":")[2],
                            kanban_run_id=int(target["kanban_run_id"]),
                        )
                    if interrupt_attempted:
                        requested = self.team.lifecycle.control(
                            "interrupt", worker_id=worker_id, run_id=run_id,
                        )
                        outcome["interrupt_requested"] = requested.get("interrupt_requested")
                    observed = self.team.lifecycle.control(
                        "wait", worker_id=worker_id, run_id=run_id, timeout_seconds=timeout,
                    )
                    worker_status = str(observed.get("status") or "UNKNOWN")
                outcome["worker_status"] = worker_status
                if worker_status in _TERMINAL_WORKER:
                    with self.team._board() as (_scope, conn):
                        workflows.record_cancel_terminal(
                            conn,
                            invocation_id,
                            owner_session_id=owner,
                            task_id=target["task_ref"].partition(":")[2],
                            kanban_run_id=int(target["kanban_run_id"]),
                            worker_status=worker_status,
                        )
            except Exception as exc:
                outcome["status"] = "effect_uncertain" if interrupt_attempted else "pending"
                outcome["error"] = str(exc)
            outcomes.append(outcome)

        finalized = False
        with self.team._board() as (_scope, conn):
            try:
                detail = workflows.finalize_cancelled(
                    conn, invocation_id, owner_session_id=owner,
                )
                finalized = detail["control_state"] == "cancelled"
            except RuntimeError:
                detail = workflows.invocation_detail(conn, invocation_id, owner_session_id=owner)
        return {
            "workflow": detail,
            "outcomes": outcomes,
            "task_disposition": "cancelled" if finalized else "pending_terminal_worker_evidence",
        }

    def dispatch(self, action: str, args: Mapping[str, Any]) -> Mapping[str, Any]:
        handlers = {
            "workflow_save": self._save,
            "workflow_list": self._list,
            "workflow_inspect": self._inspect,
            "workflow_invoke": self._invoke,
            "workflow_pause": self._pause,
            "workflow_resume": self._resume,
            "workflow_cancel": self._cancel,
        }
        handler = handlers.get(action)
        if handler is None:
            raise ValueError(f"Unsupported workflow action '{action}'.")
        return {**handler(args), "workflow_service": WORKFLOW_SERVICE_VERSION}
