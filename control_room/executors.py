"""Default executors for Control Room actions (CR-202..CR-205).

Every executor wraps ONE authoritative existing operation. If the underlying
capability is not available in this runtime (plugin absent, gateway RPC not
reachable, capability not wired), the executor returns a typed ``unavailable``
result — never a fake success and never a dead control.

Contract notes:
- Approval allow/deny: routes to the existing session-scoped ``approval.respond``
  RPC path. No second approval state machine exists here.
- Process kill: routes to ``process_registry.kill_process`` with session
  ownership semantics preserved by the registry.
- Subagent/delegation: routes to the gateway RPC methods where present.
- Peer messaging: routes through the PUBLIC Hermes Peer API only. Never
  ``hermes_peer.plugin._manager``.
- Kanban: routes through the established locked write route. No raw SQLite
  writes from Control Room.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .actions import ExecutorFn
from .contract import (
    ActionTarget,
    ControlRoomActionResult,
    ErrorCode,
)


def _unavailable(kind: str, detail: str) -> ControlRoomActionResult:
    return ControlRoomActionResult(
        status="unavailable",
        message=f"{kind} unavailable: {detail}",
        receipt={"error": ErrorCode.UNAUTHORIZED.value, "detail": detail},
    )


# ---------------------------------------------------------------------------
# Approvals (CR-202)
# ---------------------------------------------------------------------------


def approval_executor_factory(respond_fn=None) -> ExecutorFn:
    """Build an approval executor over the existing respond path.

    ``respond_fn`` is the authoritative allow/deny callable
    ``(request_id, decision) -> dict``. When omitted, the executor lazily
    imports the gateway RPC-style handler; if unavailable it returns
    ``unavailable`` — a renderer must show no approval control in that case.
    """

    def executor(target: ActionTarget, params: Dict[str, Any], context: Dict[str, Any]) -> ControlRoomActionResult:
        decision = params.get("decision")
        if decision not in ("allow", "deny"):
            return ControlRoomActionResult(
                status="failed",
                message="decision must be 'allow' or 'deny'",
                receipt={"error": ErrorCode.UNKNOWN_ACTION.value},
            )
        request_id = target.id

        responder = respond_fn or _default_approval_responder
        if responder is None:
            return _unavailable("approval", "no approval respond path in this runtime")

        try:
            receipt = responder(request_id, decision, context=context)
        except Exception as exc:  # noqa: BLE001
            return ControlRoomActionResult(
                status="failed",
                message=f"approval respond failed: {exc}",
                receipt={"error": ErrorCode.BACKEND_FAILED.value},
            )

        if isinstance(receipt, dict) and receipt.get("ok") is False:
            return ControlRoomActionResult(
                status="failed",
                message=str(receipt.get("error") or "approval respond returned failure"),
                receipt=receipt,
            )
        return ControlRoomActionResult(
            status="completed",
            message=f"approval {request_id} {decision}d",
            receipt={"request_id": request_id, "decision": decision, "result": receipt},
        )

    return executor


def _default_approval_responder(request_id: str, decision: str, context: Dict[str, Any]) -> Optional[dict]:
    """Default approval responder.

    Prefer the gateway RPC handler when it can be imported; otherwise report
    the capability as unavailable rather than faking a success.
    """
    try:
        from tui_gateway.methods_prompt import _respond_approval  # type: ignore

        return _respond_approval(request_id, decision)
    except Exception:  # noqa: BLE001
        return {"ok": False, "error": "approval.respond not reachable from this runtime"}


# ---------------------------------------------------------------------------
# Process control (CR-203)
# ---------------------------------------------------------------------------


def process_kill_executor() -> ExecutorFn:
    """Kill an owned background process through the process registry."""

    def executor(target: ActionTarget, params: Dict[str, Any], context: Dict[str, Any]) -> ControlRoomActionResult:
        try:
            from tools.process_registry import process_registry
        except Exception as exc:  # noqa: BLE001
            return _unavailable("process.kill", str(exc))

        session_id = target.id
        try:
            result = process_registry.kill_process(session_id)
        except Exception as exc:  # noqa: BLE001
            return ControlRoomActionResult(
                status="failed",
                message=f"kill failed: {exc}",
                receipt={"error": ErrorCode.BACKEND_FAILED.value},
            )

        if isinstance(result, dict) and result.get("status") == "not_found":
            return ControlRoomActionResult(
                status="failed",
                message=f"no process with id {session_id}",
                receipt={"error": ErrorCode.UNKNOWN_TARGET.value, "result": result},
            )
        return ControlRoomActionResult(
            status="completed",
            message=f"process {session_id} killed",
            receipt={"session_id": session_id, "result": result},
        )

    return executor


def subagent_interrupt_executor() -> ExecutorFn:
    """Interrupt a subagent through the gateway RPC path (if present)."""

    def executor(target: ActionTarget, params: Dict[str, Any], context: Dict[str, Any]) -> ControlRoomActionResult:
        try:
            from tui_gateway.methods_session import _subagent_interrupt  # type: ignore

            receipt = _subagent_interrupt(target.id, context.get("session_id"))
        except Exception as exc:  # noqa: BLE001
            return _unavailable("subagent.interrupt", str(exc))
        return ControlRoomActionResult(
            status="completed",
            message=f"subagent {target.id} interrupted",
            receipt={"subagent_id": target.id, "result": receipt},
        )

    return executor


def subagent_steer_executor() -> ExecutorFn:
    """Steer a subagent through the gateway RPC path (if present)."""

    def executor(target: ActionTarget, params: Dict[str, Any], context: Dict[str, Any]) -> ControlRoomActionResult:
        text = str(params.get("text") or "")
        if not text:
            return ControlRoomActionResult(
                status="failed",
                message="steer requires 'text' parameter",
                receipt={"error": ErrorCode.UNKNOWN_ACTION.value},
            )
        try:
            from tui_gateway.methods_session import _subagent_steer  # type: ignore

            receipt = _subagent_steer(target.id, text, context.get("session_id"))
        except Exception as exc:  # noqa: BLE001
            return _unavailable("subagent.steer", str(exc))
        return ControlRoomActionResult(
            status="completed",
            message=f"subagent {target.id} steered",
            receipt={"subagent_id": target.id, "result": receipt},
        )

    return executor


def delegation_pause_executor() -> ExecutorFn:
    """Pause new delegation spawning through the gateway RPC path (if present)."""

    def executor(target: ActionTarget, params: Dict[str, Any], context: Dict[str, Any]) -> ControlRoomActionResult:
        pause = bool(params.get("pause", True))
        try:
            from tui_gateway.methods_session import _delegation_pause  # type: ignore

            receipt = _delegation_pause(target.id, pause=pause)
        except Exception as exc:  # noqa: BLE001
            return _unavailable("delegation.pause", str(exc))
        return ControlRoomActionResult(
            status="completed",
            message=f"delegation {target.id} {'paused' if pause else 'resumed'}",
            receipt={"delegation_id": target.id, "pause": pause, "result": receipt},
        )

    return executor


# ---------------------------------------------------------------------------
# Hermes Peer (CR-204) — public API only
# ---------------------------------------------------------------------------


def _peer_tools_module():
    try:
        import hermes_peer.tools as peer_tools  # type: ignore

        return peer_tools
    except Exception as exc:  # noqa: BLE001
        return None


def peer_send_executor() -> ExecutorFn:
    """Send a peer message through the public Hermes Peer API."""

    def executor(target: ActionTarget, params: Dict[str, Any], context: Dict[str, Any]) -> ControlRoomActionResult:
        tools = _peer_tools_module()
        if tools is None:
            return _unavailable("peer.send", "hermes_peer plugin not available")
        text = str(params.get("message") or "")
        if not text:
            return ControlRoomActionResult(
                status="failed",
                message="peer.send requires 'message' parameter",
                receipt={"error": ErrorCode.UNKNOWN_ACTION.value},
            )
        try:
            raw = tools.peer_send_message({"target": target.id, "message": text})
        except Exception as exc:  # noqa: BLE001
            return ControlRoomActionResult(
                status="failed",
                message=f"peer.send failed: {exc}",
                receipt={"error": ErrorCode.BACKEND_FAILED.value},
            )
        try:
            receipt = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:  # noqa: BLE001
            receipt = {"raw": raw}
        if isinstance(receipt, dict) and receipt.get("error"):
            return ControlRoomActionResult(
                status="failed",
                message=str(receipt["error"]),
                receipt={"error": ErrorCode.BACKEND_FAILED.value, "result": receipt},
            )
        return ControlRoomActionResult(
            status="completed",
            message=f"message sent to {target.id}",
            receipt={"target": target.id, "result": receipt},
        )

    return executor


def peer_inbox_action_executor() -> ExecutorFn:
    """Release or refuse a held peer message via the public inbox API."""

    def executor(target: ActionTarget, params: Dict[str, Any], context: Dict[str, Any]) -> ControlRoomActionResult:
        tools = _peer_tools_module()
        if tools is None:
            return _unavailable("peer.inbox", "hermes_peer plugin not available")
        action = params.get("action")
        if action not in ("release", "refuse"):
            return ControlRoomActionResult(
                status="failed",
                message="peer.inbox action must be 'release' or 'refuse'",
                receipt={"error": ErrorCode.UNKNOWN_ACTION.value},
            )
        try:
            raw = tools.peer_read_inbox({"action": action, "message_id": target.id})
        except Exception as exc:  # noqa: BLE001
            return ControlRoomActionResult(
                status="failed",
                message=f"peer.inbox {action} failed: {exc}",
                receipt={"error": ErrorCode.BACKEND_FAILED.value},
            )
        try:
            receipt = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:  # noqa: BLE001
            receipt = {"raw": raw}
        if isinstance(receipt, dict) and receipt.get("error"):
            return ControlRoomActionResult(
                status="failed",
                message=str(receipt["error"]),
                receipt={"error": ErrorCode.BACKEND_FAILED.value, "result": receipt},
            )
        return ControlRoomActionResult(
            status="completed",
            message=f"peer message {target.id} {action}d",
            receipt={"message_id": target.id, "action": action, "result": receipt},
        )

    return executor


# ---------------------------------------------------------------------------
# Kanban (CR-205) — locked write route only
# ---------------------------------------------------------------------------


def kanban_create_executor() -> ExecutorFn:
    """Create a Kanban task through the established locked write route."""

    def executor(target: ActionTarget, params: Dict[str, Any], context: Dict[str, Any]) -> ControlRoomActionResult:
        try:
            from hermes_cli import kanban_db

            with kanban_db.connect_closing() as conn:
                task = kanban_db.create_task(
                    conn,
                    title=str(params.get("title") or ""),
                    body=str(params.get("body") or "") or None,
                    assignee=str(params.get("assignee") or "") or None,
                    created_by=str(params.get("created_by") or "") or None,
                    tenant=str(params.get("tenant") or "") or None,
                    priority=int(params.get("priority") or 0),
                    initial_status=str(params.get("initial_status") or "running"),
                )
        except Exception as exc:  # noqa: BLE001
            return ControlRoomActionResult(
                status="failed",
                message=f"kanban create failed: {exc}",
                receipt={"error": ErrorCode.BACKEND_FAILED.value},
            )
        return ControlRoomActionResult(
            status="completed",
            message=f"task {getattr(task, 'id', '?')} created",
            receipt={"task_id": getattr(task, "id", None), "result": task.model_dump() if hasattr(task, "model_dump") else str(task)},
        )

    return executor


def kanban_comment_executor() -> ExecutorFn:
    """Add a comment to a task through the locked write route."""

    def executor(target: ActionTarget, params: Dict[str, Any], context: Dict[str, Any]) -> ControlRoomActionResult:
        try:
            from hermes_cli import kanban_db

            with kanban_db.connect_closing() as conn:
                task = kanban_db.get_task(conn, target.id)
                if task is None:
                    return ControlRoomActionResult(
                        status="failed",
                        message=f"task {target.id} not found",
                        receipt={"error": ErrorCode.UNKNOWN_TARGET.value},
                    )
                receipt = kanban_db.add_comment(
                    conn,
                    task_id=target.id,
                    author=str(params.get("author") or "control-room"),
                    body=str(params.get("text") or params.get("body") or ""),
                )
        except Exception as exc:  # noqa: BLE001
            return ControlRoomActionResult(
                status="failed",
                message=f"kanban comment failed: {exc}",
                receipt={"error": ErrorCode.BACKEND_FAILED.value},
            )
        return ControlRoomActionResult(
            status="completed",
            message=f"comment added to task {target.id}",
            receipt={"task_id": target.id, "result": receipt},
        )

    return executor


def kanban_move_executor() -> ExecutorFn:
    """Move a task through a VALIDATED state transition (CR-205).

    Only explicit validated transitions exist — never a generic status
    setter, and never a generic "retry" button. Supported transitions:
    ``block`` (block_task), ``unblock`` (unblock_task), ``complete``
    (complete_task). Anything else is rejected.
    """

    def executor(target: ActionTarget, params: Dict[str, Any], context: Dict[str, Any]) -> ControlRoomActionResult:
        transition = str(params.get("transition") or "")
        if transition not in ("block", "unblock", "complete"):
            return ControlRoomActionResult(
                status="failed",
                message="kanban move transition must be one of: block, unblock, complete",
                receipt={"error": ErrorCode.UNKNOWN_ACTION.value},
            )
        try:
            from hermes_cli import kanban_db

            with kanban_db.connect_closing() as conn:
                task = kanban_db.get_task(conn, target.id)
                if task is None:
                    return ControlRoomActionResult(
                        status="failed",
                        message=f"task {target.id} not found",
                        receipt={"error": ErrorCode.UNKNOWN_TARGET.value},
                    )
                if transition == "block":
                    reason = str(params.get("reason") or "blocked via Control Room")
                    receipt = kanban_db.block_task(conn, target.id, reason=reason)
                elif transition == "unblock":
                    receipt = kanban_db.unblock_task(conn, target.id)
                else:  # complete
                    receipt = kanban_db.complete_task(conn, target.id)
        except Exception as exc:  # noqa: BLE001
            return ControlRoomActionResult(
                status="failed",
                message=f"kanban move failed: {exc}",
                receipt={"error": ErrorCode.BACKEND_FAILED.value},
            )
        if isinstance(receipt, tuple) and receipt and receipt[0] is False:
            return ControlRoomActionResult(
                status="failed",
                message=f"task {target.id} cannot be {transition}ed (validated transition refused)",
                receipt={"error": ErrorCode.BACKEND_FAILED.value, "result": receipt},
            )
        return ControlRoomActionResult(
            status="completed",
            message=f"task {target.id} {transition}ed",
            receipt={"task_id": target.id, "transition": transition, "result": receipt},
        )

    return executor


def agent_run_executor() -> ExecutorFn:
    """Start a new agent run via the surface's background/delegation route.

    The authoritative runner is surface-bound (the CLI's ``/background``
    handler, a gateway delegation dispatch, etc.). The executing surface
    injects it as ``context["agent_run_runner"]`` — a callable
    ``(prompt, profile) -> dict``. Without a runner the action is typed
    unavailable, never a fake success.
    """

    def executor(target: ActionTarget, params: Dict[str, Any], context: Dict[str, Any]) -> ControlRoomActionResult:
        runner = context.get("agent_run_runner")
        if not callable(runner):
            return _unavailable("agent_run", "no background/delegation runner in this runtime")
        prompt = str(params.get("prompt") or "")
        if not prompt:
            return ControlRoomActionResult(
                status="failed",
                message="agent_run requires 'prompt' parameter",
                receipt={"error": ErrorCode.UNKNOWN_ACTION.value},
            )
        try:
            receipt = runner(prompt, str(params.get("profile") or context.get("profile") or "default"))
        except Exception as exc:  # noqa: BLE001
            return ControlRoomActionResult(
                status="failed",
                message=f"agent_run failed: {exc}",
                receipt={"error": ErrorCode.BACKEND_FAILED.value},
            )
        return ControlRoomActionResult(
            status="completed",
            message="agent run started",
            receipt={"result": receipt},
        )

    return executor


def default_executors() -> Dict[str, ExecutorFn]:
    """Wire every executor Control Room advertises. Each degrades to typed
    unavailable when its authoritative backend is absent."""
    return {
        "approval": approval_executor_factory(),
        "process": process_kill_executor(),
        "subagent_interrupt": subagent_interrupt_executor(),
        "subagent_steer": subagent_steer_executor(),
        "delegation": delegation_pause_executor(),
        "peer_send": peer_send_executor(),
        "peer_inbox": peer_inbox_action_executor(),
        "kanban_create": kanban_create_executor(),
        "kanban_comment": kanban_comment_executor(),
        "kanban_move": kanban_move_executor(),
        "agent_run": agent_run_executor(),
    }
