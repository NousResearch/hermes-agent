"""Turn-end guard for kanban workers, which must end with a terminal board tool that hands
the card to whoever owns it next (``kanban_complete``, ``kanban_block``,
``kanban_request_review``, ``kanban_request_changes``). Some models narrate the next step
and stop with no tool calls; Hermes treats that as a clean exit → ``rc=0`` → dispatcher
``protocol_violation``. Policy-only: return a bounded synthetic nudge so the loop continues
instead of exiting.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Optional

from agent.delegation_context import owned_kanban_task


# Every tool that ends this worker's responsibility for the card, not just the two that
# close it out: ``kanban_request_review`` moves it to ``review`` (goals.py's continuation /
# finalize prompts tell builders to call it) and ``kanban_request_changes`` returns it to
# ``ready`` (the sdlc-review skill tells reviewers to). Nudging after either asks a worker
# that did the right thing to ``kanban_complete`` a card it must not close.
_TERMINAL_KANBAN_TOOLS = frozenset({
    "kanban_complete",
    "kanban_block",
    "kanban_request_review",
    "kanban_request_changes",
})

_DEFAULT_MAX_ATTEMPTS = 2
_MAX_REVIEW_SUMMARY_CHARS = 4000


def _auto_review_on_stop_enabled() -> bool:
    """Return whether unverified worker prose may enter the review lane.

    A missing terminal Kanban action is normally a protocol violation, not
    evidence that a reviewer can use.  Retrying the original assignee keeps
    that worker accountable for its receipt and prevents unrelated review
    workers from becoming a queue sink.  Deployments that deliberately want
    the old evidence-preservation behavior can opt in per worker or config.
    """

    raw = os.environ.get("HERMES_KANBAN_AUTO_REVIEW_ON_STOP")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    try:
        from hermes_cli.config import load_config

        kanban = load_config().get("kanban") or {}
        return kanban.get("auto_review_on_stop") is True
    except Exception:
        return False


def _configured_review_profile() -> str | None:
    """Return an installed independent reviewer configured for stop handoffs."""

    configured = (os.environ.get("HERMES_KANBAN_REVIEWER_PROFILE") or "").strip()
    if not configured:
        try:
            from hermes_cli.config import load_config

            kanban = load_config().get("kanban") or {}
            configured = str(kanban.get("reviewer_profile") or "").strip()
        except Exception:
            return None
    if not configured:
        return None
    try:
        from hermes_constants import get_default_hermes_root

        profile_dir = get_default_hermes_root() / "profiles" / configured
        return configured if profile_dir.is_dir() else None
    except Exception:
        return None


def kanban_stop_nudge_enabled() -> bool:
    """On when ``HERMES_KANBAN_TASK`` is set for the dispatcher-owned worker, unless
    ``HERMES_KANBAN_STOP_NUDGE`` disables it. In-process delegate_task children and cron runs
    inherit the env var but own no board task and carry no kanban toolset."""
    if (os.environ.get("HERMES_KANBAN_STOP_NUDGE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(owned_kanban_task())


def kanban_shutdown_drain_requested() -> bool:
    """Return whether this worker was asked to pause at a turn boundary."""
    if not (os.environ.get("HERMES_KANBAN_TASK") or "").strip():
        return False
    try:
        from hermes_cli import kanban_db

        return kanban_db.shutdown_drain_requested()
    except Exception:
        # A missing/unreadable control path must not turn a normal worker
        # response into a shutdown protocol failure. The dispatcher reclaim
        # remains the bounded fallback when the cooperative path is unavailable.
        return False


def pause_current_kanban_run(*, reason: str = "dispatcher shutdown") -> bool:
    """Release this worker's active run so the card can be resumed later."""
    task_id = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    raw_run_id = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    claim_lock = (os.environ.get("HERMES_KANBAN_CLAIM_LOCK") or "").strip()
    if not task_id or not raw_run_id or not claim_lock:
        return False
    try:
        run_id = int(raw_run_id)
        from hermes_cli.kanban_db_connect import connect
        from hermes_cli.kanban_db_recovery import pause_task

        conn = connect()
        try:
            return pause_task(
                conn,
                task_id,
                expected_run_id=run_id,
                claimer=claim_lock,
                reason=reason,
            )
        finally:
            conn.close()
    except Exception:
        return False


def _tool_call_name(tc: Any) -> str:
    """Tool name from a dict or object tool call (``function.name`` first, then ``name``)."""
    if isinstance(tc, dict):
        fn = tc.get("function")
        return str((fn.get("name") if isinstance(fn, dict) else tc.get("name")) or "")
    fn = getattr(tc, "function", None)
    return str((getattr(fn, "name", "") if fn is not None else getattr(tc, "name", "")) or "")


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True if this conversation has a successful terminal Kanban result."""
    review_call_ids: set[str] = set()
    review_tools = {"kanban_request_review", "kanban_request_changes"}
    for msg in filter(lambda m: isinstance(m, dict), messages or ()):
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                name = _tool_call_name(tc)
                if name in {"kanban_complete", "kanban_block"}:
                    return True
                if name in review_tools:
                    call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if call_id:
                        review_call_ids.add(str(call_id))
        elif role == "tool":
            name = str(msg.get("name") or "")
            if name in {"kanban_complete", "kanban_block"}:
                return True
            if name in review_tools and (
                not review_call_ids or str(msg.get("tool_call_id") or "") in review_call_ids
            ):
                try:
                    payload = json.loads(msg.get("content") or "")
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict) and payload.get("ok") is True:
                    return True
    return False


def successful_kanban_terminal_transition(
    *,
    messages: Iterable[dict] | None,
    tool_calls: Iterable[Any] | None,
) -> bool:
    """Return whether this worker's current tool batch durably transitioned it.

    The conversation loop calls this only after the executor has persisted
    every tool-result row.  Match the current batch by tool-call id and require
    the canonical Kanban ``{"ok": true}`` response; merely attempting a
    terminal tool (or receiving an error) must not stop the worker before it
    can correct the handoff.
    """
    if not kanban_stop_nudge_enabled():
        return False
    try:
        from agent.delegation_context import is_dispatcher_owned_worker_context

        if not is_dispatcher_owned_worker_context():
            return False
    except Exception:
        return False

    terminal_ids: set[str] = set()
    for tool_call in tool_calls or []:
        if _tool_call_name(tool_call) not in _TERMINAL_KANBAN_TOOLS:
            continue
        if isinstance(tool_call, dict):
            call_id = tool_call.get("id") or tool_call.get("tool_call_id")
        else:
            call_id = getattr(tool_call, "id", None)
        if call_id:
            terminal_ids.add(str(call_id))
    if not terminal_ids:
        return False

    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        if str(message.get("tool_call_id") or "") not in terminal_ids:
            continue
        if str(message.get("name") or message.get("tool_name") or "") not in (
            _TERMINAL_KANBAN_TOOLS
        ):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and payload.get("ok") is True:
            return True
    return False


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a terminal tool; ``None`` when
    the guard should not fire (not a kanban worker, already completed/blocked, budget exhausted)."""
    if (
        not kanban_stop_nudge_enabled()
        or attempts >= max_attempts
        or session_called_kanban_terminal(messages)
    ):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    # The transcript is the status source: this text is only reached when the session made no
    # handoff call, so it never tells a worker to close a card it already sent to review.
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` has not been handed off: this session made no terminal board "
        "call (`kanban_complete` / `kanban_request_review` / `kanban_block`). Ending now "
        "causes a protocol violation (clean exit with the card still `running`).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work is done "
        "and needs no review, `kanban_request_review(summary=...)` if it is a code "
        "change that needs same-card review, OR `kanban_block(reason=...)` if you are "
        "blocked. Reviewers approve with `kanban_complete` or send the card back with "
        "`kanban_request_changes(reason=...)`.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


def reconcile_kanban_stop_to_review(
    *, messages: Iterable[dict] | None, final_response: Any,
    attempts: int, max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> bool:
    """Preserve bounded final evidence through the configured independent review gate."""
    if (not kanban_stop_nudge_enabled() or not _auto_review_on_stop_enabled()
            or attempts < max_attempts or session_called_kanban_terminal(messages)):
        return False
    response = str(final_response or "").strip()
    reviewer = _configured_review_profile()
    if not response or reviewer is None:
        return False
    try:
        from tools.kanban_tools import _handle_request_review
        raw = _handle_request_review({
            "summary": f"Automatic terminal handoff after {attempts} unanswered Kanban stop "
                       f"nudges. Worker final output:\n\n{response[:4000]}",
            "reviewer": reviewer,
            "metadata": {"source": "kanban_stop_guard", "terminal_nudges": attempts,
                         "completion_inferred": False},
        }, _reviewer_already_validated=True)
        payload = json.loads(raw) if isinstance(raw, str) else raw
        return isinstance(payload, dict) and payload.get("ok") is True
    except Exception:
        return False


__all__ = ["build_kanban_stop_nudge", "kanban_stop_nudge_enabled", "session_called_kanban_terminal"]
