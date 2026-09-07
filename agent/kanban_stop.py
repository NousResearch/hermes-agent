"""Turn-end guard for kanban workers.

Kanban workers must end with a terminal board transition. Models
(especially GLM / Qwen families) sometimes narrate the next step
("Let me write the report now") and stop with ``finish_reason=stop`` and no
tool calls. Hermes treats that as a clean exit → ``rc=0`` → dispatcher
``protocol_violation``.

This module is policy-only: when a kanban worker tries to finish without a
terminal board tool, return a bounded synthetic nudge so the conversation
loop continues instead of exiting.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Optional


_TERMINAL_KANBAN_TOOLS = frozenset(
    {
        "kanban_complete",
        "kanban_block",
        "kanban_request_review",
        "kanban_request_changes",
    }
)

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
    """Return whether the kanban stop-guard is active for this process.

    On when ``HERMES_KANBAN_TASK`` is set (dispatcher-spawned worker), unless
    ``HERMES_KANBAN_STOP_NUDGE`` explicitly disables it.
    """
    env = os.environ.get("HERMES_KANBAN_STOP_NUDGE")
    if env is not None and env.strip().lower() in {"0", "false", "no", "off"}:
        return False
    task = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    return bool(task)


def _tool_call_name(tc: Any) -> str:
    if isinstance(tc, dict):
        fn = tc.get("function")
        if isinstance(fn, dict):
            return str(fn.get("name") or "")
        return str(tc.get("name") or "")
    fn = getattr(tc, "function", None)
    if fn is not None:
        return str(getattr(fn, "name", "") or "")
    return str(getattr(tc, "name", "") or "")


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True if this conversation already invoked a terminal kanban tool."""
    if not messages:
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                if _tool_call_name(tc) in _TERMINAL_KANBAN_TOOLS:
                    return True
        elif role == "tool":
            name = str(msg.get("name") or "")
            if name in _TERMINAL_KANBAN_TOOLS:
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
    """Return a synthetic follow-up when a kanban worker exits without a terminal tool.

    Returns ``None`` when the guard should not fire (not a kanban worker,
    already completed, blocked, or handed off across review, or nudge budget
    exhausted).
    """
    if not kanban_stop_nudge_enabled():
        return None
    if attempts >= max_attempts:
        return None
    if session_called_kanban_terminal(messages):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` is still `running`. Ending now without a board tool "
        "causes a protocol violation (clean exit with no "
        "`kanban_complete` / `kanban_block`).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work "
        "is done, OR `kanban_block(reason=...)` if you are blocked.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


def reconcile_kanban_stop_to_review(
    *,
    messages: Iterable[dict] | None,
    final_response: Any,
    attempts: int,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> bool:
    """Safely terminate a narrated worker after its nudge budget is spent.

    The stop guard already gave the model a bounded same-session opportunity
    to call a terminal board tool.  If it still returns useful text, route
    that text through the *existing* ``kanban_request_review`` handler instead
    of letting the process exit cleanly while its card remains ``running``.

    This deliberately requests review rather than completing the task: the
    final prose is evidence for a reviewer, never authority to infer success.
    The normal handler retains task/run ownership checks, redaction, and the
    goal-mode acceptance judge.  Any rejection or exception leaves lifecycle
    ownership with the dispatcher and returns ``False``.
    """
    if not kanban_stop_nudge_enabled():
        return False
    if attempts < max_attempts:
        return False
    if session_called_kanban_terminal(messages):
        return False
    if not _auto_review_on_stop_enabled():
        return False
    response_text = str(final_response or "").strip()
    if not response_text:
        return False

    bounded = response_text[:_MAX_REVIEW_SUMMARY_CHARS]
    reviewer = _configured_review_profile()
    if reviewer is None:
        # A reviewerless handoff is not autonomous: the same implementer gets
        # claimed from the review lane and tends to block while asking for a
        # human verdict. Leave lifecycle ownership with the dispatcher so the
        # attempt is recorded as a protocol violation and can follow normal
        # bounded retry/watchdog policy instead.
        return False
    summary = (
        f"Automatic terminal handoff after {attempts} unanswered Kanban stop "
        f"nudges. Worker final output:\n\n{bounded}"
    )
    try:
        from tools.kanban_tools import _handle_request_review

        raw = _handle_request_review(
            {
                "summary": summary,
                "reviewer": reviewer,
                "metadata": {
                    "source": "kanban_stop_guard",
                    "terminal_nudges": attempts,
                    "completion_inferred": False,
                },
            }
        )
        payload = json.loads(raw) if isinstance(raw, str) else raw
        return bool(isinstance(payload, dict) and payload.get("ok") is True)
    except Exception:
        return False


__all__ = [
    "build_kanban_stop_nudge",
    "kanban_stop_nudge_enabled",
    "reconcile_kanban_stop_to_review",
    "session_called_kanban_terminal",
    "successful_kanban_terminal_transition",
]
