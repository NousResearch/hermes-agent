"""``tool_decision`` plugin hook: one structured allow/refuse record per tool call.

Gates that resolve *inside* a call (approval prompts, smart approval, allowlists, ``approvals.deny``,
hardline floors) :func:`note` their verdict against the call's correlation ids (the approval
observability context). The terminal ``post_tool_call`` funnel
(``model_tools._emit_post_tool_call_hook``), which runs exactly once per call on every dispatch path,
calls :func:`emit_for_call` just before ``post_tool_call``: it folds that note with the call's own
blocked status and fires ``tool_decision``. Core keeps no sink; arguments travel only as a sha256 digest.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

HOOK = "tool_decision"

# Refusals outrank approvals (the first refusal noted for a call wins); among approvals a human's
# answer outranks an automatic one.
_APPROVAL_RANK = {"auto_approved": 1, "user_approved": 2}
# Blocked statuses the dispatch layers report on the terminal post_tool_call -> (verdict, source).
_BY_ERROR_TYPE = {
    "plugin_block": ("blocked_by_hook", "pre_tool_call"),
    "tool_scope_block": ("blocked_by_policy", "toolset_scope"),
    "guardrail_block": ("blocked_by_policy", "tool_guardrail"),
    "suspected_pruned_tool_arguments": ("blocked_by_policy", "pruned_arguments"),
    "edit_approval_denied": ("denied_by_user", "acp_edit_approval"),
    "edit_approval_error": ("blocked_by_policy", "acp_edit_approval"),
    "invalid_tool_arguments": ("blocked_by_policy", "invalid_arguments"),
}
# Calls skipped by an interrupt before any gate ran: no decision was made, so no event.
_SKIPPED_ERROR_TYPES = frozenset({"user_interrupt"})
# approval result ``outcome`` -> verdict (anything else that refused is a policy block).
_OUTCOME_VERDICTS = {"denied": "denied_by_user", "timeout": "timeout"}
_MAX_PENDING = 1024  # notes whose call never reached the funnel are dropped oldest-first

_pending: OrderedDict[tuple[str, str, str], tuple[str, str]] = OrderedDict()
# (session_id, tool_call_id, dispatch id) already reported (ids can recur after compaction), so a call
# whose terminal post_tool_call is published twice still yields one decision. A call published as
# abandoned (deadline / interrupt) is also held in ``_abandoned`` until its late worker reports, so a
# worker that settles long after its deadline is never re-reported by eviction from ``_emitted``.
_emitted: OrderedDict[tuple[str, str, str], None] = OrderedDict()
_abandoned: OrderedDict[tuple[str, str, str], None] = OrderedDict()
_ABANDON_ERROR_TYPES = frozenset({"tool_timeout", "tool_interrupted", "keyboard_interrupt", "thread_missing_result"})
_MAX_EMITTED = 8192
_lock = threading.Lock()
_deferred: ContextVar[bool] = ContextVar("tool_decision_deferred", default=False)
_dispatch_ids = count()


def new_dispatch_id() -> str:
    """Process-wide monotonic identity; next(count) is atomic in CPython."""
    return str(next(_dispatch_ids))


class DispatchTurn(str):
    """Internal turn carrier; hook boundaries strip the private dispatch identity."""

    def __new__(cls, turn_id):
        if isinstance(turn_id, cls):
            return turn_id
        value = super().__new__(cls, turn_id or "")
        value.dispatch_id = new_dispatch_id()
        return value


def _has_subscriber() -> bool:
    from hermes_cli.lifecycle import has_hook
    return has_hook(HOOK)


def args_digest(args: Any) -> str:
    """sha256 hex of the canonical JSON of *args* (sorted keys, compact separators)."""
    try:
        canonical = json.dumps(args if args is not None else {}, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, default=str)
    except (TypeError, ValueError, RecursionError):
        canonical = repr(args)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _merge(old: Optional[tuple[str, str]], new: tuple[str, str]) -> tuple[str, str]:
    if old is None or (old[0] in _APPROVAL_RANK and new[0] not in _APPROVAL_RANK):
        return new
    if old[0] not in _APPROVAL_RANK or new[0] not in _APPROVAL_RANK:
        return old
    return new if _APPROVAL_RANK[new[0]] >= _APPROVAL_RANK[old[0]] else old


@contextmanager
def notes_deferred() -> Iterator[None]:
    """Drop notes made in this block: a guard evaluated ahead of its call (terminal batch
    preparation) is reported when the call consumes it, not when it is prepared."""
    token = _deferred.set(True)
    try:
        yield
    finally:
        _deferred.reset(token)


def _store(session_id: str, tool_call_id: str, decision: tuple[str, str], turn_id=None) -> None:
    if not tool_call_id or _deferred.get() or not _has_subscriber():
        return
    from tools.approval_context import _approval_turn_id
    turn_id = _approval_turn_id.get() if turn_id is None else turn_id
    key = (session_id or "", tool_call_id, getattr(turn_id, "dispatch_id", ""))
    with _lock:
        _pending[key] = _merge(_pending.pop(key, None), decision)
        while len(_pending) > _MAX_PENDING:
            _pending.popitem(last=False)


def note(verdict: str, source: str) -> None:
    """Record a gate verdict for the tool call bound to the approval observability context.
    No-op without a ``tool_decision`` subscriber or outside a call that has a tool_call_id."""
    try:
        from tools.approval_context import _approval_session_id, _approval_tool_call_id
        _store(_approval_session_id.get(), _approval_tool_call_id.get(), (verdict, source))
    except Exception:
        logger.warning("tool_decision note failed", exc_info=True)


def note_suppressed_outcome(*, status: Optional[str], error_type: Optional[str], session_id: str,
                            tool_call_id: str, turn_id: str = "") -> None:
    """A dispatch whose terminal post_tool_call an outer executor owns hands its own block
    (e.g. an ACP edit denial) to that executor's single emission."""
    try:
        if status == "blocked" or (error_type in _BY_ERROR_TYPE and error_type not in _SKIPPED_ERROR_TYPES):
            _store(session_id, tool_call_id, _resolve(status, error_type, None), turn_id)
    except Exception:
        logger.warning("tool_decision note failed", exc_info=True)


def note_approval_result(result: Any, gate: str) -> None:
    """Classify an approval-gate result dict and :func:`note` it. Plain approvals note nothing:
    the sites that approve without a human note ``auto_approved`` themselves."""
    if not isinstance(result, dict):
        return
    if result.get("approved"):
        if result.get("user_approved"):
            note("user_approved", "approval_prompt")
        elif result.get("smart_approved"):
            note("auto_approved", "smart_approval")
        return
    if result.get("user_deny"):
        note("denied_by_rule", "approvals.deny")
    elif result.get("hardline"):
        note("blocked_by_policy", "hardline")
    elif result.get("status") in ("pending_approval", "approval_required"):
        note("approval_pending", "approval_prompt")
    elif result.get("smart_denied"):
        note("blocked_by_policy", "smart_approval")
    elif result.get("outcome") in _OUTCOME_VERDICTS:
        note(_OUTCOME_VERDICTS[result["outcome"]], "approval_prompt")
    else:
        note("blocked_by_policy", f"{gate}:{result.get('outcome') or 'blocked'}")


def _pop(session_id: str, tool_call_id: str, turn_id: str) -> Optional[tuple[str, str]]:
    if not tool_call_id:
        return None
    with _lock:
        return _pending.pop((session_id or "", tool_call_id, getattr(turn_id, "dispatch_id", "")), None)


def _already_emitted(session_id: str, tool_call_id: str, turn_id: str, error_type: Optional[str]) -> bool:
    key = (session_id or "", tool_call_id, getattr(turn_id, "dispatch_id", ""))
    with _lock:
        if key in _abandoned:
            del _abandoned[key]  # the abandoned call's worker settled; its late report is dropped
            return True
        if key in _emitted:
            return True
        _emitted[key] = None
        if error_type in _ABANDON_ERROR_TYPES:
            _abandoned[key] = None
        for table, cap in ((_emitted, _MAX_EMITTED), (_abandoned, _MAX_PENDING)):
            while len(table) > cap:
                table.popitem(last=False)
    return False


def _initiator() -> str:
    """Best-effort initiator kind: cron, subagent, api, else session (a user or injected turn)."""
    from agent.delegation_context import is_delegated_child_process_context
    from tools.approval_context import _UNATTENDED_APPROVAL_PLATFORMS, _get_session_platform, _is_cron_approval_context
    if _is_cron_approval_context():
        return "cron"
    if is_delegated_child_process_context():
        return "subagent"
    return "api" if _get_session_platform() in _UNATTENDED_APPROVAL_PLATFORMS else "session"


def _resolve(status: Optional[str], error_type: Optional[str], noted: Optional[tuple[str, str]]) -> tuple[str, str]:
    if noted is not None and noted[0] not in _APPROVAL_RANK:
        return noted  # a gate's refusal is more specific than the dispatcher's block status
    if error_type in _BY_ERROR_TYPE:
        return _BY_ERROR_TYPE[error_type]
    if error_type in _ABANDON_ERROR_TYPES:
        return "abandoned", error_type
    if status == "blocked":
        return "blocked_by_policy", error_type or "blocked"
    return noted or ("allowed", "")


def emit_for_call(*, tool_name: str, args: Any, status: Optional[str], error_type: Optional[str],
                  session_id: str, tool_call_id: str, turn_id: str, task_id: str, api_request_id: str) -> None:
    """Fire ``tool_decision`` for one call from its terminal post_tool_call. Never raises. Dispatches
    without a tool_call_id (tools an ``execute_code`` script calls back into, direct programmatic calls)
    are not model tool calls and their gate verdicts cannot be attributed, so they fire nothing."""
    try:
        if (not tool_call_id or error_type in _SKIPPED_ERROR_TYPES or not _has_subscriber()
                or _already_emitted(session_id, tool_call_id, turn_id, error_type)):
            return
        noted = _pop(session_id, tool_call_id, turn_id)
        verdict, source = _resolve(status, error_type, noted)
        from hermes_cli.lifecycle import invoke_hook
        invoke_hook(HOOK, tool_name=tool_name, args_sha256=args_digest(args), verdict=verdict, source=source,
                    initiator=_initiator(), session_id=session_id, turn_id=str(turn_id or ""), tool_call_id=tool_call_id,
                    task_id=task_id, api_request_id=api_request_id)
    except Exception:
        logger.warning("tool_decision hook dispatch failed for %s", tool_name, exc_info=True)
