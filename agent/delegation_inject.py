"""Opted-in completion units carried by a NEW tool result before its first commit.

The existing async-delegation row owns delivery. SessionDB acknowledges its claim
in the SAME transaction as the carrier, so neither a second ledger nor a
provider-retry heartbeat is needed. Late results use the unchanged idle rail.

Behavior reference: NousResearch/hermes-agent#76230. Unit identity/grouping:
NousResearch/hermes-agent#104299.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

logger = logging.getLogger(__name__)
_MARKER = (
    "\n\n[DELEGATION RESULT READY — background evidence for the current task; "
    "not a new user request]\n"
)
_CLAIMS = "_delegation_delivery_claims"
_CHECKED = "_delegation_boundary_checked"


def _completed_tool_batch_size(messages: list) -> int:
    """Only the last, still-uncommitted result of a complete tool batch is eligible."""
    if not messages or not isinstance(messages[-1], dict):
        return 0
    tail = messages[-1]
    if tail.get("role") != "tool" or tail.get(_CHECKED) or tail.get("_db_persisted") is True:
        return 0
    start = len(messages) - 1
    while start >= 0 and isinstance(messages[start], dict) and messages[start].get("role") == "tool":
        start -= 1
    if start < 0 or messages[start].get("role") != "assistant":
        return 0
    expected = [str(c.get("id") if isinstance(c, dict) else getattr(c, "id", ""))
                for c in messages[start].get("tool_calls") or []]
    actual = [str(m.get("tool_call_id") or "") for m in messages[start + 1:]]
    if not expected or any(i in ("", "None") for i in expected):
        return 0
    if len(expected) != len(actual) or len(set(expected)) != len(expected) or set(expected) != set(actual):
        return 0
    return len(actual)


def _text_size(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(b) if isinstance(b, str) else len(b.get("text", ""))
                   if isinstance(b, dict) and isinstance(b.get("text"), str) else 0 for b in content)
    return 0


def _normal_budget_available(agent: Any) -> bool:
    if getattr(agent, "_interrupt_requested", False) is True or getattr(agent, "_budget_grace_call", False) is True:
        return False
    maximum, budget = getattr(agent, "max_iterations", None), getattr(agent, "iteration_budget", None)
    if maximum is None or budget is None:
        return True
    return int(getattr(agent, "_api_call_count", 0) or 0) < int(maximum) and int(budget.remaining) > 0


def _bounded_report(text: str, available: int, event_id: str, env: Any, config: Any) -> str | None:
    """Spill through the canonical storage path; inability to fit is a deferral, never truncation."""
    if len(text) <= available:
        return text
    if available <= 0:
        return None
    from tools.tool_result_storage import PERSISTED_OUTPUT_TAG, maybe_persist_tool_result
    preview = max(0, min(int(config.preview_size), available))
    for _ in range(3):
        bounded = maybe_persist_tool_result(
            content=text, tool_name="__delegation_carrier__", tool_use_id=f"delegation-{event_id}",
            env=env, config=replace(config, preview_size=preview), threshold=0,
        )
        if PERSISTED_OUTPUT_TAG not in bounded:
            return None
        if len(bounded) <= available:
            return bounded
        if preview == 0:
            return None
        preview = max(0, preview - (len(bounded) - available) - 16)
    return None


def _release(claims: dict[str, str]) -> None:
    from tools.async_delegation import release_completion_delivery
    for event_id, token in claims.items():
        try:
            release_completion_delivery(event_id, token)
        except Exception:
            # The existing lease/restart rail still owns a claim whose DB is temporarily unavailable.
            logger.exception("Could not release delegation carrier claim %s", event_id)


@dataclass
class _Prepared:
    target: dict
    original: dict
    claims: dict[str, str]

    def finish(self) -> None:
        """Restore an uncommitted carrier; never undo evidence that is already durable."""
        if self.target.get("_db_persisted") is not True:
            from tools.async_delegation import get_durable_delegation
            # A failure after SQLite COMMIT but before marker publication must not undo the carrier.
            delivered = all((get_durable_delegation(i) or {}).get("delivery_state") == "delivered"
                            for i in self.claims)
            if not delivered:
                self.target.clear()
                self.target.update(self.original)
                _release(self.claims)
                return
        self.target.pop(_CLAIMS, None)


def _prepare(agent: Any, messages: list, env: Any, config: Any) -> _Prepared | None:
    if (getattr(agent, "_persist_disabled", False) is True or getattr(agent, "_session_db", None) is None
            or config is None or not _normal_budget_available(agent)):
        return None
    turn_id, session_id = getattr(agent, "_active_turn_id", ""), getattr(agent, "session_id", "")
    if not isinstance(turn_id, str) or not turn_id or not isinstance(session_id, str) or not session_id:
        return None
    n = _completed_tool_batch_size(messages)
    if not n:
        return None
    target = messages[-1]
    target[_CHECKED] = True
    if not isinstance(target.get("content", ""), (str, list)):
        return None
    if target.get("display_metadata") is not None and not isinstance(target["display_metadata"], dict):
        return None
    tool_name = str(target.get("tool_name") or target.get("name") or "")
    result_limit = min(config.default_result_size, config.resolve_threshold(tool_name))
    capacity = int(min(result_limit - _text_size(target.get("content", "")),
                       config.turn_budget - sum(_text_size(m.get("content", "")) for m in messages[-n:]))) - len(_MARKER)
    if capacity <= 0:
        return None

    from tools.async_delegation import claim_event_delivery, ready_inject_events
    from tools.process_registry_notifications import format_process_notification
    events = ready_inject_events(session_id, turn_id)
    if not events:
        return None
    original = deepcopy(target)
    claims: dict[str, str] = {}
    reports: list[str] = []
    try:
        for event in events:
            event_id = str(event.get("delegation_id") or "")
            if not event_id or event_id in claims:
                continue
            text = format_process_notification(event)
            if not text:
                continue
            available = capacity - (2 if reports else 0)
            bounded = _bounded_report(text, available, event_id, env, config)
            if bounded is None:
                continue
            token = claim_event_delivery(event, "tool-boundary")
            if not token:
                continue
            claims[event_id] = token
            reports.append(bounded)
            capacity = available - len(bounded)
        if not claims:
            return None
        prepared = _Prepared(target, original, claims)
        content = target.get("content", "")
        evidence = _MARKER + "\n\n".join(reports)
        target["content"] = content + evidence if isinstance(content, str) else [*content, {"type": "text", "text": evidence}]
        metadata = dict(target.get("display_metadata") or {})
        metadata.update(delegation_event_ids=list(claims), delegation_delivery="tool_boundary")
        target["display_metadata"] = metadata
        # Private RAM-only tokens. _db_flush_write passes them out-of-band to the SQLite transaction;
        # neither persisted display metadata nor the provider request ever contains a claim token.
        target[_CLAIMS] = claims
        return prepared
    except BaseException:
        target.clear()
        target.update(original)
        _release(claims)
        raise


@contextmanager
def tool_result_carrier(agent: Any, messages: list, *, storage_env: Any = None, budget_config: Any = None):
    """Enrich at most one new result, then settle immediately around its ordinary flush.

    No queue is drained here: the existing queue entry becomes unclaimable after the
    atomic transcript/ack commit, or remains available after a failed/no-op flush.
    Nothing waits for unfinished children or forces a further provider request.
    """
    prepared = None
    try:
        prepared = _prepare(agent, messages, storage_env, budget_config)
    except Exception:
        logger.warning("Could not prepare a delegation carrier; leaving normal delivery intact", exc_info=True)
    try:
        yield
    finally:
        if prepared is not None:
            prepared.finish()
