"""Tool-boundary delivery of completed background delegations.

The async registry and its durable claims remain the authority for delivery.
This module only carries already-ready ``result_delivery=inject`` events on a
new tool result before that result's append-only transcript commit. It never
waits for a child or manufactures a conversation role.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from copy import deepcopy
from dataclasses import replace
from typing import Any

logger = logging.getLogger(__name__)


_PENDING_CLAIMS_ATTR = "_pending_delegation_inject_claims"
_CLAIM_HEARTBEAT_ATTR = "_delegation_inject_claim_heartbeat"
_CLAIM_HEARTBEAT_INTERVAL_SECONDS = 60.0
_CLAIM_ABANDONED_KEY = "_delegation_local_claim_abandoned"
_CARRIER_SPILL_TOOL_NAME = "__delegation_carrier__"
_CARRIER_MARKER = (
    "\n\n[DELEGATION RESULT READY — background evidence for the current task; "
    "not a new user request]\n"
)


def _event_identity(event: dict[str, Any]) -> str:
    return str(event.get("delegation_id") or "")


def _message_event_ids(message: dict[str, Any]) -> set[str]:
    metadata = message.get("display_metadata")
    if isinstance(metadata, dict):
        values = metadata.get("delegation_event_ids") or []
    else:
        values = message.get("_delegation_event_ids") or []
    return {str(value) for value in values if value}


def _clear_carrier_metadata(message: dict[str, Any]) -> None:
    message.pop("_delegation_delivery_original_content", None)
    message.pop("_delegation_event_ids", None)


def _remove_carrier_identity(message: dict[str, Any]) -> None:
    _clear_carrier_metadata(message)
    metadata = message.get("display_metadata")
    if not isinstance(metadata, dict):
        return
    metadata.pop("delegation_event_ids", None)
    metadata.pop("delegation_delivery", None)
    if not metadata:
        message.pop("display_metadata", None)


def _append_carrier_text(message: dict[str, Any], text: str) -> None:
    marker = _CARRIER_MARKER + text
    content = message.get("content", "")
    if isinstance(content, str):
        message["content"] = content + marker
        return
    try:
        blocks = list(content) if content else []
        blocks.append({"type": "text", "text": marker.lstrip()})
        message["content"] = blocks
    except Exception:
        message["content"] = f"{content}{marker}"


def _content_text_size(content: Any) -> int:
    """Count provider-visible text without charging binary image payloads."""

    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, str):
                total += len(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    total += len(text)
        return total
    return len(str(content)) if content is not None else 0


def _bounded_carrier_text(
    text: str,
    *,
    max_chars: int | None,
    event_id: str,
    storage_env: Any,
    budget_config: Any,
) -> str | None:
    """Return a bounded carrier or persist its full report in the active env.

    A missing/failed sandbox is a delivery deferral, not permission to lose the
    full child report through inline truncation. The durable event therefore
    remains pending for the ordinary after-turn rail.
    """

    if max_chars is None or len(text) <= max_chars:
        return text
    if max_chars <= 0 or storage_env is None or budget_config is None:
        return None

    from tools.tool_result_storage import PERSISTED_OUTPUT_TAG, maybe_persist_tool_result

    preview_size = max(0, min(int(budget_config.preview_size), max_chars))
    for _ in range(3):
        carrier_budget = replace(budget_config, preview_size=preview_size)
        bounded = maybe_persist_tool_result(
            content=text,
            tool_name=_CARRIER_SPILL_TOOL_NAME,
            tool_use_id=f"delegation-{event_id}",
            env=storage_env,
            config=carrier_budget,
            threshold=0,
        )
        if PERSISTED_OUTPUT_TAG not in bounded:
            return None
        if len(bounded) <= max_chars:
            return bounded
        if preview_size == 0:
            return None
        preview_size = max(0, preview_size - (len(bounded) - max_chars) - 16)
    return None


def _durable_event_is_in_history(
    messages: list[dict[str, Any]], event_id: str
) -> bool:
    return any(
        message.get("_db_persisted") is True
        and event_id in _message_event_ids(message)
        for message in messages
        if isinstance(message, dict)
    )


def _entry_is_renewable(agent: Any, entry: dict[str, Any]) -> bool:
    """Whether this RAM claim may still be renewed for the active parent turn.

    A claim is useful only to the turn that created its carrier. Once a cached
    agent advances, retaining or renewing the old local latch can only delay the
    durable fallback and must never block a new carrier.
    """
    if entry.get(_CLAIM_ABANDONED_KEY) or entry.get("queue_retry"):
        return False
    active_turn_id = str(getattr(agent, "_active_turn_id", "") or "")
    entry_turn_id = str(entry.get("turn_id") or "")
    if active_turn_id and entry_turn_id and entry_turn_id != active_turn_id:
        entry[_CLAIM_ABANDONED_KEY] = True
        return False
    return True


def _claim_status(entry: dict[str, Any]) -> tuple[str | None, bool]:
    """Return durable state and whether this exact local token still owns it."""
    from tools.async_delegation import get_event_delivery_claim_status

    return get_event_delivery_claim_status(entry["event"], entry["claim_id"])


def _emit_injected_delivery_notice(agent: Any, entries: list[dict[str, Any]]) -> None:
    """Tell interactive surfaces that durable same-turn delivery completed.

    The completion payload is already inside an ordinary tool result.  This is
    display-only: it must never add another conversation message or trigger a
    model turn.  Unknown progress-event consumers simply ignore the event.
    """
    callback = getattr(agent, "tool_progress_callback", None)
    if not callable(callback) or not entries:
        return
    events = [entry.get("event") for entry in entries if isinstance(entry.get("event"), dict)]
    if not events:
        return
    task_count = 0
    delegation_ids: list[str] = []
    for event in events:
        results = event.get("results")
        task_count += len(results) if isinstance(results, list) and results else 1
        delegation_id = str(event.get("delegation_id") or "")
        if delegation_id and delegation_id not in delegation_ids:
            delegation_ids.append(delegation_id)
    try:
        callback(
            "delegation.injected",
            "_delegation",
            None,
            None,
            task_count=max(1, task_count),
            unit_count=len(events),
            delegation_ids=delegation_ids,
        )
    except Exception:
        # Delivery is already durable; display failure must never roll it back.
        logger.debug("Failed to render same-turn delegation delivery notice", exc_info=True)


def _stop_claim_heartbeat_if_idle(agent: Any) -> None:
    pending = list(getattr(agent, _PENDING_CLAIMS_ATTR, []) or [])
    if any(_entry_is_renewable(agent, entry) for entry in pending):
        return
    heartbeat = getattr(agent, _CLAIM_HEARTBEAT_ATTR, None)
    if isinstance(heartbeat, dict):
        stop = heartbeat.get("stop")
        if isinstance(stop, threading.Event):
            stop.set()


def ensure_pending_inject_heartbeat(agent: Any) -> bool:
    """Renew live same-turn claims throughout provider retries and backoff."""

    pending = list(getattr(agent, _PENDING_CLAIMS_ATTR, []) or [])
    if not any(_entry_is_renewable(agent, entry) for entry in pending):
        _stop_claim_heartbeat_if_idle(agent)
        return False
    existing = getattr(agent, _CLAIM_HEARTBEAT_ATTR, None)
    if isinstance(existing, dict):
        thread = existing.get("thread")
        existing_stop = existing.get("stop")
        if (
            isinstance(thread, threading.Thread)
            and thread.is_alive()
            and isinstance(existing_stop, threading.Event)
            and not existing_stop.is_set()
        ):
            return True

    stop = threading.Event()

    def _heartbeat() -> None:
        from tools.async_delegation import renew_event_delivery

        try:
            while not stop.wait(_CLAIM_HEARTBEAT_INTERVAL_SECONDS):
                pending = list(getattr(agent, _PENDING_CLAIMS_ATTR, []) or [])
                renewable = [
                    entry for entry in pending if _entry_is_renewable(agent, entry)
                ]
                if not renewable:
                    break
                for entry in renewable:
                    try:
                        renewed = renew_event_delivery(
                            entry["event"], entry["claim_id"]
                        )
                    except Exception:
                        logger.warning(
                            "Failed to renew same-turn delegation claim %s",
                            entry.get("event_id"),
                            exc_info=True,
                        )
                        continue
                    if renewed:
                        continue
                    # A False UPDATE is authoritative: the row is terminal/missing
                    # or another token owns it. Stop presenting this RAM entry as
                    # live. Cleanup restores an uncommitted carrier without touching
                    # the row now owned by another consumer.
                    entry[_CLAIM_ABANDONED_KEY] = True
                    logger.warning(
                        "Lost ownership of same-turn delegation claim %s; "
                        "retiring its local latch",
                        entry.get("event_id"),
                    )
                if not any(_entry_is_renewable(agent, entry) for entry in pending):
                    break
        finally:
            stop.set()

    from tools.thread_context import propagate_context_to_thread

    thread = threading.Thread(
        target=propagate_context_to_thread(_heartbeat),
        daemon=True,
        name="delegation-inject-claim-heartbeat",
    )
    setattr(agent, _CLAIM_HEARTBEAT_ATTR, {"stop": stop, "thread": thread})
    thread.start()
    return True


def acknowledge_pending_injects(agent: Any, *, turn_id: str | None = None) -> int:
    """Acknowledge tool-boundary claims after durable transcript persistence."""

    from tools.async_delegation import complete_event_delivery

    pending = list(getattr(agent, _PENDING_CLAIMS_ATTR, []) or [])
    keep: list[dict[str, Any]] = []
    settled_messages: list[dict[str, Any]] = []
    acknowledged_entries: list[dict[str, Any]] = []
    acknowledged = 0
    for entry in pending:
        if turn_id is not None and str(entry.get("turn_id") or "") != str(turn_id):
            keep.append(entry)
            continue
        message = entry.get("message")
        # A successful no-op flush (persistence-disabled agents) is NOT a receipt.
        if not isinstance(message, dict) or message.get("_db_persisted") is not True:
            keep.append(entry)
            continue
        try:
            committed = complete_event_delivery(entry["event"], entry["claim_id"])
        except Exception:
            logger.warning("Could not acknowledge persisted delegation carrier", exc_info=True)
            committed = False
        if committed:
            acknowledged += 1
            acknowledged_entries.append(entry)
            settled_messages.append(message)
            continue
        try:
            state, still_owned = _claim_status(entry)
        except Exception:
            keep.append(entry)
            logger.warning(
                "Could not classify failed delegation acknowledgement %s; "
                "retaining local claim for retry",
                entry.get("event_id"),
                exc_info=True,
            )
            continue
        if still_owned:
            keep.append(entry)
            logger.warning(
                "Delegation carrier persisted but durable event ack did not commit: %s",
                entry.get("event_id"),
            )
            continue
        entry[_CLAIM_ABANDONED_KEY] = True
        settled_messages.append(message)
        logger.warning(
            "Retired stale local delegation claim %s after failed acknowledgement "
            "(durable state=%s)",
            entry.get("event_id"),
            state,
        )
    setattr(agent, _PENDING_CLAIMS_ATTR, keep)
    still_pending_ids = {str(entry.get("event_id") or "") for entry in keep}
    for message in settled_messages:
        if not (_message_event_ids(message) & still_pending_ids):
            _clear_carrier_metadata(message)
    _stop_claim_heartbeat_if_idle(agent)
    _emit_injected_delivery_notice(agent, acknowledged_entries)
    return acknowledged


def release_pending_injects(
    agent: Any,
    messages: list[dict[str, Any]],
    *,
    turn_id: str | None = None,
) -> int:
    """Roll back unconsumed RAM injects, preserving already-durable copies."""

    from tools.async_delegation import (
        complete_event_delivery,
        get_event_delivery_state,
        release_event_delivery,
    )
    from tools.process_registry import process_registry

    pending = list(getattr(agent, _PENDING_CLAIMS_ATTR, []) or [])
    keep: list[dict[str, Any]] = []
    removable_event_ids: set[str] = set()
    settled_messages: list[dict[str, Any]] = []
    settled = 0
    for entry in pending:
        if turn_id is not None and str(entry.get("turn_id") or "") != str(turn_id):
            keep.append(entry)
            continue
        event, event_id = entry["event"], str(entry["event_id"])
        durable = _durable_event_is_in_history(messages, event_id)
        if not durable:
            removable_event_ids.add(event_id)
        try:
            if entry.get("queue_retry"):
                with process_registry.completion_routing_lock:
                    process_registry.completion_queue.put(event)
                settled += 1
                continue
            if durable:
                committed = complete_event_delivery(event, entry["claim_id"])
            else:
                committed = release_event_delivery(event, entry["claim_id"])
                if committed:
                    # Ownership is now the ordinary queue's. If its publication or
                    # status read fails, retain an explicit RAM retry owner instead.
                    entry["queue_retry"] = True
                    if get_event_delivery_state(event) == "pending":
                        with process_registry.completion_routing_lock:
                            process_registry.completion_queue.put(event)
            if not committed:
                state, still_owned = _claim_status(entry)
                if still_owned:
                    keep.append(entry)
                    continue
                entry[_CLAIM_ABANDONED_KEY] = True
                logger.debug("Retired local delegation claim %s (state=%s)", event_id, state)
            settled += 1
            if durable and isinstance(entry.get("message"), dict):
                settled_messages.append(entry["message"])
        except Exception:
            keep.append(entry)
            logger.warning("Could not settle delegation claim %s; retained for retry", event_id, exc_info=True)

    if removable_event_ids:
        retained: list[dict[str, Any]] = []
        for message in messages:
            if not (_message_event_ids(message) & removable_event_ids):
                retained.append(message)
                continue
            if "_delegation_delivery_original_content" in message:
                message["content"] = deepcopy(
                    message["_delegation_delivery_original_content"]
                )
                _remove_carrier_identity(message)
            retained.append(message)
        messages[:] = retained
        agent._session_messages = messages
    still_pending_ids = {str(entry.get("event_id") or "") for entry in keep}
    for message in settled_messages:
        if not (_message_event_ids(message) & still_pending_ids):
            _clear_carrier_metadata(message)
    setattr(agent, _PENDING_CLAIMS_ATTR, keep)
    _stop_claim_heartbeat_if_idle(agent)
    return settled


def _normal_budget_available(agent: Any) -> bool:
    """Mirror the conversation-loop's normal iteration-budget predicate."""

    max_iterations = getattr(agent, "max_iterations", None)
    budget = getattr(agent, "iteration_budget", None)
    # Lightweight helper users/tests do not necessarily expose loop-budget
    # state. In production both attributes exist; absent state must not make a
    # non-blocking queue drain manufacture a grace-call contract of its own.
    if max_iterations is None or budget is None:
        return True
    api_calls = int(getattr(agent, "_api_call_count", 0) or 0)
    remaining = int(getattr(budget, "remaining", 0) or 0)
    return api_calls < int(max_iterations or 0) and remaining > 0


def attach_ready_injects_to_tool_results(
    agent: Any,
    messages: list[dict[str, Any]],
    num_tool_msgs: int,
    *,
    turn_id: str | None = None,
    storage_env: Any = None,
    budget_config: Any = None,
) -> int:
    """Carry ready delegation results on a newly produced tool result.

    The carrier is restricted to the tail slice produced by the current tool
    batch. Historical messages are never scanned or rewritten. Claims stay
    live only through the append-only transcript commit; a failed commit lets
    ``release_pending_injects`` restore the original tool content and return
    the durable event to the ordinary after-turn rail.
    """

    if num_tool_msgs <= 0 or not messages:
        return 0
    active_turn_id = str(turn_id or getattr(agent, "_active_turn_id", "") or "")
    if not active_turn_id:
        return 0
    if not _normal_budget_available(agent):
        return 0

    # The single-flight latch is scoped to the carrier's originating turn. A
    # cached agent may retain an older entry after an uncertain DB failure, but
    # that obsolete local bookkeeping must never disable injection forever.
    pending = list(getattr(agent, _PENDING_CLAIMS_ATTR, []) or [])
    for entry in pending:
        _entry_is_renewable(agent, entry)  # marks prior-turn entries abandoned
    _stop_claim_heartbeat_if_idle(agent)
    if any(
        str(entry.get("turn_id") or "") in {"", active_turn_id}
        for entry in pending
    ):
        return 0

    # Only the newest result is eligible. Never search backwards past an already
    # committed tail or mutate a replayed tool result after a process restart.
    target = messages[-1]
    if (not isinstance(target, dict) or target.get("role") != "tool"
            or target.get("_db_persisted") is True):
        return 0
    tail_start = max(0, len(messages) - num_tool_msgs)

    carrier_capacity: int | None = None
    if budget_config is not None:
        target_size = _content_text_size(target.get("content", ""))
        batch_size = sum(
            _content_text_size(message.get("content", ""))
            for message in messages[tail_start:] if isinstance(message, dict)
        )
        # The executor enforces the aggregate budget after per-result flushes.
        # Reserve space for ALL siblings so it cannot later truncate a carrier
        # whose durable event has already been acknowledged.
        carrier_capacity = min(
            int(budget_config.default_result_size) - target_size,
            int(budget_config.turn_budget) - batch_size,
        ) - len(_CARRIER_MARKER)
        if carrier_capacity <= 0:
            return 0

    from tools.async_delegation import (
        claim_event_delivery,
        get_event_delivery_state,
    )
    from tools.process_registry import process_registry
    from tools.process_registry_notifications import _format_async_delegation

    accepted: list[tuple[dict[str, Any], str]] = []
    pending = list(getattr(agent, _PENDING_CLAIMS_ATTR, []) or [])
    setattr(agent, _PENDING_CLAIMS_ATTR, pending)  # establish owner before dequeue/claim
    unowned: dict[int, dict] = {}
    original_target = None
    handed_off = False

    def requeue(event):
        try:
            with process_registry.completion_routing_lock:
                process_registry.completion_queue.put(event)
        except Exception:
            # Queue failure must not leave the durable row as the only owner until restart.
            pending.append({"event": event, "event_id": _event_identity(event), "claim_id": "",
                            "message": None, "turn_id": active_turn_id, "queue_retry": True})
            logger.exception("Could not requeue delegation; retained explicit retry owner")
        unowned.pop(id(event), None)

    try:
        candidates = []
        with process_registry.completion_routing_lock:
            for _ in range(process_registry.completion_queue.qsize()):
                try:
                    event = process_registry.completion_queue.get_nowait()
                except queue.Empty:
                    break
                unowned[id(event)] = event
                if (
                    event.get("type") != "async_delegation"
                    or event.get("task_failure_notice")  # interim notices are not final unit receipts
                    or str(event.get("result_delivery") or "after_turn").strip().lower() != "inject"
                    or str(event.get("parent_turn_id") or "") != active_turn_id
                    or str(event.get("parent_session_id") or "") != str(getattr(agent, "session_id", "") or "")
                    or not event.get("parent_session_id")
                ):
                    requeue(event)
                else:
                    candidates.append(event)

        # Formatting/storage holds no routing or database lock. Every event still
        # belongs to unowned until a claim is registered in the existing RAM owner.
        for event in candidates:
            try:
                text = _format_async_delegation(event)
                if not text:
                    continue
                event_id = _event_identity(event)
                separator_size = 2 if accepted else 0
                available = None if carrier_capacity is None else carrier_capacity - separator_size
                bounded_text = _bounded_carrier_text(
                    text, max_chars=available, event_id=event_id,
                    storage_env=storage_env, budget_config=budget_config,
                )
                if bounded_text is None:
                    continue
                claim_id = claim_event_delivery(event, f"tool-boundary:{os.getpid()}")
                if claim_id is None:
                    # An uncertain read is NOT permission to discard this event.
                    # Leave it in unowned so the outer finally restores queue ownership.
                    if get_event_delivery_state(event) not in (None, "pending"):
                        unowned.pop(id(event), None)
                    continue
            except Exception:
                logger.warning("Failed to prepare tool-boundary delegation event", exc_info=True)
                continue
            entry = {"event": event, "claim_id": claim_id, "event_id": event_id,
                     "message": None, "turn_id": active_turn_id}
            pending.append(entry)
            unowned.pop(id(event), None)
            accepted.append((entry, bounded_text))
            if carrier_capacity is not None:
                carrier_capacity -= separator_size + len(bounded_text)

        if not accepted:
            return 0
        # Snapshot and mutation failures are covered by the SAME ownership boundary.
        original_target = deepcopy(target)
        original_content = deepcopy(target.get("content", ""))
        event_ids = [entry["event_id"] for entry, _text in accepted]
        _append_carrier_text(target, "\n\n".join(text for _entry, text in accepted))
        target["_delegation_delivery_original_content"] = original_content
        target["_delegation_event_ids"] = event_ids
        metadata = dict(target.get("display_metadata") or {})
        metadata.update(delegation_event_ids=event_ids, delegation_delivery="tool_boundary")
        target["display_metadata"] = metadata
        agent._session_messages = messages
        for entry, _text in accepted:
            entry["message"] = target
        ensure_pending_inject_heartbeat(agent)
        handed_off = True
        return len(accepted)
    finally:
        for event in list(unowned.values()):
            requeue(event)
        if not handed_off:
            if original_target is not None:
                target.clear()
                target.update(original_target)
            # All acquired tokens are already registered, including ones whose
            # release/status read fails. Never release a token that we didn't acquire.
            release_pending_injects(agent, messages, turn_id=active_turn_id)
