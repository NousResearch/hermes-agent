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
_CARRIER_SPILL_TOOL_NAME = "__delegation_carrier__"
_CARRIER_MARKER = (
    "\n\n[DELEGATION RESULT READY — background evidence for the current task; "
    "not a new user request]\n"
)


def _event_identity(event: dict[str, Any]) -> str:
    return (
        f"{event.get('delegation_id') or ''}:"
        f"{event.get('delivery_event_key') or 'aggregate'}"
    )


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


def _stop_claim_heartbeat_if_idle(agent: Any) -> None:
    if getattr(agent, _PENDING_CLAIMS_ATTR, None):
        return
    heartbeat = getattr(agent, _CLAIM_HEARTBEAT_ATTR, None)
    if isinstance(heartbeat, dict):
        stop = heartbeat.get("stop")
        if isinstance(stop, threading.Event):
            stop.set()


def ensure_pending_inject_heartbeat(agent: Any) -> bool:
    """Renew live same-turn claims throughout provider retries and backoff."""

    if not getattr(agent, _PENDING_CLAIMS_ATTR, None):
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

        while not stop.wait(_CLAIM_HEARTBEAT_INTERVAL_SECONDS):
            pending = list(getattr(agent, _PENDING_CLAIMS_ATTR, []) or [])
            if not pending:
                break
            for entry in pending:
                try:
                    if not renew_event_delivery(entry["event"], entry["claim_id"]):
                        logger.warning(
                            "Could not renew same-turn delegation claim %s",
                            entry.get("event_id"),
                        )
                except Exception:
                    logger.warning(
                        "Failed to renew same-turn delegation claim %s",
                        entry.get("event_id"),
                        exc_info=True,
                    )

    thread = threading.Thread(
        target=_heartbeat,
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
    acknowledged_messages: list[dict[str, Any]] = []
    acknowledged = 0
    for entry in pending:
        if turn_id is not None and str(entry.get("turn_id") or "") != str(turn_id):
            keep.append(entry)
            continue
        if complete_event_delivery(entry["event"], entry["claim_id"]):
            acknowledged += 1
            message = entry.get("message")
            if isinstance(message, dict):
                acknowledged_messages.append(message)
        else:
            keep.append(entry)
            logger.warning(
                "Delegation carrier persisted but durable event ack did not commit: %s",
                entry.get("event_id"),
            )
    setattr(agent, _PENDING_CLAIMS_ATTR, keep)
    still_pending_ids = {str(entry.get("event_id") or "") for entry in keep}
    for message in acknowledged_messages:
        if not (_message_event_ids(message) & still_pending_ids):
            _clear_carrier_metadata(message)
    _stop_claim_heartbeat_if_idle(agent)
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
    settled = 0
    for entry in pending:
        if turn_id is not None and str(entry.get("turn_id") or "") != str(turn_id):
            keep.append(entry)
            continue
        event = entry["event"]
        event_id = str(entry["event_id"])
        if _durable_event_is_in_history(messages, event_id):
            if complete_event_delivery(event, entry["claim_id"]):
                settled += 1
            else:
                keep.append(entry)
        else:
            # Remove the unconsumed marker by durable identity even when
            # compression replaced the Python dict object.
            removable_event_ids.add(event_id)
            committed = release_event_delivery(event, entry["claim_id"])
            state = get_event_delivery_state(event)
            if committed:
                # At the attempt cap release transitions to dropped, not pending.
                if state == "pending":
                    process_registry.completion_queue.put(event)
                settled += 1
            elif state == "delivered":
                settled += 1
            else:
                keep.append(entry)

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
    if getattr(agent, _PENDING_CLAIMS_ATTR, None):
        return 0

    target: dict[str, Any] | None = None
    tail_start = max(0, len(messages) - num_tool_msgs)
    for message in reversed(messages[tail_start:]):
        if isinstance(message, dict) and message.get("role") == "tool":
            target = message
            break
    if target is None:
        return 0

    carrier_capacity: int | None = None
    if budget_config is not None:
        carrier_limit = min(
            int(budget_config.default_result_size),
            int(budget_config.turn_budget),
        )
        carrier_capacity = (
            carrier_limit
            - _content_text_size(target.get("content", ""))
            - len(_CARRIER_MARKER)
        )
        if carrier_capacity <= 0:
            return 0

    from tools.async_delegation import (
        claim_event_delivery,
        get_event_delivery_state,
        release_event_delivery,
    )
    from tools.process_registry import _format_async_delegation, process_registry

    accepted: list[tuple[dict[str, Any], str, str, str]] = []
    completion_queue = process_registry.completion_queue
    with process_registry.completion_routing_lock:
        try:
            scan_count = completion_queue.qsize()
        except Exception:
            return 0
        for _ in range(max(0, scan_count)):
            try:
                event = completion_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            delivery = str(event.get("result_delivery") or "after_turn").strip().lower()
            if (
                event.get("type") != "async_delegation"
                or delivery != "inject"
                or str(event.get("parent_turn_id") or "") != active_turn_id
            ):
                completion_queue.put(event)
                continue
            parent_session_id = str(event.get("parent_session_id") or "")
            agent_session_id = str(getattr(agent, "session_id", "") or "")
            if parent_session_id and parent_session_id != agent_session_id:
                completion_queue.put(event)
                continue
            try:
                text = _format_async_delegation(event)
            except Exception:
                logger.debug("Failed to format tool-boundary delegation event", exc_info=True)
                completion_queue.put(event)
                continue
            if not text:
                completion_queue.put(event)
                continue
            event_id = _event_identity(event)
            separator_size = 2 if accepted else 0
            available = (
                None
                if carrier_capacity is None
                else carrier_capacity - separator_size
            )
            bounded_text = _bounded_carrier_text(
                text,
                max_chars=available,
                event_id=event_id,
                storage_env=storage_env,
                budget_config=budget_config,
            )
            if bounded_text is None:
                completion_queue.put(event)
                continue
            claim_id = claim_event_delivery(event, f"tool-boundary:{os.getpid()}")
            if claim_id is None:
                process_registry.defer_unclaimed_delivery(event)
                continue
            accepted.append((event, claim_id, bounded_text, event_id))
            if carrier_capacity is not None:
                carrier_capacity -= separator_size + len(bounded_text)

    if not accepted:
        return 0

    original_target = deepcopy(target)
    original_content = deepcopy(target.get("content", ""))
    original_pending = list(getattr(agent, _PENDING_CLAIMS_ATTR, []) or [])
    event_ids = [item[3] for item in accepted]
    try:
        _append_carrier_text(target, "\n\n".join(item[2] for item in accepted))
        target["_delegation_delivery_original_content"] = original_content
        target["_delegation_event_ids"] = event_ids
        display_metadata = dict(target.get("display_metadata") or {})
        display_metadata["delegation_event_ids"] = event_ids
        display_metadata["delegation_delivery"] = "tool_boundary"
        target["display_metadata"] = display_metadata
        agent._session_messages = messages

        pending = list(original_pending)
        pending.extend(
            {
                "event": event,
                "claim_id": claim_id,
                "event_id": event_id,
                "message": target,
                "turn_id": active_turn_id,
            }
            for event, claim_id, _text, event_id in accepted
        )
        setattr(agent, _PENDING_CLAIMS_ATTR, pending)
        ensure_pending_inject_heartbeat(agent)
    except Exception:
        target.clear()
        target.update(original_target)
        try:
            setattr(agent, _PENDING_CLAIMS_ATTR, original_pending)
            _stop_claim_heartbeat_if_idle(agent)
        except Exception:
            logger.debug("Failed to restore delegation claim bookkeeping", exc_info=True)
        for event, claim_id, _text, _event_id in accepted:
            try:
                released = release_event_delivery(event, claim_id)
            except Exception:
                logger.exception("Failed to release rolled-back delegation claim")
                continue
            if released and get_event_delivery_state(event) == "pending":
                completion_queue.put(event)
        raise

    return len(accepted)
