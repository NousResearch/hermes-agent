"""Bounded, observer-only events for completed agent runs.

The result observer deliberately receives neither the result object nor any
prompt, message history, tool call, or transcript. It gets one fresh mapping
of capped primitives. Delivery is process-local and best effort; callbacks run
on isolated plugin observer workers and can never rewrite or delay the live
result. Persistence-isolated internal review/curator forks are not worker
results and are deliberately excluded.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from typing import Any, Optional

from hermes_cli.middleware import OBSERVER_SCHEMA_VERSION


logger = logging.getLogger(__name__)

POST_AGENT_RESULT_HOOK = "post_agent_result"
MAX_OUTPUT_CHARS = 32_768
MAX_ERROR_CHARS = 2_048
MAX_IDENTIFIER_CHARS = 256
MAX_LABEL_CHARS = 64
MAX_LINEAGE_HASH_CHARS_PER_FIELD = 1_024
_TRUNCATION_MARKER = "\n...[truncated]...\n"


def _bounded_string(value: Any, limit: int) -> tuple[str, int, bool]:
    """Return a head/tail excerpt without coercing arbitrary objects."""
    if not isinstance(value, str):
        return "", 0, False
    length = len(value)
    if length <= limit:
        return value, length, False
    remaining = max(0, limit - len(_TRUNCATION_MARKER))
    head = remaining // 2
    tail = remaining - head
    excerpt = value[:head] + _TRUNCATION_MARKER + value[-tail:]
    return excerpt, length, True


def _identifier(value: Any, *, limit: int = MAX_IDENTIFIER_CHARS) -> str:
    if not isinstance(value, str):
        return ""
    return value[:limit]


def _exception_message(exception: BaseException) -> str:
    """Extract a bounded-safe message without invoking custom ``__str__``."""
    for arg in exception.args:
        if isinstance(arg, str):
            return arg
    return ""


def _raw_string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _lineage_digest(
    raw_identity: Mapping[str, str],
) -> tuple[str, list[str]]:
    """Hash a deterministically bounded representation of raw lineage.

    Ordinary identifiers are fully bound even when their public preview is
    capped more tightly. Pathological identifiers are represented by their
    character length and a capped head/tail excerpt, with explicit loss fields
    in the event. Full raw values are never serialized on the result path.
    """
    hash_identity: dict[str, dict[str, Any]] = {}
    truncated_fields: list[str] = []
    for field, value in raw_identity.items():
        excerpt, chars, truncated = _bounded_string(
            value, MAX_LINEAGE_HASH_CHARS_PER_FIELD
        )
        hash_identity[field] = {
            "chars": chars,
            "excerpt": excerpt,
            "truncated": truncated,
        }
        if truncated:
            truncated_fields.append(field)
    payload = json.dumps(
        hash_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest(), sorted(truncated_fields)


def _turn_identity(
    agent: Any,
    *,
    previous_turn_id: str,
    requested_task_id: Optional[str],
) -> tuple[str, str, bool]:
    """Avoid attributing a prologue exception to the previous turn."""
    current_turn_id = _raw_string(getattr(agent, "_current_turn_id", ""))
    turn_started = bool(current_turn_id and current_turn_id != previous_turn_id)
    if turn_started:
        task_id = _raw_string(getattr(agent, "_current_task_id", ""))
        return task_id, current_turn_id, True
    return _raw_string(requested_task_id), "", False


def _is_result_observer_exempt(agent: Any) -> bool:
    """Identify internal AIAgent forks that are not worker/root responses."""
    return bool(
        getattr(agent, "_result_observer_exempt", False)
        or getattr(agent, "_persist_disabled", False)
        or getattr(agent, "platform", None) == "curator"
        or getattr(agent, "_memory_write_origin", None) == "background_review"
    )


def build_agent_result_event(
    agent: Any,
    *,
    result: Any = None,
    exception: Optional[BaseException] = None,
    previous_turn_id: str = "",
    requested_task_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build the sole allowlisted packet exposed to result observers."""
    result_map: Mapping[str, Any]
    if isinstance(result, Mapping):
        result_map = result
    else:
        result_map = {}

    output_value = result_map.get("final_response")
    output, output_chars, output_truncated = _bounded_string(
        output_value, MAX_OUTPUT_CHARS
    )
    output_type = (
        "none"
        if output_value is None
        else "text"
        if isinstance(output_value, str)
        else _identifier(type(output_value).__name__, limit=MAX_LABEL_CHARS)
    )

    error_value = (
        _exception_message(exception)
        if exception is not None
        else result_map.get("error")
    )
    error, error_chars, error_truncated = _bounded_string(error_value, MAX_ERROR_CHARS)

    raw_subagent_id = _raw_string(getattr(agent, "_subagent_id", ""))
    raw_platform = _raw_string(getattr(agent, "platform", ""))
    actor = "subagent" if raw_subagent_id or raw_platform == "subagent" else "root"
    raw_role = (
        _raw_string(getattr(agent, "_delegate_role", ""))
        if actor == "subagent"
        else "root"
    )
    if actor == "subagent" and not raw_role:
        raw_role = "subagent"

    raw_task_id, raw_turn_id, turn_started = _turn_identity(
        agent,
        previous_turn_id=previous_turn_id,
        requested_task_id=requested_task_id,
    )
    raw_identity = {
        "session_id": _raw_string(getattr(agent, "session_id", "")),
        "task_id": raw_task_id,
        "turn_id": raw_turn_id,
        "subagent_id": raw_subagent_id,
        "parent_session_id": _raw_string(getattr(agent, "_parent_session_id", "")),
        "parent_turn_id": _raw_string(getattr(agent, "_parent_turn_id", "")),
        "parent_subagent_id": _raw_string(getattr(agent, "_parent_subagent_id", "")),
        "platform": raw_platform,
        "role": raw_role,
    }
    identity_limits = {
        "platform": MAX_LABEL_CHARS,
        "role": MAX_LABEL_CHARS,
    }
    truncated_fields = sorted(
        field
        for field, value in raw_identity.items()
        if len(value) > identity_limits.get(field, MAX_IDENTIFIER_CHARS)
    )
    lineage_sha256, lineage_hash_truncated_fields = _lineage_digest(raw_identity)
    exception_type = (
        _identifier(type(exception).__name__, limit=MAX_LABEL_CHARS)
        if exception is not None
        else ""
    )

    # Keep this an explicit allowlist. In particular, never spread ``result``
    # into the event: it contains messages, reasoning, tool data, and provider
    # metadata that observers neither need nor should retain by accident.
    return {
        "schema_version": OBSERVER_SCHEMA_VERSION,
        "event": "agent.result",
        "actor": actor,
        "role": _identifier(raw_role, limit=MAX_LABEL_CHARS),
        "session_id": _identifier(raw_identity["session_id"]),
        "task_id": _identifier(raw_task_id),
        "turn_id": _identifier(raw_turn_id),
        "turn_started": turn_started,
        "subagent_id": _identifier(raw_subagent_id),
        "parent_session_id": _identifier(raw_identity["parent_session_id"]),
        "parent_turn_id": _identifier(raw_identity["parent_turn_id"]),
        "parent_subagent_id": _identifier(raw_identity["parent_subagent_id"]),
        "platform": _identifier(raw_platform, limit=MAX_LABEL_CHARS),
        "identity_complete": not truncated_fields,
        "identity_truncated_fields": ",".join(truncated_fields),
        "lineage_sha256": lineage_sha256,
        "lineage_hash_input_complete": not lineage_hash_truncated_fields,
        "lineage_hash_input_truncated_fields": ",".join(lineage_hash_truncated_fields),
        "result_kind": "raised" if exception is not None else "returned",
        "completed": result_map.get("completed") is True,
        "failed": result_map.get("failed") is True,
        "partial": result_map.get("partial") is True,
        "interrupted": result_map.get("interrupted") is True,
        "response_transformed": result_map.get("response_transformed") is True,
        "output": output,
        "output_type": output_type,
        "output_chars": output_chars,
        "output_truncated": output_truncated,
        "output_excerpt_sha256": hashlib.sha256(
            output.encode("utf-8", errors="replace")
        ).hexdigest(),
        "error": error,
        "error_chars": error_chars,
        "error_truncated": error_truncated,
        "exception_type": exception_type,
    }


def result_observer_enabled(agent: Any = None) -> bool:
    """Cheap no-listener gate that does not create plugin state."""
    if agent is not None and _is_result_observer_exempt(agent):
        return False
    try:
        from hermes_cli.plugins import has_observer_hook

        return has_observer_hook(POST_AGENT_RESULT_HOOK)
    except BaseException:
        return False


def observe_agent_result(
    agent: Any,
    *,
    result: Any = None,
    exception: Optional[BaseException] = None,
    previous_turn_id: str = "",
    requested_task_id: Optional[str] = None,
) -> None:
    """Build and enqueue an event, failing open to the live result path."""
    if _is_result_observer_exempt(agent):
        return
    try:
        from hermes_cli.plugins import emit_observer_hook

        event = build_agent_result_event(
            agent,
            result=result,
            exception=exception,
            previous_turn_id=previous_turn_id,
            requested_task_id=requested_task_id,
        )
        emit_observer_hook(POST_AGENT_RESULT_HOOK, event)
    except BaseException:
        # Observation must never replace a returned result or mask the live
        # exception. Even a hostile logging handler is outside that authority.
        try:
            logger.debug("post_agent_result observation failed", exc_info=True)
        except BaseException:
            pass
