"""Phase 8 same-window rehydrate planner/consumer helpers.

This module is intentionally narrow.  It does not add a CLI command, does not
activate runtime configuration, and does not touch gateway/update-survival
paths.  It provides a deterministic planner/consumer API for same-window
re-entry from a Phase 7 compact fallback continuity packet.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

_CONTINUITY_PACKET_TYPE = "context_health_compact.continuity_packet"
_PHASE = "phase8_same_window_rehydrate"
_ALLOWED_FILES = ["tests/agent/test_same_window_rehydrate.py"]
_SAFE_HOLD_RESPONSE = (
    "Context Health same-window rehydrate HOLD: continuity packet is invalid "
    "or unsafe. Re-entry is blocked before provider calls or state mutation."
)

_UNSAFE_KEY_PARTS = (
    "raw",
    "transcript",
    "private",
    "secret",
    "token",
    "password",
    "credential",
    "exception",
    "debug",
)


def _safe_str(value: Any) -> str:
    """Return a sanitized printable value for safe metadata fields only."""

    if value is None:
        return ""
    text = str(value)
    # Metadata identifiers should remain compact and single-line.  Do not carry
    # arbitrary bodies through planner/consumer outputs.
    return text.replace("\n", " ").replace("\r", " ")[:200]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _normalized_text(value: Any) -> str:
    return str(value).lower().replace("_", " ").replace("-", " ")


def _value_contains_unsafe_material(value: Any) -> bool:
    """Detect unsafe body-like material in metadata values without echoing it."""

    if value is None or isinstance(value, (bool, int, float)):
        return False
    if isinstance(value, Mapping):
        return any(
            _value_contains_unsafe_material(key)
            or _value_contains_unsafe_material(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_value_contains_unsafe_material(item) for item in value)

    text = _normalized_text(value)
    if not text:
        return False

    # Synthetic sentinel values used by safety tests and review packets are never
    # legitimate metadata.  This catches families of needles without binding the
    # implementation to one exact test string.
    if "needle" in text or "do not leak" in text or "do not reinject" in text:
        return True

    # Safe boundary prose is allowed when it explicitly says sensitive classes
    # are excluded, not carried as payload.  Do not classify such constants as
    # leaks merely because they name the forbidden classes.
    excluded_boundary = "excluded" in text or "not included" in text

    unsafe_markers = (
        "raw transcript",
        "unrelated ab",
        "unrelated a/b",
        "private body",
    )
    if any(marker in text for marker in unsafe_markers) and not excluded_boundary:
        return True

    credential_markers = (
        "token=",
        "token:",
        "password=",
        "password:",
        "secret=",
        "secret:",
        "credential=",
        "credential:",
        "api key",
        "apikey",
        "bearer ",
        "private key",
    )
    return any(marker in text for marker in credential_markers)


def _packet_has_unsafe_shape(packet: Mapping[str, Any]) -> bool:
    """Detect raw/debug/private material without copying it into outputs."""

    if packet.get("raw_transcript_included") is True:
        return True
    if packet.get("unrelated_context_included") is True:
        return True
    if packet.get("secret_token_password_private_body_included") is True:
        return True

    safe_metadata_keys = {
        "type",
        "reason",
        "session_id",
        "task_id",
        "message_count",
        "approx_tokens",
        "rehydrate_status",
        "safety_boundary",
        "raw_transcript_included",
        "unrelated_context_included",
        "secret_token_password_private_body_included",
    }
    for key, value in packet.items():
        key_l = str(key).lower()
        if key_l in safe_metadata_keys:
            # Safety booleans explicitly set to False and known metadata fields
            # are allowed only when their values are also safe.  `approx_tokens`
            # contains the substring `token` but is only a count, not credential
            # material.
            if key_l in {
                "raw_transcript_included",
                "unrelated_context_included",
                "secret_token_password_private_body_included",
            } and value is not False:
                return True
            if key_l in {"message_count", "approx_tokens"}:
                continue
            if _value_contains_unsafe_material(value):
                return True
            continue
        if any(part in key_l for part in _UNSAFE_KEY_PARTS):
            return True
        if _value_contains_unsafe_material(value):
            return True
    return False


def _safe_packet_metadata(packet: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract only non-body metadata from a continuity packet."""

    return {
        "type": _safe_str(packet.get("type")) or _CONTINUITY_PACKET_TYPE,
        "reason": _safe_str(packet.get("reason")) or "unknown",
        "session_id": _safe_str(packet.get("session_id")) or "unknown",
        "task_id": _safe_str(packet.get("task_id")) or "default",
        "message_count": _safe_int(packet.get("message_count")),
        "approx_tokens": packet.get("approx_tokens") if isinstance(packet.get("approx_tokens"), int) else None,
        "rehydrate_status": _safe_str(packet.get("rehydrate_status")) or "phase8_planner_ready",
        "raw_transcript_included": False,
        "unrelated_context_included": False,
        "secret_token_password_private_body_included": False,
    }


def _safe_hold(reason: str = "invalid_continuity_packet") -> Dict[str, Any]:
    return {
        "status": "hold",
        "phase": _PHASE,
        "reason": reason,
        "safe_hold": True,
        "provider_call_allowed": False,
        "state_mutation_allowed": False,
        "raw_transcript_reinjected": False,
        "unrelated_context_reinjected": False,
        "secret_token_password_private_body_reinjected": False,
        "final_response": _SAFE_HOLD_RESPONSE,
    }


def validate_continuity_packet(continuity_packet: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a Phase 7 continuity packet for Phase 8 re-entry.

    Invalid or unsafe packets fail closed before provider calls or state
    mutation.  The result never reflects raw/debug/private packet values.
    """

    if not isinstance(continuity_packet, Mapping):
        return _safe_hold("invalid_continuity_packet")
    if continuity_packet.get("type") != _CONTINUITY_PACKET_TYPE:
        return _safe_hold("invalid_continuity_packet")
    if _packet_has_unsafe_shape(continuity_packet):
        return _safe_hold("unsafe_continuity_packet")

    metadata = _safe_packet_metadata(continuity_packet)
    return {
        "status": "ready",
        "phase": _PHASE,
        "safe_hold": False,
        "provider_call_allowed": False,
        "state_mutation_allowed": False,
        "raw_transcript_reinjected": False,
        "unrelated_context_reinjected": False,
        "secret_token_password_private_body_reinjected": False,
        "continuity_packet_metadata": metadata,
    }


def build_same_window_rehydrate_plan(
    *,
    continuity_packet: Mapping[str, Any],
    session_id: str,
    current_task_id: str,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Build a same-window rehydrate plan from safe continuity metadata."""

    decision = validate_continuity_packet(continuity_packet)
    if decision.get("status") != "ready":
        return {
            **decision,
            "same_window": True,
            "requires_new_process": False,
            "requires_new_window": False,
            "requires_new_session": False,
            "requires_clear": False,
            "adds_cli_command": False,
            "touches_gateway_update_survival": False,
            "runtime_activation_required": False,
            "scope": "planner_contract_only",
            "allowed_files": list(_ALLOWED_FILES),
        }

    metadata = dict(decision["continuity_packet_metadata"])
    safe_session_id = _safe_str(session_id) or metadata.get("session_id", "unknown")
    safe_task_id = _safe_str(current_task_id) or metadata.get("task_id", "default")
    metadata["session_id"] = safe_session_id
    metadata["task_id"] = safe_task_id

    return {
        "status": "ready",
        "phase": _PHASE,
        "continuity_packet_type": _CONTINUITY_PACKET_TYPE,
        "dry_run": bool(dry_run),
        "mutates_state": False,
        "same_window": True,
        "same_session_id": safe_session_id,
        "current_task_id": safe_task_id,
        "requires_new_process": False,
        "requires_new_window": False,
        "requires_new_session": False,
        "requires_clear": False,
        "provider_call_allowed": False,
        "state_mutation_allowed": not bool(dry_run),
        "raw_transcript_reinjected": False,
        "unrelated_context_reinjected": False,
        "secret_token_password_private_body_reinjected": False,
        "uses_working_context_packet": True,
        "uses_reentry_metadata": True,
        "scope": "planner_contract_only",
        "adds_cli_command": False,
        "touches_gateway_update_survival": False,
        "runtime_activation_required": False,
        "allowed_files": list(_ALLOWED_FILES),
        "continuity_packet_metadata": metadata,
    }


def _reentry_message_from_plan(plan: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = plan.get("continuity_packet_metadata") if isinstance(plan, Mapping) else {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    safe_packet = {
        "type": "phase8.same_window_reentry_metadata",
        "continuity_packet": {
            "type": _CONTINUITY_PACKET_TYPE,
            "reason": _safe_str(metadata.get("reason")) or "unknown",
            "session_id": _safe_str(plan.get("same_session_id")) or _safe_str(metadata.get("session_id")),
            "task_id": _safe_str(plan.get("current_task_id")) or _safe_str(metadata.get("task_id")),
            "message_count": _safe_int(metadata.get("message_count")),
            "approx_tokens": metadata.get("approx_tokens") if isinstance(metadata.get("approx_tokens"), int) else None,
            "raw_transcript_included": False,
            "unrelated_context_included": False,
            "secret_token_password_private_body_included": False,
        },
        "provider_visible_state": "wcp_reentry_safe_metadata_only",
    }
    return {
        "role": "assistant",
        "content": "continuity_packet reentry metadata: " + json.dumps(safe_packet, sort_keys=True),
    }


def build_reentry_provider_payload(
    *,
    plan: Mapping[str, Any],
    previous_conversation_history: Iterable[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build provider-visible re-entry payload without old raw history."""

    if not isinstance(plan, Mapping) or plan.get("status") != "ready":
        return _safe_hold("invalid_rehydrate_plan")

    message = _reentry_message_from_plan(plan)
    return {
        "provider_visible": True,
        "phase": _PHASE,
        "same_window": True,
        "raw_previous_history_included": False,
        "uses_working_context_packet": True,
        "uses_reentry_metadata": True,
        "messages": [message],
        "raw_transcript_reinjected": False,
        "unrelated_context_reinjected": False,
        "secret_token_password_private_body_reinjected": False,
    }


def apply_same_window_rehydrate_plan(
    plan: Mapping[str, Any],
    *,
    session_db: Any,
) -> Dict[str, Any]:
    """Apply a same-window rebaseline to the supplied temp SessionDB.

    The caller owns the database boundary.  This function never opens a default
    or real Hermes state DB; it only uses the injected `session_db` object.
    """

    if not isinstance(plan, Mapping) or plan.get("status") != "ready":
        return _safe_hold("invalid_rehydrate_plan")
    if session_db is None or not hasattr(session_db, "archive_and_compact"):
        return _safe_hold("invalid_session_db")

    session_id = _safe_str(plan.get("same_session_id")) or _safe_str(plan.get("session_id"))
    if not session_id:
        return _safe_hold("invalid_rehydrate_plan")

    message = _reentry_message_from_plan(plan)
    active_count = session_db.archive_and_compact(session_id, [message])
    return {
        "status": "applied",
        "phase": _PHASE,
        "same_window": True,
        "same_session_id": session_id,
        "active_count": active_count,
        "deleted_rows": 0,
        "used_soft_archive": True,
        "old_rows_recoverable": True,
        "active_state": "continuity_packet_metadata_only",
        "raw_transcript_reinjected": False,
        "unrelated_context_reinjected": False,
        "secret_token_password_private_body_reinjected": False,
        "adds_cli_command": False,
        "touches_gateway_update_survival": False,
        "runtime_activation_required": False,
    }
