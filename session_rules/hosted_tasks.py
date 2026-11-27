"""The pure driver-task rules the store applies inside its driver, work-record and policy transactions, shared by the gateway and every backend.

The task payload normalization with its field sets and limits, the bound task manifest and the task input context. ``DriverValidationError`` is the persistence contract's class, and ``TASK_STATUSES`` and ``TERMINAL_STATUSES`` live with the contract in ``persistence.gateway.hosted_rooms``. Standard library, ``persistence.gateway.hosted_rooms`` and the sibling ``session_rules`` modules only.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from functools import partial
from typing import Any

from persistence.gateway.hosted_rooms import DriverValidationError
from session_rules.hosted_attachments import validate_task_manifest
from session_rules.hosted_rooms import bounded_int, compact_json, identifier, text


MAX_IDENTIFIER_CHARS = 128
MAX_PROMPT_BYTES = 128 * 1024
MAX_RESULT_JSON_BYTES = 256 * 1024
TERMINAL_TASK_RETENTION_SECONDS = 30 * 24 * 60 * 60
MAX_RETAINED_TERMINAL_TASKS = 2048
MAX_TASK_PRUNE_BATCH = 1000


_TASK_PAYLOAD_REQUIRED_FIELDS = frozenset({"target_profile", "prompt", "source_event_seq"})
_TASK_PAYLOAD_OPTIONAL_FIELDS = frozenset({"target_member_id", "input_context", "attachments", "lead_routing"})


_identifier = partial(identifier, error=DriverValidationError, max_chars=MAX_IDENTIFIER_CHARS)
_bounded_int = partial(bounded_int, error=DriverValidationError)


MAX_TASK_INPUT_EVENTS = 128


def validate_task_input(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"watermark", "event_seqs"}:
        raise ValueError("invalid task input context")
    watermark = value["watermark"]
    seqs = value["event_seqs"]
    if type(watermark) is not int or watermark < 0:
        raise ValueError("invalid task input watermark")
    if not isinstance(seqs, list) or not 1 <= len(seqs) <= MAX_TASK_INPUT_EVENTS:
        raise ValueError("task input context exceeded its bound")
    previous = watermark
    for seq in seqs:
        if type(seq) is not int or not previous < seq <= 2**63 - 1:
            raise ValueError(
                "task input sequences must be increasing after the watermark"
            )
        previous = seq
    return {"watermark": watermark, "event_seqs": list(seqs)}


def validate_bound_task_manifest(value: Any) -> list[dict[str, Any]]:
    """Validate store metadata plus the exact source event for each task input."""
    if not isinstance(value, list) or not value:
        raise DriverValidationError("attachments must be a non-empty list")
    if any(not isinstance(item, Mapping) or "event_id" not in item for item in value):
        raise DriverValidationError("task attachment requires event_id")
    event_ids = [_identifier(item["event_id"], label="attachment event_id") for item in value]
    try:
        manifest = validate_task_manifest([
            {key: val for key, val in item.items() if key != "event_id"} for item in value])
    except ValueError as exc:
        raise DriverValidationError(str(exc)) from exc
    return [{**item, "event_id": event_id} for item, event_id in zip(manifest, event_ids)]


def normalize_task_payload(value: Any) -> tuple[dict[str, Any], str, str]:
    if not isinstance(value, dict):
        raise DriverValidationError("payload must be an object")
    unknown = set(value) - _TASK_PAYLOAD_REQUIRED_FIELDS - _TASK_PAYLOAD_OPTIONAL_FIELDS
    missing = _TASK_PAYLOAD_REQUIRED_FIELDS - set(value)
    if unknown:
        raise DriverValidationError(f"unknown payload fields: {', '.join(sorted(unknown))}")
    if missing:
        raise DriverValidationError(f"missing payload fields: {', '.join(sorted(missing))}")
    target_profile = _identifier(value["target_profile"], label="target_profile")
    prompt = text(value["prompt"], error=DriverValidationError, label="prompt", max_bytes=MAX_PROMPT_BYTES, strip=False)
    source_event_seq = _bounded_int(
        value["source_event_seq"], message="source_event_seq must be a positive integer", low=1)
    normalized = {"target_profile": target_profile, "prompt": prompt, "source_event_seq": source_event_seq}
    if "input_context" in value:
        try:
            normalized["input_context"] = validate_task_input(value["input_context"])
        except ValueError as exc:
            raise DriverValidationError(str(exc)) from exc
    if "target_member_id" in value:
        normalized["target_member_id"] = _identifier(value["target_member_id"], label="target_member_id")
    if "attachments" in value:
        normalized["attachments"] = validate_bound_task_manifest(value["attachments"])
    if "lead_routing" in value:
        # Present only on a lead's routing turn, and then exactly true: absence is an ordinary member turn.
        if value["lead_routing"] is not True:
            raise DriverValidationError("lead_routing must be true when present")
        normalized["lead_routing"] = True
    encoded = compact_json(normalized)
    return normalized, encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()
