"""Deterministic admission gate for Truth Ledger candidates."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .identity import has_stable_speaker_id
from .redaction import contains_sensitive_material, redact_text

ALLOWED_SCOPES = {"user", "agent", "project", "world"}
ALLOWED_KINDS = {
    "preference",
    "identity",
    "decision",
    "constraint",
    "environment",
    "workflow",
    "project",
    "relationship",
    "deadline",
    "correction",
    "lesson",
    "commitment",
}
ALLOWED_EVIDENCE_TYPES = {"user_stated", "tool_verified", "joint_decision", "assistant_inferred"}
ALLOWED_OPERATIONS = {"assert", "confirm", "supersede", "retract"}

MAX_VALUE_BYTES = 2048

_PRIVATE_SOURCE_CHANNELS = {
    "private_tool_output",
    "private_document",
    "private_doc",
    "tool_private",
}

_DO_NOT_REMEMBER_RE = re.compile(
    r"\b(don['’]t remember|do not remember|off the record|forget (this|that))\b",
    re.IGNORECASE,
)

_NON_DURABLE_RE = re.compile(
    r"\b(running|working|in progress|for now|temporary|tmp|soon|in a minute|just for this run)\b",
    re.IGNORECASE,
)


def _reject(reason: str, detail: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {"admit": False, "status": "none", "reason": reason}
    if detail:
        payload["detail"] = detail
    return payload


def evaluate_candidate(candidate: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a candidate against hard admission gates.

    Returns a deterministic dict with admit/status/reason keys.
    """
    source_channel = str(metadata.get("source_channel") or "").strip().lower()
    if source_channel in _PRIVATE_SOURCE_CHANNELS:
        return _reject("private_source")

    source_text = str(metadata.get("source_text") or "")
    if _DO_NOT_REMEMBER_RE.search(source_text):
        return _reject("do_not_remember")

    scope = str(candidate.get("scope") or "").strip().lower()
    if scope not in ALLOWED_SCOPES:
        return _reject("invalid_scope")

    kind = str(candidate.get("kind") or "").strip().lower()
    if kind not in ALLOWED_KINDS:
        return _reject("invalid_kind")

    evidence_type = str(candidate.get("evidence_type") or "").strip().lower()
    if evidence_type not in ALLOWED_EVIDENCE_TYPES:
        return _reject("invalid_evidence")
    if evidence_type == "assistant_inferred":
        return _reject("inferred_evidence")

    operation = str(candidate.get("operation") or "").strip().lower()
    if operation and operation not in ALLOWED_OPERATIONS:
        return _reject("invalid_operation")

    if scope == "user" and not has_stable_speaker_id(metadata):
        return _reject("unknown_speaker")

    value = str(candidate.get("value") or "")
    if len(value.encode("utf-8")) > MAX_VALUE_BYTES:
        return _reject("oversize")

    if _DO_NOT_REMEMBER_RE.search(value):
        return _reject("do_not_remember")

    if contains_sensitive_material(value):
        redacted, _ = redact_text(value)
        return _reject("sensitive_value", detail=redacted)

    if _NON_DURABLE_RE.search(value):
        return _reject("non_durable")

    return {"admit": True, "status": "admit", "reason": None}
