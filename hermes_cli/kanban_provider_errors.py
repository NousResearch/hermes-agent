"""Bounded current-attempt provider diagnostics for worker supervision."""
from __future__ import annotations
import os
import re
from hermes_cli import kanban_db as kb

_PROVIDER_EGRESS_BLOCK_RE = re.compile(
    r"LLM\s+egress\s+blocked\s*:\s*([A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*)",
    re.IGNORECASE,
)

_KNOWN_PROVIDER_EGRESS_BLOCK_REASONS = frozenset(
    {
        "base64_payload",
        "base64_scan_failed",
        "exact_secret_detected",
        "exact_secret_scan_failed",
        "invalid_codex_reasoning_replay",
        "invalid_display_path",
        "invalid_generated_context_key",
        "invalid_generated_context_segment",
        "invalid_literal_segment",
        "invalid_request_key",
        "invalid_source_grant",
        "invalid_source_presentation",
        "invalid_source_segment",
        "invalid_typed_request_root",
        "missing_request_identity",
        "payload_digest_mismatch",
        "private_absolute_path",
        "private_path_scan_failed",
        "policy_digest_mismatch",
        "receipt_unavailable",
        "redaction_failed",
        "sanitized_bytes_exceeded",
        "sanitized_segment_bytes_exceeded",
        "secret_detected",
        "sensitive_path",
        "serialized_bytes_exceeded",
        "serialization_failed",
        "source_bytes_in_literal",
        "source_bytes_in_sanitized_segment",
        "source_hash_mismatch",
        "source_path_not_canonical",
        "source_policy_unavailable",
        "source_range_mismatch",
        "source_segment_grant_mismatch",
        "source_segment_not_text",
        "source_unavailable",
        "static_literal_not_allowed",
        "token_cap_exceeded",
        "typed_request_required",
        "untrusted_provenance",
    }
)

_PROVIDER_UNSUPPORTED_THINKING_RE = re.compile(
    r"(?:does\s+not\s+support\s+thinking|thinking\s+is\s+not\s+supported|unsupported\s+thinking)",
    re.IGNORECASE,
)

_PROVIDER_UNRESPONSIVE_RE = re.compile(
    r"Provider\s+has\s+been\s+unresponsive.*?aborting\s+this\s+call",
    re.IGNORECASE | re.DOTALL,
)

def _current_worker_log_tail(task_id: str) -> str | None:
    """Read only the current worker session from the shared task log.

    Task logs are append-only across retries.  Looking for a terminal provider
    error in the whole tail can therefore attribute an old egress denial to a
    newer, unrelated run (for example a local-model timeout).  Workers emit
    ``Initializing agent...`` at the start of each process; use the last such
    marker as the run boundary.  Keep the unmarked fallback for older logs and
    small callers/tests that predate the marker.
    """
    try:
        log_path = kb.worker_log_path(task_id)
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 16_384))
            tail = handle.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return None

    marker = "Initializing agent..."
    marker_index = tail.rfind(marker)
    if marker_index >= 0:
        return tail[marker_index:]
    return tail

def _provider_egress_error_text(task_id: str) -> str | None:
    """Classify a typed firewall denial before ordinary retry logic."""

    tail = _current_worker_log_tail(task_id)
    if tail is None:
        return None
    match = _PROVIDER_EGRESS_BLOCK_RE.search(tail)
    if match is None:
        return None
    reasons = tuple(dict.fromkeys(match.group(1).casefold().split(",")))
    if not reasons or not all(
        reason in _KNOWN_PROVIDER_EGRESS_BLOCK_REASONS for reason in reasons
    ):
        return None
    return f"provider egress blocked: LLM egress blocked: {','.join(reasons)}"

def _provider_terminal_error_text(task_id: str) -> tuple[str, str] | None:
    """Return a deterministic provider failure requiring a handoff."""

    tail = _current_worker_log_tail(task_id)
    if tail is None:
        return None
    match = _PROVIDER_EGRESS_BLOCK_RE.search(tail)
    if match is not None:
        reasons = tuple(dict.fromkeys(match.group(1).casefold().split(",")))
    else:
        reasons = ()
    if reasons and all(
        reason in _KNOWN_PROVIDER_EGRESS_BLOCK_REASONS for reason in reasons
    ):
        return (
            f"provider egress blocked: LLM egress blocked: {','.join(reasons)}",
            "provider_egress_blocked",
        )
    if _PROVIDER_UNSUPPORTED_THINKING_RE.search(tail):
        return (
            "provider rejected reasoning: selected model does not support thinking",
            "unsupported_thinking",
        )
    if _PROVIDER_UNRESPONSIVE_RE.search(tail):
        return (
            "provider unresponsive: aborted after repeated stale attempts",
            "provider_unresponsive",
        )
    return None
