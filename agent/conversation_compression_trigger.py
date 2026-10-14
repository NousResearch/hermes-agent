"""Human-readable attribution for a compaction's ``trigger`` label.

Sibling of ``agent/conversation_compression.py`` (the facade imports it at module level; this module must
never import the facade). The labels are the ones each call site already passes to ``compress_context``
(see ``hermes_cli.observability.shared_metrics_fields._COMPRESSION_TRIGGERS``); this module only renders
them for people, on the compacting status line. Telemetry keeps the raw label.
"""

from __future__ import annotations

from typing import Optional

# The label manual ``/compress`` surfaces pass explicitly (the same value ``force=True`` defaults to).
MANUAL_TRIGGER_REASON = "manual"

_TRIGGER_REASON_CLAUSES = {
    MANUAL_TRIGGER_REASON: "manual — you ran /compress",
    "idle": "deferred maintenance on idle resume",
    "turn_start_threshold": "crossed the compaction threshold at turn start",
    "engine_preflight": "context-engine preflight maintenance",
    "pre_api": "context pressure before the API call",
    "post_tool": "crossed the compaction threshold after tool output",
    "gateway_hygiene": "session-hygiene maintenance",
    "overflow": "the provider rejected an oversize request",
}
# ``overflow`` attempts also carry the error classifier's reason; it names which rejection fired.
_OVERFLOW_REASON_CLAUSES = {
    "payload_too_large": "the API rejected an oversize request — 413",
    "context_overflow": "context length exceeded",
    "long_context_tier": "long-context tier window reduction",
}


def compaction_reason_clause(trigger: Optional[str], overflow_reason: Optional[str] = None) -> str:
    """The ``" (why this fired)"`` suffix for a compaction status line.

    ``""`` for no label or the bare ``"auto"`` default (says nothing an automatic status line does not); an
    unrecognized label renders raw (``" (trigger: x)"``) so a new arm degrades to something readable."""
    label = (trigger or "").strip()
    if label in ("", "auto"):
        return ""
    if label == "overflow" and overflow_reason in _OVERFLOW_REASON_CLAUSES:
        return f" ({_OVERFLOW_REASON_CLAUSES[overflow_reason]})"
    described = _TRIGGER_REASON_CLAUSES.get(label)
    return f" ({described})" if described else f" (trigger: {label})"
