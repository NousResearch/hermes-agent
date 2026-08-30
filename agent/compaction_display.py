"""Client-facing projection helpers for model-only compaction carriers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from agent.context_compressor import (
    ContextCompressor,
    is_compaction_summary_message,
)


_COMPACTION_INTERNAL_FIELDS = (
    "tool_calls",
    "finish_reason",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
)

# Legacy context-governor releases promoted orphaned raw tool outputs into
# assistant rows. The migration stamps only proven rows with this typed marker;
# display code must never infer the marker from the message text itself.
LEGACY_TOOL_CARRIER_QUARANTINE_METADATA_KEY = "legacy_tool_carrier_quarantine"
LEGACY_TOOL_CARRIER_QUARANTINE_SCHEMA = "LegacyToolCarrierQuarantineV1"


def _is_legacy_tool_carrier_quarantine(message: Dict[str, Any]) -> bool:
    """Return true for any legacy-carrier marker, including an unknown revision.

    ``display_metadata`` is the typed origin boundary.  A malformed or newer
    marker must fail closed: showing the accompanying historical content would
    recreate the leak if a producer forgets ``display_kind='hidden'``.
    """
    metadata = message.get("display_metadata")
    return bool(
        isinstance(metadata, dict)
        and isinstance(metadata.get(LEGACY_TOOL_CARRIER_QUARANTINE_METADATA_KEY), dict)
    )


def project_compaction_message_for_display(
    message: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return authentic transcript content, or ``None`` for a pure handoff.

    Model-facing recovery history retains the complete carrier. Display
    projections instead remove the handoff, inherited tool state, and internal
    reasoning while preserving any real prior-tail content or live user ask
    embedded in the carrier.
    """
    if not isinstance(message, dict):
        return None
    # ``display_kind`` is the durable presentation authority. The typed marker
    # below narrows legacy-carrier handling further, but every hidden control
    # row must stay out of a client transcript regardless of origin.
    if message.get("display_kind") == "hidden" or _is_legacy_tool_carrier_quarantine(
        message
    ):
        return None
    if not is_compaction_summary_message(message):
        return message.copy()

    projected = ContextCompressor._strip_context_summary_handoff_message(message)
    if projected is None:
        return None

    projected = projected.copy()
    for key in _COMPACTION_INTERNAL_FIELDS:
        projected.pop(key, None)
    projected.pop("display_kind", None)
    return projected
