"""Identity gating helpers for Truth Ledger admission."""

from __future__ import annotations

from typing import Any, Mapping, Optional

_UNKNOWN_SPEAKER_VALUES = {
    "",
    "unknown",
    "none",
    "null",
    "n/a",
    "na",
    "anonymous",
    "assistant",
}


def normalize_speaker_id(value: Any) -> Optional[str]:
    """Return a normalized stable speaker id or None when identity is unknown."""
    if value is None:
        return None
    speaker = str(value).strip()
    if not speaker:
        return None
    if speaker.lower() in _UNKNOWN_SPEAKER_VALUES:
        return None
    return speaker


def has_stable_speaker_id(metadata: Mapping[str, Any]) -> bool:
    """True when metadata includes a non-empty, non-unknown speaker id."""
    return normalize_speaker_id(metadata.get("speaker_id")) is not None
