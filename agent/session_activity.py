"""Shared session activity observation contract: timestamp + bounded description/provenance, observation
only (notification, timeout, kill and retry policy live elsewhere). Provenance is a small closed enum of
*noun* sources; the default agent clock stamps ``unknown`` unless a writer passes ``provenance=``."""

from __future__ import annotations

import time
from contextlib import suppress
from enum import Enum
from typing import Any, Mapping, Optional

ACTIVITY_DESCRIPTION_MAX = 120

# Durable SessionDB heartbeat cadence. Contract: MUST stay >= 30s — the SessionDB write path is contended and
# this observation-only projection never justifies extra write pressure. A code constant on purpose (no config
# can make it a high-frequency writer); matches the kanban auto-heartbeat. force_persist (terminal stamps) is the only bypass.
SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS = 60.0


class ActivityProvenance(str, Enum):
    """Where a durable/in-memory activity stamp came from."""

    UNKNOWN = "unknown"
    # Compression writers: heartbeat, host timeout, cooldown, turn hold.
    # See #72424.
    AGENT_COMPRESSION = "agent.compression"
    AGENT_COMPRESSION_TIMEOUT = "agent.compression_timeout"
    AGENT_COMPRESSION_COOLDOWN = "agent.compression_cooldown"
    AGENT_COMPRESSION_TURNHOLD = "agent.compression_turnhold"


def bound_activity_description(description: Optional[str]) -> str:
    """Clamp free-form activity text to the shared description budget."""
    text = (description or "").strip()
    return text if len(text) <= ACTIVITY_DESCRIPTION_MAX else text[: ACTIVITY_DESCRIPTION_MAX - 1] + "…"


def normalize_activity_provenance(provenance: Optional[ActivityProvenance | str]) -> ActivityProvenance:
    """Return a known provenance, or ``UNKNOWN`` when unset/unrecognized."""
    if isinstance(provenance, ActivityProvenance):
        return provenance
    try:
        return ActivityProvenance((provenance or "").strip())
    except ValueError:
        return ActivityProvenance.UNKNOWN


def reset_session_activity_persist_window(agent: Any) -> None:
    """Clear the persist rate-limit so the next stamp writes through (terminal compression labels must not stick on mid-compress text)."""
    with suppress(Exception):
        agent._session_activity_last_persist_mono = 0.0


def build_activity_snapshot(
    *,
    last_activity_at: Optional[float],
    last_activity_description: Optional[str],
    last_activity_provenance: Optional[ActivityProvenance | str] = None,
    now: Optional[float] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the shared activity snapshot (plus optional caller extras)."""
    when = float(last_activity_at) if last_activity_at is not None else None
    clock = float(now if now is not None else time.time())
    desc = bound_activity_description(last_activity_description)
    prov = normalize_activity_provenance(last_activity_provenance).value
    return {
        "last_activity_at": when,
        "last_activity_description": desc,
        "last_activity_provenance": prov,
        "seconds_since_activity": round(clock - when, 1) if when is not None else None,
        # Short aliases used by existing gateway/delegate readers.
        "last_activity_ts": when, "last_activity_desc": desc, "description": desc, "provenance": prov,
        **(extra or {}),
    }
