"""Helpers for keeping forensic/quarantined rows out of live model context."""

from __future__ import annotations

from typing import Any, Mapping


def is_live_context_message(message: Mapping[str, Any]) -> bool:
    """Return True when *message* may be replayed to the model.

    ``active=False`` / ``active=0`` rows are forensic records (rewind tails,
    interrupted partial assistant output, malformed provider responses).  They
    must remain queryable with include_inactive=True, but must not be sent back
    to providers or summarized into compression output.
    """
    return message.get("active", True) not in (False, 0)


def live_context_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not messages:
        return []
    return [m for m in messages if isinstance(m, dict) and is_live_context_message(m)]
