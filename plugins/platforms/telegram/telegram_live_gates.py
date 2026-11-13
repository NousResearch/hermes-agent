"""Telegram allowlists that follow config.yaml without a reconnect.

The adapter's ``config.extra`` (``allow_from`` and friends) is built once, when the gateway loads
config; the single-profile YAML bridge copies the same lists into ``TELEGRAM_*`` env at that load.
:func:`refresh_telegram_gates` runs at every intake auth check and reloads both when the owning
profile's config.yaml changed since connect, so the next update is judged by the edited lists. An
unchanged file costs a stat.
"""
from __future__ import annotations

from typing import Any, Optional

from gateway.platforms._shared import platform_gate_env, refresh_live_gate_extra

# The ``extra`` keys an intake verdict reads (the adapter prefilter and the gateway authz union).
_LIVE_GATE_EXTRA_KEYS = ("allow_from", "group_allow_from", "allowed_chats", "group_allowed_chats", "allowed_topics",
                         "allow_bots")


def refresh_telegram_gates(adapter: Any) -> None:
    refresh_live_gate_extra(adapter, _LIVE_GATE_EXTRA_KEYS)


def env_allowlist_decision(user_id: str) -> Optional[bool]:
    """TELEGRAM_ALLOWED_USERS decision; None when no allowlist is configured."""
    allowed_csv = platform_gate_env("TELEGRAM_ALLOWED_USERS").strip()
    if not allowed_csv:
        return None
    allowed_ids = {uid.strip() for uid in allowed_csv.split(",") if uid.strip()}
    return "*" in allowed_ids or user_id in allowed_ids
