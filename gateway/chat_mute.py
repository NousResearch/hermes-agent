"""Per-chat mute store for the gateway (``/mute`` / ``/unmute``).

Inspired by Poke (poke.com) and Devin's Slack-etiquette work at Cognition:
"be quiet" is a *state command handled by the harness*, not conversational
input the model gets to argue with. While a chat is muted the gateway drops
inbound conversational messages for that chat deterministically — no agent
turn, no tokens, no reply. Slash commands always pierce the mute so
``/unmute`` (and everything else) keeps working.
(Source: https://devin.ai/blog/devins-slack-etiquette)

State lives in ``{HERMES_HOME}/.chat_mutes.json`` so a mute survives gateway
restarts. Entries are keyed by ``"<platform>:<chat_id>"`` — chat-scoped, not
user-scoped: muting a group chat silences the agent for that whole chat.

Each entry: ``{"muted_at": <epoch>, "expires_at": <epoch|null>}``. A missing
or unparsable ``expires_at`` means indefinite. Expired entries are treated as
unmuted on read and pruned on the next write.

Reads never raise: a corrupt store reads as "nothing muted" (fail-open — a
broken marker file must never silence every chat).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_MUTE_STORE_FILENAME = ".chat_mutes.json"

# /mute duration argument: bare integer = minutes; suffixed forms m/h/d.
_DURATION_RE = re.compile(r"^(\d+)\s*(m|min|mins|minutes?|h|hr|hrs|hours?|d|days?)?$", re.IGNORECASE)

_UNIT_SECONDS = {
    "m": 60,
    "h": 3600,
    "d": 86400,
}


def _store_path() -> Path:
    return Path(get_hermes_home()) / _MUTE_STORE_FILENAME


def mute_key(platform_value: str, chat_id: Any) -> str:
    """Canonical store key for a chat: ``<platform>:<chat_id>``."""
    return f"{platform_value}:{chat_id}"


def _load_store() -> Dict[str, Dict[str, Any]]:
    path = _store_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as e:
        logger.debug("chat_mute: store read failed (%s); treating as empty", e)
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("chat_mute: corrupt store at %s; treating as empty", path)
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _save_store(data: Dict[str, Dict[str, Any]]) -> None:
    path = _store_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _entry_active(entry: Dict[str, Any], now: Optional[float] = None) -> bool:
    """True when the entry is a live (non-expired) mute."""
    now = time.time() if now is None else now
    expires_at = entry.get("expires_at")
    if expires_at is None:
        return True
    try:
        return float(expires_at) > now
    except (TypeError, ValueError):
        # Unparsable expiry degrades to indefinite: the user explicitly asked
        # for silence and only an explicit /unmute (or a valid expiry) ends it.
        return True


def is_chat_muted(platform_value: str, chat_id: Any) -> bool:
    """Return True when the chat is currently muted."""
    entry = _load_store().get(mute_key(platform_value, chat_id))
    if entry is None:
        return False
    return _entry_active(entry)


def get_mute_entry(platform_value: str, chat_id: Any) -> Optional[Dict[str, Any]]:
    """Return the live mute entry for a chat, or None when unmuted/expired."""
    entry = _load_store().get(mute_key(platform_value, chat_id))
    if entry is not None and _entry_active(entry):
        return entry
    return None


def set_chat_mute(
    platform_value: str,
    chat_id: Any,
    duration_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Mute a chat (indefinitely, or for ``duration_seconds``). Returns the entry."""
    now = time.time()
    entry: Dict[str, Any] = {
        "muted_at": now,
        "expires_at": (now + duration_seconds) if duration_seconds else None,
    }
    store = _prune_expired(_load_store(), now)
    store[mute_key(platform_value, chat_id)] = entry
    _save_store(store)
    return entry


def clear_chat_mute(platform_value: str, chat_id: Any) -> bool:
    """Unmute a chat. Returns True when a live mute was removed."""
    now = time.time()
    store = _load_store()
    key = mute_key(platform_value, chat_id)
    entry = store.pop(key, None)
    was_live = entry is not None and _entry_active(entry, now)
    _save_store(_prune_expired(store, now))
    return was_live


def _prune_expired(
    store: Dict[str, Dict[str, Any]], now: Optional[float] = None
) -> Dict[str, Dict[str, Any]]:
    now = time.time() if now is None else now
    return {k: v for k, v in store.items() if _entry_active(v, now)}


def parse_mute_duration(arg: str) -> Tuple[bool, Optional[float]]:
    """Parse a ``/mute`` duration argument.

    Returns ``(ok, seconds)``. ``seconds`` is None for an indefinite mute
    (empty arg or "on"). ``ok`` False means the argument was unrecognized.
    """
    arg = (arg or "").strip().lower()
    if arg in {"", "on"}:
        return True, None
    m = _DURATION_RE.match(arg)
    if not m:
        return False, None
    value = int(m.group(1))
    if value <= 0:
        return False, None
    unit = (m.group(2) or "m")[0].lower()
    return True, float(value * _UNIT_SECONDS[unit])


def format_remaining(entry: Dict[str, Any], now: Optional[float] = None) -> str:
    """Human-readable remaining time for a mute entry ('' when indefinite)."""
    expires_at = entry.get("expires_at")
    if expires_at is None:
        return ""
    now = time.time() if now is None else now
    try:
        # Ceil so a freshly set "2h" mute reads "2h 0m", not "1h 59m".
        remaining = max(0, -int(-(float(expires_at) - now) // 1))
    except (TypeError, ValueError):
        return ""
    if remaining >= 86400:
        return f"{remaining // 86400}d {(remaining % 86400) // 3600}h"
    if remaining >= 3600:
        return f"{remaining // 3600}h {(remaining % 3600) // 60}m"
    return f"{max(1, remaining // 60)}m"
