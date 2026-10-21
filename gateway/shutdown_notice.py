"""Notice-target keys and the record of chats warned that the gateway is going down.

Split out of ``gateway.run_shutdown`` (which is over its line cap) so the shutdown→startup
pair can share one module: ``run_shutdown`` records who it warned as the process stops, and
``run_notifications`` tells those chats the gateway is back on the next boot.

Only an in-chat ``/restart`` requester (``.restart_notify.json``) or a configured home channel
used to learn that the gateway had returned; a restart from outside the process left the
active chats that did get a "going down" notice waiting to ask "are you back?".
"""

from __future__ import annotations

import json
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def notice_target_key(platform_value: str, chat_id, thread_id) -> tuple:
    """Dedup key for one notice destination: thread/topic platforms share a chat but route apart."""
    return (platform_value, str(chat_id), str(thread_id) if thread_id else None)


def delivery_target_key(platform_value: str, chat_id, thread_id, *, profile: Optional[str] = None) -> tuple:
    """Dedupe key for one DELIVERED chat: profile-independent, except Telegram private chats.

    Two served profiles can share one home chat (one Telegram group for the whole host) and owe it
    ONE notice per host restart. A positive Telegram chat id names the USER, though: the same id
    under two bot tokens is two conversations, so those stay keyed per served profile (#118233).
    """
    from gateway.delivery import looks_like_telegram_private_chat_id

    if (profile and profile != "default" and platform_value == "telegram"
            and looks_like_telegram_private_chat_id(chat_id)):
        platform_value = f"{profile}:{platform_value}"
    return notice_target_key(platform_value, chat_id, thread_id)


def _shutdown_notice_marker(home: Path) -> Path:
    """Path of the record of chats warned that the gateway is going down."""
    return home / ".shutdown_notify.json"


def mark_shutdown_notified(home: Path, entries: list[dict]) -> None:
    """Persist the chats that received the shutdown notice (atomic, best-effort).

    Never raises: a shutdown must not be blocked by a bookkeeping write.
    """
    if not entries:
        return
    from utils import atomic_json_write

    try:
        atomic_json_write(_shutdown_notice_marker(home), {"targets": entries})
    except (OSError, TypeError, ValueError):
        logger.debug("Could not record the shutdown notice targets", exc_info=True)


def read_shutdown_notice_marker(home: Path) -> list[dict]:
    """Targets recorded by the last shutdown, or [] when absent/unreadable."""
    path = _shutdown_notice_marker(home)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return []
    targets = data.get("targets") if isinstance(data, dict) else None
    return [t for t in targets if isinstance(t, dict)] if isinstance(targets, list) else []


def clear_shutdown_notice_marker(home: Path) -> None:
    with suppress(OSError):
        _shutdown_notice_marker(home).unlink(missing_ok=True)


def record_shutdown_notice(home: Path, notified: Any, restart_source: Any) -> None:
    """Record the chats just told the gateway is going down, for the startup return notice.

    ``notified`` maps each delivery key ``(platform, chat_id, thread_id)`` to the profile whose bot
    delivered it. The key's platform may be ``<profile>:telegram`` (see ``delivery_target_key``), so
    the entry stores the bare platform plus the profile: the startup half needs both to resolve a
    ``Platform`` and the right bot. An in-chat ``/restart`` owns its own marker
    (``.restart_notify.json``), so it records nothing here.
    """
    if not notified or restart_source is not None:
        return
    entries = []
    for (platform, chat_id, thread_id), profile in notified.items():
        entry = {"platform": platform.rpartition(":")[2], "chat_id": chat_id, "thread_id": thread_id}
        if profile and profile != "default":
            entry["profile"] = profile
        entries.append(entry)
    mark_shutdown_notified(home, entries)
