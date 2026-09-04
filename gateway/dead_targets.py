"""Persistent registry of delivery targets confirmed unreachable. Re-sending to a permanently gone chat
(deleted group, bot kicked/blocked, deactivated user) every cron tick wastes flood-control budget; delivery
short-circuits targets proven dead and any later successful send clears the flag. Only *whole-chat* deaths
(``forbidden``, chat-level ``not_found``) are recorded: adapters self-heal thread/topic-level ``not_found``
by retrying without ``reply_to``. Storage is a per-profile JSON file; reads/writes are best-effort (a
corrupt or unwritable file degrades to in-memory-only rather than raising on the delivery path).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

# classify_send_error kinds meaning the whole chat is unreachable (not transient / thread-level).
_DEAD_ERROR_KINDS = frozenset({"forbidden", "not_found"})


def _normalize(platform: str, chat_id: str) -> str:
    """Canonical key for a (platform, chat_id) pair."""
    return f"{str(platform).strip().lower()}:{str(chat_id).strip()}"


def classify_dead_error(error_text: Optional[str]) -> Optional[str]:
    """Best-effort dead-target error_kind from a raised error's text, else None. ``_deliver_to_platform``
    raises on hard failure (no SendResult), so ``deliver()`` only has the exception string. ``not_found``
    collapses chat-level and thread/topic/message-level failures: only a whole-chat not_found means the
    target is dead — a deleted forum topic or edited-away message must not mark the entire chat dead."""
    if not error_text:
        return None
    try:
        from .platforms.base import classify_send_error, is_chat_level_not_found
    except Exception:  # pragma: no cover - import guard
        return None
    kind = classify_send_error(None, error_text=error_text)
    dead = kind in _DEAD_ERROR_KINDS and (kind != "not_found" or is_chat_level_not_found(error_text=error_text))
    return kind if dead else None


class DeadTargetRegistry:
    """Thread-safe, persistent set of confirmed-dead targets keyed ``platform:chat_id``. Each entry stores
    reason + timestamp for observability; :meth:`clear` (called on a successful send) removes the flag."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._lock = threading.RLock()
        self._dead: Dict[str, Dict[str, object]] = {}
        self._path = path if path is not None else get_hermes_home() / "gateway" / "dead_targets.json"
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8")) if self._path.exists() else {}
            self._dead = {k: v for k, v in raw.items() if isinstance(v, dict)} if isinstance(raw, dict) else {}
        except (OSError, ValueError) as exc:
            logger.debug("dead_targets: could not load %s (%s) — starting empty", self._path, exc)

    def _flush_locked(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._dead, indent=2), encoding="utf-8")
            tmp.replace(self._path)
        except OSError as exc:  # best-effort: keep in-memory state, never break delivery
            logger.debug("dead_targets: could not persist %s (%s)", self._path, exc)

    def is_dead(self, platform: str, chat_id: Optional[str]) -> bool:
        with self._lock:
            return bool(chat_id) and _normalize(platform, chat_id) in self._dead

    def mark_dead(self, platform: str, chat_id: Optional[str], reason: str = "") -> bool:
        """Record a target as confirmed-dead. Returns True if newly added."""
        if not chat_id:
            return False
        key = _normalize(platform, chat_id)
        with self._lock:
            existed = key in self._dead
            self._dead[key] = {"platform": str(platform).strip().lower(), "chat_id": str(chat_id),
                               "reason": str(reason)[:200], "marked_at": time.time()}
            self._flush_locked()
        if not existed:
            logger.info("dead_targets: marked %s as unreachable (%s) — future deliveries "
                        "to this target will be skipped until a send succeeds", key, reason or "no reason given")
        return not existed

    def clear(self, platform: str, chat_id: Optional[str]) -> bool:
        """Remove a target's dead flag (self-healing). Returns True if it was set."""
        if not chat_id:
            return False
        key = _normalize(platform, chat_id)
        with self._lock:
            if self._dead.pop(key, None) is None:
                return False
            self._flush_locked()
            logger.info("dead_targets: cleared %s (delivery succeeded again)", key)
        return True
