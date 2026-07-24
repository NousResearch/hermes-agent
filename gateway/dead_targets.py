"""Persistent registry of delivery targets that are confirmed unreachable.

When a messaging platform reports that a target chat is permanently gone — a
deleted group (``Forbidden: the group chat was deleted``), a bot kicked/blocked,
or a deactivated user — re-sending to it on every cron tick or every fan-out
delivery wastes a send attempt against the platform's flood-control envelope and
spams the logs.  This registry lets the delivery layer short-circuit a target it
has already proven dead, while staying self-healing: any successful send to that
target clears the flag, so a user who re-adds the bot (or restores the chat)
recovers automatically with no manual cleanup.

Scope is deliberately narrow.  Only *whole-chat* deaths are recorded — the
``forbidden`` and chat-level ``not_found`` (``chat not found``) error kinds.
Thread/topic-level ``not_found`` is NOT recorded here: the adapters already
self-heal that by retrying without ``reply_to`` (see the Telegram adapter's
reply-target-deleted path), and a deleted topic does not mean the parent chat is
dead.

The store is a small JSON file under the active profile's HERMES_HOME so each
profile keeps its own dead set.  Reads/writes are best-effort: a corrupt or
unwritable file degrades to an in-memory-only registry rather than raising on
the delivery path.
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

# Error kinds (from gateway.platforms.base.classify_send_error) that mean the
# *whole chat* is unreachable, not a transient or thread-level problem.
_DEAD_ERROR_KINDS = frozenset({"forbidden", "not_found"})


def _normalize(platform: str, chat_id: str) -> str:
    """Canonical key for a (platform, chat_id) pair."""
    return f"{str(platform).strip().lower()}:{str(chat_id).strip()}"


class DeadTargetRegistry:
    """Thread-safe, persistent set of confirmed-dead delivery targets.

    Keyed on ``platform:chat_id``.  Stores the reason and a timestamp for
    observability.  Self-healing: :meth:`clear` (called on a successful send)
    removes the flag.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._lock = threading.RLock()
        self._dead: Dict[str, Dict[str, object]] = {}
        if path is not None:
            self._path = path
        else:
            self._path = get_hermes_home() / "gateway" / "dead_targets.json"
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text())
                if isinstance(raw, dict):
                    # Only keep well-shaped entries.
                    self._dead = {
                        k: v for k, v in raw.items() if isinstance(v, dict)
                    }
        except (OSError, ValueError) as exc:
            logger.debug("dead_targets: could not load %s (%s) — starting empty",
                         self._path, exc)
            self._dead = {}

    def _flush_locked(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._dead, indent=2), encoding="utf-8")
            tmp.replace(self._path)
        except OSError as exc:
            # Best-effort: keep the in-memory state, don't break delivery.
            logger.debug("dead_targets: could not persist %s (%s)", self._path, exc)

    # -- public API --------------------------------------------------------

    @staticmethod
    def is_dead_error_kind(error_kind: Optional[str]) -> bool:
        """Return True when ``error_kind`` denotes a permanent whole-chat death."""
        return bool(error_kind) and error_kind in _DEAD_ERROR_KINDS

    def is_dead(self, platform: str, chat_id: Optional[str]) -> bool:
        if not chat_id:
            return False
        with self._lock:
            return _normalize(platform, chat_id) in self._dead

    def mark_dead(self, platform: str, chat_id: Optional[str],
                  reason: str = "") -> bool:
        """Record a target as confirmed-dead.  Returns True if newly added."""
        if not chat_id:
            return False
        key = _normalize(platform, chat_id)
        with self._lock:
            existed = key in self._dead
            self._dead[key] = {
                "platform": str(platform).strip().lower(),
                "chat_id": str(chat_id),
                "reason": str(reason)[:200],
                "marked_at": time.time(),
            }
            self._flush_locked()
        if not existed:
            logger.info(
                "dead_targets: marked %s as unreachable (%s) — future deliveries "
                "to this target will be skipped until a send succeeds",
                key, reason or "no reason given",
            )
        return not existed

    def clear(self, platform: str, chat_id: Optional[str]) -> bool:
        """Remove a target's dead flag (self-healing).  Returns True if it was set."""
        if not chat_id:
            return False
        key = _normalize(platform, chat_id)
        with self._lock:
            if key in self._dead:
                del self._dead[key]
                self._flush_locked()
                logger.info("dead_targets: cleared %s (delivery succeeded again)", key)
                return True
        return False

    def all_dead(self) -> Dict[str, Dict[str, object]]:
        """Snapshot of the current dead set (for diagnostics / `hermes` CLI)."""
        with self._lock:
            return {k: dict(v) for k, v in self._dead.items()}
