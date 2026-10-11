"""Per-target backoff for recoverable Weixin sessions that are not ready."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SessionBackoffRegistry:
    """Persisted ``(account_id, chat_id)`` suppression with bounded re-probes."""

    def __init__(self, home: str, base_seconds: float = 30.0, max_seconds: float = 1800.0):
        self._path = Path(home) / "gateway" / "weixin_session_backoff.json"
        self._base_seconds = max(0.0, float(base_seconds))
        self._max_seconds = max(self._base_seconds, float(max_seconds))
        self._states: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        try:
            loaded = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self._states = {str(key): state for key, state in loaded.items()
                                if isinstance(state, dict)}
        except (OSError, ValueError) as exc:
            logger.debug("weixin session backoff: could not load %s (%s)", self._path, exc)

    def _flush_locked(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_suffix(self._path.suffix + ".tmp")
            temporary.write_text(json.dumps(self._states, indent=2), encoding="utf-8")
            temporary.replace(self._path)
        except OSError as exc:
            logger.debug("weixin session backoff: could not persist %s (%s)", self._path, exc)

    def should_suppress(self, account_id: str, chat_id: str,
                        now: Optional[float] = None) -> Tuple[bool, float, int, str]:
        state = self._states.get(self._key(account_id, chat_id))
        if not state:
            return False, 0.0, 0, ""
        current = time.time() if now is None else now
        remaining = max(0.0, float(state["next_probe_at"]) - current)
        return remaining > 0, remaining, int(state.get("consecutive_failures", 0)), str(state.get("last_error", ""))

    def record_failure(self, account_id: str, chat_id: str, error: str,
                       now: Optional[float] = None, *, threshold: int = 3) -> bool:
        key = self._key(account_id, chat_id)
        current = time.time() if now is None else now
        with self._lock:
            state = self._states.setdefault(key, {"short_failures": 0, "alerted": False})
            consecutive = int(state.get("consecutive_failures", 0)) + 1
            short_failures = int(state.get("short_failures", 0))
            delay = self._max_seconds if short_failures >= 2 else self._base_seconds * (2 ** short_failures)
            state.update(
                consecutive_failures=consecutive,
                short_failures=short_failures + 1,
                next_probe_at=current + min(delay, self._max_seconds),
                last_error=str(error)[:500],
                updated_at=current,
            )
            self._flush_locked()
            should_alert = consecutive >= threshold and not bool(state.get("alerted"))
            if should_alert:
                state["alerted"] = True
            return should_alert

    def clear(self, account_id: str, chat_id: str) -> bool:
        key = self._key(account_id, chat_id)
        with self._lock:
            if self._states.pop(key, None) is None:
                return False
            self._flush_locked()
            return True

    @staticmethod
    def _key(account_id: str, chat_id: str) -> str:
        return f"weixin:{account_id}:{chat_id}"
