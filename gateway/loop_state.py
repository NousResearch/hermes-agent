"""Persistent runtime state for gateway loop protection.

The store is intentionally small and JSON-backed: gateways need a durable,
crash-safe way to remember recent agent-agent exchanges and temporary
quarantines, but this path must not introduce a database dependency or block
startup if the file is malformed.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_cli.config import get_hermes_home


_DEFAULT_STATE_FILE = "gateway-loop-guard-state.json"


def default_loop_state_path() -> Path:
    """Return the default loop guard state path for the active profile."""

    return get_hermes_home() / _DEFAULT_STATE_FILE


def text_fingerprint(text: str) -> str:
    """Stable hash for loop-novelty comparisons."""

    normalized = " ".join((text or "").lower().split())
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


class LoopStateStore:
    """JSON-backed rolling event/quarantine store.

    Shape on disk::

        {
          "version": 1,
          "events": {"email:a<->b": [{...}, ...]},
          "quarantines": {"email:a<->b": {"expires_at": 123, ...}}
        }
    """

    def __init__(self, path: str | os.PathLike[str] | None = None, *, max_events_per_pair: int = 80):
        self.path = Path(path) if path else default_loop_state_path()
        self.max_events_per_pair = max(10, int(max_events_per_pair or 80))

    @staticmethod
    def pair_key(platform: str, left: str, right: str) -> str:
        """Return a stable key for an unordered communication pair."""

        a = (left or "").strip().lower()
        b = (right or "").strip().lower()
        ordered = sorted([a, b])
        return f"{(platform or '').strip().lower()}:{ordered[0]}<->{ordered[1]}"

    def _empty(self) -> Dict[str, Any]:
        return {"version": 1, "events": {}, "quarantines": {}}

    def _load(self) -> Dict[str, Any]:
        try:
            if not self.path.exists():
                return self._empty()
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return self._empty()
            data.setdefault("version", 1)
            if not isinstance(data.get("events"), dict):
                data["events"] = {}
            if not isinstance(data.get("quarantines"), dict):
                data["quarantines"] = {}
            return data
        except Exception:
            # Corrupt state must never prevent the gateway from starting.
            return self._empty()

    def _save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp_name, self.path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    def _prune_expired_quarantines(self, data: Dict[str, Any], *, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        quarantines = data.setdefault("quarantines", {})
        expired = [key for key, item in quarantines.items() if float(item.get("expires_at", 0) or 0) <= now]
        for key in expired:
            quarantines.pop(key, None)
        return bool(expired)

    def get_quarantine(self, pair_key: str) -> Optional[Dict[str, Any]]:
        data = self._load()
        changed = self._prune_expired_quarantines(data)
        item = data.get("quarantines", {}).get(pair_key)
        if changed:
            self._save(data)
        if not item:
            return None
        return dict(item)

    def set_quarantine(
        self,
        pair_key: str,
        *,
        ttl_seconds: int,
        reason: str,
        category: str,
        now: float | None = None,
    ) -> Dict[str, Any]:
        now = time.time() if now is None else now
        ttl = max(1, int(ttl_seconds or 1))
        item = {
            "created_at": now,
            "expires_at": now + ttl,
            "reason": str(reason or "loop_guard"),
            "category": str(category or "unclear"),
        }
        data = self._load()
        self._prune_expired_quarantines(data, now=now)
        data.setdefault("quarantines", {})[pair_key] = item
        self._save(data)
        return dict(item)

    def clear_quarantine(self, pair_key: str) -> bool:
        data = self._load()
        removed = data.setdefault("quarantines", {}).pop(pair_key, None) is not None
        if removed:
            self._save(data)
        return removed

    def add_event(
        self,
        pair_key: str,
        *,
        direction: str,
        text: str,
        subject: str = "",
        risk: str = "low",
        category: str = "normal",
        action: str = "allow",
        stage: str = "",
        now: float | None = None,
    ) -> Dict[str, Any]:
        now = time.time() if now is None else now
        event = {
            "ts": now,
            "direction": str(direction or ""),
            "hash": text_fingerprint(text),
            "chars": len(text or ""),
            "subject": (subject or "")[:160],
            "risk": risk,
            "category": category,
            "action": action,
            "stage": stage,
        }
        data = self._load()
        events = data.setdefault("events", {}).setdefault(pair_key, [])
        events.append(event)
        if len(events) > self.max_events_per_pair:
            del events[:-self.max_events_per_pair]
        self._prune_expired_quarantines(data, now=now)
        self._save(data)
        return dict(event)

    def recent_events(self, pair_key: str, *, window_seconds: int = 1800, now: float | None = None) -> List[Dict[str, Any]]:
        now = time.time() if now is None else now
        cutoff = now - max(0, int(window_seconds or 0))
        data = self._load()
        events = data.get("events", {}).get(pair_key, [])
        if not isinstance(events, list):
            return []
        return [dict(item) for item in events if float(item.get("ts", 0) or 0) >= cutoff]

    def duplicate_count(
        self,
        pair_key: str,
        text: str,
        *,
        direction: str | None = None,
        window_seconds: int = 1800,
        now: float | None = None,
    ) -> int:
        fp = text_fingerprint(text)
        count = 0
        for event in self.recent_events(pair_key, window_seconds=window_seconds, now=now):
            if direction is not None and event.get("direction") != direction:
                continue
            if event.get("hash") == fp:
                count += 1
        return count
