"""Atomic rolling-window reservations for MCP write tools."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from hermes_trader.config import TRADER_HOME_SUBDIR

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def default_rate_limit_state_path() -> Path:
    return _hermes_home() / TRADER_HOME_SUBDIR / "write_rate_limit.json"


@contextmanager
def _state_lock(path: Path) -> Iterator[None]:
    """Serialize reservations in-process and across Hermes processes."""
    key = str(path.resolve())
    with _LOCKS_GUARD:
        thread_lock = _LOCKS.setdefault(key, threading.Lock())
    with thread_lock:
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + 5.0
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                break
            except FileExistsError:
                try:
                    stale = time.time() - lock_path.stat().st_mtime > 30.0
                    if stale:
                        lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out acquiring rate-limit lock {lock_path}")
                time.sleep(0.01)
        try:
            os.write(fd, str(os.getpid()).encode("ascii"))
            yield
        finally:
            os.close(fd)
            lock_path.unlink(missing_ok=True)


@dataclass
class WriteToolRateLimiter:
    max_per_hour: int = 10
    state_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.state_path is None:
            self.state_path = default_rate_limit_state_path()

    def _load_entries(self) -> list[dict[str, Any]]:
        path = self.state_path
        assert path is not None
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(data, list):
            return []
        entries: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, (int, float)):
                entries.append({"id": None, "timestamp": float(item), "status": "committed"})
            elif isinstance(item, dict):
                try:
                    entries.append({
                        "id": item.get("id"),
                        "timestamp": float(item["timestamp"]),
                        "status": str(item.get("status", "committed")),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
        return entries

    def _save_entries(self, entries: list[dict[str, Any]]) -> None:
        path = self.state_path
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(entries), encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _pruned(entries: list[dict[str, Any]], now: float) -> list[dict[str, Any]]:
        cutoff = now - 3600.0
        return [entry for entry in entries if entry["timestamp"] >= cutoff]

    def prune(self, now: Optional[float] = None) -> list[float]:
        current = now if now is not None else time.time()
        path = self.state_path
        assert path is not None
        with _state_lock(path):
            kept = self._pruned(self._load_entries(), current)
            self._save_entries(kept)
        return [entry["timestamp"] for entry in kept]

    def allow(self, now: Optional[float] = None) -> bool:
        return self.remaining(now=now) > 0

    def reserve(self, now: Optional[float] = None) -> Optional[str]:
        """Atomically consume capacity before dispatch; return a reservation id."""
        current = now if now is not None else time.time()
        path = self.state_path
        assert path is not None
        with _state_lock(path):
            kept = self._pruned(self._load_entries(), current)
            if len(kept) >= self.max_per_hour:
                self._save_entries(kept)
                return None
            reservation_id = uuid.uuid4().hex
            kept.append({"id": reservation_id, "timestamp": current, "status": "pending"})
            self._save_entries(kept)
            return reservation_id

    def reconcile(
        self,
        reservation_id: str,
        *,
        succeeded: bool,
        now: Optional[float] = None,
    ) -> None:
        """Commit successful dispatches and release capacity for failed calls."""
        path = self.state_path
        assert path is not None
        with _state_lock(path):
            current = now if now is not None else time.time()
            entries = self._pruned(self._load_entries(), current)
            updated: list[dict[str, Any]] = []
            for entry in entries:
                if entry.get("id") != reservation_id:
                    updated.append(entry)
                elif succeeded:
                    entry["status"] = "committed"
                    updated.append(entry)
            self._save_entries(updated)

    def record(self, now: Optional[float] = None) -> None:
        current = now if now is not None else time.time()
        reservation_id = self.reserve(now=current)
        if reservation_id is not None:
            self.reconcile(reservation_id, succeeded=True, now=current)

    def remaining(self, now: Optional[float] = None) -> int:
        return max(0, self.max_per_hour - len(self.prune(now=now)))


def reserve_write_rate_limit(
    *, max_per_hour: int = 10, state_path: Optional[Path] = None, now: Optional[float] = None
) -> tuple[Optional[str], str]:
    limiter = WriteToolRateLimiter(max_per_hour=max_per_hour, state_path=state_path)
    reservation_id = limiter.reserve(now=now)
    if reservation_id is not None:
        return reservation_id, ""
    return None, (
        f"Write tool rate limit exceeded ({max_per_hour}/hour). "
        "Retry after rolling window clears."
    )


def check_write_rate_limit(
    *, max_per_hour: int = 10, state_path: Optional[Path] = None, now: Optional[float] = None
) -> tuple[bool, str]:
    limiter = WriteToolRateLimiter(max_per_hour=max_per_hour, state_path=state_path)
    if limiter.allow(now=now):
        return True, ""
    return False, (
        f"Write tool rate limit exceeded ({max_per_hour}/hour). "
        "Retry after rolling window clears."
    )
