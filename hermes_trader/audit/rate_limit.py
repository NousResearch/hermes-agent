"""Rate limit MCP write tools (execute_swap, submit_gasless_swap)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from hermes_trader.config import TRADER_HOME_SUBDIR


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def default_rate_limit_state_path() -> Path:
    return _hermes_home() / TRADER_HOME_SUBDIR / "write_rate_limit.json"


@dataclass
class WriteToolRateLimiter:
    max_per_hour: int = 10
    state_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.state_path is None:
            self.state_path = default_rate_limit_state_path()

    def _load_timestamps(self) -> List[float]:
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
        return [float(x) for x in data]

    def _save_timestamps(self, stamps: List[float]) -> None:
        path = self.state_path
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stamps), encoding="utf-8")

    def prune(self, now: Optional[float] = None) -> List[float]:
        current = now if now is not None else time.time()
        cutoff = current - 3600.0
        kept = [ts for ts in self._load_timestamps() if ts >= cutoff]
        self._save_timestamps(kept)
        return kept

    def allow(self, now: Optional[float] = None) -> bool:
        kept = self.prune(now=now)
        return len(kept) < self.max_per_hour

    def record(self, now: Optional[float] = None) -> None:
        current = now if now is not None else time.time()
        kept = self.prune(now=current)
        kept.append(current)
        self._save_timestamps(kept)

    def remaining(self, now: Optional[float] = None) -> int:
        kept = self.prune(now=now)
        return max(0, self.max_per_hour - len(kept))


def check_write_rate_limit(
    *,
    max_per_hour: int = 10,
    state_path: Optional[Path] = None,
    now: Optional[float] = None,
) -> tuple[bool, str]:
    limiter = WriteToolRateLimiter(max_per_hour=max_per_hour, state_path=state_path)
    if limiter.allow(now=now):
        return True, ""
    return (
        False,
        f"Write tool rate limit exceeded ({max_per_hour}/hour). "
        f"Retry after rolling window clears.",
    )