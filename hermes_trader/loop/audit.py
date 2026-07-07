"""Append-only cycle audit log (JSONL) until P3 episodic memory ships."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_trader.config import TRADER_HOME_SUBDIR


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def default_cycle_log_path() -> Path:
    return _hermes_home() / TRADER_HOME_SUBDIR / "cycles.jsonl"


def _serialize(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return asdict(value)
    if hasattr(value, "value"):  # Enum
        return value.value
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    return value


class CycleAuditLog:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or default_cycle_log_path()

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            **record,
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_serialize(entry), ensure_ascii=False))
            handle.write("\n")