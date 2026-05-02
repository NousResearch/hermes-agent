from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .profile import DEFAULT_RUNTIME_PROFILE, RuntimeProfile

DEFAULT_EXECUTION_STATE = {
    "trade_date": "",
    "sell_count": 0,
    "buy_count": 0,
    "actions": [],
    "active_signal": None,
    "last_signal_id": None,
    "last_signal_action": None,
    "last_signal_status": None,
    "last_signal_at": None,
}


class TradingStateStore:
    def __init__(self, base_dir: str | Path, profile: RuntimeProfile = DEFAULT_RUNTIME_PROFILE):
        self.base_dir = Path(base_dir)
        self.profile = profile
        self.profile_dir = self.base_dir / "profiles" / self.profile.profile_id
        self.state_dir = self.profile_dir / "state" / "realtime"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.state_dir / name

    def load_json(self, name: str, default: Any = None) -> Any:
        path = self._path(name)
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def save_json(self, name: str, payload: Any) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._path(name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def append_jsonl(self, name: str, payload: dict) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self._path(name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def read_jsonl(self, name: str) -> list[dict]:
        path = self._path(name)
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    def load_execution_state(self) -> dict:
        state = self.load_json("execution_state.json", {}) or {}
        merged = dict(DEFAULT_EXECUTION_STATE)
        merged.update(state)
        merged["actions"] = list(merged.get("actions") or [])
        if merged.get("active_signal") in ({}, []):
            merged["active_signal"] = None
        return merged

    def save_execution_state(self, state: dict) -> None:
        merged = dict(DEFAULT_EXECUTION_STATE)
        merged.update(state or {})
        merged["actions"] = list(merged.get("actions") or [])
        if merged.get("active_signal") in ({}, []):
            merged["active_signal"] = None
        self.save_json("execution_state.json", merged)

    def load_pending_signal(self) -> dict:
        return self.load_json("pending_signal.json", {}) or {}

    def save_pending_signal(self, state: dict) -> None:
        self.save_json("pending_signal.json", state)

    def clear_pending_signal(self) -> None:
        self.save_pending_signal({})

    def load_push_state(self) -> dict:
        return self.load_json("push_state.json", {}) or {}

    def save_push_state(self, state: dict) -> None:
        self.save_json("push_state.json", state)

    def append_signal_send_history(self, event_type: str, payload: dict) -> None:
        self.append_jsonl(
            "signal_send_history.jsonl",
            {
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "event": event_type,
                **payload,
            },
        )

    def record_dispatch_event(self, event_type: str, payload: dict) -> None:
        self.append_jsonl(
            "dispatch_ledger.jsonl",
            {
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "event": event_type,
                **payload,
            },
        )


class OlinStateStore(TradingStateStore):
    def __init__(self, base_dir: str | Path):
        super().__init__(base_dir, profile=DEFAULT_RUNTIME_PROFILE)
