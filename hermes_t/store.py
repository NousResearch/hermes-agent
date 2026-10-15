"""Generic profile-scoped state store for hermes_t runtime."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_DEFAULT_BASE_DIRNAME = ".hermes_t_runtime"
_EXECUTION_STATE_FILENAME = "execution_state.json"
_PENDING_SIGNAL_FILENAME = "pending_signal.json"
_POSITION_FILENAME = "position.json"
_PUSH_STATE_FILENAME = "push_state.json"
_ACTIVE_SIGNAL_FILENAME = "active_signal.json"
_DISPATCH_LEDGER_FILENAME = "dispatch_ledger.jsonl"
_SIGNAL_SEND_HISTORY_FILENAME = "signal_send_history.jsonl"
_VALID_PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_profile_id(profile_id: str) -> str:
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("profile_id must be a non-blank string")
    normalized = profile_id.strip()
    if not _VALID_PROFILE_ID_RE.fullmatch(normalized) or normalized in {".", ".."}:
        raise ValueError("profile_id must contain only letters, digits, dot, underscore, or hyphen")
    return normalized


class TradingStateStore:
    def __init__(self, base_dir: str | Path | None = None, *, profile_id: str) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else Path.home() / _DEFAULT_BASE_DIRNAME
        self.profile_id = validate_profile_id(profile_id)
        self.state_dir = self.base_dir / self.profile_id

    def _ensure_state_dir(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _json_path(self, filename: str) -> Path:
        return self.state_dir / filename

    def _jsonl_path(self, filename: str) -> Path:
        return self.state_dir / filename

    def _load_json(self, filename: str, *, default: Any) -> Any:
        path = self._json_path(filename)
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_json(self, filename: str, payload: Any) -> None:
        self._ensure_state_dir()
        self._json_path(filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_jsonl(self, filename: str) -> list[dict[str, Any]]:
        path = self._jsonl_path(filename)
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    def _append_jsonl(self, filename: str, row: dict[str, Any]) -> None:
        self._ensure_state_dir()
        path = self._jsonl_path(filename)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def load_execution_state(self) -> dict[str, Any]:
        return self._load_json(_EXECUTION_STATE_FILENAME, default={})

    def save_execution_state(self, payload: dict[str, Any]) -> None:
        self._save_json(_EXECUTION_STATE_FILENAME, payload)

    def load_pending_signal(self) -> dict[str, Any]:
        return self._load_json(_PENDING_SIGNAL_FILENAME, default={})

    def save_pending_signal(self, payload: dict[str, Any]) -> None:
        self._save_json(_PENDING_SIGNAL_FILENAME, payload)

    def clear_pending_signal(self) -> None:
        self.save_pending_signal({})

    def load_position(self) -> dict[str, Any] | None:
        return self._load_json(_POSITION_FILENAME, default=None)

    def save_position(self, payload: dict[str, Any]) -> None:
        self._save_json(_POSITION_FILENAME, payload)

    def load_push_state(self) -> dict[str, Any]:
        return self._load_json(_PUSH_STATE_FILENAME, default={})

    def save_push_state(self, payload: dict[str, Any]) -> None:
        self._save_json(_PUSH_STATE_FILENAME, payload)

    def load_active_signal(self) -> dict[str, Any]:
        return self._load_json(_ACTIVE_SIGNAL_FILENAME, default={})

    def save_active_signal(self, payload: dict[str, Any]) -> None:
        self._save_json(_ACTIVE_SIGNAL_FILENAME, payload)

    def read_dispatch_ledger(self) -> list[dict[str, Any]]:
        return self._read_jsonl(_DISPATCH_LEDGER_FILENAME)

    def append_dispatch_ledger(self, row: dict[str, Any]) -> None:
        self._append_jsonl(_DISPATCH_LEDGER_FILENAME, row)

    def read_signal_send_history(self) -> list[dict[str, Any]]:
        return self._read_jsonl(_SIGNAL_SEND_HISTORY_FILENAME)

    def append_signal_send_history(self, row: dict[str, Any]) -> None:
        self._append_jsonl(_SIGNAL_SEND_HISTORY_FILENAME, row)
