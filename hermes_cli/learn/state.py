"""Profile-local Learn mode state.

This module owns the small control-plane state for Learn. It deliberately
resolves ``get_hermes_home()`` at call time so dashboard requests scoped to a
profile write under that profile's HERMES_HOME.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now
from utils import atomic_replace

VALID_MODES = frozenset({"off", "learn"})
START_MODES = frozenset({"learn"})
VALID_STATES = frozenset({"stopped", "running", "paused"})

_STATE_FILE = "state.json"
_EVENTS_FILE = "events.jsonl"
_DATA_FILES = (_EVENTS_FILE, "opportunities.json", "reports.json", "suggestions.json")
_lock = threading.RLock()


def _learn_dir(home: Path | None = None) -> Path:
    return (home or get_hermes_home()).resolve() / "learn"


def _state_path(home: Path | None = None) -> Path:
    return _learn_dir(home) / _STATE_FILE


def _events_path(home: Path | None = None) -> Path:
    return _learn_dir(home) / _EVENTS_FILE


def _secure_file(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _now_iso() -> str:
    return _hermes_now().isoformat()


def _default_state() -> Dict[str, Any]:
    return {
        "mode": "off",
        "state": "stopped",
        "retention_days": 14,
        "allowlist": [],
        "denylist": [],
        "started_at": None,
        "paused_at": None,
        "resumed_at": None,
        "stopped_at": None,
        "data_deleted_at": None,
        "updated_at": None,
    }


def _load_state(home: Path | None = None) -> Dict[str, Any]:
    path = _state_path(home)
    if not path.exists():
        return _default_state()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_state()
    if not isinstance(raw, dict):
        return _default_state()

    state = _default_state()
    state.update(raw)
    if state.get("mode") not in VALID_MODES:
        state["mode"] = "off"
    if state.get("state") not in VALID_STATES:
        state["state"] = "stopped"
    state["allowlist"] = state.get("allowlist") if isinstance(state.get("allowlist"), list) else []
    state["denylist"] = state.get("denylist") if isinstance(state.get("denylist"), list) else []
    try:
        state["retention_days"] = max(1, min(int(state.get("retention_days", 14)), 365))
    except (TypeError, ValueError):
        state["retention_days"] = 14
    return state


def _save_state(state: Dict[str, Any], home: Path | None = None) -> None:
    learn_dir = _learn_dir(home)
    learn_dir.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["updated_at"] = _now_iso()
    fd, tmp_path = tempfile.mkstemp(dir=str(learn_dir), suffix=".tmp", prefix=".learn_state_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, _state_path(home))
        _secure_file(_state_path(home))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _count_events(home: Path | None = None) -> int:
    path = _events_path(home)
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def _public_status(state: Dict[str, Any], home: Path | None = None) -> Dict[str, Any]:
    resolved_home = (home or get_hermes_home()).resolve()
    runtime_state = state.get("state", "stopped")
    mode = state.get("mode", "off")
    return {
        **state,
        "mode": mode,
        "state": runtime_state,
        "enabled": mode != "off",
        "running": runtime_state == "running",
        "paused": runtime_state == "paused",
        "collected_event_count": _count_events(resolved_home),
        "hermes_home": str(resolved_home),
        "storage_path": str(_learn_dir(resolved_home)),
    }


def get_status() -> Dict[str, Any]:
    """Return public Learn status without exposing collected payloads."""
    home = get_hermes_home().resolve()
    with _lock:
        return _public_status(_load_state(home), home)


def start(*, mode: str = "learn") -> Dict[str, Any]:
    """Start Learn in ``mode``."""
    mode = (mode or "").strip().lower()
    if mode not in START_MODES:
        raise ValueError("mode must be 'learn'; other Learn modes are planned but not implemented in this MVP")
    home = get_hermes_home().resolve()
    with _lock:
        state = _load_state(home)
        state["mode"] = mode
        state["state"] = "running"
        state["started_at"] = state.get("started_at") or _now_iso()
        state["resumed_at"] = _now_iso()
        state["paused_at"] = None
        state["stopped_at"] = None
        _save_state(state, home)
        return _public_status(state, home)


def pause() -> Dict[str, Any]:
    """Pause Learn collection/analysis without changing the selected mode."""
    home = get_hermes_home().resolve()
    with _lock:
        state = _load_state(home)
        if state.get("mode") != "off":
            state["state"] = "paused"
            state["paused_at"] = _now_iso()
        _save_state(state, home)
        return _public_status(state, home)


def resume() -> Dict[str, Any]:
    """Resume Learn if a non-off mode is selected."""
    home = get_hermes_home().resolve()
    with _lock:
        state = _load_state(home)
        if state.get("mode") != "off":
            state["state"] = "running"
            state["resumed_at"] = _now_iso()
            state["paused_at"] = None
        _save_state(state, home)
        return _public_status(state, home)


def stop() -> Dict[str, Any]:
    """Stop Learn runtime activity while preserving mode/configuration."""
    home = get_hermes_home().resolve()
    with _lock:
        state = _load_state(home)
        state["state"] = "stopped"
        state["paused_at"] = None
        state["stopped_at"] = _now_iso()
        _save_state(state, home)
        return _public_status(state, home)


def _clean_filter_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item).strip().lower()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def update_config(
    *,
    allowlist: list[str] | None = None,
    denylist: list[str] | None = None,
    retention_days: int | None = None,
) -> Dict[str, Any]:
    """Update Learn collection controls for the active profile."""
    home = get_hermes_home().resolve()
    with _lock:
        state = _load_state(home)
        if allowlist is not None:
            state["allowlist"] = _clean_filter_list(allowlist)
        if denylist is not None:
            state["denylist"] = _clean_filter_list(denylist)
        if retention_days is not None:
            state["retention_days"] = max(1, min(int(retention_days), 365))
        _save_state(state, home)
        return _public_status(state, home)


def delete_data() -> Dict[str, Any]:
    """Delete collected Learn artifacts and stop runtime activity."""
    home = get_hermes_home().resolve()
    learn_dir = _learn_dir(home)
    with _lock:
        for name in _DATA_FILES:
            path = learn_dir / name
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        state = _load_state(home)
        state["state"] = "stopped"
        state["paused_at"] = None
        state["data_deleted_at"] = _now_iso()
        _save_state(state, home)
        return _public_status(state, home)
