import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("VIRTUAL_OFFICE_DATA_ROOT", str(PROJECT_ROOT / "data")))
SETTINGS_PATH = DATA_ROOT / "config" / "settings.json"
DEFAULT_SETTINGS: dict[str, Any] = {
    "hermes_adapter_path": "adapters/hermes-adapter/main.py",
    "codex_adapter_path": "adapters/codex-adapter/main.py",
    "backend_port": 8647,
    "frontend_port": 5173,
    "codex_workdir": "D:\\Codex",
}


def _ensure_store() -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not SETTINGS_PATH.exists():
        SETTINGS_PATH.write_text(json.dumps(DEFAULT_SETTINGS, indent=2), encoding="utf-8")


def get_settings() -> dict[str, Any]:
    _ensure_store()
    try:
        raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    merged = dict(DEFAULT_SETTINGS)
    merged.update(raw)
    return merged


def save_settings(payload: dict[str, Any]) -> dict[str, Any]:
    current = get_settings()
    for key, value in payload.items():
        if key not in DEFAULT_SETTINGS:
            continue
        current[key] = value
    SETTINGS_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current
