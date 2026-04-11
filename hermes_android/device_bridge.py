from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_DEVICE_STATE_FILE = "android-device-state.json"
_WORKSPACE_LIMIT = 20
_DEFAULT_GLOBAL_ACTIONS = ["home", "back", "recents", "notifications", "quicksettings"]


def _hermes_home() -> Path:
    raw = os.getenv("HERMES_HOME", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.cwd().resolve()


def workspace_dir() -> Path:
    workspace = _hermes_home() / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _device_state_path() -> Path:
    return _hermes_home() / _DEVICE_STATE_FILE


def _scan_workspace_files(limit: int = _WORKSPACE_LIMIT) -> list[dict[str, Any]]:
    workspace = workspace_dir()
    candidates = [path for path in workspace.rglob("*") if path.is_file()]
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)

    payload: list[dict[str, Any]] = []
    for path in candidates[:limit]:
        stat = path.stat()
        payload.append(
            {
                "name": path.name,
                "relative_path": path.relative_to(workspace).as_posix(),
                "size_bytes": stat.st_size,
                "modified_epoch_ms": int(stat.st_mtime * 1000),
            }
        )
    return payload


def read_device_capabilities() -> dict[str, Any]:
    workspace = workspace_dir()
    payload: dict[str, Any] = {
        "workspace_path": str(workspace),
        "shared_tree_uri": "",
        "shared_tree_label": "",
        "accessibility_enabled": False,
        "accessibility_connected": False,
        "available_global_actions": list(_DEFAULT_GLOBAL_ACTIONS),
    }

    state_path = _device_state_path()
    if state_path.is_file():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            payload["state_error"] = f"Could not read Android device state: {exc}"
        else:
            for key in (
                "workspace_path",
                "shared_tree_uri",
                "shared_tree_label",
                "accessibility_enabled",
                "accessibility_connected",
                "available_global_actions",
            ):
                if key in loaded:
                    payload[key] = loaded[key]

    workspace_files = _scan_workspace_files()
    payload["workspace_files"] = workspace_files
    payload["workspace_file_count"] = len([path for path in workspace.rglob("*") if path.is_file()])
    payload["guide"] = (
        "Call android_device_status first to discover workspace_path, then use "
        "read_file/write_file/search_files/patch within that workspace."
    )
    return payload


def read_device_capabilities_json() -> str:
    return json.dumps(read_device_capabilities())
