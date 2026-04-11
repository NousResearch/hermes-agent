from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_DEVICE_STATE_FILE = "android-device-state.json"
_WORKSPACE_LIMIT = 20
_DEFAULT_GLOBAL_ACTIONS = ["home", "back", "recents", "notifications", "quicksettings"]
_SHARED_FOLDER_TOOLS = [
    "android_shared_folder_list",
    "android_shared_folder_read",
    "android_shared_folder_write",
]
_ACCESSIBILITY_TOOLS = [
    "android_ui_snapshot",
    "android_ui_action",
]
_SHARED_FOLDER_BRIDGE_CLASS = "com.nousresearch.hermesagent.device.HermesSharedFolderBridge"
_ACCESSIBILITY_BRIDGE_CLASS = "com.nousresearch.hermesagent.device.HermesAccessibilityUiBridge"


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


def _load_json_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        return json.loads(raw)
    raise TypeError(f"Unsupported Android bridge payload type: {type(raw).__name__}")


def _call_android_bridge(class_name: str, method_name: str, *args: Any) -> Any:
    try:
        from java import jclass  # type: ignore
    except Exception as exc:  # pragma: no cover - only exercised on Android
        raise RuntimeError(
            "Chaquopy Java bridge is unavailable in this environment"
        ) from exc

    bridge = jclass(class_name)
    method = getattr(bridge, method_name)
    return method(*args)


def _call_shared_folder_bridge(method_name: str, *args: Any) -> Any:
    return _call_android_bridge(_SHARED_FOLDER_BRIDGE_CLASS, method_name, *args)


def _call_accessibility_bridge(method_name: str, *args: Any) -> Any:
    return _call_android_bridge(_ACCESSIBILITY_BRIDGE_CLASS, method_name, *args)


def _bridge_json_result(caller, method_name: str, *args: Any) -> dict[str, Any]:
    try:
        return _load_json_payload(caller(method_name, *args))
    except Exception as exc:
        return {"error": str(exc)}


def list_shared_folder_entries(relative_path: str = "", *, recursive: bool = False, limit: int = 100) -> dict[str, Any]:
    return _bridge_json_result(_call_shared_folder_bridge, "listEntriesJson", relative_path, recursive, limit)


def read_shared_folder_file(relative_path: str, *, max_chars: int = 100_000) -> dict[str, Any]:
    return _bridge_json_result(_call_shared_folder_bridge, "readTextFileJson", relative_path, max_chars)


def write_shared_folder_file(relative_path: str, content: str, *, create_directories: bool = True) -> dict[str, Any]:
    return _bridge_json_result(_call_shared_folder_bridge, "writeTextFileJson", relative_path, content, create_directories)


def read_accessibility_snapshot(*, limit: int = 80) -> dict[str, Any]:
    return _bridge_json_result(_call_accessibility_bridge, "snapshotJson", limit)


def perform_accessibility_action(
    *,
    action: str,
    text_contains: str = "",
    content_description_contains: str = "",
    view_id: str = "",
    package_name: str = "",
    value: str = "",
    index: int = 0,
) -> dict[str, Any]:
    return _bridge_json_result(
        _call_accessibility_bridge,
        "performActionJson",
        action,
        text_contains,
        content_description_contains,
        view_id,
        package_name,
        value,
        index,
    )


def read_device_capabilities() -> dict[str, Any]:
    workspace = workspace_dir()
    payload: dict[str, Any] = {
        "workspace_path": str(workspace),
        "shared_tree_uri": "",
        "shared_tree_label": "",
        "accessibility_enabled": False,
        "accessibility_connected": False,
        "available_global_actions": list(_DEFAULT_GLOBAL_ACTIONS),
        "shared_folder_tools": list(_SHARED_FOLDER_TOOLS),
        "ui_targeting_tools": list(_ACCESSIBILITY_TOOLS),
        "linux_enabled": False,
        "linux_android_abi": "",
        "linux_termux_arch": "",
        "linux_prefix_path": "",
        "linux_bash_path": "",
        "linux_home_path": "",
        "linux_tmp_path": "",
        "linux_package_count": 0,
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
                "linux_enabled",
                "linux_android_abi",
                "linux_termux_arch",
                "linux_prefix_path",
                "linux_bash_path",
                "linux_home_path",
                "linux_tmp_path",
                "linux_package_count",
            ):
                if key in loaded:
                    payload[key] = loaded[key]

    workspace_files = _scan_workspace_files()
    payload["workspace_files"] = workspace_files
    payload["workspace_file_count"] = len([path for path in workspace.rglob("*") if path.is_file()])
    payload["shared_folder_granted"] = bool(payload["shared_tree_uri"])
    payload["guide"] = (
        "Grant a shared folder in the Device tab, then use android_shared_folder_list/read/write "
        "to work on those files directly. Use workspace read_file/write_file/search_files/patch only "
        "for imported copies or scratch files."
    )
    payload["linux_guide"] = (
        "The Android Linux suite exposes a real local bash-based command subsystem. Use terminal/process "
        "for direct CLI execution inside the extracted command prefix shown in linux_prefix_path."
    )
    payload["accessibility_guide"] = (
        "Enable Hermes accessibility, inspect the visible UI with android_ui_snapshot, then trigger a "
        "targeted click/focus/set_text/scroll action with android_ui_action."
    )
    return payload


def read_device_capabilities_json() -> str:
    return json.dumps(read_device_capabilities())
