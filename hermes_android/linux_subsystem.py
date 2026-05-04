from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_STATE_FILE_NAME = "linux-subsystem-state.json"


def linux_state_file(files_dir: str | Path) -> Path:
    return Path(files_dir).expanduser().resolve() / "hermes-home" / "linux" / _STATE_FILE_NAME


def load_linux_subsystem_state(files_dir: str | Path) -> dict[str, Any] | None:
    path = linux_state_file(files_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def apply_linux_subsystem_env(files_dir: str | Path) -> dict[str, str]:
    state = load_linux_subsystem_state(files_dir)
    if not state or not state.get("enabled"):
        return {}
    shell_path = str(state.get("shell_path") or state.get("bash_path") or "/system/bin/sh")
    native_library_dir = str(state.get("native_library_dir", ""))
    lib_path = str(state.get("lib_path", ""))
    ld_library_path = ":".join(
        item for item in [native_library_dir, lib_path, os.environ.get("LD_LIBRARY_PATH", "")] if item
    )
    execution_mode = str(state.get("execution_mode", "android_system_shell"))
    return {
        "TERMINAL_ENV": "android_linux",
        "HERMES_ANDROID_EXECUTION_MODE": execution_mode,
        "HERMES_ANDROID_SHELL": shell_path,
        "HERMES_ANDROID_LINUX_PREFIX": str(state.get("prefix_path", "")),
        "HERMES_ANDROID_LINUX_BASH": shell_path,
        "HERMES_ANDROID_LINUX_BIN": str(state.get("bin_path", "")),
        "HERMES_ANDROID_LINUX_LIB": lib_path,
        "HERMES_ANDROID_NATIVE_LIB": native_library_dir,
        "HERMES_ANDROID_ALLOW_PREFIX_BIN": "",
        "LD_LIBRARY_PATH": ld_library_path,
        "HERMES_ANDROID_LINUX_HOME": str(state.get("home_path", "")),
        "HERMES_ANDROID_LINUX_TMP": str(state.get("tmp_path", "")),
        "TERMINAL_CWD": str(state.get("home_path", "")),
    }
