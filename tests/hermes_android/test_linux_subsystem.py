import json
from pathlib import Path

from hermes_android.linux_subsystem import load_linux_subsystem_state, apply_linux_subsystem_env


def test_load_linux_subsystem_state_returns_none_when_missing(tmp_path):
    assert load_linux_subsystem_state(tmp_path / "files") is None


def test_apply_linux_subsystem_env_sets_terminal_backend_markers(tmp_path):
    files_dir = tmp_path / "files"
    state_dir = files_dir / "hermes-home" / "linux"
    state_dir.mkdir(parents=True)
    state = {
        "enabled": True,
        "android_abi": "arm64-v8a",
        "termux_arch": "aarch64",
        "prefix_path": str(state_dir / "prefix"),
        "bash_path": str(state_dir / "prefix" / "bin" / "bash"),
        "bin_path": str(state_dir / "prefix" / "bin"),
        "lib_path": str(state_dir / "prefix" / "lib"),
        "native_library_dir": str(state_dir / "native-lib"),
        "home_path": str(state_dir / "prefix" / "home"),
        "tmp_path": str(state_dir / "prefix" / "tmp"),
        "execution_mode": "embedded_termux",
        "packages": [{"name": "bash"}],
    }
    (state_dir / "linux-subsystem-state.json").write_text(json.dumps(state), encoding="utf-8")

    env_updates = apply_linux_subsystem_env(files_dir)

    assert env_updates["TERMINAL_ENV"] == "android_linux"
    assert env_updates["HERMES_ANDROID_LINUX_PREFIX"] == state["prefix_path"]
    assert env_updates["HERMES_ANDROID_SHELL"] == "/system/bin/sh"
    assert env_updates["HERMES_ANDROID_NATIVE_SHELL"] == state["bash_path"]
    assert env_updates["HERMES_ANDROID_LINUX_BASH"] == "/system/bin/sh"
    assert env_updates["HERMES_ANDROID_LINUX_NATIVE_BASH"] == state["bash_path"]
    assert env_updates["HERMES_ANDROID_NATIVE_LIB"] == state["native_library_dir"]
    assert env_updates["HERMES_ANDROID_ALLOW_PREFIX_BIN"] == ""
    assert str(Path(state["bash_path"]).parent) in env_updates["LD_LIBRARY_PATH"]
    assert state["native_library_dir"] in env_updates["LD_LIBRARY_PATH"]
    assert state["lib_path"] in env_updates["LD_LIBRARY_PATH"]
    assert env_updates["HERMES_ANDROID_LINUX_HOME"] == state["home_path"]
    assert env_updates["HERMES_ANDROID_LINUX_TMP"] == state["tmp_path"]
    assert env_updates["HOME"] == state["home_path"]
    assert env_updates["TMPDIR"] == state["tmp_path"]
    assert env_updates["TERMINAL_CWD"] == state["home_path"]


def test_apply_linux_subsystem_env_derives_native_lib_dir_from_shell_path(tmp_path):
    files_dir = tmp_path / "files"
    state_dir = files_dir / "hermes-home" / "linux"
    state_dir.mkdir(parents=True)
    native_dir = state_dir / "apk-lib"
    shell_path = native_dir / "libhermes_android_bash.so"
    state = {
        "enabled": True,
        "prefix_path": str(state_dir / "prefix"),
        "shell_path": str(shell_path),
        "bash_path": str(shell_path),
        "bin_path": str(state_dir / "prefix" / "bin"),
        "lib_path": str(state_dir / "prefix" / "lib"),
        "native_library_dir": "",
        "home_path": str(state_dir / "prefix" / "home"),
        "tmp_path": str(state_dir / "prefix" / "tmp"),
        "execution_mode": "embedded_termux",
    }
    (state_dir / "linux-subsystem-state.json").write_text(json.dumps(state), encoding="utf-8")

    env_updates = apply_linux_subsystem_env(files_dir)

    assert env_updates["HERMES_ANDROID_NATIVE_LIB"] == str(native_dir)
    assert env_updates["HERMES_ANDROID_SHELL"] == "/system/bin/sh"
    assert env_updates["HERMES_ANDROID_NATIVE_SHELL"] == str(shell_path)
    assert env_updates["LD_LIBRARY_PATH"].startswith(str(native_dir))
