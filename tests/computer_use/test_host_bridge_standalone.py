"""Unit tests for the standalone CUA host bridge launcher.

Pure unit: no network, no real X11 server, no real cua-driver. Xvfb spawns are
faked to capture the child env, which is exactly the leak the H4 fix guards.
"""

import os
import sys
import subprocess
from unittest.mock import MagicMock

import pytest

from tools.computer_use import host_bridge_standalone as standalone


@pytest.fixture(autouse=True)
def _launcher_state(monkeypatch):
    """Fresh process-tracking state per test; the module keeps globals."""
    monkeypatch.setattr(standalone, "_spawned_procs", [])
    monkeypatch.setattr(standalone, "_display_env_modified", False)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HERMES_CUA_REMOTE_TOKEN", raising=False)
    monkeypatch.delenv("CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS", raising=False)
    monkeypatch.delenv("CUA_DRIVER_PERMISSION_MODE", raising=False)
    monkeypatch.delenv("HERMES_CUA_BRIDGE_ALLOW_PLAINTEXT", raising=False)


def test_sanitize_standalone_env_strips_secrets_keeps_standard():
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/u",
        "DISPLAY": ":42",
        "LANG": "C.UTF-8",
        "LC_ALL": "en_AU.UTF-8",
        "HERMES_CUA_REMOTE_TOKEN": "supersecret",
        "CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS": "1",
        "ANTHROPIC_API_KEY": "sk-secret",
    }
    child = standalone._sanitize_standalone_env(env)
    assert "HERMES_CUA_REMOTE_TOKEN" not in child
    assert "CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS" not in child
    assert "ANTHROPIC_API_KEY" not in child
    assert child["PATH"] == "/usr/bin:/bin"
    assert child["DISPLAY"] == ":42"
    assert child["LC_ALL"] == "en_AU.UTF-8"


def _fake_popen_record_env():
    """Return (factory, captured) — factory records env= kwarg of every Popen."""
    captured: list[dict] = []

    def fake_popen(cmd, **kwargs):
        captured.append({"cmd": cmd, **kwargs})
        proc = MagicMock()
        proc.pid = 4242
        return proc

    return fake_popen, captured


def test_xvfb_popen_env_has_no_token(monkeypatch):
    """H4 regression: Xvfb children must never inherit the bridge token."""
    monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", "supersecret-token")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(standalone.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(standalone, "_pick_free_display", lambda: ":99")
    monkeypatch.setattr(standalone, "_wait_for_x_ready", lambda display, env: None)
    fake_popen, captured = _fake_popen_record_env()
    monkeypatch.setattr(standalone.subprocess, "Popen", fake_popen)

    display = standalone._ensure_xvfb()
    assert display == ":99"
    assert len(captured) == 2  # Xvfb + openbox, both with explicit env=
    for entry in captured:
        env = entry["env"]
        assert env is not None, "every spawn must pass an explicit sanitized env="
        assert "HERMES_CUA_REMOTE_TOKEN" not in env
        assert env["DISPLAY"] == ":99"


def test_pick_free_display_skips_locked_and_stale_sockets(tmp_path):
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    socket_dir = tmp_path / "sockets"
    socket_dir.mkdir()
    # :99 and :100 have lock files; :101 has a stale socket — all skipped.
    (lock_dir / ".X99-lock").write_text("1234\n")
    (lock_dir / ".X100-lock").write_text("5678\n")
    (socket_dir / "X101").touch()

    picked = standalone._pick_free_display(
        lock_dir=str(lock_dir), socket_dir=str(socket_dir)
    )
    assert picked == ":102"

    # No free candidate left in :99-:109 → fail closed with the tried range.
    for candidate in range(99, 110):
        (lock_dir / f".X{candidate}-lock").write_text("1\n")
    with pytest.raises(RuntimeError, match=":99-:109"):
        standalone._pick_free_display(lock_dir=str(lock_dir), socket_dir=str(socket_dir))


def test_wait_for_x_ready_polls_xdpyinfo(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return MagicMock(returncode=0 if len(calls) >= 3 else 1)

    monkeypatch.setattr(standalone.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(standalone.subprocess, "run", fake_run)
    monkeypatch.setattr(standalone.time, "sleep", lambda s: None)
    monkeypatch.setattr(standalone.time, "monotonic", lambda: 0.0)  # never times out

    standalone._wait_for_x_ready(":99", {})
    assert len(calls) == 3
    assert calls[0] == ["/usr/bin/xdpyinfo", "-display", ":99"]


def test_split_list_arg_strips_and_drops_empties():
    value = " localhost:8765 , 127.0.0.1:8765 , , http://localhost:8765, "
    assert standalone._split_list_arg(value) == [
        "localhost:8765",
        "127.0.0.1:8765",
        "http://localhost:8765",
    ]


def test_split_list_arg_all_empty_yields_empty_list():
    assert standalone._split_list_arg(" , ,, ") == []


def _patch_bridge_imports(monkeypatch):
    """Standalone imports the bridge via sys.path insertion; no-op both."""
    fake_validate = lambda hosts, origins: (list(hosts), list(origins))
    fake_app = lambda **kwargs: object()
    import types

    fake_validation = types.ModuleType("host_validation")
    fake_validation.validate_security_allowlists = fake_validate
    fake_bridge = types.ModuleType("host_bridge")
    fake_bridge.create_host_bridge_app = fake_app
    monkeypatch.setitem(sys.modules, "host_validation", fake_validation)
    monkeypatch.setitem(sys.modules, "host_bridge", fake_bridge)


def test_main_refuses_non_loopback_plaintext(monkeypatch, capsys):
    """Fail-closed TLS gate: 0.0.0.0 without the ack never reaches Xvfb."""
    monkeypatch.setattr(sys, "argv", [
        "host_bridge_standalone.py",
        "--port", "8765",
        "--bind", "0.0.0.0",
        "--allowed-hosts", "localhost",
        "--allowed-origins", "http://localhost:8765",
    ])
    spawn_calls: list[str] = []
    monkeypatch.setattr(standalone.shutil, "which", lambda name: spawn_calls.append(name) or "/usr/bin/x")
    monkeypatch.setattr(standalone.subprocess, "Popen", MagicMock())

    with pytest.raises(RuntimeError, match="plaintext"):
        standalone.main()
    assert spawn_calls == []  # gate fires before anything is spawned


def test_main_allows_plaintext_with_ack_and_serves(monkeypatch):
    """With the ack + no-op'd heavy deps, main() completes and serves."""
    monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", "a" * 64)
    monkeypatch.setattr(sys, "argv", [
        "host_bridge_standalone.py",
        "--port", "8765",
        "--bind", "0.0.0.0",
        "--allowed-hosts", " localhost:8765 , 127.0.0.1:8765 ",
        "--allowed-origins", " http://localhost:8765 , http://127.0.0.1:8765 ",
    ])
    monkeypatch.setenv("HERMES_CUA_BRIDGE_ALLOW_PLAINTEXT", "1")
    monkeypatch.setenv("DISPLAY", ":77")  # skip the Xvfb path entirely
    fake_driver = os.path.join(os.path.expanduser("~"), ".local", "bin", "cua-driver")
    monkeypatch.setattr(standalone.os.path, "isfile", lambda p: p == fake_driver)
    monkeypatch.setattr(standalone.os, "access", lambda p, mode: True)
    _patch_bridge_imports(monkeypatch)

    serve_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        standalone, "_serve_app", lambda app, host, port: serve_calls.append((host, port))
    )

    standalone.main()
    assert serve_calls == [("0.0.0.0", 8765)]


def test_ensure_xvfb_noop_on_macos_without_display(monkeypatch):
    """M3: darwin has no X11 requirement — unset DISPLAY must not spawn/raise."""
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    spawn_calls: list[str] = []
    monkeypatch.setattr(
        standalone.shutil, "which", lambda name: spawn_calls.append(name) or "/usr/bin/x"
    )
    monkeypatch.setattr(standalone.subprocess, "Popen", MagicMock())

    assert standalone._ensure_xvfb() == ""
    assert spawn_calls == []  # nothing probed, nothing spawned
    assert "DISPLAY" not in os.environ


def test_ensure_xvfb_missing_binary_fails_closed(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(standalone.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Xvfb is not installed"):
        standalone._ensure_xvfb()


def test_shutdown_terminates_procs_and_restores_display(monkeypatch):
    procs = [MagicMock(), MagicMock()]
    monkeypatch.setattr(standalone, "_spawned_procs", list(procs))
    monkeypatch.setattr(standalone, "_display_env_modified", True)
    monkeypatch.setenv("DISPLAY", ":99")

    standalone._shutdown_xvfb()

    for proc in procs:
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=5)
    assert "DISPLAY" not in os.environ