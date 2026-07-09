"""Tests for the generic cua-driver spawn_prefix on CuaDriverBackend.

PR: ``computer_use: pluggable per-task backend provider``.

``spawn_prefix`` lets any container/remote runtime own the cua-driver
process — core stays runtime-agnostic. These tests pin the two synchronous,
easily-isolated surfaces that depend on the prefix:

* ``CuaDriverBackend.is_available()`` — True when a prefix is set (the
  binary lives in-target, not on the host PATH) and falls back to the host
  binary check otherwise.
* ``_CuaDriverSession._call_tool_via_cli`` — the CLI-fallback command is
  wrapped through the prefix and uses the prefix's env; without a prefix
  the legacy ``[_CUA_DRIVER_CMD, "call", ...]`` shape is preserved.

The async MCP spawn path (``_lifecycle_coro``) is exercised end-to-end in
the cua-driver integration suite; here we pin the command-shape contract
that the integration suite depends on.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.computer_use.cua_backend import CuaDriverBackend, _CuaDriverSession, _AsyncBridge


def _fake_proc(stdout: str) -> MagicMock:
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = ""
    proc.returncode = 0
    return proc


def test_is_available_true_with_spawn_prefix_even_without_host_binary():
    # Prefix mode: binary is in-target; host PATH/HERMES_CUA_DRIVER_CMD is N/A.
    with patch("tools.computer_use.cua_backend.cua_driver_binary_available", return_value=False), \
         patch("tools.computer_use.cua_backend.sys.platform", "linux"):
        b = CuaDriverBackend(spawn_prefix=["docker", "exec", "-i", "c1", "env", "DISPLAY=:1"])
        assert b.is_available() is True


def test_is_available_falls_back_to_host_binary_without_prefix():
    with patch("tools.computer_use.cua_backend.cua_driver_binary_available", return_value=True), \
         patch("tools.computer_use.cua_backend.sys.platform", "linux"):
        assert CuaDriverBackend().is_available() is True
    with patch("tools.computer_use.cua_backend.cua_driver_binary_available", return_value=False), \
         patch("tools.computer_use.cua_backend.sys.platform", "linux"):
        assert CuaDriverBackend().is_available() is False


def test_is_available_false_on_unsupported_platform_even_with_prefix():
    # Platform gate still applies in prefix mode — cua-driver itself must
    # support the host platform's input primitives.
    with patch("tools.computer_use.cua_backend.sys.platform", "freebsd13"):
        b = CuaDriverBackend(spawn_prefix=["docker", "exec", "-i", "c1"])
        assert b.is_available() is False


def _make_session(spawn_prefix=None, driver_path=None, spawn_env=None) -> _CuaDriverSession:
    # _call_tool_via_cli is synchronous and does not touch the bridge loop,
    # so an un-started bridge is fine.
    return _CuaDriverSession(
        _AsyncBridge(),
        spawn_prefix=spawn_prefix,
        driver_path=driver_path,
        spawn_env=spawn_env,
    )


def test_cli_fallback_cmd_wrapped_through_prefix():
    sess = _make_session(
        spawn_prefix=["docker", "exec", "-i", "hermes-wt-abc", "env", "DISPLAY=:1", "HOME=/config"],
        driver_path="/config/bin/cua-driver",
    )
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = list(cmd)
        captured["env"] = kw.get("env")
        return _fake_proc(json.dumps({"tree_markdown": "x", "element_count": 0, "screenshot_png_b64": ""}))

    with patch("subprocess.run", side_effect=fake_run):
        sess._call_tool_via_cli("get_window_state", {"x": 1}, timeout=15.0)
    cmd = captured["cmd"]
    assert cmd[:6] == ["docker", "exec", "-i", "hermes-wt-abc", "env", "DISPLAY=:1"]
    assert cmd[6] == "HOME=/config"
    assert cmd[7] == "/config/bin/cua-driver"
    assert cmd[8:11] == ["call", "get_window_state", json.dumps({"x": 1})]
    # Prefix mode does NOT route the screenshot through a host temp file
    # (the in-target driver can't write a host path); screenshot_out_file
    # must be absent from the call payload.
    payload = json.loads(cmd[10])
    assert "screenshot_out_file" not in payload


def test_cli_fallback_cmd_legacy_shape_without_prefix():
    sess = _make_session()
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = list(cmd)
        return _fake_proc(json.dumps({"tree_markdown": "x", "element_count": 0}))

    with patch("tools.computer_use.cua_backend._CUA_DRIVER_CMD", "cua-driver"), \
         patch("subprocess.run", side_effect=fake_run):
        sess._call_tool_via_cli("list_windows", {}, timeout=15.0)
    cmd = captured["cmd"]
    assert cmd[0] == "cua-driver"
    assert cmd[1:4] == ["call", "list_windows", json.dumps({})]


def test_cli_fallback_legacy_uses_shot_file_for_get_window_state():
    # Without a prefix, get_window_state still routes the screenshot through
    # a host temp file (the optimization the prefix path deliberately skips).
    sess = _make_session()
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = list(cmd)
        return _fake_proc(json.dumps({"tree_markdown": "x", "element_count": 0}))

    with patch("tools.computer_use.cua_backend._CUA_DRIVER_CMD", "cua-driver"), \
         patch("subprocess.run", side_effect=fake_run):
        sess._call_tool_via_cli("get_window_state", {}, timeout=15.0)
    payload = json.loads(captured["cmd"][3])
    assert "screenshot_out_file" in payload
    assert payload["screenshot_out_file"].startswith("/tmp/")  # mkstemp default dir
