"""An empty capture diagnoses its constructed desktop, not later configuration."""

import pytest

from tools.computer_use import cua_backend as cb


@pytest.mark.parametrize("remote", [False, True], ids=["local", "remote"])
def test_empty_capture_keeps_constructed_desktop(monkeypatch, remote):
    monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", "t" * 64)
    config = {"remote": {"enabled": remote, "url": "https://desktop.example.test/mcp"}}
    monkeypatch.setattr(cb, "_computer_use_cfg", lambda: config)
    backend = cb.CuaDriverBackend()
    monkeypatch.setattr(backend, "list_windows", lambda: [])
    monkeypatch.setattr(cb, "_linux_session_locked", lambda: True)

    config["remote"]["enabled"] = not remote
    capture = backend.capture(mode="ax")

    assert ("remote desktop returned no windows" in capture.window_title) is remote
    assert ("LOCKED" in capture.window_title) is not remote
