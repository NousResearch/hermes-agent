"""Remote empty results and direct CLI helpers must never query the local desktop."""

from types import SimpleNamespace

import pytest

from tools.computer_use import cua_backend, cua_backend_session
from tools.computer_use.cua_backend_capture import _CaptureMixin
from tools.computer_use.remote import RemoteCuaConfig


@pytest.fixture
def local_cli(monkeypatch):
    calls = []
    local_windows = [{"window_id": 41, "pid": 42, "app_name": "local-only-canary"}]
    monkeypatch.setattr(cua_backend_session._driver, "resolve_cua_driver_cmd", lambda: "/fake/cua-driver")
    monkeypatch.setattr(cua_backend, "cua_driver_child_env", lambda: {})

    def run(cmd, env, name, timeout):
        calls.append(name)
        return {"windows": local_windows}

    monkeypatch.setattr(cua_backend_session, "_cli_run_json", run)
    return calls


def _session(remote):
    config = RemoteCuaConfig("https://remote.example/mcp", "t" * 32) if remote else None
    return cua_backend_session._CuaDriverSession(SimpleNamespace(), remote_config=config)


@pytest.mark.parametrize("remote", [True, False], ids=["remote-empty-is-authoritative", "local-fallback-control"])
def test_empty_window_discovery_preserves_selected_machine(monkeypatch, local_cli, remote, caplog):
    session = _session(remote)
    monkeypatch.setattr(session, "call_tool", lambda *args, **kwargs: {"windows": []})

    class CaptureBackend(_CaptureMixin):
        _session = session
        _remote_config = session._remote_config
        _session_id = "test-session"

    windows = CaptureBackend().list_windows()
    if remote:
        assert local_cli == [], "empty remote discovery queried the local CLI"
        assert windows == [], "local windows must never replace an empty remote result"
        assert "re-fetching via CLI" not in caplog.text
    else:
        assert local_cli == ["list_windows"]
        assert [window["pid"] for window in windows] == [42]


def test_direct_remote_cli_call_is_rejected_before_local_io(monkeypatch, local_cli):
    resolved = []
    monkeypatch.setattr(cua_backend_session._driver, "resolve_cua_driver_cmd",
                        lambda: resolved.append(True) or "/fake/cua-driver")
    with pytest.raises(RuntimeError, match="local CLI.*remote"):
        _session(True)._call_tool_via_cli("click", {"x": 1, "y": 1}, 1)
    assert not resolved
    assert local_cli == []
