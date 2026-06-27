"""Focused tests for dashboard PTY reconnect breadcrumbs."""

import json
import sys
from pathlib import Path
from urllib.parse import urlencode

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="PTY bridge is POSIX-only"
)


class _OneFrameBridge:
    def __init__(self):
        self._sent = False

    @classmethod
    def spawn(cls, *args, **kwargs):
        return cls()

    def read(self, timeout):
        if not self._sent:
            self._sent = True
            return b"ready"
        return None

    def resize(self, *, cols, rows):
        pass

    def write(self, raw):
        pass

    def close(self):
        pass


@pytest.fixture
def pty_client(monkeypatch, _isolate_hermes_home):
    from starlette.testclient import TestClient

    import hermes_cli.web_server as ws

    monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    monkeypatch.setattr(ws.PtyBridge, "spawn", _OneFrameBridge.spawn)
    ws.app.state.pty_active_session_files = {}

    client = TestClient(ws.app)
    return ws, client, ws._SESSION_TOKEN


def _url(token: str, **params: str) -> str:
    return f"/api/pty?{urlencode({'token': token, **params})}"


def test_resolve_chat_argv_sets_active_session_file_env(monkeypatch):
    """Dashboard chat gives the TUI a breadcrumb file for reconnect resume."""
    import hermes_cli.main as main_mod
    import hermes_cli.web_server as ws

    monkeypatch.setattr(
        main_mod,
        "_make_tui_argv",
        lambda project_root, tui_dev=False: (["node", "dist/entry.js"], "/tmp/ui-tui"),
    )

    _argv, _cwd, env = ws._resolve_chat_argv(
        active_session_file="/tmp/hermes-active-session.json"
    )

    assert env["HERMES_TUI_ACTIVE_SESSION_FILE"] == "/tmp/hermes-active-session.json"


def test_channel_reconnect_resumes_active_session_file(pty_client, monkeypatch):
    """A new /api/pty socket on the same channel resumes the last TUI sid."""
    ws, client, token = pty_client
    captured = []

    def fake_resolve(resume=None, sidecar_url=None, profile=None, active_session_file=None):
        captured.append(
            {
                "active_session_file": active_session_file,
                "resume": resume,
                "sidecar_url": sidecar_url,
            }
        )
        if active_session_file and not resume:
            Path(active_session_file).write_text(
                json.dumps({"session_id": "sess-live"}),
                encoding="utf-8",
            )
        return (["fake-hermes-tui"], None, None)

    monkeypatch.setattr(ws, "_resolve_chat_argv", fake_resolve)

    with client.websocket_connect(_url(token, channel="reconnect-chan")) as conn:
        assert conn.receive_bytes() == b"ready"

    with client.websocket_connect(_url(token, channel="reconnect-chan")) as conn:
        assert conn.receive_bytes() == b"ready"

    assert captured[0]["resume"] is None
    assert captured[0]["active_session_file"]
    assert captured[1]["resume"] == "sess-live"
    assert captured[1]["active_session_file"] == captured[0]["active_session_file"]


def test_fresh_param_ignores_channel_active_session_file(pty_client, monkeypatch):
    """Explicit fresh starts must not resurrect the prior channel session."""
    ws, client, token = pty_client
    channel = "fresh-chan"
    active_file = ws._active_session_file_for_channel(ws.app, channel)
    active_file.write_text(json.dumps({"session_id": "sess-old"}), encoding="utf-8")
    captured = {}

    def fake_resolve(resume=None, sidecar_url=None, profile=None, active_session_file=None):
        captured["active_session_file"] = active_session_file
        captured["resume"] = resume
        return (["fake-hermes-tui"], None, None)

    monkeypatch.setattr(ws, "_resolve_chat_argv", fake_resolve)

    with client.websocket_connect(_url(token, channel=channel, fresh="1")) as conn:
        assert conn.receive_bytes() == b"ready"

    assert captured["resume"] is None
    assert captured["active_session_file"] == str(active_file)
    assert not active_file.exists()
