"""Dashboard PTY websocket start-failure close-reason contract.

Regression for #71349: when the server cannot start the chat (missing Node,
TUI build failure, bad profile, ...), ``/api/pty`` writes the reason into the
terminal AND closes with 1011 carrying the same user-facing message in the
close frame's reason, so the dashboard overlay can say why chat could not
start instead of the generic "the reason is printed above". Before this, the
close carried code 1011 with an empty reason.
"""

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import hermes_cli.web_server_chat as _web_server_chat


pytestmark = pytest.mark.platforms("posix")  # PTY bridge is POSIX-only


@pytest.fixture
def pty_client(monkeypatch, _isolate_hermes_home):
    from hermes_cli import web_server as ws

    monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    ws.app.state.pty_active_session_files = {}
    client = TestClient(ws.app)
    return ws, client, ws._SESSION_TOKEN


def test_pty_start_failure_close_carries_reason(pty_client, monkeypatch):
    """A launch SystemExit must reach the browser inside the 1011 close frame."""
    _ws, client, token = pty_client

    def boom(**kw):
        raise SystemExit(1)

    monkeypatch.setattr(_web_server_chat, "_resolve_chat_argv", boom)

    from hermes_cli.web_routers import chat_ws, chat_ws_errors

    expected = chat_ws_errors.CHAT_NEEDS_NODE
    with client.websocket_connect(
        f"/api/pty?token={token}&channel=launch-fail-chan"
    ) as conn:
        # The terminal pane gets the red sentence first...
        assert expected in conn.receive_text()
        # ... then the server closes 1011; the close frame repeats the
        # sentence (clamped to RFC 6455's 123-byte reason limit) so the
        # overlay can surface it even when the terminal text never
        # rendered (scrolled away, missed frame).
        with pytest.raises(WebSocketDisconnect) as exc:
            conn.receive_text()
    assert exc.value.code == 1011
    assert exc.value.reason == chat_ws._ws_close_reason(expected)


def test_pty_start_failure_reason_is_bounded(pty_client, monkeypatch):
    """The close reason is clamped to RFC 6455's 123-byte limit — a huge
    failure message must not crash the close handler."""
    from fastapi import HTTPException

    _ws, client, token = pty_client

    def boom(**kw):
        raise HTTPException(status_code=400, detail="x" * 500)

    monkeypatch.setattr(_web_server_chat, "_resolve_chat_argv", boom)

    with client.websocket_connect(
        f"/api/pty?token={token}&channel=huge-fail-chan"
    ) as conn:
        conn.receive_text()
        with pytest.raises(WebSocketDisconnect) as exc:
            conn.receive_text()
    assert exc.value.code == 1011
    assert 0 < len(exc.value.reason.encode("utf-8")) <= 123
