"""What ``hermes peer dm`` tells the sender when the turn does not come back.

A peer DM is one synchronous ``POST /api/sessions/{id}/chat`` with a 600 s client timeout, and the
receiving gateway runs the turn to completion whatever the client does. So a timeout while awaiting
the response means the message IS in the peer's Bot Chat and is being answered — the opposite of
unreachable. Reporting it as unreachable makes the sending bot resend, and the teammate runs the same
turn twice. A timeout while connecting means the request never arrived, and must stay unreachable.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import urllib.error
from types import SimpleNamespace

import pytest

from hermes_cli.subcommands import peer as peer_mod

SESSION = "20260916_bot_chat"


def _run(monkeypatch, capsys, *, base="http://192.168.2.55:8642", raise_on_post=None, raise_on_session=None):
    def _ensure(base, key):
        if raise_on_session is not None:
            raise raise_on_session
        return SESSION

    monkeypatch.setattr(peer_mod, "_ensure_bot_chat", _ensure)
    if raise_on_post is not None:
        def _request(url, key, **kwargs):
            raise raise_on_post

        monkeypatch.setattr(peer_mod, "_request", _request)
    code = peer_mod._peer_dm(SimpleNamespace(json=False), "hello", "mini", None, base, "key")
    return code, capsys.readouterr().err


@pytest.mark.parametrize(
    ("raise_on_post", "raise_on_session", "expected", "forbidden"),
    [
        # The ruling's fix: a bare post-accept timeout is the response-phase shape, so the
        # replay path reports the branch's "do not resend - outcome unknown", not upstream's
        # "accepted ... still running" (which is the queued-in-the-other-lane wording).
        (TimeoutError("timed out"), None, "do not resend", "Could not reach"),
        (urllib.error.URLError(TimeoutError("timed out")), None, "Could not reach peer", "still running"),
        (urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")), None, "Could not reach peer", "still running"),
        (OSError(51, "Network is unreachable"), None, "Could not reach peer", "still running"),
        (TimeoutError("timed out"), TimeoutError("timed out"), "Could not reach peer", "still running"),
        (urllib.error.HTTPError("http://peer/chat", 503, "Service Unavailable", {}, None), None,
         "rejected the request (HTTP 503)", "still running"),
    ],
    ids=["response-timeout", "connect-timeout", "connection-refused", "network-unreachable",
         "timeout-before-the-session-is-known", "http-rejection"],
)
def test_only_a_timeout_on_an_accepted_turn_is_reported_as_delivered(
    monkeypatch, capsys, raise_on_post, raise_on_session, expected, forbidden
):
    """Two things must both hold to call the message delivered: the Bot Chat was already resolved
    (the peer answered a moment ago) and the timeout hit while awaiting the response (urllib raises
    that one bare). A connect-phase timeout, a refusal, or a status code proves nothing arrived."""
    code, err = _run(monkeypatch, capsys, raise_on_post=raise_on_post, raise_on_session=raise_on_session)

    assert code == 1
    assert expected in err
    assert forbidden not in err
    if "still running" in expected:
        assert SESSION in err
        assert "Do NOT resend" in err


@pytest.mark.parametrize("phase", ["response", "connect"])
def test_the_real_urllib_stack_raises_the_signatures_the_branch_relies_on(monkeypatch, capsys, phase):
    """Through the real opener, not a hand-built exception: urllib wraps a connect/send failure in
    ``URLError`` and lets a timeout on the response through bare. Getting that backwards reports a
    message that never arrived as delivered, and the sender is told not to resend it."""
    for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(peer_mod, "DM_TIMEOUT_S", 0.5)

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    held = []
    done = threading.Event()

    def _accept_and_never_answer():
        with contextlib.suppress(OSError):  # the connect phase never dials, so close() ends accept()
            conn, _ = server.accept()
            held.append(conn)
            conn.recv(65536)
            done.wait(10)

    thread = threading.Thread(target=_accept_and_never_answer, daemon=True)
    thread.start()
    if phase == "connect":
        def _connect_times_out(*args, **kwargs):
            raise TimeoutError("timed out")

        monkeypatch.setattr(socket, "create_connection", _connect_times_out)
    try:
        code, err = _run(monkeypatch, capsys, base=f"http://127.0.0.1:{server.getsockname()[1]}")
    finally:
        done.set()
        server.close()
        for conn in held:
            conn.close()

    assert code == 1
    if phase == "response":
        assert "do not resend" in err and "outcome unknown" in err
    else:
        assert "Could not reach peer" in err and "still running" not in err
