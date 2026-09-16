"""What ``hermes peer dm`` tells the sender when the turn does not come back.

A peer DM is one synchronous ``POST /api/sessions/{id}/chat`` with a 600 s client timeout, and the
receiving gateway runs the turn to completion whatever the client does. So a read timeout means the
message IS in the peer's Bot Chat and is being answered — the opposite of unreachable. Reporting it
as unreachable makes the sending bot resend, and the teammate runs the same turn twice.
"""

from __future__ import annotations

import urllib.error
from types import SimpleNamespace

import pytest

from hermes_cli.subcommands import peer as peer_mod

SESSION = "20260916_bot_chat"


def _run(monkeypatch, capsys, *, raise_on_post, raise_on_session=None):
    def _ensure(base, key):
        if raise_on_session is not None:
            raise raise_on_session
        return SESSION

    def _request(url, key, **kwargs):
        raise raise_on_post

    monkeypatch.setattr(peer_mod, "_ensure_bot_chat", _ensure)
    monkeypatch.setattr(peer_mod, "_request", _request)
    code = peer_mod._peer_dm(SimpleNamespace(json=False), "hello", "mini", None,
                             "http://192.168.2.55:8642", "key")
    return code, capsys.readouterr().err


@pytest.mark.parametrize(
    ("raise_on_post", "raise_on_session", "expected", "forbidden"),
    [
        (TimeoutError("timed out"), None, "accepted the message but its turn is still running", "Could not reach"),
        (urllib.error.URLError(TimeoutError("timed out")), None, "accepted the message but its turn is still running", "Could not reach"),
        (urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")), None, "Could not reach peer", "still running"),
        (OSError(51, "Network is unreachable"), None, "Could not reach peer", "still running"),
        (TimeoutError("timed out"), TimeoutError("timed out"), "Could not reach peer", "still running"),
    ],
    ids=["read-timeout", "wrapped-read-timeout", "connection-refused", "network-unreachable",
         "timeout-before-the-session-is-known"],
)
def test_only_a_timeout_on_an_accepted_turn_is_reported_as_delivered(
    monkeypatch, capsys, raise_on_post, raise_on_session, expected, forbidden
):
    """The discriminator is whether the Bot Chat was already resolved: that proves the peer answered
    a moment ago, so a timeout on the turn itself is a slow answer, not an unreachable host. A
    timeout while still looking the session up proves nothing and stays 'could not reach'."""
    code, err = _run(monkeypatch, capsys, raise_on_post=raise_on_post, raise_on_session=raise_on_session)

    assert code == 1
    assert expected in err
    assert forbidden not in err
    if "still running" in expected:
        assert SESSION in err
        assert "Do NOT resend" in err


def test_an_http_rejection_still_reads_as_a_rejection(monkeypatch, capsys):
    """A status code is the peer's own answer: it never means the turn is running."""
    rejection = urllib.error.HTTPError("http://peer/chat", 503, "Service Unavailable", {}, None)

    code, err = _run(monkeypatch, capsys, raise_on_post=rejection)

    assert code == 1
    assert "rejected the request (HTTP 503)" in err
    assert "still running" not in err
