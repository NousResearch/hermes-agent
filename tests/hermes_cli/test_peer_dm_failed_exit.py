"""#136472: ``hermes peer dm`` exits 1 when the peer's agent failed the turn it accepted.

The message IS delivered (it stays in the peer's transcript, so a resend would run a second turn),
but nothing answered it — the command's own epilog contract ("1 delivery/peer error") applies. The
signal rides the completion object's additive ``failed`` / ``error`` / ``failure_reason`` keys.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli.subcommands import peer as peer_mod

SESSION = "20261011_bot_chat"


def _run(monkeypatch, capsys, payload):
    monkeypatch.setattr(peer_mod, "_ensure_bot_chat", lambda base, key: SESSION)
    monkeypatch.setattr(peer_mod, "_request", lambda url, key, **kwargs: payload)
    code = peer_mod._peer_dm(
        SimpleNamespace(json=False), "hello", "mini", None, "http://192.168.2.55:8642", "key")
    return code, capsys.readouterr().err


def _completion(**extra) -> dict:
    payload = {"object": "hermes.session.chat.completion", "session_id": SESSION,
               "message": {"role": "assistant", "content": "Billing or credits exhausted: HTTP 429"}}
    payload.update(extra)
    return payload


def test_a_failed_turn_exits_1_and_names_the_peer_failure(monkeypatch, capsys):
    code, err = _run(monkeypatch, capsys, _completion(
        failed=True, error="HTTP 429: monthly usage quota exceeded",
        failure_reason="provider_quota_limit"))
    assert code == 1, "a peer whose agent died is a delivery/peer error, not a success"
    assert "accepted the message but its turn failed" in err
    assert "HTTP 429: monthly usage quota exceeded" in err, "the peer's own failure text is shown"
    assert "provider_quota_limit" in err, "the typed reason rides along"


def test_a_failed_turn_without_error_fields_still_exits_1(monkeypatch, capsys):
    code, err = _run(monkeypatch, capsys, _completion(failed=True))
    assert code == 1
    assert "turn failed" in err, "no error text from the peer still names the failure"


def test_an_answered_turn_still_exits_0(monkeypatch, capsys):
    code, err = _run(monkeypatch, capsys, _completion())
    assert code == 0
    assert err == "", "an answered exchange reports nothing on stderr"


def test_a_queued_turn_is_not_mistaken_for_a_failed_one(monkeypatch, capsys):
    """The queued shape carries a ``status`` field but no ``failed`` flag; it must keep its own
    do-not-resense message and exit 0."""
    code, err = _run(monkeypatch, capsys, {
        "object": "hermes.session.chat.queued", "session_id": SESSION, "status": "queued"})
    assert code == 0
    assert "Bot Chat open" in err or err == ""
