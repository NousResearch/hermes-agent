"""UNIT 2 — ``hermes peer dm/run`` client side of the delivery envelope (§1, §3, §5.1).

Covers: the three delivery headers, the derived idempotency key, ENUM A -> exit
code mapping (a receipt is SUCCESS), and the do-not-resend text contract.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hermes_cli.subcommands import peer as peer_cmd


def _args(**kw):
    base = {
        "peer_action": "dm",
        "target": "spark",
        "message": "ping",
        "json": True,
        "idempotency_key": None,
        "wait_seconds": None,
    }
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def peer_env(monkeypatch):
    """Isolate the peer registry + Bot Chat session, capturing every request."""
    calls: list[dict] = []

    monkeypatch.setattr(peer_cmd, "_load_peers", lambda: {"spark": {"url": "http://spark.lan:8377"}})
    monkeypatch.setattr(peer_cmd, "_peer_secret", lambda name: "k" * 20)
    monkeypatch.setattr(peer_cmd, "_ensure_bot_chat", lambda base, key: "bc_1")
    monkeypatch.setattr(peer_cmd, "_sender_profile", lambda: "yoyodine-coder")

    def fake_request(url, api_key, **kw):
        calls.append({"url": url, "api_key": api_key, **kw})
        return peer_env.response

    peer_env.response = {}
    monkeypatch.setattr(peer_cmd, "_request", fake_request)
    peer_env.calls = calls
    return peer_env


def _envelope(**kw):
    env = {
        "object": "hermes.peer.send_result",
        "result": "receipt",
        "status": "queued",
        "delivery_id": "d" * 32,
        "detail": "Target busy — delivered to its queue. Do not resend; receipt is retained.",
        "attempts": 0,
    }
    env.update(kw)
    return env


# ── headers (§5.1) ───────────────────────────────────────────────────────────


def test_dm_sends_the_three_delivery_headers(peer_env, capsys):
    peer_env.response = _envelope()
    assert peer_cmd.cmd_peer(_args()) == 0
    headers = peer_env.calls[0]["headers"]
    assert set(headers) == {
        "Idempotency-Key", "X-Hermes-Sender-Profile", "X-Hermes-Wait-Seconds",
    }
    assert headers["X-Hermes-Sender-Profile"] == "yoyodine-coder"
    assert headers["X-Hermes-Wait-Seconds"] == "600"  # peer.dm_wait_seconds default
    assert headers["Idempotency-Key"]


def test_wait_flag_sets_header_and_extends_http_timeout(peer_env, capsys):
    peer_env.response = _envelope()
    assert peer_cmd.cmd_peer(_args(wait_seconds=30.0)) == 0
    call = peer_env.calls[0]
    assert call["headers"]["X-Hermes-Wait-Seconds"] == "30"
    # The HTTP timeout must outlast the caller's budget so the peer's own receipt
    # window (bot_mode.receipt_after_seconds) always fits inside it.
    assert call["timeout"] >= 30.0


def test_derived_key_is_stable_across_retries(peer_env, capsys):
    """Same sender+target+session+message -> same key, so the peer dedups the retry."""
    keys = []
    for _ in range(2):
        peer_env.response = _envelope()
        assert peer_cmd.cmd_peer(_args()) == 0
        keys.append(peer_env.calls[-1]["headers"]["Idempotency-Key"])
    assert keys[0] == keys[1]
    assert len(keys[0]) >= 32


def test_explicit_idempotency_key_wins(peer_env, capsys):
    peer_env.response = _envelope()
    assert peer_cmd.cmd_peer(_args(idempotency_key="ticket-123")) == 0
    assert peer_env.calls[0]["headers"]["Idempotency-Key"] == "ticket-123"


def test_control_characters_in_key_are_refused_before_any_request(peer_env, capsys):
    peer_env.response = _envelope()
    assert peer_cmd.cmd_peer(_args(idempotency_key="bad\nkey")) == 2
    assert peer_env.calls == []
    assert "Idempotency key" in capsys.readouterr().err


def test_run_without_key_derives_instead_of_generating_a_uuid(peer_env, capsys):
    peer_env.response = {"run_id": "run_1", "status": "started", "replayed": False}
    assert peer_cmd.cmd_peer(_args(peer_action="run")) == 0
    post = [c for c in peer_env.calls if c["url"].endswith("/v1/runs")][-1]
    key = post["headers"]["Idempotency-Key"]
    assert not key.startswith("peer-")
    assert key.startswith("auto:")
    assert len(key) >= 32


# ── ENUM A -> exit codes (§1.5) ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("result", "code"),
    [("delivered", 0), ("receipt", 0), ("failed", 1), ("unknown", 1), ("refused", 2)],
)
def test_result_to_exit_code_mapping(peer_env, capsys, result, code):
    peer_env.response = _envelope(result=result, status="queued", reply="hi")
    assert peer_cmd.cmd_peer(_args()) == code


def test_receipt_is_success_and_prints_the_envelope(peer_env, capsys):
    peer_env.response = _envelope()
    assert peer_cmd.cmd_peer(_args()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["object"] == "hermes.peer.send_result"
    assert payload["result"] == "receipt"
    assert payload["status"] == "queued"
    assert payload["attempts"] == 0
    assert "peer" in payload and "profile" in payload
    assert "reoffer_count" not in payload
    assert "send_id" not in payload


def test_receipt_text_mode_says_do_not_resend_and_never_failed(peer_env, capsys):
    peer_env.response = _envelope()
    assert peer_cmd.cmd_peer(_args(json=False)) == 0
    out = capsys.readouterr().out
    assert "do not resend" in out.lower()
    assert "failed" not in out.lower()
    assert "idempotency_key:" in out
    assert "session_id: bc_1" in out


def test_failed_envelope_exits_1_and_still_emits_stdout_json(peer_env, capsys):
    peer_env.response = _envelope(result="failed", status="failed", error="queue_full")
    assert peer_cmd.cmd_peer(_args()) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"] == "failed"
    assert payload["error"] == "queue_full"


def test_refused_envelope_exits_2(peer_env, capsys):
    peer_env.response = _envelope(result="refused", status="cancelled", reason="lease_lost")
    assert peer_cmd.cmd_peer(_args()) == 2


def test_failed_text_mode_reports_on_stderr(peer_env, capsys):
    peer_env.response = _envelope(result="failed", status="expired", error="lease_expired")
    assert peer_cmd.cmd_peer(_args(json=False)) == 1
    captured = capsys.readouterr()
    assert "expired" in captured.err
    assert captured.out == ""


# ── legacy peers (no envelope) ───────────────────────────────────────────────


def test_legacy_peer_without_object_keeps_reply_shape(peer_env, capsys):
    peer_env.response = {"session_id": "bc_1", "message": {"content": "pong"}}
    assert peer_cmd.cmd_peer(_args()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["reply"] == "pong"
    assert payload["peer"] == "spark"


def test_legacy_peer_text_mode_prints_reply(peer_env, capsys):
    peer_env.response = {"session_id": "bc_1", "message": {"content": "pong"}}
    assert peer_cmd.cmd_peer(_args(json=False)) == 0
    assert capsys.readouterr().out.strip() == "pong"


# ── wait resolution ──────────────────────────────────────────────────────────


def test_wait_defaults_to_peer_dm_wait_seconds():
    assert peer_cmd._resolve_wait_seconds(_args(wait_seconds=None)) == 600.0


def test_wait_clamps_bad_values():
    assert peer_cmd._resolve_wait_seconds(_args(wait_seconds=-5)) == 1.0
