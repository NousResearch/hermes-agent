"""UNIT 2 — ``hermes peer dm/run`` client side of the delivery envelope (§1, §3, §5.1).

Covers: the three delivery headers, the derived idempotency key, ENUM A -> exit
code mapping (a receipt is SUCCESS), and the do-not-resend text contract.
"""

from __future__ import annotations

import io
import json
import urllib.error
from types import SimpleNamespace

import pytest

from hermes_cli import urllib_security
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


# ── T6: a post-retention 404 on the READ path is not a delivery failure ──────


def _http_error(code: int, message: str = "boom") -> urllib.error.HTTPError:
    body = json.dumps({"error": {"message": message}}).encode()
    return urllib.error.HTTPError(
        "http://spark.lan:8377/v1/runs/run_gone", code, message,
        {"Content-Type": "application/json"}, io.BytesIO(body))


def _scripted_request(monkeypatch, steps: list):
    """Replace ``_request`` with a scripted responder; returns the call log."""
    calls: list[dict] = []

    def fake(url, api_key, **kw):
        calls.append({"url": url, "api_key": api_key, **kw})
        step = steps[min(len(calls) - 1, len(steps) - 1)]
        if isinstance(step, BaseException):
            raise step
        return step

    monkeypatch.setattr(peer_cmd, "_request", fake)
    return calls


def test_status_404_is_not_a_delivery_failure(peer_env, monkeypatch, capsys):
    """Spec T6/§7: a GC'd run row (404) is not a delivery signal."""
    _scripted_request(monkeypatch, [_http_error(404, "Run not found")])
    rc = peer_cmd.cmd_peer(_args(peer_action="status", run_id="run_gone", json=False))
    assert rc == 0
    out = capsys.readouterr().out
    assert "run_gone" in out
    assert "not a delivery failure" in out
    assert "HTTP 404" not in out


def test_status_404_json_is_an_unknown_envelope(peer_env, monkeypatch, capsys):
    _scripted_request(monkeypatch, [_http_error(404, "Run not found")])
    rc = peer_cmd.cmd_peer(_args(peer_action="status", run_id="run_gone"))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["object"] == "hermes.peer.run"
    assert payload["result"] == "unknown"
    assert payload["status"] == "unknown"
    assert payload["run_id"] == "run_gone"
    assert payload["reason"] == "run_not_found"
    assert payload["retryable"] is False
    assert payload["peer"] == "spark"


def test_status_500_still_exits_nonzero(peer_env, monkeypatch, capsys):
    """Guard the 404 carve-out against over-reach: every other code is unchanged."""
    _scripted_request(monkeypatch, [_http_error(500, "internal error")])
    rc = peer_cmd.cmd_peer(_args(peer_action="status", run_id="run_1"))
    assert rc == 1
    captured = capsys.readouterr()
    assert "HTTP 500" in captured.err
    assert captured.out == ""


def test_stop_404_keeps_todays_failure_semantics(peer_env, monkeypatch, capsys):
    """The carve-out is the READ path only — ``peer stop`` is untouched."""
    _scripted_request(monkeypatch, [_http_error(404, "Run not found")])
    rc = peer_cmd.cmd_peer(_args(peer_action="stop", run_id="run_gone"))
    assert rc == 1
    assert "HTTP 404" in capsys.readouterr().err


# ── D3: a read timeout on an accepted delivery is not "unreachable" ──────────


def _timeout() -> urllib.error.URLError:
    return urllib.error.URLError(TimeoutError("timed out"))


def test_dm_read_timeout_replays_the_same_key_and_reports_the_envelope(peer_env, monkeypatch, capsys):
    """The retry re-issues the identical request (same key, wait pinned to 1s)."""
    calls = _scripted_request(monkeypatch, [_timeout(), _envelope()])
    assert peer_cmd.cmd_peer(_args()) == 0
    assert len(calls) == 2
    first, second = calls
    assert first["url"] == second["url"]
    assert first["body"] == second["body"]
    assert first["headers"]["Idempotency-Key"] == second["headers"]["Idempotency-Key"]
    assert second["headers"]["X-Hermes-Wait-Seconds"] == "1"
    assert second["headers"]["X-Hermes-Sender-Profile"] == "yoyodine-coder"
    captured = capsys.readouterr()
    assert json.loads(captured.out)["result"] == "receipt"
    assert "unreachable" not in (captured.out + captured.err).lower()


def test_dm_replay_can_report_a_settled_result(peer_env, monkeypatch, capsys):
    """The replay returns whatever the reservation has since settled to."""
    settled = _envelope(result="delivered", status="delivered", reply="pong")
    _scripted_request(monkeypatch, [_timeout(), settled])
    assert peer_cmd.cmd_peer(_args()) == 0
    assert json.loads(capsys.readouterr().out)["reply"] == "pong"


def test_dm_double_timeout_reports_unknown_and_never_unreachable(peer_env, monkeypatch, capsys):
    """Both attempts lost: the delivery MAY have landed — never say unreachable."""
    calls = _scripted_request(monkeypatch, [_timeout(), _timeout()])
    rc = peer_cmd.cmd_peer(_args())
    assert rc == peer_cmd.RESULT_EXIT_CODES["unknown"] == 1
    assert len(calls) == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["object"] == peer_cmd.SEND_RESULT_OBJECT
    assert payload["result"] == "unknown"
    assert payload["status"] == "unknown"
    assert payload["retryable"] is False
    assert "do not resend" in payload["detail"].lower()
    assert payload["idempotency_key"] == calls[0]["headers"]["Idempotency-Key"]
    text = (captured.out + captured.err).lower()
    assert "unreachable" not in text
    assert "could not reach peer" not in text


def test_dm_double_timeout_text_mode_says_do_not_resend(peer_env, monkeypatch, capsys):
    _scripted_request(monkeypatch, [_timeout(), _timeout()])
    assert peer_cmd.cmd_peer(_args(json=False)) == 1
    captured = capsys.readouterr()
    text = (captured.out + captured.err).lower()
    assert "do not resend" in text
    assert "unreachable" not in text


def test_bare_socket_timeout_is_also_a_read_timeout(peer_env, monkeypatch, capsys):
    """Real evidence: a timeout during the body read surfaces as ``socket.timeout``
    (== ``TimeoutError``), not wrapped in ``URLError``."""
    calls = _scripted_request(monkeypatch, [TimeoutError("timed out"), _envelope()])
    assert peer_cmd.cmd_peer(_args()) == 0
    assert len(calls) == 2
    assert calls[1]["headers"]["X-Hermes-Wait-Seconds"] == "1"


def test_non_delivery_transport_failure_still_reports_unreachable(monkeypatch, capsys):
    """No idempotency key ⇒ no reservation ⇒ the old honest wording stands,
    and no replay is attempted (a lookup timeout delivered nothing)."""
    monkeypatch.setattr(peer_cmd, "_load_peers", lambda: {"spark": {"url": "http://spark.lan:8377"}})
    monkeypatch.setattr(peer_cmd, "_peer_secret", lambda name: "k" * 20)
    monkeypatch.setattr(peer_cmd, "_sender_profile", lambda: "yoyodine-coder")
    calls = _scripted_request(monkeypatch, [_timeout()])
    assert peer_cmd.cmd_peer(_args()) == 1
    assert len(calls) == 1  # the Bot Chat lookup is not a delivery: no replay
    assert "Could not reach peer 'spark'" in capsys.readouterr().err


class _FakeResponse:
    """Stand-in for the ``urlopen`` response used by ``_request``."""

    def __init__(self, *, payload: dict | None = None, error: BaseException | None = None):
        self._body = json.dumps(payload).encode() if payload is not None else b""
        self._error = error

    def read(self, *args):
        if self._error is not None:
            raise self._error
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_delivery_replay_reuses_the_identical_wire_request(monkeypatch, capsys):
    """End-to-end through the real ``_request``: the replay re-sends the same
    URL, body and idempotency key, with a short wait budget and a short read
    ceiling. The first attempt times out mid-body-read (the real evidence)."""
    monkeypatch.setattr(peer_cmd, "_load_peers", lambda: {"spark": {"url": "http://spark.lan:8377"}})
    monkeypatch.setattr(peer_cmd, "_peer_secret", lambda name: "k" * 20)
    monkeypatch.setattr(peer_cmd, "_ensure_bot_chat", lambda base, key: "bc_1")
    monkeypatch.setattr(peer_cmd, "_sender_profile", lambda: "yoyodine-coder")

    wire: list[dict] = []

    def fake_open(request, *, timeout, **kw):
        wire.append({"url": request.full_url, "body": request.data, "timeout": timeout,
                     "headers": dict(request.header_items())})
        if len(wire) == 1:
            return _FakeResponse(error=TimeoutError("timed out"))
        return _FakeResponse(payload=_envelope())

    monkeypatch.setattr(urllib_security, "open_credentialed_url", fake_open)
    assert peer_cmd.cmd_peer(_args()) == 0

    assert len(wire) == 2
    first, second = wire
    first_headers = {k.lower(): v for k, v in first["headers"].items()}
    second_headers = {k.lower(): v for k, v in second["headers"].items()}
    assert first["url"] == second["url"]
    assert first["body"] == second["body"]
    assert first_headers["idempotency-key"] == second_headers["idempotency-key"]
    assert second_headers["x-hermes-wait-seconds"] == "1"
    assert second["timeout"] <= first["timeout"]
    assert json.loads(capsys.readouterr().out)["result"] == "receipt"
