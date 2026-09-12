"""UNIT 3 — the gateway side of a peer delivery: admission, receipt, in-call turn.

Drives the real ``_handle_session_chat`` handler on a real adapter whose model
turn is stubbed (the same seam ``tests/gateway/test_peer_dm_hidden_e2e.py``
uses): HTTP-shaped requests, a real ``RunIdempotencyStore``, and the real
on-disk delivery queue under a tmp HERMES_HOME.
"""

import asyncio
import json
import os
import time
from types import SimpleNamespace

import pytest

from gateway.platforms import api_server as api
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from tools import bot_delivery_queue as q


@pytest.fixture()
def home(tmp_path, monkeypatch):
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(path))
    return path


class _FakeResponse:
    def __init__(self, payload, status=200, headers=None):
        self.status = status
        self.body = json.dumps(payload).encode()
        self.headers = headers or {}


class _FakeWeb:
    """Minimal ``aiohttp.web`` stand-in: aiohttp is not installed in this venv, and
    every assertion here is on the real JSON payload and status code."""

    @staticmethod
    def json_response(payload, status=200, headers=None):
        return _FakeResponse(payload, status, headers)


@pytest.fixture()
def adapter(home, monkeypatch):
    monkeypatch.setattr(api, "web", _FakeWeb)
    a = api.APIServerAdapter.__new__(api.APIServerAdapter)
    a._run_idempotency_store = RunIdempotencyStore(":memory:")
    a._run_statuses = {}
    a._run_idempotency_ids = set()
    a._run_owners = {}
    a._run_owner_pid = os.getpid()
    a._run_owner_started = 0
    a._model_name = "test-model"
    a._api_key = ""            # no key configured -> the auth check passes through
    a._pending_agent_requests = 0
    a._prepare_session_chat = _prepared
    a._run_idempotency_scope = lambda request: "peer-scope"
    a.calls = []

    async def _history(session_id):
        return []

    async def _run_agent(conversation_history=None, **kwargs):
        a.calls.append(kwargs)
        return {"final_response": "pong"}, {}

    a._conversation_history_for_session = _history
    a._run_agent = _run_agent
    # Keep the probe loop honest but fast: the real durations are asserted separately.
    monkeypatch.setattr(q, "receipt_after_seconds", lambda: 0.05)
    monkeypatch.setattr(q, "probe_delay", lambda n, **kw: 0.01)
    yield a
    a._run_idempotency_store.close()


async def _prepared(request):
    return ({"session_id": "sess-1",
             "user_message": getattr(request, "user_message", "hi"),
             "gateway_session_key": None, "body": {}, "runtime_request": {},
             "lock_active": False, "run_kwargs": {"session_id": "sess-1"}}, None)


def _headers(key=q.validate_idempotency_key("auto:test:1"), sender="sender",
             wait="30"):
    headers = {}
    if key is not None:
        headers["Idempotency-Key"] = key
    if sender is not None:
        headers["X-Hermes-Sender-Profile"] = sender
    if wait is not None:
        headers["X-Hermes-Wait-Seconds"] = wait
    return headers


def _request(**kwargs):
    message = kwargs.pop("message", "hi")
    return SimpleNamespace(
        headers=_headers(**kwargs), path="/api/sessions/sess-1/chat",
        user_message=message)


def _send(adapter, **kwargs):
    body = asyncio.run(adapter._handle_session_chat(_request(**kwargs))).body
    return json.loads(body)


def _hold_turn_lock(home, profile="default"):
    from tools.bot_mode_probe import _hermes_root
    from tools.bot_relay import acquire_turn_lock

    return acquire_turn_lock(_hermes_root(home), profile, timeout_seconds=0)


# ── the additive seam ────────────────────────────────────────────────────────

def test_plain_chat_without_delivery_headers_is_untouched(adapter, home):
    """No headers -> the pre-existing chat completion, and no queue record."""
    body = _send(adapter, key=None, sender=None, wait=None)
    assert body["object"] == "hermes.session.chat.completion"
    assert not (home / "runtime" / "bot_delivery").exists()
    assert adapter.calls, "the ordinary turn still ran"


def test_either_header_alone_is_not_a_delivery(adapter):
    assert adapter._bot_send_request(_request(sender=None)) is None
    assert adapter._bot_send_request(_request(key=None)) is None


def test_wait_header_parsing(adapter):
    assert adapter._bot_send_request(_request(wait="1800"))["wait_seconds"] == 1800.0
    assert adapter._bot_send_request(_request(wait=""))["wait_seconds"] == 0.0
    assert adapter._bot_send_request(_request(wait="nonsense"))["wait_seconds"] == 0.0
    assert adapter._bot_send_request(_request(wait="-5"))["wait_seconds"] == 0.0


# ── delivered in-call ────────────────────────────────────────────────────────

def test_free_target_delivers_in_call_and_settles_the_record(adapter, home):
    body = _send(adapter)
    assert list(body.keys()) == list(q.ENVELOPE_FIELDS)  # 23 fields, §1.3 order
    assert body["object"] == q.OBJECT_NAME
    assert body["result"] == "delivered"
    assert body["status"] == "delivered"
    assert body["reply"] == "pong"
    assert body["attempts"] == 1
    assert body["replayed"] is False
    assert len(adapter.calls) == 1
    settled = list((q.delivery_root(home) / q.SETTLED_DIR).glob("*.json"))
    assert len(settled) == 1
    record = json.loads(settled[0].read_text())
    assert record["status"] == "delivered" and record["attempts"] == 1
    assert record["reply"] == "pong"
    # One delivery id is also the run registry id (§4.5: run_id == delivery_id).
    assert body["delivery_id"] in adapter._run_statuses


def test_delivery_record_is_twenty_fields(adapter, home):
    body = _send(adapter)
    record = json.loads((q.delivery_root(home) / q.SETTLED_DIR
                         / f"{body['delivery_id']}.json").read_text())
    assert len(record) == 20
    assert "reoffer_count" in record and "reoffer_count" not in body


def test_send_id_is_diagnostics_only(adapter, home):
    body = _send(adapter)
    assert len(body["send_id"]) == 32 and body["send_id"] != body["delivery_id"]
    record = json.loads((q.delivery_root(home) / q.SETTLED_DIR
                         / f"{body['delivery_id']}.json").read_text())
    assert "send_id" not in record and body["send_id"] not in json.dumps(record)


# ── busy target: receipt, no late turn ───────────────────────────────────────

def test_locked_target_returns_an_acceptance_receipt(adapter, home):
    with _hold_turn_lock(home):
        body = _send(adapter)
    assert body["result"] == "receipt"
    assert body["status"] == "queued"
    assert body["busy"] is True
    assert body["retryable"] is False
    assert body["attempts"] == 0
    assert body["queue_position"] == 1
    assert body["status_detail"] == q.STATUS_DETAIL_TARGET_BUSY
    assert body["detail"] == q.DETAIL_RECEIPT
    assert 0 < body["queued_seconds"] < 1
    assert body["reply"] is None
    assert adapter.calls == [], "no turn may start while the slot is held"
    # Still queued, so the drainer can run it.
    assert len(list((q.delivery_root(home) / q.QUEUE_DIR).glob("*.json"))) == 1


def test_turn_lease_timeout_never_reaches_the_sender(adapter, home):
    async def _agent(conversation_history=None, **kwargs):
        return ({"failed": True, "error": "session_turn_lease_timeout:sess-1",
                 "final_response": "busy"}, {})

    adapter._run_agent = _agent
    body = _send(adapter)
    assert body["result"] == "receipt"
    assert body["status"] == "queued"
    assert body["attempts"] == 0
    assert body["status_detail"] != "session_turn_lease_timeout"
    record = json.loads(next((q.delivery_root(home) / q.QUEUE_DIR).glob("*.json")).read_text())
    assert record["status"] == "queued" and record["attempts"] == 0


def test_failed_turn_is_terminal_failed_with_a_reason_code(adapter, home):
    async def _agent(conversation_history=None, **kwargs):
        raise RuntimeError("429 rate limit exceeded")

    adapter._run_agent = _agent
    body = _send(adapter)
    assert body["result"] == "failed"
    assert body["status"] == "failed"
    assert body["reason"] == "provider_rate_limit"
    assert body["attempts"] == 1
    assert body["error"]


# ── idempotency ──────────────────────────────────────────────────────────────

def test_duplicate_key_replays_and_never_runs_a_second_turn(adapter, home):
    first = _send(adapter)
    second = _send(adapter)
    assert len(adapter.calls) == 1
    assert second["result"] == "delivered"
    assert second["replayed"] is True
    assert second["reply"] == "pong"
    assert second["delivery_id"] == first["delivery_id"]
    assert len(list((q.delivery_root(home) / q.SETTLED_DIR).glob("*.json"))) == 1


def test_duplicate_key_after_a_receipt_replays_the_queue_record(adapter, home):
    with _hold_turn_lock(home):
        first = _send(adapter)
    second = _send(adapter)
    assert second["delivery_id"] == first["delivery_id"]
    assert second["status"] == "queued"
    assert second["replayed"] is True
    assert second["attempts"] == 0


def test_same_key_different_message_is_a_conflict(adapter):
    _send(adapter)
    body = asyncio.run(adapter._handle_session_chat(_request(message="different")))
    assert body.status == 409
    assert b"idempotency_key_conflict" in body.body


def test_invalid_idempotency_key_is_rejected(adapter, home):
    body = asyncio.run(adapter._handle_session_chat(
        _request(key="bad\nkey")))
    assert body.status == 400
    assert b"invalid_idempotency_key" in body.body
    assert not (q.delivery_root(home) / q.QUEUE_DIR).exists()


# ── capacity ─────────────────────────────────────────────────────────────────

def test_queue_full_refuses_without_taking_a_reservation(adapter, home):
    from hermes_cli import delivery_keys

    for i in range(q.max_per_profile()):
        # Distinct senders so only the per-profile cap is in play.
        q.admit(home, sender_profile=f"filler-{i}", target_profile="default",
                target_session_id="sess-1", idempotency_key=f"auto:other:{i}",
                fingerprint="f" * 64,
                delivery_id=delivery_keys.delivery_id_from("peer-scope", f"auto:other:{i}"),
                message="filler")
    body = _send(adapter, key="auto:test:full")
    assert body["result"] == "failed"
    assert body["status"] == "failed"
    assert body["reason"] == "queue_full"
    assert body["retryable"] is True
    assert body["retry_after_seconds"] == 60
    assert body["detail"] == q.queue_full_detail()
    assert adapter.calls == []
    # No reservation was taken (B-T3): the key is still free and the queue is intact.
    assert len(adapter._run_idempotency_ids) == 0
    assert q.queue_depth(home, "default") == q.max_per_profile()


def test_per_sender_cap_is_enforced(adapter, home):
    from hermes_cli import delivery_keys

    for i in range(q.max_per_sender()):
        q.admit(home, sender_profile="sender", target_profile="default",
                target_session_id="sess-1", idempotency_key=f"auto:sender:{i}",
                fingerprint="f" * 64,
                delivery_id=delivery_keys.delivery_id_from("peer-scope", f"auto:sender:{i}"),
                message="filler")
    body = _send(adapter, key="auto:test:cap")
    assert body["reason"] == "queue_full"
    assert body["retryable"] is True


# ── status handle (§4.5) ─────────────────────────────────────────────────────

def test_run_id_equals_delivery_id_and_status_row_carries_the_projection(adapter, home):
    body = _send(adapter)
    row = adapter._run_statuses[body["delivery_id"]]
    assert row["run_id"] == body["delivery_id"]
    assert row["delivery_status"] == "delivered"
    assert row["result"] == "delivered"
    assert row["attempts"] == 1
    # D4: the delivery handle lives for the registry's retention window like every
    # other run row. With ``retention_until=0`` the row sat on the default age window
    # and no later write extended it, so a sender still polling the delivery could get
    # 404 "Run not found" for a delivery the receiver had accepted.
    assert row["retention_until"] > time.time() + 60
    assert row["retention_until"] == pytest.approx(
        time.time() + RunIdempotencyStore.RETENTION_SECONDS, abs=120)
    assert {"object", "run_id", "status"} <= set(row)


# ── envelope contract + ambiguous recovery (§1.3, §4.9) ──────────────────────

def test_gateway_envelope_is_twenty_three_fields_in_spec_order(adapter, home):
    """§1.3: every sender-visible envelope is the same 23 fields, in the same order."""
    assert len(q.ENVELOPE_FIELDS) == 23
    delivered = _send(adapter)
    assert list(delivered) == list(q.ENVELOPE_FIELDS)
    assert "reoffer_count" not in delivered and "send_id" in delivered

    with _hold_turn_lock(home):
        receipt = _send(adapter, key=q.validate_idempotency_key("auto:test:held"),
                        message="held")
    assert list(receipt) == list(q.ENVELOPE_FIELDS)
    assert receipt["busy"] is True and receipt["attempts"] == 0
    assert receipt["status_detail"] == q.STATUS_DETAIL_TARGET_BUSY


def _make_ambiguous(home, delivery_id):
    """Simulate a crash mid-turn: the record settles as ``ambiguous`` (§4.9)."""
    queued = q.delivery_root(home) / q.QUEUE_DIR / f"{delivery_id}.json"
    record = json.loads(queued.read_text())
    record["status"] = q.STATUS_AMBIGUOUS
    (q.delivery_root(home) / q.SETTLED_DIR / f"{delivery_id}.json").write_text(
        json.dumps(record), encoding="utf-8")
    queued.unlink()


def test_ambiguous_record_is_recovered_once_and_never_twice(adapter, home):
    """§4.9 rules 1-2: an ``ambiguous`` record is re-offered under its own key, ONCE.

    The recovery re-uses the original delivery id (never a second delivery), runs the
    re-offered work to completion, and a second ambiguity is NOT re-offered again.
    """
    with _hold_turn_lock(home):
        first = _send(adapter)
    _make_ambiguous(home, first["delivery_id"])

    second = _send(adapter)  # the single recovery replay re-queues the work
    assert second["delivery_id"] == first["delivery_id"]
    assert second["result"] == "receipt" and second["status"] == "queued"
    assert second["attempts"] == 0
    assert adapter.calls == [], "recovery re-queues: the drainer runs it, never this call"

    # The re-offered delivery is live work, not a stranded row.
    queued = json.loads((q.delivery_root(home) / q.QUEUE_DIR
                         / f"{first['delivery_id']}.json").read_text())
    assert queued["status"] == "queued" and queued["reoffer_count"] == 1
    assert queued["reoffer_count"] not in second  # internal bookkeeping, never an envelope field

    # Phase 2: the recovery is spent — a second ambiguity is a replay, never a re-offer.
    _make_ambiguous(home, first["delivery_id"])
    third = _send(adapter)
    assert third["delivery_id"] == first["delivery_id"]
    assert third["replayed"] is True
    assert third["result"] == "unknown"  # honest: the outcome really is unknown
    assert q.read_record(home, first["delivery_id"])["reoffer_count"] == 1
    assert adapter.calls == [], "there is exactly one recovery replay, never a second one"
