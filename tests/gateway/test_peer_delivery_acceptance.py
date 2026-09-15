"""ACCEPTANCE — ONE scripted peer delivery walks all three card criteria in order.

The three criteria (peer-send-result-spec §1.5 ENUM A, §2.x queue/backoff, §7):

  1. B-T1  a send to a BUSY target is ACCEPTED, never failed: ``result == "receipt"``,
     ``status == "queued"``, ``busy is True``, ``retryable is False``, ``attempts == 0``,
     and the client maps ``receipt`` to exit code 0 (a receipt is success).
  2. the sender must NOT enter a retry loop on ``target_busy``: the outcome is not
     retryable, and a second identical send (a sender that retries) starts no extra turn
     and creates no second delivery — ``adapter.calls`` is unchanged.
  3. B-T2  same-key retries deliver EXACTLY ONCE: three identical sends under one
     idempotency key (one of them a retry while the target is busy) run exactly ONE turn,
     leave exactly ONE delivery record, share ONE ``delivery_id``, and every response
     after the first is a replay.

Everything is driven through the real gateway handler (``_handle_session_chat`` +
``_bot_send_probe_and_run`` / ``_bot_send_turn``), the real ``hermes peer dm`` client
command, the real idempotency store and the real on-disk delivery queue under a tmp
HERMES_HOME. Assertions are made on returned envelopes, files under the delivery root
and ``adapter.calls`` — never on source text.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from types import SimpleNamespace

import pytest

from gateway.platforms import api_server as api
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from hermes_cli.subcommands import peer as peer_cmd
from tools import bot_delivery_queue as q


# ── the gateway seam (same fixture/helper style as test_peer_delivery_gateway.py) ──

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
    # Keep the probe loop honest but fast: the real durations are asserted elsewhere.
    monkeypatch.setattr(q, "receipt_after_seconds", lambda: 0.05)
    monkeypatch.setattr(q, "probe_delay", lambda n, **kw: 0.01)
    yield a
    a._run_idempotency_store.close()


async def _prepared(request):
    return ({"session_id": "sess-1",
             "user_message": getattr(request, "user_message", "hi"),
             "gateway_session_key": None, "body": {}, "runtime_request": {},
             "lock_active": False, "run_kwargs": {"session_id": "sess-1"}}, None)


def _headers(key, sender="sender", wait="30"):
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
    """One real HTTP-shaped peer send through the gateway handler."""
    body = asyncio.run(adapter._handle_session_chat(_request(**kwargs))).body
    return json.loads(body)


def _hold_turn_lock(home, profile="default"):
    """Hold the target profile's turn lock, which is what makes a target BUSY."""
    from tools.bot_mode_probe import _hermes_root
    from tools.bot_relay import acquire_turn_lock

    return acquire_turn_lock(_hermes_root(home), profile, timeout_seconds=0)


def _ctx(session_id="sess-1"):
    """The prepared-request context a receiver-side drainer hands to the turn runner."""
    return {"session_id": session_id, "user_message": "drain", "gateway_session_key": None,
            "body": {}, "runtime_request": {}, "lock_active": False,
            "run_kwargs": {"session_id": session_id}}


# ── the delivery root as a real artifact (exactly one delivery record?) ──────────

def _delivery_records(home):
    """Every on-disk delivery record, as (lifecycle-dir, filename) pairs."""
    root = q.delivery_root(home)
    return sorted(
        (sub, path.name)
        for sub in (q.QUEUE_DIR, q.CLAIMED_DIR, q.SETTLED_DIR)
        for path in (root / sub).glob("*.json")
    )


# ── the client seam (same fixture style as test_peer_receipt_client.py) ──────────

def _client_args(**kw):
    base = {"peer_action": "dm", "target": "spark", "message": "ping", "json": True,
            "idempotency_key": None, "wait_seconds": None}
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture()
def peer_client(monkeypatch):
    """Isolate the peer registry + Bot Chat session, capturing every request."""
    calls: list[dict] = []

    monkeypatch.setattr(peer_cmd, "_load_peers",
                        lambda: {"spark": {"url": "http://spark.lan:8377"}})
    monkeypatch.setattr(peer_cmd, "_peer_secret", lambda name: "k" * 20)
    monkeypatch.setattr(peer_cmd, "_ensure_bot_chat", lambda base, key: "bc_1")
    monkeypatch.setattr(peer_cmd, "_sender_profile", lambda: "yoyodine-coder")

    def fake_request(url, api_key, **kw):
        calls.append({"url": url, "api_key": api_key, **kw})
        return peer_client.response

    peer_client.response = {}
    monkeypatch.setattr(peer_cmd, "_request", fake_request)
    peer_client.calls = calls
    return peer_client


# ── the scripted acceptance walk ────────────────────────────────────────────────

MESSAGE = "acceptance walk: receipt on busy, no retry loop, exactly once"
KEY = q.validate_idempotency_key("auto:acceptance:walk:1")


def test_busy_receipt_no_retry_loop_and_same_key_exactly_once(
        adapter, home, peer_client, capsys):
    """One delivery, three criteria, walked end to end in sequence."""
    profile = adapter._bot_send_target_profile(home)

    # ── (1) BUSY target -> accepted receipt, never a failure ────────────────────
    with _hold_turn_lock(home):
        first = _send(adapter, key=KEY, message=MESSAGE)

    assert first["object"] == q.OBJECT_NAME
    assert first["result"] == q.RESULT_RECEIPT
    assert first["status"] == q.STATUS_QUEUED
    assert first["busy"] is True
    assert first["retryable"] is False
    assert first["attempts"] == 0
    assert first["replayed"] is False
    assert first["reply"] is None
    assert first["queue_position"] == 1
    assert first["status_detail"] == q.STATUS_DETAIL_TARGET_BUSY
    assert first["detail"] == q.DETAIL_RECEIPT
    assert adapter.calls == [], "the held slot must not run the payload in-call"

    delivery_id = first["delivery_id"]
    assert q.read_record(home, delivery_id)["status"] == q.STATUS_QUEUED
    assert _delivery_records(home) == [(q.QUEUE_DIR, f"{delivery_id}.json")]

    # ...and the client-side mapping of `receipt` is SUCCESS: exit code 0 (§1.5).
    # Feed the REAL gateway envelope through the REAL `hermes peer dm` command.
    peer_client.response = dict(first)
    assert peer_cmd.cmd_peer(_client_args(message=MESSAGE)) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["result"] == "receipt"
    assert emitted["status"] == "queued"
    assert emitted["delivery_id"] == delivery_id
    assert emitted["attempts"] == 0

    # ── (2) no retry loop: a sender that retries starts no extra turn ───────────
    calls_before = len(adapter.calls)
    with _hold_turn_lock(home):
        second = _send(adapter, key=KEY, message=MESSAGE)

    assert second["retryable"] is False, "target_busy is never handed back as retryable"
    assert second["delivery_id"] == delivery_id
    assert second["replayed"] is True
    assert second["status"] == q.STATUS_QUEUED
    assert second["attempts"] == 0
    assert len(adapter.calls) == calls_before, "a retry must not start a turn"
    assert adapter.calls == []
    assert _delivery_records(home) == [(q.QUEUE_DIR, f"{delivery_id}.json")]
    assert q.queue_depth(home, profile) == 1

    # ── (3) same-key retries deliver EXACTLY ONCE ──────────────────────────────
    # 3rd identical send, same key: the slot is free now, and dedup is key-bound,
    # not lock-bound — it still may not run the payload a second time.
    third = _send(adapter, key=KEY, message=MESSAGE)
    assert third["delivery_id"] == delivery_id
    assert third["replayed"] is True
    assert third["result"] == q.RESULT_RECEIPT
    assert third["status"] == q.STATUS_QUEUED
    assert adapter.calls == [], "no send may ever re-run an accepted delivery"

    # The durable queue is the only thing allowed to run it: the receiver drains it
    # exactly once, through the same turn runner the in-call curve uses.
    with _hold_turn_lock(home):
        drained = json.loads(asyncio.run(adapter._bot_send_turn(
            _request(key=KEY, message=MESSAGE), _ctx(), home=home,
            delivery_id=delivery_id, target_profile=profile,
            session_id="sess-1", waited=0.0)).body)
    assert drained["result"] == q.RESULT_DELIVERED
    assert drained["status"] == q.STATUS_DELIVERED
    assert drained["attempts"] == 1
    assert drained["reply"] == "pong"
    assert drained["delivery_id"] == delivery_id

    # exactly ONE turn ran, for exactly ONE delivery record...
    assert len(adapter.calls) == 1
    assert _delivery_records(home) == [(q.SETTLED_DIR, f"{delivery_id}.json")]
    record = q.read_record(home, delivery_id)
    assert record["status"] == q.STATUS_DELIVERED and record["attempts"] == 1

    # ...every response carried the SAME delivery_id, and every response after the
    # first was a replay (never a re-run).
    assert {r["delivery_id"] for r in (first, second, third)} == {delivery_id}
    assert [r["replayed"] for r in (first, second, third)] == [False, True, True]

    # Extra guard: a retry after delivery replays the settled outcome, still one turn.
    fourth = _send(adapter, key=KEY, message=MESSAGE)
    assert fourth["delivery_id"] == delivery_id
    assert fourth["replayed"] is True
    assert fourth["result"] == q.RESULT_DELIVERED
    assert len(adapter.calls) == 1
    assert _delivery_records(home) == [(q.SETTLED_DIR, f"{delivery_id}.json")]


# ── D1: the queue is DRAINED, not merely written ────────────────────────────────

def test_drained_head_runs_even_when_its_own_request_is_gone(adapter, home):
    """D1: an accepted record must RUN -- including one whose sender has left.

    Live evidence: 53 records sat in ``queued`` with a full body each and no runner,
    because the in-call curve only ever ran the delivery its own request admitted.
    Here the OLDER delivery is accepted first and its request then abandons it (a
    restart or a timed-out sender does exactly that). When a LATER request takes the
    slot, the queue head must run -- not be handed back as one more receipt.
    """
    profile = adapter._bot_send_target_profile(home)
    older_key = q.validate_idempotency_key("auto:d1:older")
    our_key = q.validate_idempotency_key("auto:d1:ours")

    with _hold_turn_lock(home):
        older = _send(adapter, key=older_key, message="older payload")
        ours = _send(adapter, key=our_key, message="our payload")
    assert (older["status"], ours["status"]) == (q.STATUS_QUEUED, q.STATUS_QUEUED)
    assert (older["queue_position"], ours["queue_position"]) == (1, 2)
    assert adapter.calls == [], "a held slot runs nothing in-call"

    # The slot frees and OUR request admits: it runs the head first, then itself.
    with _hold_turn_lock(home):
        last = json.loads(asyncio.run(adapter._bot_send_turn(
            _request(key=our_key, message="our payload"), _ctx(), home=home,
            delivery_id=ours["delivery_id"], target_profile=profile,
            session_id="sess-1", waited=0.0)).body)

    assert [call.get("user_message") for call in adapter.calls] == ["older payload", None], (
        "the drained head must RUN, and run FIRST")
    assert last["delivery_id"] == ours["delivery_id"]
    assert last["status"] == q.STATUS_DELIVERED
    assert q.read_record(home, older["delivery_id"])["status"] == q.STATUS_DELIVERED, (
        "the foreign head is delivered, never requeued into a queue nothing drains")
    assert q.read_record(home, ours["delivery_id"])["status"] == q.STATUS_DELIVERED
    assert _delivery_records(home) == sorted(
        (q.SETTLED_DIR, f"{d['delivery_id']}.json") for d in (older, ours))
    assert q.queue_depth(home, profile) == 0


def test_sweep_drains_a_free_slot_and_never_expires_a_held_one(adapter, home):
    """D1 triggers 3-4 + the TTL rule: no live request needed, no holder fought.

    The live slot was held by a UI/desktop session, not a card worker (a 1800.3s
    lease wait reproduced against one), so the sweep skips a held slot, leaves the
    record queued -- and the over-age sweep must NOT expire it merely because its
    target is mid-turn: the holder is the turn that drains it next.
    """
    from gateway.platforms import api_server_bot_delivery as drain

    profile = adapter._bot_send_target_profile(home)
    with _hold_turn_lock(home):
        first = _send(adapter, key=q.validate_idempotency_key("auto:d1:s1"), message="one")
        second = _send(adapter, key=q.validate_idempotency_key("auto:d1:s2"), message="two")
    assert q.queue_depth(home, profile) == 2

    # (a) an over-age record whose slot is HELD is kept, not expired.
    future = time.time_ns() + 10 * 24 * 3600 * 1_000_000_000
    assert q.sweep_delivery_queue(home, now_ns=future,
                                  slot_held_fn=lambda h, p: True) == 0
    assert q.read_record(home, first["delivery_id"])["status"] == q.STATUS_QUEUED

    # (b) the drainer itself skips a held slot; it never fights for it.
    with _hold_turn_lock(home):
        assert asyncio.run(drain.drain_once(adapter, home)) == 0
    assert adapter.calls == []
    assert q.queue_depth(home, profile) == 2

    # (c) the slot frees -> the whole backlog runs, in FIFO order.
    assert asyncio.run(drain.drain_once(adapter, home)) == 2
    assert [call["user_message"] for call in adapter.calls] == ["one", "two"]
    assert q.queue_depth(home, profile) == 0
    assert [q.read_record(home, r["delivery_id"])["status"]
            for r in (first, second)] == [q.STATUS_DELIVERED, q.STATUS_DELIVERED]


def test_queued_receipt_reports_the_slot_state_observed(adapter, home):
    """D2: a receipt says WHY it queued, from the slot's real state.

    The old receipt hard-coded "target busy", so a 53-deep backlog on a free slot
    read as one busy turn. With the slot free the honest cause is the backlog, and
    ``busy`` follows the same field the envelope prints (D7).
    """
    with _hold_turn_lock(home):
        mine = _send(adapter, key=q.validate_idempotency_key("auto:d2"), message="hi")
    assert mine["status"] == q.STATUS_QUEUED

    receipt = adapter._bot_send_receipt(
        mine["delivery_id"], q.read_record(home, mine["delivery_id"]),
        waited=0.0, home=home)
    body = json.loads(receipt.body)
    assert body["status_detail"] == q.STATUS_DETAIL_BACKLOG
    assert body["busy"] is False
    stored = q.read_record(home, mine["delivery_id"])
    assert stored["status_detail"] == q.STATUS_DETAIL_BACKLOG
    assert q.busy_from_detail(stored) is False


# ── D1 sub-item 2: the drain invariant (N keys, free slot -> N turns, once) ────

def test_drain_invariant_n_keys_free_lock_n_turns_exactly_once(adapter, home):
    """D1 sub-item 2: with a free slot, N admitted keys run N turns exactly once.

    Sub-item 2 of the verification was PARTIAL because "delivered exactly once"
    could not be exercised while zero turns executed for receipted deliveries.
    Now that a receipt is a promise of execution, the invariant is testable
    end-to-end: six distinct keys admitted while the slot is held all run --
    each exactly once, in FIFO order -- and a same-key replay afterwards starts
    no second turn.
    """
    from gateway.platforms import api_server_bot_delivery as drain

    profile = adapter._bot_send_target_profile(home)
    count = 6
    keys = [q.validate_idempotency_key(f"auto:inv:{i}") for i in range(count)]
    messages = [f"invariant payload {i}" for i in range(count)]

    # The slot is held, so every send is receipted rather than run in-call.
    with _hold_turn_lock(home):
        admitted = [_send(adapter, key=k, message=m)
                    for k, m in zip(keys, messages)]
    assert [r["result"] for r in admitted] == [q.RESULT_RECEIPT] * count
    assert len({r["delivery_id"] for r in admitted}) == count
    assert adapter.calls == [], "a held slot runs nothing in-call"
    assert q.queue_depth(home, profile) == count

    # The slot frees with no live request: the drainer runs the whole backlog.
    assert asyncio.run(drain.drain_once(adapter, home)) == count
    ran = [call.get("user_message") for call in adapter.calls]
    assert ran == messages, "FIFO, one turn per admitted key"
    assert len(ran) == len(set(ran)) == count, "each key ran exactly once"
    assert q.queue_depth(home, profile) == 0
    for record in admitted:
        assert q.read_record(home, record["delivery_id"])["status"] == q.STATUS_DELIVERED
    assert _delivery_records(home) == sorted(
        (q.SETTLED_DIR, f"{r['delivery_id']}.json") for r in admitted)

    # A same-key replay after settlement must not start a second turn.
    before = len(adapter.calls)
    replay = _send(adapter, key=keys[0], message=messages[0])
    assert replay["delivery_id"] == admitted[0]["delivery_id"]
    assert replay["replayed"] is True
    assert len(adapter.calls) == before, "a replay never starts a second turn"
    assert q.queue_depth(home, profile) == 0
