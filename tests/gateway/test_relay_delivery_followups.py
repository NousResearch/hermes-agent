"""Follow-up regressions for the 2026-08-09 relay delivery fixes (#82592).

Four review findings on the original branch:

1. HIGH — classifier/resolver mismatch: ``_classify_completion_target``
   returned "deliver" for idle-ended parents, but
   ``_resolve_async_delegation_session`` (untouched) still dropped every
   non-compression-ended pin. The durable row was acked at adapter
   acceptance, then the injection silently died inside the pipeline — a
   falsely-acknowledged permanent loss, strictly worse than the honest
   terminal drop on main. Fix: the resolver retargets non-user-boundary
   ends to the chat's current session; both sides share
   ``_USER_BOUNDARY_END_REASONS`` so they cannot drift again.

2. HIGH — the drain-grace clamp only budgeted drain + 3×teardown, but
   ``RelayAdapter.disconnect`` spends monitor-teardown + go_idle time
   BEFORE the transport drain inside the same runner wait_for. Fix: the
   adapter measures its own elapsed time and threads the REMAINING budget
   into ``transport.disconnect(budget_s=...)``.

3. P1 — a ``_request_response`` racing disconnect could register a future
   after the fail-pending loop and strand its caller for the full
   30s outbound timeout. Fix: fast-fail when ``_closing`` is set.

4. P1 — ``_build_process_event_source``'s last-resort reconstruction
   omitted ``scope_id``, so a scoped relay completion whose session-store
   origin was unavailable primed no scope discriminator and could still
   bounce off the connector's tenant guard. Fix: thread ``scope_id``
   through the reconstruction (and warn when it's absent for scoped chats).
"""

import asyncio
import os
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.relay.ws_transport import (
    _DISCONNECT_DRAIN_GRACE_S,
    _TEARDOWN_AWAIT_TIMEOUT_S,
    _disconnect_drain_grace_s,
    WebSocketRelayTransport,
)
from gateway.run import GatewayRunner, _USER_BOUNDARY_END_REASONS
from gateway.session import AsyncSessionStore, SessionEntry


# ---------------------------------------------------------------------------
# 1. Classifier/resolver coherence (the falsely-acked-loss class)
# ---------------------------------------------------------------------------

def _entry(session_id, session_key="agent:main:slack:dm:U1"):
    return SessionEntry(
        session_key=session_key,
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.SLACK,
        chat_type="dm",
    )


def _runner_with_rows(rows, *, switched_entry=None):
    runner = object.__new__(GatewayRunner)
    db = MagicMock()
    db.get_session = AsyncMock(side_effect=lambda session_id: rows.get(session_id))
    db.get_compression_tip = AsyncMock(return_value=None)
    runner._session_db = db
    runner.session_store = MagicMock()
    runner.session_store.switch_session = MagicMock(return_value=switched_entry)
    runner.session_store.advance_compression_session = MagicMock(
        return_value=switched_entry
    )
    runner._async_session_store = AsyncSessionStore(runner.session_store)
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("end_reason", ["idle", "idle_timeout", "timeout", None, ""])
async def test_resolver_retargets_idle_ended_pin_to_current_session(end_reason):
    """The delivery leg the classifier's "deliver" verdict promises: an
    idle-ended pin must resolve to the chat's CURRENT session, not drop."""
    current = _entry("sess_current")
    runner = _runner_with_rows(
        {
            "sess_idle": {
                "id": "sess_idle",
                "ended_at": "2026-08-09T00:00:00",
                "end_reason": end_reason,
            }
        }
    )

    resolved = await runner._resolve_async_delegation_session(
        current, "sess_idle"
    )

    assert resolved is current, (
        f"end_reason={end_reason!r}: idle-ended pin must retarget to the "
        f"chat's current session (got {resolved!r}); dropping here after "
        "the classifier said 'deliver' acks the durable row for a message "
        "that never went out"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("end_reason", sorted(_USER_BOUNDARY_END_REASONS))
async def test_resolver_still_drops_user_boundary_ends(end_reason):
    current = _entry("sess_current")
    runner = _runner_with_rows(
        {
            "sess_closed": {
                "id": "sess_closed",
                "ended_at": "2026-08-09T00:00:00",
                "end_reason": end_reason,
            }
        }
    )
    resolved = await runner._resolve_async_delegation_session(
        current, "sess_closed"
    )
    assert resolved is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "end_reason",
    ["idle", "idle_timeout", "timeout", "", None, "agent_close", "cron_complete"],
)
async def test_classifier_and_resolver_agree_on_ended_parents(end_reason):
    """Coherence invariant: whenever the pre-flight classifier says
    "deliver" for an ended parent, the in-pipeline resolver must actually
    deliver (return a session), and when it says "terminal" the resolver
    must drop. Divergence in the deliver->drop direction acks the durable
    row for an injection the pipeline then discards."""
    row = {
        "id": "sess_x",
        "ended_at": "2026-08-09T00:00:00",
        "end_reason": end_reason,
    }
    current = _entry("sess_current")
    runner = _runner_with_rows({"sess_x": row})

    verdict = await runner._classify_completion_target("sess_x")
    resolved = await runner._resolve_async_delegation_session(current, "sess_x")

    if verdict == "deliver":
        assert resolved is not None, (
            f"classifier said deliver for end_reason={end_reason!r} but the "
            "resolver dropped — durable row would be falsely acknowledged"
        )
    elif verdict == "terminal":
        assert resolved is None


# ---------------------------------------------------------------------------
# 2. Disconnect budget threading (adapter -> transport drain)
# ---------------------------------------------------------------------------

def test_drain_grace_uses_threaded_remaining_budget(monkeypatch):
    """An explicit remaining budget must override the env-mirrored default:
    the adapter has already spent monitor/go_idle time out of the runner's
    wait_for, so the transport can only drain what is actually left."""
    monkeypatch.delenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", raising=False)
    reserved = 3 * _TEARDOWN_AWAIT_TIMEOUT_S + 0.5
    # Plenty of remaining budget: grace caps at the constant.
    assert _disconnect_drain_grace_s(100.0) == _DISCONNECT_DRAIN_GRACE_S
    # Exactly reserved left: no drain.
    assert _disconnect_drain_grace_s(reserved) == 0.0
    # Less than reserved / nothing left: clamped to zero, never negative.
    assert _disconnect_drain_grace_s(0.0) == 0.0
    # Partial remainder: drain gets exactly the surplus.
    assert _disconnect_drain_grace_s(reserved + 1.0) == pytest.approx(1.0)


def test_drain_grace_env_fallback_unchanged(monkeypatch):
    """No threaded budget -> the env-mirrored runner default still applies
    (the original #82592 clamp semantics are preserved)."""
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "10")
    reserved = 3 * _TEARDOWN_AWAIT_TIMEOUT_S + 0.5
    assert _disconnect_drain_grace_s() == pytest.approx(
        min(_DISCONNECT_DRAIN_GRACE_S, 10.0 - reserved)
    )


@pytest.mark.asyncio
async def test_adapter_disconnect_threads_remaining_budget():
    """RelayAdapter.disconnect must pass budget_s to the transport, and the
    value must be <= the full budget (its own spend subtracted)."""
    from gateway.relay.adapter import RelayAdapter

    adapter = object.__new__(RelayAdapter)
    adapter._revocation_monitor = None

    seen = {}

    class _Transport:
        async def go_idle(self, timeout_s=10.0):
            await asyncio.sleep(0.05)
            return True

        def disconnect(self, budget_s=None):
            seen["budget_s"] = budget_s

            async def _noop():
                return None

            return _noop()

    adapter._transport = _Transport()
    await adapter.disconnect()

    assert "budget_s" in seen, "transport.disconnect never received budget_s"
    from gateway.relay.ws_transport import _env_disconnect_budget_s

    full = _env_disconnect_budget_s()
    assert seen["budget_s"] is not None
    assert seen["budget_s"] <= full
    # go_idle slept 0.05s, so some budget must have been consumed.
    assert seen["budget_s"] < full


@pytest.mark.asyncio
async def test_adapter_disconnect_tolerates_legacy_transport_signature():
    """A transport without the budget_s keyword (stubs) must still be torn
    down through the legacy no-arg call, not crash."""
    from gateway.relay.adapter import RelayAdapter

    adapter = object.__new__(RelayAdapter)
    adapter._revocation_monitor = None
    called = {}

    class _LegacyTransport:
        def disconnect(self):  # no budget_s
            called["legacy"] = True

            async def _noop():
                return None

            return _noop()

    adapter._transport = _LegacyTransport()
    await adapter.disconnect()
    assert called.get("legacy") is True


# ---------------------------------------------------------------------------
# 3. _request_response vs disconnect race
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_request_response_fails_fast_when_closing():
    """A request racing teardown must fail immediately instead of
    registering a future the fail-pending loop may already have missed
    (which would strand the caller for _OUTBOUND_TIMEOUT_S)."""
    transport = object.__new__(WebSocketRelayTransport)
    transport._closing = True
    transport._ws = object()  # socket still nominally open
    transport._pending = {}

    result = await transport._request_response({"a": 1})

    assert result == {"success": False, "error": "relay transport closed"}
    assert not transport._pending, (
        "no future may be registered once _closing is set"
    )


# ---------------------------------------------------------------------------
# 4. scope_id in fallback source reconstruction
# ---------------------------------------------------------------------------

def _fallback_runner():
    runner = object.__new__(GatewayRunner)
    store = MagicMock()
    store._ensure_loaded = MagicMock(side_effect=RuntimeError("store down"))
    store._entries = {}
    runner.session_store = store
    runner._session_sources = None
    return runner


def test_fallback_source_reconstruction_carries_scope_id():
    """When the session-store origin is unavailable, the reconstructed
    SessionSource must carry the event's scope_id so relay egress priming
    still captures the tenant discriminator."""
    runner = _fallback_runner()
    evt = {
        "session_key": "agent:main:discord:group:C123:U9",
        "platform": "discord",
        "chat_type": "group",
        "chat_id": "C123",
        "user_id": "U9",
        "scope_id": "G777",
        "type": "async_delegation",
    }
    source = runner._build_process_event_source(evt)
    assert source is not None
    assert source.scope_id == "G777"
    assert source.user_id == "U9"


def test_fallback_source_reconstruction_without_scope_still_routes():
    """Absent scope_id must not fail the reconstruction (DMs and
    author-bound scoped chats still route via user_id)."""
    runner = _fallback_runner()
    evt = {
        "session_key": "agent:main:slack:dm:D42",
        "platform": "slack",
        "chat_type": "dm",
        "chat_id": "D42",
        "user_id": "U1",
        "type": "async_delegation",
    }
    source = runner._build_process_event_source(evt)
    assert source is not None
    assert source.scope_id is None
    assert source.user_id == "U1"


# ---------------------------------------------------------------------------
# 5. Cancellation-safe pending-future failure (round-3 finding 1)
# ---------------------------------------------------------------------------

def _bare_transport():
    t = object.__new__(WebSocketRelayTransport)
    t._closing = False
    t._supervisor = None
    t._reader = None
    t._ws = None
    t._pending = {}
    t._going_idle_ack = None
    return t


@pytest.mark.asyncio
async def test_disconnect_cancellation_still_fails_pending_futures():
    """A cancellation landing during the drain (outer cleanup deadline,
    runner wait_for) must NOT leave registered futures unresolved — a
    stranded waiter would block until _OUTBOUND_TIMEOUT_S (30s). The
    fail-pending loop runs in a finally, so cancellation cannot skip it."""
    t = _bare_transport()
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    t._pending["req-1"] = fut

    task = asyncio.create_task(t.disconnect(budget_s=60.0))
    # Let the drain start waiting on the pending future...
    await asyncio.sleep(0.05)
    assert not task.done(), "disconnect should be inside the drain wait"
    # ...then cancel it mid-drain, as the runner's wait_for would.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert fut.done(), (
        "pending outbound future left unresolved after cancelled disconnect"
    )
    with pytest.raises(RuntimeError, match="relay transport closed"):
        fut.result()
    assert not t._pending


@pytest.mark.asyncio
async def test_disconnect_idempotent_second_pass():
    """A second disconnect() (adapter and outer cleanup can both run one)
    must be safe: done futures are skipped, cleared map stays cleared."""
    t = _bare_transport()
    await t.disconnect()
    await t.disconnect()  # must not raise
    assert not t._pending


# ---------------------------------------------------------------------------
# 6. Durable routing origin: scope_id survives dispatch -> restart -> replay
# ---------------------------------------------------------------------------

def test_durable_dispatch_persists_and_recovers_scope_id(tmp_path, monkeypatch):
    """End-to-end restart shape: dispatch with a scoped session context bound,
    simulate owner death, recover — the recovered completion event must carry
    scope_id/user_id, and the reconstructed SessionSource must prime them."""
    import tools.async_delegation as ad
    from gateway.session_context import clear_session_vars, set_session_vars

    ad._reset_for_tests()
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")

    tokens = set_session_vars(
        platform="discord",
        chat_id="C123",
        chat_type="group",
        user_id="U9",
        scope_id="G777",
        session_key="agent:main:discord:group:C123:U9",
    )
    try:
        record = {
            "delegation_id": "d-scope-1",
            "session_key": "agent:main:discord:group:C123:U9",
            "origin_ui_session_id": "",
            "origin_session_id": "",
            "parent_session_id": "sess-p",
            "goal": "scoped goal",
            "dispatched_at": 100.0,
            **ad._capture_routing_origin(),
        }
        assert record.get("scope_id") == "G777", (
            "dispatch-time capture must snapshot HERMES_SESSION_SCOPE_ID"
        )
        ad._persist_dispatch(record)
    finally:
        clear_session_vars(tokens)

    # Simulate the owner process being gone: recovery marks the row unknown
    # and rebuilds the completion event from the durable task_json.
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: False)
    ad.recover_abandoned_delegations()

    with ad._transaction() as conn:
        row = conn.execute(
            "SELECT event_json FROM async_delegations WHERE delegation_id='d-scope-1'"
        ).fetchone()
    assert row and row[0], "recovered row must have an event_json"
    import json as _json

    evt = _json.loads(row[0])
    assert evt.get("scope_id") == "G777", (
        "recovered completion event lost scope_id — post-restart scoped "
        "relay egress would be declined by the connector's tenant guard"
    )
    assert evt.get("user_id") == "U9"

    # The gateway-side fallback reconstruction must carry it into the source.
    runner = _fallback_runner()
    source = runner._build_process_event_source(evt)
    assert source is not None
    assert source.scope_id == "G777"
    assert source.user_id == "U9"


def test_live_completion_event_carries_scope_id(tmp_path, monkeypatch):
    """The live (non-restart) completion event must carry the dispatch-time
    routing origin too, so priming works even when the in-memory source
    cache was evicted."""
    import tools.async_delegation as ad

    record = {
        "delegation_id": "d-live-1",
        "session_key": "agent:main:discord:group:C123:U9",
        "scope_id": "G777",
        "user_id": "U9",
        "goal": "g",
        "dispatched_at": 100.0,
        "completed_at": 101.0,
    }

    captured = {}

    class _Q:
        def put(self, evt):
            captured.update(evt)

    class _PR:
        completion_queue = _Q()

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(
        "tools.process_registry.process_registry", _PR(), raising=False
    )
    ad._push_completion_event(record, {"summary": "ok"}, "completed")
    assert captured.get("scope_id") == "G777"
    assert captured.get("user_id") == "U9"
