"""Regression: synthetic injections must prime relay egress routing metadata.

Staging incident 2026-08-09 (defect #4, post-restart replay): restored
async-delegation completions injected fine after the routing fix, but their
egress bounced at the connector — "slack egress declined: target not routed
to an onboarded tenant". The relay adapter re-attaches the tenant
discriminators (metadata.scope_id / metadata.user_id) from per-chat caches
warmed ONLY by the inbound path (_on_inbound -> _capture_scope). Synthetic
completion turns call handle_message directly, so right after a restart the
caches are cold and every reply the injected turn produces is declined by
the fail-closed egress guard until real inbound traffic arrives.

The session-store origin carried on the synthetic event already holds
scope_id/user_id — the fix is priming the adapter's routing caches from the
synthetic event before dispatching it.

Also pins the replay staleness cap: restored completions older than the cap
must be terminally dropped, not replayed as full-context turns (one July
session replayed in August burned a 102K-token context).
"""

import json
import time
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.relay.adapter import RelayAdapter
from gateway.session import SessionSource


def _bare_relay_adapter():
    a = object.__new__(RelayAdapter)
    a._scope_by_chat = {}
    a._dm_user_by_chat = {}
    a._platform_by_chat = {}
    a._chat_type_by_chat = {}
    a._last_inbound_ts_by_chat = {}
    return a


def _synth_event_with_origin():
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="D0BJTDCSR7C",
        chat_type="dm",
        user_id="U0B5F8EEYAD",
        scope_id="T0AAAA111",
    )
    return SimpleNamespace(source=source, message_id="1786298425.877239")


def test_prime_routing_cache_warms_egress_discriminators():
    """Priming from a synthetic event must give _with_scope everything the
    connector's egress guard needs — scope_id AND user_id."""
    adapter = _bare_relay_adapter()
    adapter.prime_routing_cache(_synth_event_with_origin())

    meta = adapter._with_scope("D0BJTDCSR7C", None)
    assert meta.get("user_id") == "U0B5F8EEYAD", (
        "DM tenant discriminator missing after priming — egress would be "
        "declined 'not routed to an onboarded tenant' (defect #4)"
    )
    assert meta.get("scope_id") == "T0AAAA111"


@pytest.mark.asyncio
async def test_injection_path_primes_before_handle_message():
    """End-to-end wiring: _inject_watch_notification must call
    prime_routing_cache on the resolved adapter BEFORE handle_message —
    a helper nobody calls fixes nothing."""
    from unittest.mock import AsyncMock
    from gateway.run import GatewayRunner

    calls = []

    class _Adapter:
        name = "relay"

        def fronts_platform(self, platform):
            return platform == Platform.SLACK

        def prime_routing_cache(self, event):
            calls.append(("prime", getattr(event.source, "chat_id", None)))

        async def handle_message(self, event):
            calls.append(("handle", getattr(event.source, "chat_id", None)))

    runner = object.__new__(GatewayRunner)
    runner._running = True
    adapter = _Adapter()
    runner.adapters = {Platform.RELAY: adapter}
    runner.config = SimpleNamespace(platforms={})

    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_prime_wiring",
        "session_key": "agent:main:slack:dm:D0BJTDCSR7C:1786298425.877239",
        "platform": "slack",
        "chat_type": "dm",
        "chat_id": "D0BJTDCSR7C",
        "status": "completed",
    }
    result = await runner._inject_watch_notification("[done]", evt)
    assert result is True
    assert calls and calls[0][0] == "prime", (
        f"injection path never primed the adapter (calls={calls}) — cold "
        "caches would bounce every post-restart reply at the egress guard"
    )
    assert calls == [("prime", "D0BJTDCSR7C"), ("handle", "D0BJTDCSR7C")]


def test_prime_routing_cache_never_raises_on_malformed_event():
    adapter = _bare_relay_adapter()
    adapter.prime_routing_cache(SimpleNamespace(source=None))
    adapter.prime_routing_cache(None)
    assert adapter._with_scope("X", {"k": "v"}).get("k") == "v"


# ---------------------------------------------------------------------------
# Staleness cap on restored completions
# ---------------------------------------------------------------------------

@pytest.fixture()
def _isolated_delegation_db(tmp_path, monkeypatch):
    import tools.async_delegation as ad

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    return ad


def _insert_pending(ad, delegation_id, completed_at):
    evt = {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": "agent:main:slack:dm:D1:2",
        "status": "completed",
    }
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            """INSERT INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id,
                parent_session_id, state, dispatched_at, completed_at,
                delivery_state, event_json, updated_at)
               VALUES (?, ?, '', ?, 'completed', ?, ?, 'pending', ?, ?)""",
            (
                delegation_id, "agent:main:slack:dm:D1:2", "parent-1",
                completed_at, completed_at, json.dumps(evt), completed_at,
            ),
        )


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_restore_drops_completions_older_than_replay_cap(_isolated_delegation_db):
    ad = _isolated_delegation_db
    now = time.time()
    _insert_pending(ad, "deleg_fresh", now - 3600)
    # ABSOLUTE age, deliberately NOT derived from the cap constant: a 30-day
    # old completion must never replay, whatever the cap is tuned to.
    _insert_pending(ad, "deleg_stale", now - 30 * 24 * 3600)
    assert ad._MAX_COMPLETION_REPLAY_AGE_S <= 7 * 24 * 3600, (
        "replay cap drifted past a week — the 102K-token stale-replay class "
        "would return"
    )

    q = _Queue()
    restored = ad.restore_undelivered_completions(q)

    ids = [e.get("delegation_id") for e in q.items]
    assert "deleg_fresh" in ids
    assert "deleg_stale" not in ids, (
        "a weeks-old completion was replayed as a fresh full-context turn "
        "(102K-token replay incident class)"
    )
    assert restored == 1
    # The stale row must converge to a terminal state, not replay forever.
    with ad._DB_LOCK, ad._transaction() as conn:
        state = conn.execute(
            "SELECT delivery_state FROM async_delegations WHERE delegation_id='deleg_stale'"
        ).fetchone()[0]
    assert state == "dropped"
