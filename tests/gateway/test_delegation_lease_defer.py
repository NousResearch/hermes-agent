"""Lease-fenced delegation deliveries must DEFER, not burn the attempt budget (#turn-lease-fix).

2026-09-14 receipt (deleg_b1078862): an async delegation completed while its parent session
held an active turn lease. ``append_delegation_delivery`` correctly refused the row
(SessionTurnLeaseLostError — "Session has an active turn lease"), but the delivery loop
treated each refusal as an ordinary failed attempt: 8 claims/releases against a lease that
outlived the whole budget, then a terminal ``delivery_state='dropped'`` — the subagent's
result was never delivered.

The fix: a lease-fence refusal in ``_self_post_api_server`` re-raises as ``WakeNotAccepted``,
which the completion-delivery settlement maps to the ``defer`` branch —
``defer_completion_delivery`` REFUNDS the attempt (``delivery_attempts-1``) and leaves the
row ``pending``. The watcher requeues and retries once the turn ends; the attempt budget is
never spent on a fence that is guaranteed to lift.
"""

import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

import gateway.wake as wake
from gateway.config import Platform
from gateway.run_notifications import GatewayNotificationsMixin
from gateway.wake import WakeNotAccepted
from hermes_state_errors import SessionTurnLeaseLostError


class _Runner(GatewayNotificationsMixin):
    """Minimal host of the notifications mixin — only what these paths touch."""

    def __init__(self):
        import threading

        self.adapters = {}
        self._session_db = None
        self._completion_delivery_lock = threading.Lock()
        self._completion_deliveries_inflight = set()
        self._completion_deliveries_delivered = {}
        self._completion_delivery_retention = 64

    def _get_cached_session_source(self, session_key):
        return None


@pytest.fixture
def runner():
    return _Runner()


def _lease_fence_error() -> Exception:
    # Exact message shape append_delegation_delivery's guard raises.
    return SessionTurnLeaseLostError(
        "Session has an active turn lease; refusing transcript mutation for 's1'"
    )


def test_lease_fence_raises_wake_not_accepted_for_delegation(runner, monkeypatch):
    """A lease-fenced delegation persist must re-raise WakeNotAccepted (the defer signal)."""
    calls = []

    async def persist(adapter, *, text, session_id, evt):
        calls.append(session_id)
        raise _lease_fence_error()

    monkeypatch.setattr(wake, "persist_delegation_delivery", persist)

    with pytest.raises(WakeNotAccepted, match="deferred"):
        asyncio.run(
            runner._self_post_api_server(
                object(), "text", "s1", {"type": "async_delegation"}
            )
        )
    assert calls == ["s1"]


def test_lease_fence_wake_maps_to_defer_settlement(runner, monkeypatch):
    """WakeNotAccepted from the inject path settles the durable claim as DEFER (refund)."""
    settled = []

    async def inject(synth_text, evt, *, raise_not_accepted=False):
        raise WakeNotAccepted("session turn lease holds 's1'; delivery deferred")

    async def preflight(evt):
        return SimpleNamespace(
            proceed=True, early_result=None, delegation_id="deleg_x", claim_id="claim_x"
        )

    monkeypatch.setattr(runner, "_inject_watch_notification", inject)
    monkeypatch.setattr(runner, "_preflight_completion_delivery", preflight)
    monkeypatch.setattr(
        runner, "_settle_durable_claim",
        lambda kind, delegation_id, claim_id: settled.append(kind),
    )
    evt = {"type": "async_delegation", "delegation_id": "deleg_x", "origin_session_id": "s1"}

    result = asyncio.run(runner._deliver_completion_notification_scoped("text", evt))
    assert result is False
    assert settled == ["defer"]


def test_non_lease_persist_failure_still_returns_false(runner, monkeypatch):
    """A non-lease persist error keeps the old WARNING + False path (no defer)."""
    import logging

    async def persist(adapter, *, text, session_id, evt):
        raise RuntimeError("disk exploded")

    monkeypatch.setattr(wake, "persist_delegation_delivery", persist)
    result = asyncio.run(
        runner._self_post_api_server(
            object(), "text", "s1", {"type": "async_delegation"}
        )
    )
    assert result is False


def test_lease_fence_detector_polarity():
    """_is_active_turn_lease_rejection: lease ownership errors True, everything else False.

    Both SessionTurnLeaseLostError shapes mean 'another live turn owns the transcript right
    now' for a holder-less delivery row — defer is correct for each. Lock errors and
    unrelated failures must stay False (they take the ordinary retry/backoff path).
    """
    assert GatewayNotificationsMixin._is_active_turn_lease_rejection(_lease_fence_error())
    assert GatewayNotificationsMixin._is_active_turn_lease_rejection(
        RuntimeError("Session has an active turn lease; refusing transcript mutation")
    )
    assert GatewayNotificationsMixin._is_active_turn_lease_rejection(
        SessionTurnLeaseLostError("Session turn lease lost; refusing transcript write")
    )
    assert not GatewayNotificationsMixin._is_active_turn_lease_rejection(
        RuntimeError("database is locked")
    )
    assert not GatewayNotificationsMixin._is_active_turn_lease_rejection(ValueError("x"))
    assert not GatewayNotificationsMixin._is_active_turn_lease_rejection(
        RuntimeError("disk exploded")
    )


def test_lease_fence_deferred_before_claim_in_preflight():
    """A live lease on the target session stops the delivery BEFORE the durable claim.

    The pre-flight probe is read-only (WAL read, no write lock) so a long turn costs zero
    claim/defer write cycles — the 2s watcher tick just requeues until the lease clears.
    """
    import time as _time

    runner = _Runner()

    class _LeaseActiveDB:
        def get_session_turn_lease_owner(self, session_id):
            return ("pid=999:turn=t:platform=webui", _time.time() + 300.0)

    class _Adapter:
        supports_async_delivery = False

        def _ensure_session_db(self):
            return object()

    runner.adapters[Platform.API_SERVER] = _Adapter()
    runner._session_db = _LeaseActiveDB()

    async def _classify(parent):
        return "deliver"

    runner._classify_completion_target = _classify

    async def _ready(e):
        return await runner._completion_delivery_ready(e)

    # Raw api_server route: bare session_key, no platform/chat metadata.
    evt = {"type": "async_delegation", "delegation_id": "d", "parent_session_id": "s1",
           "session_key": "s1"}
    assert asyncio.run(_ready(evt)) is False  # lease live → requeue without claiming

    class _LeaseFreeDB:
        def get_session_turn_lease_owner(self, session_id):
            return None

    runner._session_db = _LeaseFreeDB()
    assert asyncio.run(_ready(evt)) is True


def test_lease_probe_fails_open_on_errors():
    """A probe that cannot read the lease fails OPEN (proceed to claim/persist), where the
    persist-side fence + defer refund still protect the row."""
    runner = _Runner()
    evt = {"type": "async_delegation", "delegation_id": "d", "parent_session_id": "s1"}

    class _BrokenDB:
        def get_session_turn_lease_owner(self, session_id):
            raise sqlite3.OperationalError("database is locked")

    class _Adapter:
        supports_async_delivery = False

        def _ensure_session_db(self):
            return object()

    runner.adapters[Platform.API_SERVER] = _Adapter()
    runner._session_db = _BrokenDB()

    async def _classify(parent):
        return "deliver"

    runner._classify_completion_target = _classify

    evt = {"type": "async_delegation", "delegation_id": "d", "parent_session_id": "s1",
           "session_key": "s1"}

    async def _ready():
        return await runner._completion_delivery_ready(evt)

    assert asyncio.run(_ready()) is True  # fail-open: no lease evidence → proceed
