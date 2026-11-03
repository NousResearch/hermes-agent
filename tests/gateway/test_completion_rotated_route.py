"""An async delegation completion pinned to a session its ROUTE has rolled past still reaches the route.

Admission follows compression lineage only, so a completion pinned to a session that ended without a user
boundary (idle/daily reset, archived thread, stale reap) while the same route runs a new session was refused
on every retry (``admission_conflict``): deferred with its attempt refunded, never delivered, never surfaced.
The policy: deliver it into that route's live, non-suspended session, labelled as a previous-session result.
Never across routes (``session_key`` is the security boundary), never past a user boundary (``/new``).
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource
from gateway.session_automation import _owned_by
from tools import async_delegation as ad

LABEL = "[Previous session's delegation result \u2014 deleg_rotated]"


def _source(chat_id):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm")


def _event(session_key, pin, delegation_id="deleg_rotated"):
    return {
        "type": "async_delegation", "delegation_id": delegation_id, "session_key": session_key,
        "parent_session_id": pin, "dispatched_at": 1.0, "status": "completed", "summary": "the result",
    }


def _persist(evt):
    ad._persist_dispatch(evt)
    ad._persist_completion(evt, {"status": evt["status"], "summary": evt["summary"]})


class _Route:
    """A runner whose adapter admits like the session authority: only the route's own current session."""

    def __init__(self, tmp_path):
        self.scope = _profile_runtime_scope(tmp_path)
        self.scope.__enter__()
        self.runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
        self.store = self.runner.session_store
        self.db = cast(Any, self.store._db)
        self.injected: list[Any] = []

        async def accept(event):
            self.injected.append(event)
            current = self.store.lookup_by_session_key(event.metadata["gateway_session_key"])
            if current is not None and _owned_by(self.db, event.metadata["gateway_session_id"], current.session_id):
                event._gateway_accepted = True

        self.runner.adapters[Platform.TELEGRAM] = cast(BasePlatformAdapter, SimpleNamespace(handle_message=accept))

    def route(self, chat_id):
        return self.store.get_or_create_session(_source(chat_id))

    def rotate(self, chat_id, end_reason=None):
        """Roll the route to a new session; ``end_reason`` ends the old one that way first (first reason wins)."""
        old = self.store.get_or_create_session(_source(chat_id))
        if end_reason:
            self.db.end_session(old.session_id, end_reason)
        new = self.store.get_or_create_session(_source(chat_id), force_new=True)
        assert new.session_id != old.session_id
        return old, new

    async def deliver(self, evt):
        _persist(evt)
        return await self.runner._deliver_completion_notification("the result", evt)

    def close(self):
        self.store.close_all_db_handles()
        self.runner.close_all_session_db_handles()
        self.scope.__exit__(None, None, None)


@pytest.fixture
def route(tmp_path, private_db_probe_cleanup):
    fixture = _Route(tmp_path)
    try:
        yield fixture
    finally:
        fixture.close()


@pytest.mark.asyncio
async def test_pinned_session_rolled_over_without_a_user_boundary_delivers_into_the_live_session(route):
    old, new = route.rotate("rotated-ok", end_reason="idle")
    assert not _owned_by(route.db, old.session_id, new.session_id)  # the stranded case: admission refuses the pin

    assert await route.deliver(_event(old.session_key, old.session_id)) is True

    (event,) = route.injected
    assert event.metadata["gateway_session_id"] == new.session_id
    assert event.metadata["gateway_proven_session_id"] == new.session_id
    assert event.text.count(LABEL) == 1 and "the result" in event.text
    row = ad.get_durable_delegation("deleg_rotated")
    assert (row["delivery_state"], row["delivery_attempts"]) == ("delivered", 1)
    assert route.store.lookup_by_session_key(old.session_key).session_id == new.session_id


@pytest.mark.asyncio
async def test_pinned_session_closed_by_new_is_still_refused(route):
    old, new = route.rotate("rotated-new", end_reason="session_reset")  # /new stamps this user boundary

    assert await route.deliver(_event(old.session_key, old.session_id)) is None

    assert route.injected == []
    row = ad.get_durable_delegation("deleg_rotated")
    assert row["delivery_state"] != "delivered"
    assert route.store.lookup_by_session_key(old.session_key).session_id == new.session_id


@pytest.mark.asyncio
async def test_a_different_route_is_never_retargeted(route):
    old_a, new_a = route.rotate("route-a", end_reason="idle")
    other = route.route("route-b")  # a live session on ANOTHER route
    assert other.session_key != old_a.session_key
    # The completion is stamped with route B but pinned to route A's ended session.
    assert await route.runner._rotated_route_successor(old_a.session_id, other.session_key) == ""

    assert await route.deliver(_event(other.session_key, old_a.session_id)) is False

    (event,) = route.injected
    assert event.metadata["gateway_session_id"] == old_a.session_id  # untouched pin, refused by admission
    assert LABEL not in event.text
    assert not _owned_by(route.db, old_a.session_id, other.session_id)
    assert not _owned_by(route.db, old_a.session_id, new_a.session_id)
    row = ad.get_durable_delegation("deleg_rotated")
    assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)
    assert route.store.lookup_by_session_key(other.session_key).session_id == other.session_id
    assert route.store.lookup_by_session_key(old_a.session_key).session_id == new_a.session_id


@pytest.mark.asyncio
async def test_no_live_session_on_the_route_keeps_todays_pending_row(route):
    old, new = route.rotate("rotated-suspended", end_reason="idle")
    route.store._entries[old.session_key].suspended = True  # nothing live to deliver into

    assert await route.deliver(_event(old.session_key, old.session_id)) is False

    (event,) = route.injected
    assert event.metadata["gateway_session_id"] == old.session_id and LABEL not in event.text
    row = ad.get_durable_delegation("deleg_rotated")
    assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)
    assert route.store.lookup_by_session_key(old.session_key).session_id == new.session_id
