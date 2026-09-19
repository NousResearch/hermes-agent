"""Invariant: a webhook delivery's one-shot session is LOCATABLE BY ROUTE NAME (#112274).

A route runs each delivery in its own session (``webhook:<route>:<delivery_id>``), so the
session id is unguessable from any other session.  When the user then says "send that to
Sam" in their DM, the DM agent has no referent at all — the event was processed seconds
earlier and its session exists, but nothing in the DM can name it.  The only workaround is
grepping gateway.log for the session id and calling session_search on it.

The contract these tests pin: the delivery's session is titled after its route, and
``session_search`` resolves that route name (from ANY other session, including the DM) to
the delivery that just ran.  Reuses the existing title store + session_search title
resolution — no new store, no new tool.

Naming rides the REAL spawn path (``_spawn_agent_run``, the seam ``_handle_webhook`` uses
for an accepted event); the runner-side message handler is stubbed exactly as the live
gateway injects it, so the session row and its title come from the adapter's own code.
"""

import json
import time

import pytest

from gateway.config import Platform
from gateway.platforms.webhook import _INSECURE_NO_AUTH
from gateway.session import SessionSource
from tools.session_search_tool import session_search
from tests.gateway.test_webhook_session_close import (  # noqa: F401  (fixtures/helpers)
    _FakeRunner,
    _drain_background_tasks,
    _make_adapter,
    _make_store,
)


ROUTES = {
    "alerts": {"secret": _INSECURE_NO_AUTH, "prompt": "Alert: {message}", "deliver": "log"},
    "pr-bot": {"secret": _INSECURE_NO_AUTH, "prompt": "PR: {message}", "deliver": "log"},
}


def _make_named_adapter(routes=None):
    return _make_adapter(routes or ROUTES)


def _spawn(adapter, route_name: str, delivery_id: str, text: str):
    """One delivery through the adapter's real dispatch seam (returns the spawned task)."""
    return adapter._spawn_agent_run(
        {"message": text}, text, delivery_id, time.time(),
        route_config=adapter._routes[route_name], route_name=route_name, profile=None,
        event_type="push",
    )


def _dm_session(store) -> str:
    """The user's DM session (the session that must be able to refer to the delivery)."""
    entry = store.get_or_create_session(
        SessionSource(platform=Platform.TELEGRAM, chat_id="424242", chat_type="dm", user_id="424242")
    )
    return entry.session_id


def _delivery_session_id(store, route_name: str, delivery_id: str) -> str:
    """The session the adapter keyed this delivery on (what the run wrote into)."""
    for row in store._db._conn.execute(
        "SELECT id FROM sessions WHERE chat_id = ?", (f"webhook:{route_name}:{delivery_id}",)
    ):
        return row[0]
    raise AssertionError(f"no session row for delivery {delivery_id}")


def _search(db, query: str, current_session_id: str) -> dict:
    return json.loads(session_search(query=query, db=db, current_session_id=current_session_id))


@pytest.mark.asyncio
async def test_delivery_session_is_named_after_its_route(tmp_path):
    """The one-shot session carries the route name, so a name→session lookup is possible."""
    store = _make_store(tmp_path)
    adapter = _make_named_adapter()
    adapter.gateway_runner = _FakeRunner(store)

    async def _message_handler(event):
        store.get_or_create_session(event.source)
        return ""

    adapter._message_handler = _message_handler
    _spawn(adapter, "alerts", "named-1", "Alert: server on fire")
    await _drain_background_tasks(adapter)

    row = store._db.get_session(_delivery_session_id(store, "alerts", "named-1"))
    assert row is not None
    assert row["title"] == "webhook:alerts", (
        "the per-delivery session has no route-derived title, so nothing can address it by "
        "route name (session_search sees an untitled row)"
    )


@pytest.mark.asyncio
async def test_dm_session_locates_the_delivery_by_route_name(tmp_path):
    """A DM session can find the delivery that just ran — no session id in hand."""
    store = _make_store(tmp_path)
    adapter = _make_named_adapter()
    adapter.gateway_runner = _FakeRunner(store)

    async def _message_handler(event):
        store.get_or_create_session(event.source)
        return ""

    adapter._message_handler = _message_handler
    _spawn(adapter, "alerts", "locate-1", "Alert: disk full on api-01")
    await _drain_background_tasks(adapter)

    delivery_session = _delivery_session_id(store, "alerts", "locate-1")
    dm_id = _dm_session(store)

    payload = _search(store._db, "webhook:alerts", dm_id)
    session_ids = [r["session_id"] for r in payload["results"]]
    assert delivery_session in session_ids, (
        f"session_search('webhook:alerts') from the DM did not surface the delivery session "
        f"{delivery_session} (got {session_ids})"
    )
    assert payload["results"][0]["session_id"] == delivery_session, "the delivery must be the top hit"
    assert payload["results"][0]["matched_role"] == "session_title"


@pytest.mark.asyncio
async def test_route_query_returns_the_latest_delivery_not_the_stale_one(tmp_path):
    """Two deliveries of one route: the route name resolves to the newer session."""
    store = _make_store(tmp_path)
    adapter = _make_named_adapter()
    adapter.gateway_runner = _FakeRunner(store)

    async def _message_handler(event):
        store.get_or_create_session(event.source)
        return ""

    adapter._message_handler = _message_handler
    _spawn(adapter, "alerts", "old-1", "Alert: first")
    await _drain_background_tasks(adapter)
    old_session = _delivery_session_id(store, "alerts", "old-1")
    # Pin the ordering: the first delivery's session is unambiguously older.
    store._db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (time.time() - 600, old_session))
    store._db._conn.commit()

    _spawn(adapter, "alerts", "new-1", "Alert: second")
    await _drain_background_tasks(adapter)
    new_session = _delivery_session_id(store, "alerts", "new-1")

    dm_id = _dm_session(store)
    top = _search(store._db, "webhook:alerts", dm_id)["results"][0]
    assert top["session_id"] == new_session, (
        f"route lookup returned {top['session_id']} — a stale delivery ({old_session}) instead of "
        f"the one that just ran ({new_session})"
    )


@pytest.mark.asyncio
async def test_route_lookup_does_not_leak_other_routes_or_plain_sessions(tmp_path):
    """The protection surface: one route's name resolves only to that route's deliveries."""
    store = _make_store(tmp_path)
    adapter = _make_named_adapter()
    adapter.gateway_runner = _FakeRunner(store)

    async def _message_handler(event):
        store.get_or_create_session(event.source)
        return ""

    adapter._message_handler = _message_handler
    _spawn(adapter, "alerts", "iso-1", "Alert: unrelated")
    _spawn(adapter, "pr-bot", "iso-2", "PR: unrelated")
    await _drain_background_tasks(adapter)

    alerts_session = _delivery_session_id(store, "alerts", "iso-1")
    pr_session = _delivery_session_id(store, "pr-bot", "iso-2")
    dm_id = _dm_session(store)

    pr_hits = [r["session_id"] for r in _search(store._db, "webhook:pr-bot", dm_id)["results"]]
    assert pr_hits and alerts_session not in pr_hits, "another route's session leaked into the result"

    alerts_hits = [r["session_id"] for r in _search(store._db, "webhook:alerts", dm_id)["results"]]
    assert pr_session not in alerts_hits and dm_id not in alerts_hits

    missing = _search(store._db, "webhook:never-configured", dm_id)
    assert missing["results"] == []
    assert [r["session_id"] for r in _search(store._db, "webhook:alerts", dm_id)["results"]][0] == alerts_session


@pytest.mark.asyncio
async def test_delivery_title_survives_the_runs_own_auto_titler(tmp_path):
    """The run's turn-start titler must not rename the delivery out of its route name.

    The run titles its session from the webhook prompt at turn start; if that won, the route
    name (and with it the only handle another session could search by) would be gone.
    """
    from agent.title_generator import apply_instant_title

    store = _make_store(tmp_path)
    adapter = _make_named_adapter()
    adapter.gateway_runner = _FakeRunner(store)

    async def _message_handler(event):
        store.get_or_create_session(event.source)
        return ""

    adapter._message_handler = _message_handler
    _spawn(adapter, "alerts", "title-1", "Alert: disk full on api-01")
    await _drain_background_tasks(adapter)

    delivery_session = _delivery_session_id(store, "alerts", "title-1")
    apply_instant_title(store._db, delivery_session, "Alert: disk full on api-01")
    assert store._db.get_session(delivery_session)["title"] == "webhook:alerts"


@pytest.mark.asyncio
async def test_naming_is_best_effort_without_a_session_store(tmp_path):
    """No store wired (default/other runners): the delivery still runs, nothing is written."""
    adapter = _make_named_adapter()
    adapter.gateway_runner = object()  # no session_store attribute

    async def _message_handler(event):
        return ""

    adapter._message_handler = _message_handler
    _spawn(adapter, "alerts", "nostore-1", "Alert: whatever")
    await _drain_background_tasks(adapter)
