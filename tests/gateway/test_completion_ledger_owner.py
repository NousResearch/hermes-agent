"""A deleted raw local parent must not strand its secondary profile's result."""

import asyncio
import queue
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.runtime_scope import bind_profile_scope
from gateway.run import GatewayRunner
from hermes_constants import get_hermes_home, hermes_home_key
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_session
from tests.gateway.test_completion_delivery import _runner
from tools import async_delegation as ledger


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    homes = {name: tmp_path / name for name in ("default", "worker", "reviewer")}
    dbs = {}
    for name, home in homes.items():
        home.mkdir()
        dbs[hermes_home_key(home)] = open_persistence(profile_scope(home=home))
    monkeypatch.setenv("HERMES_HOME", str(homes["default"]))
    monkeypatch.setattr(GatewayRunner, "persistence", property(
        lambda self: dbs[hermes_home_key(get_hermes_home())]))
    runner = _runner(None)
    runner.config = SimpleNamespace(multiplex_profiles=True)
    runner._primary_profile_name = "default"
    runner._served_profile_homes = homes
    # A removed local parent has no source from which to rediscover its owner.
    runner._build_process_event_source = lambda evt: None
    runner._served_api_server_wake_profile = lambda evt, sid: None
    runner._get_cached_session_source = lambda key: None
    runner._completion_delivery_ready = AsyncMock(return_value=True)
    monkeypatch.setattr("gateway.session_local_recovery.restore_completion_route", lambda *a: None)
    ledger._reset_for_tests()
    try:
        yield runner, homes, dbs
    finally:
        ledger._reset_for_tests()
        for db in dbs.values():
            db.close()


def seed(home, delegation_id, *, parent="local-shared", stored_owner="/forged"):
    evt = {"type": "async_delegation", "delegation_id": delegation_id,
           "session_key": parent, "parent_session_id": parent, "status": "completed",
           "summary": "kept result", "completed_at": time.time(),
           "_completion_ledger_home": stored_owner}
    with bind_profile_scope(home, hydrate_secrets=False):
        ledger._persist_dispatch({**evt, "dispatched_at": time.time()})
        ledger._persist_completion(evt, {"summary": "kept result"})
        q = queue.Queue()
        assert ledger.restore_undelivered_completions(q) == 1
        return q.get_nowait()


def row(home, delegation_id):
    with bind_profile_scope(home, hydrate_secrets=False):
        saved = ledger.get_durable_delegation(delegation_id)
        return None if saved is None else (saved["delivery_state"], saved["delivery_attempts"], saved["result"])


def test_deleted_raw_parent_settles_only_owning_ledger(fleet):
    runner, homes, _ = fleet
    evt = seed(homes["worker"], "deleg_gone")
    before = row(homes["worker"], "deleg_gone")
    runner._inject_watch_notification = AsyncMock(side_effect=AssertionError("deleted parent must not wake"))
    assert asyncio.run(runner._deliver_completion_notification("done", evt)) is None
    after = row(homes["worker"], "deleg_gone")
    assert after == ("dropped", 1, before[2])
    assert row(homes["default"], "deleg_gone") is None
    with bind_profile_scope(homes["worker"], hydrate_secrets=False):
        q = queue.Queue()
        assert ledger.restore_undelivered_completions(q) == 0
        assert q.empty()


def test_healthy_same_raw_parent_keeps_profile_and_recovers_refusal(fleet):
    runner, homes, dbs = fleet
    seen = []

    async def inject(text, evt, **kwargs):
        seen.append(hermes_home_key(get_hermes_home()))
        return len(seen) > 1

    runner._inject_watch_notification = inject
    for name in ("worker", "reviewer"):
        seed_session(dbs[hermes_home_key(homes[name])], "local-shared", "api_server")
    evt = seed(homes["worker"], "deleg_live")
    original_result = row(homes["worker"], "deleg_live")[2]
    assert asyncio.run(runner._deliver_completion_notification("done", evt)) is False
    assert row(homes["worker"], "deleg_live")[0] == "pending"
    assert asyncio.run(runner._deliver_completion_notification("done", evt)) is True
    assert seen == [hermes_home_key(homes["worker"])] * 2
    assert row(homes["worker"], "deleg_live")[0] == "delivered"
    assert row(homes["worker"], "deleg_live")[2] == original_result
    assert row(homes["reviewer"], "deleg_live") is None
    other = seed(homes["reviewer"], "deleg_live")
    assert runner._event_route_key(evt, runner._ASYNC_GROUP_KEY_FIELDS) != runner._event_route_key(
        other, runner._ASYNC_GROUP_KEY_FIELDS)
    assert asyncio.run(runner._deliver_completion_notification("later", other)) is True
    assert seen[-1] == hermes_home_key(homes["reviewer"])


def test_replay_overwrites_stored_owner_hint(fleet):
    _, homes, _ = fleet
    evt = seed(homes["worker"], "deleg_hint", stored_owner=hermes_home_key(homes["reviewer"]))
    assert evt.get("_completion_ledger_home") == hermes_home_key(homes["worker"])


def test_unserved_owner_is_refused_without_settlement(fleet):
    runner, homes, _ = fleet
    evt = seed(homes["worker"], "deleg_unknown")
    evt["_completion_ledger_home"] = str(homes["worker"].parent / "unserved")

    async def attempt():
        async with runner._completion_event_scope(evt):
            pytest.fail("unserved ledger owner was admitted")

    with pytest.raises(LookupError):
        asyncio.run(attempt())
    target = queue.Queue()
    asyncio.run(runner._deliver_async_delegation_groups(
        {"unserved": [evt]}, SimpleNamespace(completion_queue=target)))
    assert target.get_nowait() == evt
    assert row(homes["worker"], "deleg_unknown")[:2] == ("pending", 0)
    assert row(homes["default"], "deleg_unknown") is None


def test_live_final_copy_carries_dispatch_owner(fleet, monkeypatch):
    _, homes, _ = fleet
    import tools.process_registry as registry

    q = queue.Queue()
    monkeypatch.setattr(registry.process_registry, "completion_queue", q)
    record = {"delegation_id": "deleg_live_copy", "session_key": "local-shared",
              "parent_session_id": "local-shared", "dispatched_at": time.time(),
              "_home": hermes_home_key(homes["worker"]), "goal": "test"}
    with bind_profile_scope(homes["worker"], hydrate_secrets=False):
        ledger._persist_dispatch(record)
        ledger._push_completion_event(record, {"summary": "kept result"}, "completed")
    assert q.get_nowait().get("_completion_ledger_home") == hermes_home_key(homes["worker"])


def test_interim_copy_keeps_owner_without_settling_final(fleet, monkeypatch):
    runner, homes, _ = fleet
    import tools.process_registry as registry

    q = queue.Queue()
    monkeypatch.setattr(registry.process_registry, "completion_queue", q)
    record = {"delegation_id": "deleg_interim", "session_key": "local-shared",
              "parent_session_id": "local-shared", "dispatched_at": time.time(),
              "_home": hermes_home_key(homes["worker"]), "status": "running"}
    monkeypatch.setitem(ledger._records, "deleg_interim", record)
    with bind_profile_scope(homes["worker"], hydrate_secrets=False):
        ledger._persist_dispatch(record)
        ledger.push_task_failure_notice("deleg_interim", {"task_index": 0, "status": "error"}, n_tasks=2)
    evt = q.get_nowait()
    assert evt["_completion_ledger_home"] == hermes_home_key(homes["worker"])
    before = row(homes["worker"], "deleg_interim")
    # No adapter is configured: leave the notice retryable and the running final row untouched.
    assert asyncio.run(runner._deliver_completion_notification("interim", evt)) is False
    assert row(homes["worker"], "deleg_interim") == before


@pytest.mark.asyncio
async def test_replay_finds_admission_committed_before_owner_stamps(tmp_path, monkeypatch):
    """A lost delivery ACK must not hide the native FIFO's pre-upgrade admission."""
    from gateway.session_automation import completion_admission
    from persistence_assembly import close_stores_under
    from tests.gateway.test_completion_admission import pending
    from tests.gateway.test_session_end_delegations import _local
    from tests.gateway.test_unviewed_completion_notes import _target

    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        route = authority.sessions[ref.session_id].route
        event = dict(pending(route, "deleg_legacy"), parent_session_id=_target(runner, authority, ref))
        assert await runner._deliver_async_delegation_group([event]) is True
        committed = completion_admission(runner, event)
        assert committed is not None
        # Older producers committed this same raw identity with no owner field.
        replay = event | {"_completion_ledger_home": authority.persistence.scope.hermes_home_key}
        assert completion_admission(runner, replay) == committed
    finally:
        close_stores_under(authority.persistence.scope.profile_dir)
