import asyncio
import queue
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner, _format_gateway_process_notification


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.async_delegation as ad
    import tools.process_registry as pr_module

    ad._reset_for_tests()
    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    yield registry
    ad._reset_for_tests()


def _runner(adapter):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(_ensure_loaded=lambda: None, _entries={})
    runner._session_source_cache = {}
    runner._completion_delivery_lock = __import__("threading").Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    runner._background_tasks = set()
    return runner


def _event():
    envelope = {
        "type": "async_delegation_work_closeout",
        "work_id": "work-1",
        "generation": 0,
        "delivery_id": "closeout-1",
        "members": [{"delegation_id": "deleg-1", "status": "completed"}],
    }
    return {
        "type": "async_delegation_work_closeout",
        "delivery_id": "closeout-1",
        "origin_work_id": "work-1",
        "work_generation": 0,
        "claim_id": "claim-1",
        "envelope": envelope,
        "session_key": "agent:main:telegram:dm:12345:678",
        "parent_session_id": "session-1",
    }


def _stop_after_sleeps(monkeypatch, runner, count=2):
    calls = 0

    async def bounded_sleep(_delay):
        nonlocal calls
        calls += 1
        if calls >= count:
            runner._running = False

    monkeypatch.setattr(asyncio, "sleep", bounded_sleep)


def test_closeout_formatter_is_bounded_internal_envelope():
    text = _format_gateway_process_notification(_event())
    assert text is not None
    assert "INTERNAL DELEGATION CLOSEOUT" in text
    assert "closeout-1" in text
    assert "deleg-1" in text


def test_closeout_watcher_injects_typed_metadata_once(
    monkeypatch, isolated_registry,
):
    work_queue = queue.Queue()
    isolated_registry.completion_queue = work_queue
    work_queue.put(_event())
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner)
    monkeypatch.setattr(
        runner, "_classify_completion_target", AsyncMock(return_value="deliver")
    )

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_awaited_once()
    delivered = adapter.handle_message.await_args.args[0]
    assert delivered.internal is True
    assert delivered.metadata["delegation_closeout"] == {
        "work_id": "work-1",
        "generation": 0,
        "delivery_id": "closeout-1",
        "claim_id": "claim-1",
    }
    assert runner._completion_delivery_identity(_event()) == (
        "async_delegation_work_closeout",
        "closeout-1",
        "",
    )


def test_gateway_closeout_deduplication_is_scoped_by_profile():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    runner._classify_completion_target = AsyncMock(return_value="deliver")
    runner._inject_watch_notification = AsyncMock(return_value=True)
    profile_a = _event()
    profile_b = _event()
    profile_a["_ledger_profile_home"] = "/tmp/profile-a"
    profile_b["_ledger_profile_home"] = "/tmp/profile-b"

    async def deliver_both():
        return (
            await runner._deliver_completion_notification("a", profile_a),
            await runner._deliver_completion_notification("b", profile_b),
        )

    assert asyncio.run(deliver_both()) == (True, True)
    assert runner._inject_watch_notification.await_count == 2
    assert runner._completion_delivery_identity(profile_a) != (
        runner._completion_delivery_identity(profile_b)
    )


def test_api_closeout_self_post_forwards_trusted_profile(monkeypatch):
    from gateway import wake

    adapter = SimpleNamespace(supports_async_delivery=False)
    runner = _runner(adapter)
    runner.adapters = {Platform.API_SERVER: adapter}
    event = _event()
    event["session_key"] = "raw-api-session"
    event["origin_session_id"] = "raw-api-session"
    deliver = AsyncMock()
    monkeypatch.setattr(wake, "deliver_wake", deliver)

    result = asyncio.run(
        runner._inject_watch_notification(
            "closeout",
            event,
            trusted_profile_name="secondary",
        )
    )

    assert result is True
    assert deliver.await_args.kwargs["profile"] == "secondary"
    assert deliver.await_args.kwargs["session_id"] == "raw-api-session"


def test_busy_session_keeps_closeout_until_real_session_becomes_idle(
    monkeypatch, isolated_registry
):
    event = _event()
    isolated_registry.completion_queue.put(event)
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    from gateway.session_state import SessionState

    state = SessionState()
    state.turn.agent = object()
    runner._sessions = {event["session_key"]: state}
    sleeps = 0

    async def busy_then_idle(_delay):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:
            state.turn.agent = None
        elif sleeps >= 3:
            runner._running = False

    monkeypatch.setattr(asyncio, "sleep", busy_then_idle)
    monkeypatch.setattr(
        runner, "_classify_completion_target", AsyncMock(return_value="deliver")
    )

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_awaited_once()
    assert isolated_registry.completion_queue.empty()


def test_multiplex_recovery_release_retry_and_close_stay_in_secondary_profile(
    monkeypatch, isolated_registry, tmp_path
):
    from gateway import run as run_module
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )
    from tools import async_delegation as ad

    default_home = tmp_path
    secondary_home = tmp_path / "profiles" / "secondary"
    secondary_home.mkdir(parents=True)
    token = set_hermes_home_override(secondary_home)
    try:
        assert ad.register_work_group_member(
            work_id="secondary-work",
            owner_turn_id="owner-turn",
            delegation_id="secondary-child",
            feature_config={"delegation": {"task_scoped_closeout": True}},
            routing={
                "origin_session": "agent:main:telegram:dm:12345:678",
                "parent_session_id": "secondary-session",
            },
            task={"goal": "review", "task_index": 0},
        )
        assert not ad.persist_group_member_completion(
            "secondary-child",
            {"delegation_id": "secondary-child", "status": "completed"},
            {"status": "completed", "summary": "done"},
        )
        assert ad.seal_work_group("secondary-work", "owner-turn")
        old = ad.claim_ready_work_group("secondary-work", "pre-crash")
        assert old is not None
        discarded = queue.Queue()
        assert ad._enqueue_claimed_work_group(old, target_queue=discarded)
        discarded.get_nowait()
        with ad._transaction() as conn:
            conn.execute(
                """UPDATE async_delegation_work_groups
                   SET closeout_claimed_at=0, closeout_owner_pid=2147483647,
                       closeout_owner_started_at=1"""
            )
    finally:
        reset_hermes_home_override(token)

    ad._reset_for_tests()
    monkeypatch.setattr(
        run_module,
        "_multiplex_profile_homes",
        lambda _config: [
            ("default", Path(default_home)),
            ("secondary", Path(secondary_home)),
        ],
    )
    runner = _runner(SimpleNamespace(handle_message=AsyncMock()))
    runner.config = SimpleNamespace(multiplex_profiles=True)
    recovered_queue = queue.Queue()

    runner._recover_closeout_work_groups(recovered_queue)
    recovered = recovered_queue.get_nowait()
    assert Path(recovered["_ledger_profile_home"]) == secondary_home.resolve()
    with runner._closeout_event_runtime_scope(recovered) as profile_name:
        assert profile_name == "secondary"
        assert ad.release_enqueued_work_group_event(recovered)

    runner._recover_closeout_work_groups(recovered_queue)
    retried = recovered_queue.get_nowait()
    assert retried["delivery_id"] == recovered["delivery_id"]
    assert retried["claim_id"] != recovered["claim_id"]
    with runner._closeout_event_runtime_scope(retried) as profile_name:
        assert profile_name == "secondary"
        assert ad.bind_work_group_closeout_turn(
            "secondary-work",
            retried["delivery_id"],
            retried["claim_id"],
            "closeout-turn",
        )
        assert ad.close_work_group(
            "secondary-work",
            0,
            retried["delivery_id"],
            retried["claim_id"],
            "closeout-turn",
        )
        with ad._transaction() as conn:
            assert conn.execute(
                "SELECT state FROM async_delegation_work_groups "
                "WHERE work_id='secondary-work'"
            ).fetchone() == ("closed",)

    with ad._transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM async_delegation_work_groups"
        ).fetchone()[0] == 0
