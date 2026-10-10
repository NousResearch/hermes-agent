"""Behavior contracts for durable ordinary inbound admission."""

import concurrent.futures
import sqlite3
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source

from gateway.chat_execution_recovery import ChatExecutionStore


def test_reservations_are_unique_and_restart_floor_is_respected(tmp_path):
    store = ChatExecutionStore(tmp_path)
    assert store.reserve("restart", 8, 4) == 9

    def reserve(_):
        return ChatExecutionStore(tmp_path).reserve("parallel", 0, 0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        generations = list(pool.map(reserve, range(24)))
    assert sorted(generations) == list(range(1, 25))


def test_reservations_are_isolated_by_profile_home(tmp_path):
    first = ChatExecutionStore(tmp_path / "first")
    second = ChatExecutionStore(tmp_path / "second")
    assert first.reserve("agent:main", 0, 0) == 1
    assert first.reserve("agent:main", 0, 0) == 2
    assert second.reserve("agent:main", 0, 0) == 1
    assert first.path != second.path


def test_reservation_honors_memory_only_floor(tmp_path):
    store = ChatExecutionStore(tmp_path)
    # Stop/reset may advance the in-memory generation without starting a worker.
    assert store.latest_generation("agent:main") == 0
    assert store.reserve("agent:main", 17, 0) == 18
    assert store.latest_generation("agent:main") == 18


def test_fenced_admission_is_refused(tmp_path):
    store = ChatExecutionStore(tmp_path)
    assert store.start("agent:main", 1, "session-1")
    assert store.begin_recovery("agent:main", 1, "recovery-1")
    assert store.reserve("agent:main", 1, 1) is None
    assert store.latest_generation("agent:main") == 1


def test_sqlite_admission_failure_is_observable(tmp_path):
    store = ChatExecutionStore(tmp_path)
    store.path.unlink()
    store.path.mkdir()
    with pytest.raises(sqlite3.OperationalError):
        store.reserve("agent:main", 0, 0)



def inbound_fixture(home):
    """Real handler, generation and cleanup; isolate unrelated ingress and model I/O."""
    runner, _ = make_restart_runner()
    source = make_restart_source(chat_id="durable-admission")
    from gateway.session_identity import RoutingIdentity
    source._identity = RoutingIdentity(
        transport_profile="transport", runtime_profile="worker",
        authorization_home=home / "authorization", runtime_home=home,
        multiplexed=False,
    )
    event = MessageEvent(text="hello", source=source, message_type=MessageType.TEXT)
    runner._hm_admit_event = AsyncMock(return_value=(event, source, True))
    runner._hm_estop_gate = lambda *args: None
    runner._hm_pending_reply_intercepts = AsyncMock(return_value=None)
    runner._hm_evict_idle_stale_agent = lambda *args: None
    runner._hm_evict_reaped_agent = lambda *args: None
    runner._hm_dispatch_idle_commands = AsyncMock(return_value=(False, None))
    runner._hm_rescue_orphaned_fifo = lambda *args: args[:3]
    slot = Mock()
    runner._claim_active_session_slot = lambda *args: (slot, None)
    runner._persist_active_agents = Mock()
    runner._handle_message_with_agent = AsyncMock(return_value="executed")
    runner._run_post_turn_hooks = AsyncMock()
    runner._clear_durable_active_turn = AsyncMock()
    return runner, event, runner._session_key_for_source(source), slot


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["fenced", "database"])
async def test_actual_inbound_refusal_releases_slot_without_executing(tmp_path, failure):
    runner, event, key, slot = inbound_fixture(tmp_path)
    store = ChatExecutionStore(tmp_path)
    if failure == "fenced":
        assert store.start(key, 1, "physical-session")
        assert store.begin_recovery(key, 1, "recovery")
    else:
        store.path.unlink()
        store.path.mkdir()
    reply = await runner._handle_message(event)
    assert reply
    runner._handle_message_with_agent.assert_not_awaited()
    slot.release.assert_called_once_with()
    assert not runner._is_session_running(key)
    assert key not in runner._running_agents
    assert key not in runner._running_agents_ts
    assert runner._session_state(key).turn.lease is None
    assert runner._session_state(key).turn.event is None


@pytest.mark.asyncio
async def test_actual_inbound_restart_and_memory_floor(tmp_path):
    store = ChatExecutionStore(tmp_path)
    runner, event, key, slot = inbound_fixture(tmp_path)
    assert store.reserve(key, 8, 0) == 9
    assert runner._current_session_run_generation(key) == 0
    assert await runner._handle_message(event) == "executed"
    assert runner._handle_message_with_agent.await_args.args[-1] == 10
    slot.release.assert_called_once_with()
    restarted, event, key, _ = inbound_fixture(tmp_path)
    restarted._session_state(key).persistent.run_generation = 17
    assert await restarted._handle_message(event) == "executed"
    assert restarted._handle_message_with_agent.await_args.args[-1] == 18
    assert store.latest_generation(key) == 18
    assert store.inspect(key, 18)["session_id"] is None


@pytest.mark.asyncio
async def test_actual_inbound_profile_fence_isolation(tmp_path):
    first_home, second_home = tmp_path / "first", tmp_path / "second"
    first, event, key, slot = inbound_fixture(first_home)
    store = ChatExecutionStore(first_home)
    assert store.start(key, 1, "session")
    assert store.begin_recovery(key, 1, "recovery")
    assert await first._handle_message(event) != "executed"
    first._handle_message_with_agent.assert_not_awaited()
    slot.release.assert_called_once_with()
    second, event, second_key, _ = inbound_fixture(second_home)
    assert key == second_key
    assert await second._handle_message(event) == "executed"
    assert second._handle_message_with_agent.await_args.args[-1] == 1
    assert store.latest_generation(key) == 1
    assert ChatExecutionStore(second_home).latest_generation(key) == 1
    assert not (first_home / "authorization" / "chat-execution.sqlite3").exists()
    assert not (second_home / "authorization" / "chat-execution.sqlite3").exists()


def test_actual_stop_invalidation_does_not_journal_execution(tmp_path):
    runner, _, key, _ = inbound_fixture(tmp_path)
    store = ChatExecutionStore(tmp_path)
    assert runner._invalidate_session_run_generation(key, reason="stop") == 1
    assert runner._begin_session_run_generation(key) == 2
    assert store.latest_generation(key) == 0
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM executions").fetchone()[0] == 0
