import asyncio
import multiprocessing
import sqlite3
import threading

import pytest

from gateway.chat_execution_recovery import ChatExecutionStore


def _concurrent_start(home, generation, barrier, output):
    store = ChatExecutionStore(home)
    barrier.wait()
    output.put(store.start("s", generation, "sid"))


def test_durable_generation_floor_survives_reopen(tmp_path):
    store = ChatExecutionStore(tmp_path)
    assert store.latest_generation('s') == 0
    assert store.start('s', 9, 'sid')
    assert ChatExecutionStore(tmp_path).latest_generation('s') == 9
    assert not ChatExecutionStore(tmp_path).start('s', 1, 'sid')


def test_reserve_generation_atomically_above_durable_and_memory_floors(tmp_path):
    store = ChatExecutionStore(tmp_path)
    first = store.reserve("s", 8, 2)
    assert first == 9
    assert store.inspect("s", first)["session_id"] is None
    assert store.reserve("s", 8, 2) == 10


def test_reservation_binds_actual_session_and_starts_once(tmp_path):
    store = ChatExecutionStore(tmp_path)
    generation = store.reserve("s", 0, 0)
    assert store.inspect("s", generation)["session_id"] is None
    assert store.bind_worker_started("s", generation, "actual-session")
    assert not store.bind_worker_started("s", generation, "actual-session")
    proof = ChatExecutionStore(tmp_path).inspect("s", generation)
    assert proof["session_id"] == "actual-session"
    assert proof["worker"] == "pending"


def test_binding_rejects_stale_and_fenced_reservations(tmp_path):
    store = ChatExecutionStore(tmp_path)
    first = store.reserve("s", 0, 0)
    second = store.reserve("s", 0, first)
    assert not store.bind_worker_started("s", first, "old")
    assert store.begin_recovery("s", second, "r") is False


def test_unbound_reservation_cannot_be_completed_or_recovered(tmp_path):
    store = ChatExecutionStore(tmp_path)
    generation = store.reserve("s", 0, 0)
    assert not store.worker_started("s", generation)
    assert not store.worker_finished("s", generation)
    assert store.recover("s", generation, "r", lambda _: pytest.fail("transfer"))["status"] == "stale"


def test_keeps_generations_and_enforces_uncommitted_fence(tmp_path):
    store = ChatExecutionStore(tmp_path)
    assert store.start("s", 1, "sid")
    assert store.worker_started("s", 1)
    assert store.worker_finished("s", 1)
    assert store.start("s", 2, "sid")
    assert store.inspect("s", 1)["run_generation"] == 1
    assert store.inspect("s", 2)["worker"] == "pending"


def test_recovery_requires_completed_worker_and_real_initial_no_tool(tmp_path):
    store = ChatExecutionStore(tmp_path)
    store.start("s", 1, "sid")
    assert not store.worker_finished("s", 1)
    assert store.inspect("s", 1)["worker"] == "pending"
    assert store.inspect("s", 1)["effects"] == "unknown"
    assert store.begin_recovery("s", 1, "r")
    assert not store.start("s", 2, "sid")
    called = []
    assert store.recover("s", 1, "r", lambda proof: called.append(proof)) ["status"] == "blocked"
    assert not called


def test_old_unknown_effects_block_later_completed_no_tool_recovery(tmp_path):
    store = ChatExecutionStore(tmp_path)
    assert store.start('s', 1, 'sid')
    assert store.worker_started('s', 1)
    assert store.observe_tool('s', 1)
    assert store.worker_finished('s', 1)
    assert store.start('s', 2, 'sid')
    assert store.effects_none('s', 2)
    assert store.worker_started('s', 2)
    assert store.worker_finished('s', 2)
    called = []
    result = store.recover('s', 2, 'r', lambda proof: called.append(proof) or True)
    assert result['status'] == 'blocked'
    assert not called


def test_tool_is_permanent_and_effects_none_cannot_reset(tmp_path):
    store = ChatExecutionStore(tmp_path)
    store.start("s", 1, "sid")
    store.worker_started("s", 1)
    assert store.observe_tool("s", 1)
    assert not store.effects_none("s", 1)
    assert store.worker_finished("s", 1)
    assert store.inspect("s", 1)["effects"] == "unknown"


def test_tool_requires_actual_started_marker_and_exact_positive_generation(tmp_path):
    store = ChatExecutionStore(tmp_path)
    store.start("s", 1, "sid")
    assert not store.worker_finished("s", 1)
    assert not store.worker_started("s", 0)
    assert not store.observe_tool("s", 2)
    assert store.inspect("s", 1)["worker"] == "pending"


def test_callback_failure_retains_fence_then_same_id_retry_commits_and_new_turn_allowed(tmp_path):
    store = ChatExecutionStore(tmp_path)
    store.start("s", 5, "sid")
    store.effects_none("s", 5)
    store.worker_started("s", 5)
    store.worker_finished("s", 5)
    with pytest.raises(RuntimeError):
        store.recover("s", 5, "rid", lambda _: (_ for _ in ()).throw(RuntimeError()))
    reopened = ChatExecutionStore(tmp_path)
    assert not reopened.start("s", 6, "sid")
    first = reopened.recover("s", 5, "rid", lambda _: {"ok": True})
    assert first["status"] == "committed"
    assert reopened.recover("s", 5, "rid", lambda _: pytest.fail("duplicate transfer")) == first
    assert reopened.start("s", 6, "sid")


def test_transfer_callback_false_is_not_commit(tmp_path):
    store = ChatExecutionStore(tmp_path)
    store.start("s", 1, "sid")
    store.effects_none("s", 1)
    store.worker_started("s", 1)
    store.worker_finished("s", 1)
    result = store.recover("s", 1, "r", lambda _: False)
    assert result["status"] != "committed"
    assert not store.start("s", 2, "sid")


def test_physical_worker_finish_can_settle_after_recovery_fence(tmp_path):
    store = ChatExecutionStore(tmp_path)
    assert store.start('s', 1, 'sid')
    assert store.effects_none('s', 1)
    assert store.worker_started('s', 1)
    assert store.begin_recovery('s', 1, 'r')
    assert not store.start('s', 2, 'sid')
    assert store.worker_finished('s', 1)
    assert store.recover('s', 1, 'r', lambda _: True)['status'] == 'committed'


def test_different_recovery_id_cannot_take_existing_fence(tmp_path):
    store = ChatExecutionStore(tmp_path)
    store.start("s", 1, "sid")
    assert store.begin_recovery("s", 1, "a")
    assert not store.begin_recovery("s", 1, "b")


def test_recovery_fence_revalidates_after_callback_against_other_process(tmp_path):
    store = ChatExecutionStore(tmp_path)
    store.start("s", 1, "sid")
    store.effects_none("s", 1)
    store.worker_started("s", 1)
    store.worker_finished("s", 1)
    reached = threading.Event()
    resume = threading.Event()
    outcome = []

    def transfer(_):
        reached.set()
        resume.wait(5)
        return True

    thread = threading.Thread(target=lambda: outcome.append(store.recover("s", 1, "r", transfer)))
    thread.start()
    assert reached.wait(5)
    competitor = ChatExecutionStore(tmp_path)
    assert not competitor.observe_tool("s", 1)
    resume.set()
    thread.join(5)
    assert outcome and outcome[0]["status"] == "committed"


def test_independent_process_start_has_one_winner(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    output = ctx.Queue()

    a = ctx.Process(target=_concurrent_start, args=(str(tmp_path), 1, barrier, output))
    b = ctx.Process(target=_concurrent_start, args=(str(tmp_path), 1, barrier, output))
    a.start(); b.start()
    a.join(10); b.join(10)
    assert a.exitcode == b.exitcode == 0
    assert sorted((output.get(timeout=2), output.get(timeout=2))) == [False, True]


def test_wrong_profile_binding_and_corrupt_or_missing_journal_fail_closed(tmp_path):
    home = tmp_path / "owner"
    store = ChatExecutionStore(home)
    assert store.recover_chat_execution("s", 1, tmp_path / "other", "r", lambda _: True)["status"] == "stale"
    store.path.write_bytes(b"not sqlite")
    with pytest.raises(sqlite3.DatabaseError):
        store.inspect("s", 1)
