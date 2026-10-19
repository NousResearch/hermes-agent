"""Physical retirement is durable but never asserts that historical tool effects were undone."""
from gateway.chat_execution_recovery import ChatExecutionStore


def completed_tool_turn(home, generation=1):
    store = ChatExecutionStore(home)
    assert store.start("old-chat", generation, f"session-{generation}")
    assert store.worker_started("old-chat", generation)
    assert store.observe_tool("old-chat", generation)
    assert store.worker_finished("old-chat", generation)
    return store


def test_quiescent_tool_user_can_retire_without_fabricating_effects_none(tmp_path):
    store = completed_tool_turn(tmp_path)
    # The stronger existing execution-recovery contract must remain refused.
    recovered = store.recover("old-chat", 1, "recovery", lambda proof: True)
    assert recovered["status"] == "blocked"
    assert recovered["effects"] == "unknown"
    retired = store.retire_for_handoff("old-chat", 1, "recovery")
    assert retired["status"] == "retired"
    assert retired["effects"] == "unknown"
    assert retired["execution_admitted"] is False
    assert not store.start("old-chat", 2, "late-old-session")
    assert ChatExecutionStore(tmp_path).retire_for_handoff("old-chat", 1, "recovery") == retired


def test_retirement_cannot_be_reused_as_execution_recovery(tmp_path):
    store = ChatExecutionStore(tmp_path)
    assert store.start("old-chat", 1, "session-1")
    assert store.worker_started("old-chat", 1)
    assert store.effects_none("old-chat", 1)
    assert store.worker_finished("old-chat", 1)
    assert store.retire_for_handoff("old-chat", 1, "handoff")["status"] == "retired"
    called = []
    result = store.recover("old-chat", 1, "handoff", lambda proof: called.append(proof))
    assert result["status"] != "committed"
    assert called == []
    assert not store.start("old-chat", 2, "late")


def test_running_worker_stays_fenced_until_its_actual_completion(tmp_path):
    store = ChatExecutionStore(tmp_path)
    assert store.start("old-chat", 1, "session-1")
    assert store.worker_started("old-chat", 1)
    assert store.retire_for_handoff("old-chat", 1, "handoff")["status"] == "blocked"
    assert store.reserve("old-chat", 1, 1) is None
    assert store.worker_finished("old-chat", 1)
    assert store.retire_for_handoff("old-chat", 1, "handoff")["status"] == "retired"


def test_claim_generation_does_not_hide_later_running_turns(tmp_path):
    store = completed_tool_turn(tmp_path)
    assert store.start("old-chat", 2, "session-2")
    assert store.worker_started("old-chat", 2)
    assert store.retire_for_handoff("old-chat", 1, "handoff")["status"] == "blocked"
    assert store.worker_finished("old-chat", 2)
    receipt = store.retire_for_handoff("old-chat", 1, "handoff")
    assert receipt["status"] == "retired"
    assert receipt["run_generation"] == 2


def test_unfinished_detached_resource_prevents_retirement(tmp_path):
    store = ChatExecutionStore(tmp_path)
    assert store.start("old-chat", 1, "session-1")
    assert store.worker_started("old-chat", 1)
    assert store.reserve_concurrent_tool_resource("old-chat", 1, "child", "async_delegation_thread")
    assert store.start_concurrent_tool_resource("old-chat", 1, "child", "session-1", "async_delegation_thread")
    assert store.worker_finished("old-chat", 1)
    assert store.retire_for_handoff("old-chat", 1, "handoff")["status"] == "blocked"
    assert store.finish_concurrent_tool_resource("old-chat", 1, "child", "session-1", "async_delegation_thread")
    assert store.retire_for_handoff("old-chat", 1, "handoff")["status"] == "retired"


def test_missing_claim_and_competing_request_fail_closed(tmp_path):
    store = completed_tool_turn(tmp_path)
    assert store.retire_for_handoff("old-chat", 9, "handoff")["status"] == "blocked"
    assert store.retire_for_handoff("old-chat", 1, "handoff")["status"] == "retired"
    assert store.retire_for_handoff("old-chat", 1, "competitor")["status"] == "stale"
