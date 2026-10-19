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


def test_desktop_turn_uses_real_session_and_tool_observation_before_handoff(tmp_path):
    from agent.chat_execution_context import bind_chat_execution, observe_tool_before_execution
    store = ChatExecutionStore(tmp_path)
    generation = store.start_local_worker("desktop-chat", "actual-desktop-session")
    assert generation == 1
    assert store.start_local_worker("desktop-chat", "competing-session") is None
    with bind_chat_execution(tmp_path, "desktop-chat", generation):
        assert observe_tool_before_execution()
    assert store.worker_finished("desktop-chat", generation)
    receipt = store.retire_for_handoff("desktop-chat", generation, "handoff")
    assert receipt["status"] == "retired"
    assert receipt["session_id"] == "actual-desktop-session"
    assert receipt["effects"] == "unknown"
    assert store.start_local_worker("desktop-chat", "late-session") is None


def test_pending_desktop_submit_blocks_retirement_until_real_worker_settles(tmp_path):
    store = ChatExecutionStore(tmp_path)
    first = store.start_local_worker("desktop-chat", "actual-session-1")
    assert first == 1
    assert store.worker_finished("desktop-chat", first)
    assert store.begin_submit("desktop-chat", "accepted-rpc")
    # Retirement must leave the door open for the already accepted submit to
    # reach native admission, while refusing to claim the chat is quiescent.
    assert store.retire_for_handoff("desktop-chat", first, "handoff")["status"] == "blocked"
    second = store.start_local_worker("desktop-chat", "actual-session-2", ("accepted-rpc",))
    assert second == 2
    assert store.inspect("desktop-chat", second)["session_id"] == "actual-session-2"
    assert store.worker_finished("desktop-chat", second)
    assert store.retire_for_handoff("desktop-chat", first, "handoff")["status"] == "retired"
    assert not store.begin_submit("desktop-chat", "late-rpc")
    assert store.start_local_worker("desktop-chat", "late-session") is None


def test_unresolved_submit_preserves_stronger_recovery_refusal(tmp_path):
    store = ChatExecutionStore(tmp_path)
    assert store.start("desktop-chat", 1, "actual-session")
    assert store.worker_started("desktop-chat", 1)
    assert store.effects_none("desktop-chat", 1)
    assert store.worker_finished("desktop-chat", 1)
    assert store.begin_submit("desktop-chat", "accepted-rpc")
    called = []
    result = store.recover("desktop-chat", 1, "recovery", lambda proof: called.append(proof))
    assert result["status"] == "blocked"
    assert result["effects"] == "unknown"
    assert called == []


def test_rotated_pending_submit_fences_both_keys_until_real_worker_admission(tmp_path):
    store = ChatExecutionStore(tmp_path)
    assert store.start_local_worker("old-key", "actual-old") == 1
    assert store.worker_finished("old-key", 1)
    assert store.begin_submit("old-key", "accepted-rpc")
    assert store.submit_claims_active("new-key", ("accepted-rpc",), source_key="old-key")
    assert store.retire_for_handoff("old-key", 1, "old-handoff")["status"] == "blocked"
    assert store.start_local_worker("new-key", "actual-new", ("accepted-rpc",)) == 1
    assert store.worker_finished("new-key", 1)
    assert store.retire_for_handoff("old-key", 1, "old-handoff")["status"] == "retired"
    assert store.retire_for_handoff("new-key", 1, "new-handoff")["status"] == "retired"


def test_cancelled_rotated_submit_settles_both_key_aliases(tmp_path):
    store = ChatExecutionStore(tmp_path)
    assert store.begin_submit("old-key", "cancelled-rpc")
    assert store.submit_claims_active("new-key", ("cancelled-rpc",), source_key="old-key")
    assert store.abandon_submit("old-key", "cancelled-rpc")
    assert not store.submit_claims_active("new-key", ("cancelled-rpc",), source_key="old-key")


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
