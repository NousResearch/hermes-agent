"""Unit tests for agent.turn_facade_lease (admission + lease bracket)."""
import threading
from types import SimpleNamespace

import pytest

from agent.turn_facade_lease import (
    LEASE_TTL_SECONDS,
    DurableTurnLease,
    admit_durable_turn_lease,
)


class _Db:
    def __init__(self, exists=True, acquired=True):
        self.exists = exists
        self.acquired = acquired
        self.events = []

    def get_session(self, session_id):
        return {"id": session_id} if self.exists else None

    def acquire_session_turn_lease(self, session_id, holder, **kwargs):
        self.events.append(("acquire", session_id, holder))
        return self.acquired

    def refresh_session_turn_lease(self, session_id, holder, **kwargs):
        return True

    def release_session_turn_lease(self, session_id, holder):
        self.events.append(("release", session_id, holder))


def _agent(db, **overrides):
    agent = SimpleNamespace(
        _session_db=db,
        session_id="s1",
        _persist_disabled=False,
        _interrupt_requested=False,
        _interrupt_message=None,
        _execution_thread_id=None,
        _session_turn_lease_refresh_interval=60.0,
        statuses=[],
    )
    agent._emit_status = agent.statuses.append
    agent._emit_warning = agent.statuses.append
    agent._touch_activity = lambda *a, **k: None
    agent._liveness_activity_lock = lambda: threading.Lock()
    for k, v in overrides.items():
        setattr(agent, k, v)
    return agent


def _admit(agent, history=None):
    return admit_durable_turn_lease(
        agent,
        session_id="s1",
        relay_turn_id="s1:t:abcd",
        task_context={"session_id": "s1", "task_id": "t", "platform": getattr(agent, "platform", "cli")},
        conversation_history=history,
    )


def test_no_lease_without_durable_row_or_when_persist_disabled():
    seed = [{"role": "user", "content": "hi"}]
    admission = _admit(_agent(_Db(exists=False)), seed)
    assert admission.lease is None and admission.early_result is None
    assert admission.conversation_history is seed

    db = _Db()
    admission = _admit(_agent(db, _persist_disabled=True), seed)
    assert admission.lease is None and db.events == []


def test_admission_sets_holder_attrs_and_release_clears_them(monkeypatch):
    monkeypatch.setattr(
        "agent.turn_liveness.resolve_turn_liveness_settings", lambda cfg: (None, 1.0)
    )
    db = _Db()
    agent = _agent(db)
    admission = _admit(agent)
    lease = admission.lease
    assert isinstance(lease, DurableTurnLease)
    assert agent._session_db_created is True
    assert agent._active_session_turn_lease_holder == lease.holder
    assert agent._active_session_turn_lease_ttl_seconds == LEASE_TTL_SECONDS
    assert lease.holder.startswith("pid=") and ":platform=cli" in lease.holder
    assert lease.watchdog is None and lease.timer_handles == []
    assert lease.is_turn_active() is False

    lease.stop_refresher()
    lease.join_threads()
    lease.clear_interrupt()
    lease.release()
    assert db.events == [("acquire", "s1", lease.holder), ("release", "s1", lease.holder)]
    assert agent._active_session_turn_lease_holder is None
    assert agent._active_session_turn_lease_ttl_seconds is None


def test_timeout_and_interrupt_early_results():
    agent = _agent(_Db(acquired=False))
    admission = _admit(agent, [{"role": "user", "content": "x"}])
    assert admission.lease is None
    assert admission.early_result["failed"] is True
    assert admission.early_result["error"] == "session_turn_lease_timeout:s1"
    # Stamped so UI descriptors show "session busy" instead of code="unknown".
    assert admission.early_result["failure_reason"] == "session_busy"
    assert admission.early_result["failure_retryable"] is True
    assert admission.early_result["messages"] == [{"role": "user", "content": "x"}]

    agent = _agent(_Db(acquired=False), _interrupt_requested=True, _interrupt_message="stop")
    agent.clear_interrupt = lambda: None
    admission = _admit(agent)
    assert admission.early_result["interrupted"] is True
    assert admission.early_result["interrupt_message"] == "stop"


def test_interrupt_turn_only_while_active():
    agent = _agent(_Db())
    calls = []
    agent.interrupt = lambda msg, **kw: calls.append(msg)
    lease = DurableTurnLease(agent, agent._session_db, "s1", "h")
    lease._interrupt_turn("lost")  # inactive: ignored
    assert calls == [] and lease.interrupt_message is None
    lease.turn_active = True
    lease._interrupt_turn("lost")
    assert calls == ["lost"] and lease.interrupt_message == "lost"
    lease.deactivate_after_liveness_abort()
    assert lease.stop.is_set() and lease.is_turn_active() is False


def test_immediate_durable_admission_hydrates_other_process_tail_before_user_staging(tmp_path, monkeypatch):
    """A finished gateway turn must be visible to a cached Desktop agent even without contention."""
    import copy
    import os
    from hermes_state import SessionDB
    from gateway.internal_events import create_gateway_system_event, gateway_system_event_message
    from tests.agent.test_turn_context import _FakeAgent, _build

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.turn_liveness.resolve_turn_liveness_settings", lambda cfg: (None, 1.0))
    path = tmp_path / "state.db"
    desktop_db, gateway_db = SessionDB(path), SessionDB(path)
    sid = "shared-physical"
    desktop_db.create_session(sid, source="telegram", system_prompt="SYSTEM")
    original = [{"role": "user", "content": "original parent request"},
                {"role": "assistant", "content": "parent completed coordination"}]
    desktop_db.append_messages_batch(sid, original)
    stale = desktop_db.get_messages_as_conversation(sid, repair_alternation=True, include_row_ids=True)
    before = copy.deepcopy(stale)
    content, marker = create_gateway_system_event(
        content="host-authored completion metadata", session_key="telegram-route",
        expected_session_id=sid, event_id="fixture-delivery", event_kind="external_tool_completed",
        plugin_id="fixture", expected_route={"profile_name":"default", "platform":"telegram", "user_id":"42", "chat_id":"42", "topic_id":"77"},
        eligibility_check=lambda: True)
    external = [gateway_system_event_message(marker, content),
        {"role":"assistant", "content":"", "tool_calls":[{"id":"exact-read", "type":"function", "function":{"name":"external_worker_read", "arguments":"{}"}}]},
        {"role":"tool", "tool_call_id":"exact-read", "tool_name":"external_worker_read", "content":"exact completion evidence"},
        {"role":"assistant", "content":"verified completion report"}]
    gateway_holder = f"pid={os.getpid()}:turn=gateway:platform=telegram"
    assert gateway_db.try_acquire_session_turn_lease(sid, gateway_holder)
    assert gateway_db.append_messages_batch(sid, external, turn_lease_holder=gateway_holder) == len(external)
    gateway_db.release_session_turn_lease(sid, gateway_holder)
    durable = desktop_db.get_messages_as_conversation(sid, repair_alternation=True, include_row_ids=True)
    # Prove the difference is real omitted durable rows, not role/projection filtering.
    assert [row['role'] for row in durable] == ['user','assistant','developer','assistant','tool','assistant']
    assert durable[-1]['content'] == 'verified completion report'
    assert len(stale) == 2 and len(durable) == 6
    agent = _agent(desktop_db, session_id=sid, _cached_system_prompt="SYSTEM")
    context = {"session_id":sid, "task_id":"desktop-followup", "platform":"desktop"}
    admission = admit_durable_turn_lease(agent, session_id=sid, relay_turn_id="desktop-next",
        task_context=context, conversation_history=stale)
    try:
        assert admission.lease is not None and admission.early_result is None
        assert not any('waiting' in status.lower() for status in agent.statuses)
        assert stale == before and agent._cached_system_prompt == "SYSTEM" and agent.session_id == sid
        assert admission.conversation_history == durable
        # Exercise real user staging after native admission, not a test-only list append.
        staged_agent = _FakeAgent()
        staged_agent.session_id = sid
        built = _build(staged_agent, user_message="next genuine user request", conversation_history=admission.conversation_history)
        assert [row['role'] for row in built.messages] == ['user','assistant','developer','assistant','tool','assistant','user']
        assert sum(row.get('content') == 'exact completion evidence' for row in built.messages) == 1
        assert sum(row.get('content') == 'next genuine user request' for row in built.messages) == 1
        assert built.messages[:2] == before
    finally:
        if admission.lease is not None:
            admission.lease.release()
        desktop_db.close()
        gateway_db.close()


@pytest.fixture
def durable_history(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.turn_liveness.resolve_turn_liveness_settings", lambda cfg: (None, 1.0))
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="telegram", system_prompt="unchanged system")
    db.append_messages_batch("s1", [
        {"role": "user", "content": "original"},
        {"role": "assistant", "content": "original answer"},
    ])
    agent = _agent(db, platform="desktop", _cached_system_prompt="unchanged system")
    history = db.get_messages_as_conversation("s1", repair_alternation=True, include_row_ids=True)
    yield db, agent, history
    db.close()


def _foreign_turn(db):
    db.append_messages_batch("s1", [
        {"role": "developer", "content": "external completion metadata"},
        {"role": "assistant", "content": "external result"},
    ])


def _assert_released(db, agent):
    assert agent._active_session_turn_lease_holder is None
    assert agent._active_session_turn_lease_ttl_seconds is None
    assert db.try_acquire_session_turn_lease("s1", "test-successor")
    db.release_session_turn_lease("s1", "test-successor")


def test_history_gap_before_later_local_append_preserves_cached_objects(durable_history):
    import copy
    from agent.session_persistence import _db_flush_write

    db, agent, history = durable_history
    history[0]["content"] = [{"type": "text", "text": "original"},
                             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
    history[1]["api_only_field"] = {"opaque": [1, 2]}
    _foreign_turn(db)
    local = [{"role": "user", "content": "later desktop request"},
             {"role": "assistant", "content": "later desktop answer"}]
    # Real flush synchronizes committed row IDs onto the original live dicts.
    _db_flush_write(agent, copy.deepcopy(local), local, [*history, *local])
    history += local
    before = copy.deepcopy(history)
    admission = _admit(agent, history)
    try:
        hydrated = admission.conversation_history
        assert len(hydrated) == 6
        assert hydrated[:2] == history[:2] and hydrated[4:] == local
        assert all(hydrated[i] is history[j] for i, j in [(0, 0), (1, 1), (4, 2), (5, 3)])
        assert hydrated[2]["role"] == "developer"
        assert history == before and agent._cached_system_prompt == "unchanged system"
    finally:
        admission.lease.release()


def test_unchanged_history_preserves_list_and_objects_with_api_only_fields(durable_history):
    db, agent, history = durable_history
    history[0]["content"] = [{"type": "text", "text": "original"}]
    history[1]["api_only_field"] = object()
    admission = _admit(agent, history)
    try:
        assert admission.conversation_history is history
    finally:
        admission.lease.release()


def test_unpersisted_prefix_suffix_and_current_user_remain_once(durable_history):
    from tests.agent.test_turn_context import _FakeAgent, _build

    db, agent, history = durable_history
    prefix = {"role": "developer", "content": "caller-only prefix"}
    suffix = {"role": "developer", "content": "caller-only suffix"}
    history = [prefix, *history, suffix]
    _foreign_turn(db)
    admission = _admit(agent, history)
    try:
        hydrated = admission.conversation_history
        assert hydrated[0] is prefix and hydrated[-1] is suffix
        staged = _build(_FakeAgent(), user_message="current human", conversation_history=hydrated)
        for content in ("caller-only prefix", "caller-only suffix", "current human", "external result"):
            assert sum(row.get("content") == content for row in staged.messages) == 1
    finally:
        admission.lease.release()


@pytest.mark.parametrize("changed", [False, True])
def test_native_gateway_marker_only_projection(durable_history, changed):
    from gateway.session_transcript import SessionTranscriptMixin

    db, agent, _ = durable_history
    store = SimpleNamespace(_db_for_session_id=lambda sid: db, _follow_reroutes=lambda sid: sid)
    history = SessionTranscriptMixin.load_transcript(store, "s1")
    assert all(row.get("_db_persisted") and "_row_id" not in row for row in history)
    from gateway.run import _build_gateway_agent_history
    history, _ = _build_gateway_agent_history(history)
    agent.platform = "telegram"
    if changed:
        _foreign_turn(db)
    admission = _admit(agent, history)
    assert admission.conversation_history is history
    admission.lease.release()


@pytest.mark.parametrize("mutation", ["foreign", "reordered", "duplicate", "rewound", "interleaved"])
def test_ambiguous_history_fails_closed_and_releases_lease(durable_history, mutation):
    db, agent, history = durable_history
    if mutation == "foreign":
        db.create_session("other", source="desktop")
        db.append_messages_batch("other", [{"role": "user", "content": "same text is not identity"}])
        history[0] = db.get_messages_as_conversation("other", include_row_ids=True)[0]
    elif mutation == "reordered":
        history.reverse()
    elif mutation == "duplicate":
        history.append(history[0])
    elif mutation == "rewound":
        db.rewind_to_message("s1", history[0]["_row_id"])
    elif mutation == "interleaved":
        history.insert(1, {"role": "developer", "content": "caller-only interior"})
        _foreign_turn(db)
    with pytest.raises(ValueError, match="session_history_conflict"):
        _admit(agent, history)
    _assert_released(db, agent)
    assert agent.session_id == "s1" and agent._cached_system_prompt == "unchanged system"


def test_replay_filtered_orphan_does_not_reenter_history(durable_history):
    db, agent, history = durable_history
    db.append_messages_batch("s1", [{"role": "tool", "tool_call_id": "missing", "content": "orphan"}])
    projected = db.get_messages_as_conversation("s1", repair_alternation=True, include_row_ids=True)
    assert projected == history
    admission = _admit(agent, history)
    assert admission.conversation_history is history
    admission.lease.release()


def test_replay_merge_into_existing_anchor_fails_closed(durable_history):
    db, agent, history = durable_history
    db.append_messages_batch("s1", [{"role": "assistant", "content": "foreign merged tail"}])
    projected = db.get_messages_as_conversation("s1", repair_alternation=True, include_row_ids=True)
    assert [row["_row_id"] for row in projected] == [row["_row_id"] for row in history]
    assert projected[-1]["content"] != history[-1]["content"]
    with pytest.raises(ValueError, match="native replay repair changed anchors"):
        _admit(agent, history)
    _assert_released(db, agent)


def test_projection_read_failure_releases_before_model(durable_history, monkeypatch):
    db, agent, history = durable_history
    def fail(*args, **kwargs):
        raise OSError("fixture read failure")
    monkeypatch.setattr(db, "get_messages_as_conversation", fail)
    with pytest.raises(OSError, match="fixture read failure"):
        _admit(agent, history)
    _assert_released(db, agent)


@pytest.mark.parametrize("waited", [False, True])
def test_compression_tip_follows_native_policy_under_same_lease(durable_history, monkeypatch, waited):
    db, agent, history = durable_history
    db.end_session("s1", "compression")
    db.create_session("tip", source="telegram", parent_session_id="s1")
    db.append_messages_batch("tip", [{"role": "user", "content": "native compressed context"}])
    if waited:
        acquire = db.acquire_session_turn_lease
        def acquire_after_wait(*args, **kwargs):
            kwargs["on_wait"](0.0)
            return acquire(*args, **kwargs)
        monkeypatch.setattr(db, "acquire_session_turn_lease", acquire_after_wait)
    admission = _admit(agent, history)
    try:
        assert agent.session_id == "tip"
        assert admission.conversation_history == db.get_messages_as_conversation("tip", repair_alternation=True, include_row_ids=True)
        assert agent._cached_system_prompt == "unchanged system"
        assert admission.lease.session_id == "s1"
    finally:
        admission.lease.release()
    _assert_released(db, agent)


def test_explicit_branch_is_not_adopted(durable_history):
    db, agent, history = durable_history
    db.create_session("branch", source="desktop", parent_session_id="s1", model_config={"_branched_from": "s1"})
    db.append_messages_batch("branch", [{"role": "user", "content": "another conversation"}])
    admission = _admit(agent, history)
    assert agent.session_id == "s1" and admission.conversation_history is history
    admission.lease.release()


def test_resume_outside_lease_fails_closed(durable_history):
    db, agent, history = durable_history
    db.create_session("continuation", source="desktop", parent_session_id="s1")
    db.append_messages_batch("continuation", [{"role": "user", "content": "unleased continuation"}])
    with pytest.raises(ValueError, match="outside acquired lease"):
        _admit(agent, history)
    _assert_released(db, agent)
    assert agent.session_id == "s1"


def test_native_desktop_resume_then_normal_tool_flush_remains_anchored(durable_history):
    from agent.replay_cleanup import sanitize_replay_history
    from tests.agent.test_cross_process_turn_lease import _flush_agent

    db, agent, _ = durable_history
    history, _ = db.get_resume_conversations("s1")
    history = sanitize_replay_history(history)
    first = _admit(agent, history)
    assert first.conversation_history is history
    first.lease.release()
    messages = [*history,
        {"role": "user", "content": "ordinary desktop followup"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "native-read", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "native-read", "tool_name": "read_file", "content": "native evidence"},
        {"role": "assistant", "content": "ordinary answer", "api_only_field": "retained"},
    ]
    flush_agent = _flush_agent(db, "s1")
    assert flush_agent._flush_messages_to_session_db(messages, history)
    assert all(row.get("_db_persisted") is True and type(row.get("_row_id")) is int for row in messages)
    admission = _admit(agent, messages)
    assert admission.conversation_history is messages
    admission.lease.release()
    _foreign_turn(db)
    admission = _admit(agent, messages)
    try:
        assert len(admission.conversation_history) == len(messages) + 2
        assert all(admission.conversation_history[i] is row for i, row in enumerate(messages))
        assert admission.conversation_history[-1]["content"] == "external result"
    finally:
        admission.lease.release()


def test_caller_seed_marked_persisted_without_row_id_is_not_inferred(durable_history):
    from tests.agent.test_cross_process_turn_lease import _flush_agent

    db, agent, history = durable_history
    seed = {"role": "developer", "content": "caller-only seed"}
    seeded_history = [seed, *history]
    messages = [*seeded_history, {"role": "user", "content": "next human"},
                {"role": "assistant", "content": "next answer"}]
    assert _flush_agent(db, "s1")._flush_messages_to_session_db(messages, seeded_history)
    # Native collect marks seeds as persisted without ever inserting them. That marker is not
    # row identity; accepting it as a prefix would also accept a missing/foreign durable row.
    assert seed.get("_db_persisted") is True and "_row_id" not in seed
    admission = _admit(agent, messages)
    assert admission.conversation_history is messages
    admission.lease.release()
    _assert_released(db, agent)


def test_unchanged_historical_repair_does_not_block_unrelated_foreign_tail(durable_history):
    db, agent, _ = durable_history
    db.append_messages_batch("s1", [{"role": "assistant", "content": "continued original answer"}])
    history, _ = db.get_resume_conversations("s1")
    assert len(history) == 2  # native replay already merged the historical assistant pair
    _foreign_turn(db)
    admission = _admit(agent, history)
    try:
        assert len(admission.conversation_history) == 4
        assert admission.conversation_history[0] is history[0]
        assert admission.conversation_history[1] is history[1]
    finally:
        admission.lease.release()


@pytest.mark.parametrize("platform", ["api", "desktop"])
def test_client_owned_second_request_keeps_supplied_history(durable_history, platform):
    db, agent, _ = durable_history
    agent.platform = platform
    caller_history = [{"role": "user", "content": "original"},
                      {"role": "assistant", "content": "client owns the response"}]
    admission = _admit(agent, caller_history)
    assert admission.conversation_history is caller_history
    admission.lease.release()


def _interrupted_block(name):
    return [{"role": "assistant", "content": "", "finish_reason": "incomplete", "tool_calls": [
        {"id": "interrupted-call", "type": "function", "function": {"name": name, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "interrupted-call", "tool_name": name,
         "content": "[command interrupted]", "api_content": "unsafe stale sidecar"}]


@pytest.mark.parametrize("name", ["read_file", "execute_code"])
def test_native_desktop_replay_cleanup_survives_foreign_tail(durable_history, name):
    from agent.replay_cleanup import sanitize_replay_history
    db, agent, _ = durable_history
    db.append_messages_batch("s1", _interrupted_block(name))
    history, _ = db.get_resume_conversations("s1")
    history = sanitize_replay_history(history)
    _foreign_turn(db)
    admission = _admit(agent, history)
    try:
        result = admission.conversation_history
        assert sum(row.get("content") == "external result" for row in result) == 1
        if name == "read_file":
            assert not any(row.get("tool_call_id") == "interrupted-call" or row.get("tool_calls") for row in result)
        else:
            notice = next(row for row in result if row.get("tool_call_id") == "interrupted-call")
            assert notice["effect_disposition"] == "unknown" and "UNKNOWN" in notice["content"]
            assert "api_content" not in notice
    finally:
        admission.lease.release()


def test_cached_raw_interrupted_result_cannot_override_native_unknown_rewrite(durable_history):
    db, agent, _ = durable_history
    db.append_messages_batch("s1", _interrupted_block("execute_code"))
    history, _ = db.get_resume_conversations("s1")
    original_tool = history[-1]
    admission = _admit(agent, history)
    try:
        notice = admission.conversation_history[-1]
        assert notice is not original_tool
        assert "UNKNOWN" in notice["content"] and "api_content" not in notice
        assert original_tool["content"] == "[command interrupted]"
    finally:
        admission.lease.release()


@pytest.mark.parametrize("foreign_tail", [False, True])
def test_synthetic_unknown_recovery_notice_keeps_anchor_position(durable_history, foreign_tail):
    from agent.replay_cleanup import sanitize_replay_history
    db, agent, _ = durable_history
    db.append_messages_batch("s1", _interrupted_block("execute_code")[:1])
    history, _ = db.get_resume_conversations("s1")
    history = sanitize_replay_history(history)
    assert history[-1]["effect_disposition"] == "unknown" and "_row_id" not in history[-1]
    if foreign_tail:
        _foreign_turn(db)
    admission = _admit(agent, history)
    try:
        result = admission.conversation_history
        notice_indexes = [i for i, row in enumerate(result) if row.get("effect_disposition") == "unknown"]
        assert len(notice_indexes) == 1
        i = notice_indexes[0]
        assert result[i - 1]["tool_calls"][0]["id"] == result[i]["tool_call_id"]
        if foreign_tail:
            assert result[i + 1]["role"] == "developer"
    finally:
        admission.lease.release()


@pytest.mark.parametrize("platform", ["telegram", "api", "cli", "unknown"])
def test_non_desktop_never_enters_new_reconciliation(durable_history, monkeypatch, platform):
    db, agent, history = durable_history
    agent.platform = platform
    _foreign_turn(db)
    def unexpected(*args, **kwargs):
        pytest.fail("non-Desktop admission must not invoke new projection")
    monkeypatch.setattr(db, "get_resume_conversations", unexpected)
    admission = _admit(agent, history)
    assert admission.conversation_history is history
    admission.lease.release()


def test_previously_expired_confirmation_cannot_regrow_from_db(durable_history):
    from agent.replay_cleanup import strip_stale_dangerous_confirmations
    db, agent, _ = durable_history
    db.append_messages_batch("s1", [{"role": "user", "content": "confirm forced restart",
                                    "timestamp": 1.0, "api_content": "confirm forced restart"},
                                   {"role": "assistant", "content": "old confirmation reply"}])
    history, _ = db.get_resume_conversations("s1")
    history = strip_stale_dangerous_confirmations(history, now=100.0)
    redacted = history[-2]
    assert "EXPIRED" in redacted["content"] and "api_content" not in redacted
    _foreign_turn(db)
    admission = _admit(agent, history)
    try:
        assert admission.conversation_history[2] is redacted
        assert "EXPIRED" in admission.conversation_history[2]["content"]
        assert "api_content" not in admission.conversation_history[2]
    finally:
        admission.lease.release()


def test_immediate_compression_tip_uses_desktop_sanitized_projection(durable_history):
    db, agent, history = durable_history
    db.end_session("s1", "compression")
    db.create_session("tip", source="telegram", parent_session_id="s1")
    db.append_messages_batch("tip", [{"role": "user", "content": "compressed user context"},
                                    *_interrupted_block("execute_code")])
    admission = _admit(agent, history)
    try:
        assert agent.session_id == "tip"
        notice = admission.conversation_history[-1]
        assert notice["effect_disposition"] == "unknown" and "UNKNOWN" in notice["content"]
        assert "api_content" not in notice
    finally:
        admission.lease.release()


def test_changed_synthetic_recovery_sidecar_is_not_replayed_as_caller_suffix(durable_history):
    from agent.replay_cleanup import sanitize_replay_history
    db, agent, _ = durable_history
    db.append_messages_batch("s1", _interrupted_block("execute_code")[:1])
    history, _ = db.get_resume_conversations("s1")
    history = sanitize_replay_history(history)
    history[-1]["api_content"] = "stale unsafe recovery sidecar"
    _foreign_turn(db)
    with pytest.raises(ValueError, match="ambiguous recovery result"):
        _admit(agent, history)
    _assert_released(db, agent)


def test_compression_adoption_removes_old_synthetic_notice_and_preserves_caller_suffix(durable_history):
    from agent.replay_cleanup import sanitize_replay_history
    db, agent, _ = durable_history
    db.append_messages_batch("s1", _interrupted_block("execute_code")[:1])
    history, _ = db.get_resume_conversations("s1")
    history = sanitize_replay_history(history)
    old_notice = history[-1]
    assert old_notice["effect_disposition"] == "unknown" and "_row_id" not in old_notice
    caller_suffix = {"role": "developer", "content": "caller-owned next-turn context"}
    history.append(caller_suffix)
    db.end_session("s1", "compression")
    db.create_session("tip", source="telegram", parent_session_id="s1")
    db.append_messages_batch("tip", [{"role": "user", "content": "compressed user context"},
                                    {"role": "assistant", "content": "compressed answer"}])
    admission = _admit(agent, history)
    try:
        result = admission.conversation_history
        assert agent.session_id == "tip"
        assert [row["role"] for row in result] == ["user", "assistant", "developer"]
        assert result[-1] is caller_suffix
        assert sum(row is caller_suffix for row in result) == 1
        assert not any(row.get("tool_call_id") == "interrupted-call" for row in result)
        assert old_notice in history and old_notice["effect_disposition"] == "unknown"
    finally:
        admission.lease.release()
    _assert_released(db, agent)


@pytest.mark.parametrize("platform,cached_history", [("telegram", "empty"), ("telegram", "stale"), ("desktop", "stale")])
def test_typed_replay_is_rejected_under_durable_lease_before_loop(
    durable_history, monkeypatch, platform, cached_history,
):
    from unittest.mock import Mock
    from run_agent import AIAgent
    from gateway.internal_events import create_gateway_system_event, gateway_system_event_message
    db, partial_agent, stale = durable_history
    content, event = create_gateway_system_event(
        content="completed external work", session_key="route", expected_session_id="s1",
        event_id="already-admitted", event_kind="external_tool_completed", plugin_id="fixture",
        expected_route={"profile_name": "default", "platform": "telegram", "user_id": "42", "chat_id": "42", "topic_id": ""},
        eligibility_check=lambda: True,
    )
    db.append_messages_batch("s1", [gateway_system_event_message(event, content),
                                   {"role": "assistant", "content": "completed"}])
    before = db.get_messages_as_conversation("s1", repair_alternation=False, include_row_ids=True)
    # A fresh agent has no gateway receipt cache (restart/eviction); Desktop's
    # supplied cache was captured before the other owner committed this event.
    agent = AIAgent.__new__(AIAgent)
    agent.__dict__.update(vars(partial_agent))
    agent.platform = platform
    model_loop = Mock(return_value={"completed": True, "messages": []})
    monkeypatch.setattr("agent.conversation_loop.run_conversation", model_loop)
    with pytest.raises(ValueError, match="already admitted"):
        agent.run_conversation(content, conversation_history=[] if cached_history == "empty" else stale,
                               gateway_system_event=event)
    model_loop.assert_not_called()
    assert db.get_messages_as_conversation("s1", repair_alternation=False, include_row_ids=True) == before
    _assert_released(db, agent)
