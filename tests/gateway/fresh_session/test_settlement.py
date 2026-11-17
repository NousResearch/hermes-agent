"""Real session-store routing with a synthetic Telegram adapter, without a bot or model API."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.session import AsyncSessionStore, SessionEntry, SessionStore
from gateway.fresh_session.settlement import settle
from tests.gateway.fresh_session.test_publication import topic
from tests.gateway.test_telegram_topic_mode import _make_runner

pytestmark = pytest.mark.asyncio


@pytest.fixture
def lane(request):
    topic_data = request.getfixturevalue("topic")
    topic_data.db.release_session_turn_lease("parent", "turn-owner")
    runner = _make_runner(session_db=topic_data.db)
    store = SessionStore(topic_data.home / "sessions", runner.config)
    store._db = topic_data.db
    entry = SessionEntry(session_key=topic_data.key, session_id="parent", created_at=datetime.now(),
                         updated_at=datetime.now(), origin=topic_data.source)
    store._entries[topic_data.key] = entry
    store._save()
    adapter = SimpleNamespace(_pending_messages={}, send=AsyncMock())
    state = SimpleNamespace(conversation=SimpleNamespace(queued_events=[]), turn=SimpleNamespace(agent=None),
                            persistent=SimpleNamespace(approvals=None, update_prompt_pending=False))
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._delivery_adapter_for = lambda source: adapter
    runner._is_session_run_current = lambda key, generation: generation == 1
    runner._peek_session_state = lambda key: state
    runner._agent_has_active_subagents = lambda agent: False
    runner._evict_cached_agent = Mock()
    from gateway.run import GatewayRunner
    runner._clear_session_boundary_security_state = GatewayRunner._clear_session_boundary_security_state.__get__(runner)
    result = {"completed": True, "final_response": "Task checkpoint complete.", "already_sent": True,
              "fresh_topic_handoff": {"session_id": "parent", "summary": "Next: verify the files."}}
    return SimpleNamespace(**vars(topic_data), runner=runner, store=store, adapter=adapter,
                           state=state, entry=entry, result=result)


async def test_successor_handles_same_topic_next_message_and_one_note(lane):
    assert await settle(lane.runner, lane.source, lane.entry, lane.key, 1, lane.result)
    successor = lane.store.peek_session_id(lane.key)
    assert successor != "parent"
    next_entry = lane.store.get_or_create_session(lane.source)
    assert next_entry.session_id == successor
    healed = await lane.runner._hmwa_heal_telegram_topic_binding(lane.source, next_entry, lane.key)
    assert healed.session_id == successor
    history = lane.store.load_transcript(successor)
    assert len(history) == 1
    assert "Next: verify the files." in history[0]["content"]
    assert lane.db.get_session("parent")["end_reason"] == "session_handoff"
    lane.adapter.send.assert_awaited_once()
    assert lane.adapter.send.call_args.kwargs["metadata"]["thread_id"] == lane.source.thread_id
    assert not await settle(lane.runner, lane.source, lane.entry, lane.key, 1, lane.result)
    lane.adapter.send.assert_awaited_once()


@pytest.mark.parametrize("blocker", ["queued", "overflow_queue", "subagent", "stale", "new", "compression"])
async def test_unsafe_settlement_leaves_existing_route_intact(lane, blocker):
    if blocker == "queued":
        lane.adapter._pending_messages[lane.key] = "new input"
    if blocker == "overflow_queue":
        lane.state.conversation.queued_events.append("queued input")
    if blocker == "subagent":
        lane.runner._agent_has_active_subagents = lambda agent: True
    if blocker == "new":
        lane.store.reset_session(lane.key)
    if blocker == "compression":
        assert lane.db.try_acquire_compression_lock("parent", "hygiene")
    original = lane.store.peek_session_id(lane.key)
    assert not await settle(lane.runner, lane.source, lane.entry, lane.key,
                            2 if blocker == "stale" else 1, lane.result)
    assert lane.store.peek_session_id(lane.key) == original
    lane.adapter.send.assert_not_awaited()


async def test_crash_after_database_commit_recovers_from_topic_binding(lane, monkeypatch):
    switch = lane.store.switch_session
    monkeypatch.setattr(lane.store, "switch_session", Mock(side_effect=OSError("simulated index-save failure")))
    assert not await settle(lane.runner, lane.source, lane.entry, lane.key, 1, lane.result)
    binding = lane.db.get_telegram_topic_binding(chat_id=lane.source.chat_id, thread_id=lane.source.thread_id,
                                                profile_name=lane.source.profile)
    assert binding["session_id"] != "parent"
    monkeypatch.setattr(lane.store, "switch_session", switch)
    healed = await lane.runner._hmwa_heal_telegram_topic_binding(lane.source, lane.entry, lane.key)
    assert healed.session_id == binding["session_id"]
    assert "Next: verify the files." in lane.store.load_transcript(healed.session_id)[0]["content"]


async def test_failed_turn_discards_handoff_even_with_queued_input(lane):
    lane.result["failed"] = True
    lane.adapter._pending_messages[lane.key] = "keep this input"
    assert not await settle(lane.runner, lane.source, lane.entry, lane.key, 1, lane.result)
    assert lane.store.peek_session_id(lane.key) == "parent"
    assert lane.adapter._pending_messages[lane.key] == "keep this input"
    lane.adapter.send.assert_not_awaited()


async def test_group_topic_does_not_adopt_another_users_isolated_lane(lane):
    other = SessionEntry(session_key="another-user-key", session_id="another-user",
                         created_at=datetime.now(), updated_at=datetime.now(), origin=lane.source)
    healed = await lane.runner._hmwa_heal_telegram_topic_binding(lane.source, other, other.session_key)
    assert healed is other


async def test_new_between_commit_and_route_adoption_wins(lane, monkeypatch):
    switch = lane.store.switch_session
    published = []

    def concurrent_new(key, child, **kwargs):
        published.append(child)
        new = lane.store.reset_session(key)
        lane.db.bind_telegram_topic(chat_id=lane.source.chat_id, thread_id=lane.source.thread_id,
                                   user_id=lane.source.user_id, profile_name=lane.source.profile,
                                   session_key=key, session_id=new.session_id)
        return switch(key, child, **kwargs)

    monkeypatch.setattr(lane.store, "switch_session", concurrent_new)
    assert not await settle(lane.runner, lane.source, lane.entry, lane.key, 1, lane.result)
    assert len(published) == 1
    child = lane.db.get_session(published[0])
    assert child["end_reason"] == "superseded_by_new"
    assert lane.store.peek_session_id(lane.key) != published[0]
    binding = lane.db.get_telegram_topic_binding(chat_id=lane.source.chat_id, thread_id=lane.source.thread_id,
                                                profile_name=lane.source.profile)
    assert binding["session_id"] == lane.store.peek_session_id(lane.key)
    lane.adapter.send.assert_not_awaited()


def _grant_session(lane):
    from tools import approval
    approval.approve_session(lane.key, 'synthetic-dangerous-command')
    approval.enable_session_yolo(lane.key)
    lane.store.set_session_yolo(lane.key, True)
    return approval


async def test_handoff_clears_session_grants_and_yolo(lane):
    approval = _grant_session(lane)
    assert approval.is_approved(lane.key, 'synthetic-dangerous-command')
    assert await settle(lane.runner, lane.source, lane.entry, lane.key, 1, lane.result)
    assert not approval.is_approved(lane.key, 'synthetic-dangerous-command')
    assert not approval.is_session_yolo_enabled(lane.key)
    assert lane.store.lookup_by_session_key(lane.key).yolo is None


async def test_restart_adoption_clears_stale_json_yolo_and_grants(lane, monkeypatch):
    approval = _grant_session(lane)
    switch = lane.store.switch_session
    monkeypatch.setattr(lane.store, 'switch_session', Mock(side_effect=OSError('index publication interrupted')))
    assert not await settle(lane.runner, lane.source, lane.entry, lane.key, 1, lane.result)
    monkeypatch.setattr(lane.store, 'switch_session', switch)
    restored = await lane.runner._hmwa_heal_telegram_topic_binding(lane.source, lane.entry, lane.key)
    assert restored.session_id != 'parent'
    assert restored.yolo is None
    assert not approval.is_approved(lane.key, 'synthetic-dangerous-command')
    assert not approval.is_session_yolo_enabled(lane.key)


async def test_new_database_and_binding_commit_before_json_adoption(lane, monkeypatch):
    save = lane.store._save
    monkeypatch.setattr(lane.store, '_save', Mock(side_effect=OSError('simulated crash before JSON save')))
    with pytest.raises(OSError):
        lane.store.reset_session(lane.key)
    binding = lane.db.get_telegram_topic_binding(chat_id=lane.source.chat_id, thread_id=lane.source.thread_id,
                                                profile_name=lane.source.profile)
    assert binding['session_id'] != 'parent'
    assert lane.db.get_session('parent')['end_reason'] == 'session_reset'
    assert lane.db.get_session(binding['session_id'])['ended_at'] is None
    monkeypatch.setattr(lane.store, '_save', save)
    lane.store._entries[lane.key] = lane.entry  # disk mirror from before the simulated crash
    recovered = await lane.runner._hmwa_heal_telegram_topic_binding(lane.source, lane.entry, lane.key)
    assert recovered.session_id == binding['session_id']


@pytest.mark.parametrize('reason', ['session_reset', 'superseded_by_new', 'session_handoff'])
async def test_stale_binding_never_reopens_retired_conversation(lane, reason):
    lane.db.end_session('parent', reason)
    lane.db.create_session('current', source='telegram')
    current = SessionEntry(session_key=lane.key, session_id='current', created_at=datetime.now(),
                           updated_at=datetime.now(), origin=lane.source)
    lane.store._entries[lane.key] = current
    lane.store._save()
    healed = await lane.runner._hmwa_heal_telegram_topic_binding(lane.source, current, lane.key)
    assert healed.session_id == 'current'
    assert lane.db.get_session('parent')['end_reason'] == reason


@pytest.mark.parametrize('work', ['process', 'delegation', 'cached_child', 'cached_review', 'cold_review', 'approval'])
async def test_detached_work_blocks_terminal_settlement(lane, monkeypatch, work):
    import threading
    from tools import async_delegation
    owner = SimpleNamespace(_background_review_run=SimpleNamespace(request_done=threading.Event()))
    lane.runner._agent_cache = {}
    if work == 'process':
        lane.store._has_active_processes_fn = lambda key: True
        monkeypatch.setattr(lane.store, '_has_active_processes_safe', lambda *args, **kwargs: True)
    if work == 'delegation':
        monkeypatch.setattr(async_delegation, 'has_live_for_session', lambda **kwargs: True)
    if work in ('cached_review', 'cached_child'):
        lane.runner._agent_cache[lane.key] = (owner, 'signature')
    if work == 'cached_child':
        owner._background_review_run = None
        lane.runner._agent_has_active_subagents = lambda agent: agent is owner
    if work in ('cold_review', 'approval'):
        live = SimpleNamespace(review=owner, controls=None)
        if work == 'approval':
            owner._background_review_run.request_done.set()
            live.controls = SimpleNamespace(pending={'prompt': 'waiting'}, remote_responders={})
        authority = SimpleNamespace(sessions={'parent': live})
        monkeypatch.setattr('gateway.session_authorities.active_authority', lambda runner: authority)
    assert not await settle(lane.runner, lane.source, lane.entry, lane.key, 1, lane.result)
    assert lane.store.peek_session_id(lane.key) == 'parent'
    lane.adapter.send.assert_not_awaited()


@pytest.mark.parametrize('reason', ['agent_close', 'ws_orphan_reap'])
async def test_topic_binding_recovers_automatic_cleanup_with_history(lane, reason):
    lane.db.end_session('parent', reason)
    lane.db.create_session('current', source='telegram')
    current = SessionEntry(session_key=lane.key, session_id='current', created_at=datetime.now(),
                           updated_at=datetime.now(), origin=lane.source)
    lane.store._entries[lane.key] = current
    lane.store._save()
    assert lane.db.get_session('parent')['ended_at'] is not None
    healed = await lane.runner._hmwa_heal_telegram_topic_binding(lane.source, current, lane.key)
    assert healed.session_id == 'parent'
    assert lane.store.peek_session_id(lane.key) == 'parent'
    assert lane.db.get_session('parent')['ended_at'] is None
    assert lane.db.get_session('parent')['end_reason'] is None
    assert lane.db.get_messages('parent') == lane.messages
    assert lane.store.get_or_create_session(lane.source).session_id == 'parent'
    assert 'predecessorword' in lane.store.load_transcript('parent')[0]['content']


@pytest.mark.parametrize('reason', ['session_handoff', 'session_reset', 'superseded_by_new'])
async def test_healer_refuses_retirement_after_automatic_target_snapshot(lane, monkeypatch, reason):
    lane.db.end_session('parent', 'agent_close')
    lane.db.create_session('current', source='telegram')
    current = SessionEntry(session_key=lane.key, session_id='current', created_at=datetime.now(),
                           updated_at=datetime.now(), origin=lane.source)
    lane.store._entries[lane.key] = current
    lane.store._save()
    reopen = lane.db.reopen_session
    def retire_before_guarded_reopen(session_id, **kwargs):
        assert kwargs == {'automatic_only': True}
        assert lane.db.promote_to_session_reset(session_id, reason)
        return reopen(session_id, **kwargs)
    monkeypatch.setattr(lane.db, 'reopen_session', retire_before_guarded_reopen)
    healed = await lane.runner._hmwa_heal_telegram_topic_binding(lane.source, current, lane.key)
    assert healed.session_id == 'current'
    assert lane.store.peek_session_id(lane.key) == 'current'
    assert lane.db.get_session('parent')['end_reason'] == reason
    assert lane.db.get_session('parent')['ended_at'] is not None
    assert lane.db.get_session('current')['ended_at'] is None
    assert lane.db.get_messages('parent') == lane.messages
