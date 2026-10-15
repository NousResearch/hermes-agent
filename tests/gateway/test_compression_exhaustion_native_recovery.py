"""Native gateway pause admission, topic recovery, and queue retention."""
import asyncio
import copy
import json
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionStore, SessionSource
from hermes_state import AsyncSessionDB, SessionDB

from tests.gateway.test_compression_exhaustion_reset_policy import KEY, HISTORY, policy, mark, reload_entry, admit, exhaust

pytest_plugins = ["tests.gateway.test_compression_exhaustion_reset_policy"]



def native_recovery_command(e, monkeypatch, command):
    """Use real target resolution, origin authorization and compression persistence."""
    if command in {"resume", "sessions"}:
        e.db.create_session("authorized-target", source="telegram", user_id=e.source.user_id,
                            chat_id=e.source.chat_id, chat_type=e.source.chat_type,
                            thread_id=e.source.thread_id)
        return f"/{command} authorized-target"
    if command == "compress":
        from agent.conversation_compression_manual import CompressResult, CompressRequest
        agent = Mock()
        agent.session_id = e.entry.session_id
        agent.context_compressor = SimpleNamespace(_last_summary_error=None,
                                                   _last_aux_model_failure_model=None,
                                                   _last_compress_aborted=False)
        def compress(*args, **kwargs):
            e.db.create_session("compressed-target", source="telegram")
            agent.session_id = "compressed-target"
            return CompressResult("compressed", HISTORY, HISTORY, 100, 90, CompressRequest(),
                                  summary={"headline": "Compressed", "token_line": "90", "note": ""})
        e.runner._build_manual_compression_agent = AsyncMock(return_value=agent)
        e.runner._cleanup_agent_resources_off_loop = AsyncMock()
        e.runner._resolve_session_agent_runtime = Mock(return_value=("test", {"api_key": "test"}))
        monkeypatch.setattr("agent.conversation_compression_manual.compress_now", compress)
        monkeypatch.setattr("agent.conversation_compression.finalize_context_engine_compression_notification", Mock())
    return f"/{command}"


async def native_message(e, text):
    return await e.runner._handle_message(MessageEvent(text=text, source=e.source))


@pytest.mark.asyncio
async def test_paused_new_cleans_one_shot_agent_after_primary_commit(native_env, monkeypatch):
    e = native_env
    mark(e)
    state = e.runner._session_state(e.key)
    state.conversation.model_override = {"model": "standing-model", "provider": "custom"}
    e.runner._claim_one_turn_restore(e.key)
    state.conversation.model_override = {"model": "temporary-model", "provider": "custom"}
    agent = Mock()
    e.runner._agent_cache_lock = threading.Lock()
    e.runner._agent_cache = {e.key: (agent, "signature")}
    del e.runner._cleanup_old_agent_for_reset

    observed = []

    def clean_after_commit(old_agent):
        current = reload_entry(e)
        observed.append((old_agent, current.session_id, current.compression_paused))

    cleanup = Mock(side_effect=clean_after_commit)
    monkeypatch.setattr(e.runner, "_cleanup_agent_resources", cleanup)
    reply = await native_message(e, "/new")
    assert "could not be persisted" not in str(reply)
    cleanup.assert_called_once_with(agent)
    assert observed == [(agent, reload_entry(e).session_id, False)]
    assert observed[0][1] != e.entry.session_id
    assert e.runner._cached_agent_for(e.key) is None
    assert state.conversation.one_turn_restore is None
    assert state.conversation.model_override is None
    assert not e.runner._is_session_running(e.key)


@pytest.mark.asyncio
async def test_failed_paused_new_preserves_old_session_resources(native_env, monkeypatch):
    e = native_env
    mark(e)
    state = e.runner._session_state(e.key)
    state.conversation.model_override = {"model": "retained-model"}
    state.conversation.service_tier_override = "priority"
    state.conversation.ephemeral_pin = ("retained", "prompt")
    state.persistent.approvals = {"command": "retained approval"}
    conversation = copy.deepcopy(state.conversation)
    approvals = copy.deepcopy(state.persistent.approvals)
    agent = Mock()
    e.runner._agent_cache_lock = threading.Lock()
    e.runner._agent_cache = {e.key: (agent, "signature")}
    del e.runner._cleanup_old_agent_for_reset
    cleanup = Mock()
    monkeypatch.setattr(e.runner, "_cleanup_agent_resources", cleanup)
    invalidate = Mock(wraps=e.runner._invalidate_session_run_generation)
    monkeypatch.setattr(e.runner, "_invalidate_session_run_generation", invalidate)
    failure = Mock(side_effect=OSError("primary unavailable"))
    monkeypatch.setattr(e.db, "replace_gateway_routing_entries", failure)
    with patch("tools.async_delegation.interrupt_for_session") as interrupt, patch(
        "gateway.slash_commands_session._reset_process_scoped_tool_state"
    ) as process_reset:
        reply = await native_message(e, "/new")
    failure.assert_called_once()
    assert "could not be persisted" in str(reply)
    assert reload_entry(e).session_id == e.entry.session_id
    assert reload_entry(e).compression_paused
    invalidate.assert_not_called()
    cleanup.assert_not_called()
    assert e.runner._cached_agent_for(e.key) is agent
    assert state.conversation == conversation
    assert state.persistent.approvals == approvals
    interrupt.assert_not_called()
    process_reset.assert_not_called()
    e.runner._fire_session_reset_hooks.assert_not_awaited()
    assert not e.runner._is_session_running(e.key)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["primary", "binding"])
@pytest.mark.parametrize("command", ["new", "resume", "compress", "sessions"])
@pytest.mark.parametrize("fail", [False, True])
async def test_native_recovery_owns_admission_through_cancelled_commit(
    native_env, monkeypatch, command, boundary, fail,
):
    e = native_env
    mark(e)
    pin = {"model": "previous-model", "provider": "custom",
           "base_url": "https://previous-provider.invalid/v1"}
    if command in {"resume", "sessions"}:
        e.store.set_model_override(e.key, pin)
        e.runner._session_state(e.key).conversation.model_override = dict(pin)
        persisted = reload_entry(e)
        assert persisted is not None and persisted.model_override == pin
    text = native_recovery_command(e, monkeypatch, command)
    entered, release = threading.Event(), threading.Event()
    obj, attr = ((e.db, "replace_gateway_routing_entries") if boundary == "primary"
                 else (e.runner, "_sync_telegram_topic_binding"))
    original = getattr(obj, attr)
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        if fail:
            raise OSError("private fixture failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(obj, attr, blocked)
    task = asyncio.create_task(native_message(e, text))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        state = e.runner._peek_session_state(e.key)
        assert state is not None and state.turn.agent is not None, "native command never claimed admission"
        owner, lease = state.turn.agent, state.turn.lease
        assert lease is not None and not lease.released
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        for incoming in ("ordinary message", "/new", "/stop", "/resume authorized-target", "/compress"):
            reply = await asyncio.wait_for(native_message(e, incoming), 1)
            assert "retry" in str(reply).lower()
            assert state.turn.agent is owner and state.turn.lease is lease
        e.runner._hmwa_prepare_turn.assert_not_awaited()
        e.runner._run_agent.assert_not_awaited()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
    assert not e.runner._is_session_running(e.key)
    assert lease.released
    persisted = reload_entry(e)
    assert persisted is not None
    assert persisted.compression_paused is (fail and boundary == "primary")
    if command in {"resume", "sessions"}:
        if fail and boundary == "primary":
            assert persisted.session_id == e.entry.session_id
            assert persisted.model_override == pin
            assert e.runner._session_model_override(e.key) == pin
        else:
            assert persisted.session_id == "authorized-target"
            assert persisted.model_override is None
            assert e.runner._session_model_override(e.key) is None
            # The next turn must not resurrect the previous conversation's billing route.
            with patch("gateway.run._resolve_runtime_agent_kwargs_for_provider") as resolve:
                e.runner._rehydrate_session_model_override(e.key)
            resolve.assert_not_called()
            assert e.runner._session_model_override(e.key) is None
    assert e.db.get_messages(e.entry.session_id), "archived history lost"


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypath", ["idle", "busy", "confirmed", "alias"])
async def test_native_new_postcommit_tail_blocks_messages(native_env, monkeypatch, entrypath):
    e = native_env
    mark(e)
    entered, release = asyncio.Event(), asyncio.Event()
    async def hooks(*args):
        entered.set()
        await release.wait()
    e.runner._fire_session_reset_hooks = hooks
    old_message = None
    if entrypath == "busy":
        # The ordinary native claim exists before pause resolution; hold its setup.
        started, finish = asyncio.Event(), asyncio.Event()
        original = e.runner._hmwa_resolve_session
        async def resolve(*args):
            started.set()
            await finish.wait()
            return await original(*args)
        e.runner._hmwa_resolve_session = resolve
        old_message = asyncio.create_task(native_message(e, "old message"))
        await asyncio.wait_for(started.wait(), 5)
        e.runner._hmwa_resolve_session = original
    if entrypath == "confirmed":
        callbacks = []
        async def request(**kwargs):
            callbacks.append(kwargs["handler"])
            return "confirm"
        e.runner._request_slash_confirm = request
        e.runner._read_user_config.return_value = {"approvals": {"destructive_slash_confirm": True}}
        assert await native_message(e, "/new") == "confirm"
        assert not e.runner._is_session_running(e.key)
        task = asyncio.create_task(callbacks[0]("once"))
    else:
        task = asyncio.create_task(native_message(e, "/reset" if entrypath == "alias" else "/new"))
    try:
        assert await asyncio.wait_for(entered.wait(), 5)
        assert not reload_entry(e).compression_paused
        reply = await native_message(e, "after commit")
        e.runner._hmwa_prepare_turn.assert_not_awaited()
        assert "retry" in str(reply).lower()
        assert e.runner._is_session_running(e.key)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        if old_message:
            old_message.cancel()
            await asyncio.gather(old_message, return_exceptions=True)
    assert not e.runner._is_session_running(e.key)
    assert await native_message(e, "after release") == "prepared"
    e.runner._hmwa_prepare_turn.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["hook", "notice"])
async def test_native_old_new_cancellation_cannot_clear_successor(native_env, boundary):
    e = native_env
    mark(e)
    entered, release_notice = threading.Event(), threading.Event()
    async def hooks(*args):
        entered.set()
        await asyncio.Event().wait()
    def notice(*args):
        entered.set()
        assert release_notice.wait(10)
        return ""
    if boundary == "hook":
        e.runner._fire_session_reset_hooks = hooks
    else:
        e.runner._reset_notice_session_info = notice
    old = asyncio.create_task(native_message(e, "/new"))
    successor = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert "retry" in str(await native_message(e, "before displacement")).lower()
        e.runner._hmwa_prepare_turn.assert_not_awaited()
        # Explicit native displacement after storage/binding settled, then actual message claim.
        state = e.runner._peek_session_state(e.key)
        if state is not None:
            generation = e.runner._invalidate_session_run_generation(e.key, reason="test-displacement")
            e.runner._release_running_agent_state(e.key, run_generation=generation)
        prepared = asyncio.Event()
        async def prepare(*args):
            prepared.set()
            await asyncio.Event().wait()
        e.runner._hmwa_prepare_turn = prepare
        successor = asyncio.create_task(native_message(e, "successor"))
        await asyncio.wait_for(prepared.wait(), 5)
        state = e.runner._peek_session_state(e.key)
        successor_agent, successor_lease = state.turn.agent, state.turn.lease
        successor_generation = state.persistent.run_generation
        old.cancel()
        await asyncio.gather(old, return_exceptions=True)
        assert old.cancelled()
        assert state.turn.agent is successor_agent, "old command cleared successor native slot"
        assert state.turn.lease is successor_lease and not successor_lease.released
        assert state.persistent.run_generation == successor_generation
    finally:
        release_notice.set()
        old.cancel()
        if successor:
            successor.cancel()
        await asyncio.gather(old, *([successor] if successor else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_native_message_already_past_busy_check_cannot_claim_over_recovery(native_env):
    e = native_env
    mark(e)
    dispatched, continue_dispatch = asyncio.Event(), asyncio.Event()
    original = e.runner._hm_dispatch_idle_commands
    async def dispatch(event, *args):
        result = await original(event, *args)
        if event.text == "racing message":
            dispatched.set()
            await continue_dispatch.wait()
        return result
    e.runner._hm_dispatch_idle_commands = dispatch
    committed, finish = asyncio.Event(), asyncio.Event()
    async def hooks(*args):
        committed.set()
        await finish.wait()
    e.runner._fire_session_reset_hooks = hooks
    incoming = asyncio.create_task(native_message(e, "racing message"))
    recovery = None
    try:
        await asyncio.wait_for(dispatched.wait(), 5)
        recovery = asyncio.create_task(native_message(e, "/new"))
        await asyncio.wait_for(committed.wait(), 5)
        continue_dispatch.set()
        reply = await incoming
        e.runner._hmwa_prepare_turn.assert_not_awaited()
        assert "retry" in str(reply).lower()
    finally:
        continue_dispatch.set()
        finish.set()
        await asyncio.gather(incoming, *([recovery] if recovery else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_native_same_target_resume_owns_validation_but_retains_pause(native_env):
    e = native_env
    mark(e)
    entered, release = asyncio.Event(), asyncio.Event()
    original = e.runner._resolve_resume_target
    async def resolve(*args):
        entered.set()
        await release.wait()
        return await original(*args)
    e.runner._resolve_resume_target = resolve
    task = asyncio.create_task(native_message(e, f"/resume {e.entry.session_id}"))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        state = e.runner._peek_session_state(e.key)
        assert state is not None and state.turn.agent is not None
        assert "retry" in str(await native_message(e, "during resume")).lower()
        # A stripped-topic alias resolves to the same effective recovery identity.
        raw_alias = replace(e.source, thread_id="raw-topic-alias")
        e.runner._normalize_source_for_session_key = lambda source: e.source
        alias_reply = await e.runner._handle_message(MessageEvent(text="alias message", source=raw_alias))
        assert "retry" in str(alias_reply).lower()
        assert state.turn.agent is not None
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    assert reload_entry(e).compression_paused
    assert not e.runner._is_session_running(e.key)
    assert "already" in task.result().lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("paused", [False, True])
async def test_native_new_cancellation_during_cleanup_settles_own_admission(native_env, paused):
    e = native_env
    if paused:
        mark(e)
    entered, release = asyncio.Event(), asyncio.Event()
    async def cleanup(*args):
        entered.set()
        await release.wait()
    e.runner._cleanup_old_agent_for_reset = cleanup
    task = asyncio.create_task(native_message(e, "/new"))
    lease = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        state = e.runner._peek_session_state(e.key)
        assert e.runner._is_session_running(e.key) is paused
        if paused:
            assert not reload_entry(e).compression_paused
            lease = state.turn.lease
            assert lease is not None and not lease.released
        task.cancel()
        await asyncio.sleep(0)
        if paused:
            assert not task.done() and not lease.released
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
    assert not e.runner._is_session_running(e.key)
    if lease:
        assert lease.released
    assert not reload_entry(e).compression_paused



@pytest.mark.asyncio
async def test_native_retry_preserves_paused_transcript(native_env):
    e = native_env
    mark(e)
    policy(e, "compression:\n  exhaustion_action: pause\n")
    session_id = e.entry.session_id
    before = e.db.get_messages(session_id)
    e.entry.last_prompt_tokens = 987

    reply = await native_message(e, "/retry")

    assert "paused" in str(reply).lower()
    e.runner._hmwa_prepare_turn.assert_not_awaited()
    e.runner._run_agent.assert_not_awaited()
    assert e.db.get_messages(session_id) == before
    assert e.entry.last_prompt_tokens == 987
    persisted = reload_entry(e)
    assert persisted is not None
    assert persisted.session_id == session_id and persisted.compression_paused
    assert not e.runner._is_session_running(e.key)


@pytest.mark.asyncio
async def test_native_compress_reports_committed_history_when_pause_clear_fails(
    native_env, monkeypatch,
):
    from agent.conversation_compression_manual import CompressRequest, CompressResult

    e = native_env
    mark(e)
    text = native_recovery_command(e, monkeypatch, "compress")
    agent = e.runner._build_manual_compression_agent.return_value
    session_id = e.entry.session_id
    before = e.db.get_messages(session_id)
    compressed = [{"role": "user", "content": "Summary of the completed inspection"}]
    failure = Mock(side_effect=OSError("routing storage unavailable"))

    def compress(*args, **kwargs):
        # Commit real in-place history before failing the later routing write.
        e.db.archive_and_compact(session_id, compressed)
        agent._last_compaction_in_place = True
        monkeypatch.setattr(e.db, "replace_gateway_routing_entries", failure)
        return CompressResult(
            "compressed", HISTORY, compressed, 100, 20, CompressRequest(),
            summary={"headline": "Compressed", "token_line": "20", "note": ""},
        )

    monkeypatch.setattr("agent.conversation_compression_manual.compress_now", compress)
    reply = await native_message(e, text)

    failure.assert_called_once()
    after = e.db.get_messages(session_id)
    assert after != before
    assert [message["content"] for message in after] == [compressed[0]["content"]]
    persisted = reload_entry(e)
    assert persisted is not None
    assert persisted.session_id == session_id and persisted.compression_paused
    assert not e.runner._is_session_running(e.key)
    e.runner._hmwa_prepare_turn.assert_not_awaited()
    e.runner._run_agent.assert_not_awaited()
    assert "unchanged" not in str(reply).lower()
    assert "persist" in str(reply).lower()
    assert "/compress" in str(reply) and "/new" in str(reply)


@pytest.mark.asyncio
@pytest.mark.parametrize("routing_committed", [False, True])
async def test_compression_child_startup_keeps_pause_until_routing_commit(env, routing_committed):
    e = env
    mark(e)
    parent = e.entry.session_id
    before = e.db.get_messages(parent)
    child = "compression-child"
    e.db.create_session(child, source="telegram", parent_session_id=parent,
                        session_key=e.key, user_id=e.source.user_id, chat_id=e.source.chat_id,
                        chat_type=e.source.chat_type, thread_id=e.source.thread_id)
    assert e.store.rewrite_transcript(child, HISTORY)
    e.db.end_session(parent, "compression")
    assert e.db.get_compression_tip(parent) == child
    if routing_committed:
        assert e.store.commit_manual_compression(e.key, parent, child)
    else:
        # Compression committed, but the route and pause release did not.
        with patch.object(e.db, "replace_gateway_routing_entries", side_effect=OSError("routing unavailable")):
            with pytest.raises(OSError):
                e.store.commit_manual_compression(e.key, parent, child)

    restarted = SessionStore(e.store.sessions_dir, e.store.config)
    restarted._db = e.db
    e.runner.session_store = e.store = restarted
    recovered = restarted.lookup_by_session_key(e.key)
    assert recovered.session_id == child
    assert recovered.compression_paused is (not routing_committed)
    assert reload_entry(e).compression_paused is (not routing_committed)
    assert e.db.get_messages(parent) == before
    if not routing_committed:
        reply, _ = await admit(e)
        assert "paused" in reply.lower()
        e.runner._hmwa_prepare_turn.assert_not_awaited()
        e.runner._run_agent.assert_not_awaited()
        # A later successful explicit compression commit can still release the pause.
        assert restarted.commit_manual_compression(e.key, child, child)
        assert not reload_entry(e).compression_paused
    assert e.db.get_messages(parent) == before
    assert len(e.db.get_messages(child)) == len(HISTORY)


@pytest.mark.asyncio
@pytest.mark.parametrize("lineage", ["unrelated", "lookup_error"])
async def test_startup_pause_transfer_requires_verified_compression_child(env, monkeypatch, lineage):
    e = env
    mark(e)
    parent = e.entry.session_id
    child = "candidate-child"
    e.db.create_session(child, source="telegram",
                        parent_session_id=parent if lineage == "lookup_error" else None,
                        session_key=e.key, user_id=e.source.user_id, chat_id=e.source.chat_id,
                        chat_type=e.source.chat_type, thread_id=e.source.thread_id)
    assert e.store.rewrite_transcript(child, HISTORY)
    e.db.end_session(parent, "compression")
    fresh_db = SessionDB(db_path=e.tmp / "state.db")
    restarted = SessionStore(e.store.sessions_dir, e.store.config)
    restarted._db = fresh_db
    try:
        if lineage == "lookup_error":
            monkeypatch.setattr(fresh_db, "get_compression_tip",
                                Mock(side_effect=OSError("lineage unavailable")))
        recovered = restarted.lookup_by_session_key(e.key)
        assert recovered is not None and recovered.session_id == parent
        assert recovered.compression_paused
        e.runner.session_store = restarted
        reply, _ = await admit(e)
        assert "paused" in reply.lower()
        e.runner._hmwa_prepare_turn.assert_not_awaited()
        e.runner._run_agent.assert_not_awaited()
    finally:
        e.runner.session_store = e.store
        restarted.close_all_db_handles()
        fresh_db.close()


def test_idle_compression_pause_survives_prune_and_store_reload(env):
    from datetime import datetime, timedelta

    e = env
    mark(e)
    ordinary = e.store.get_or_create_session(replace(e.source, chat_id="ordinary-idle"))
    assert ordinary.session_key != e.key
    old = datetime.now() - timedelta(days=40)
    e.entry.updated_at = ordinary.updated_at = old
    e.store._save_entries()

    assert e.store.prune_old_entries(max_age_days=30) == 1
    assert e.store.lookup_by_session_key(e.key).session_id == e.entry.session_id
    assert e.store.lookup_by_session_key(ordinary.session_key) is None
    db = SessionDB(db_path=e.tmp / "state.db")
    store = SessionStore(e.store.sessions_dir, e.store.config)
    store._db = db
    try:
        paused = store.lookup_by_session_key(e.key)
        assert paused is not None and paused.session_id == e.entry.session_id
        assert paused.metadata[KEY] is True and paused.compression_paused
        assert paused.updated_at == old
        assert store.lookup_by_session_key(ordinary.session_key) is None
    finally:
        store.close_all_db_handles()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("end_reason", [None, "agent_close"])
async def test_recovery_hint_accepts_current_recoverable_peer(env, end_reason):
    e = env
    sid = e.entry.session_id
    if end_reason:
        e.db.end_session(sid, end_reason)
    e.store._entries.pop(e.key)
    e.store._save_entries()
    db = SessionDB(db_path=e.tmp / "state.db")
    store = SessionStore(e.store.sessions_dir, e.store.config)
    store._db = db
    try:
        recovered = await asyncio.to_thread(
            store.get_or_create_session, e.source, recovery_session_id=sid)
        assert recovered.session_id == sid
        row = db.get_session(sid)
        assert row is not None and row["end_reason"] is None
    finally:
        store.close_all_db_handles()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("foreign_field", ["source", "user_id", "chat_id", "session_key"])
async def test_recovery_hint_cannot_adopt_foreign_peer_without_primary_route(env, foreign_field):
    e = env
    incoming = replace(e.source, chat_id="incoming-chat", thread_id="17585")
    incoming_key = e.store._generate_session_key(incoming)
    foreign = dict(source="telegram", user_id=incoming.user_id, chat_id=incoming.chat_id,
                   chat_type=incoming.chat_type, thread_id=incoming.thread_id,
                   session_key=incoming_key)
    foreign[foreign_field] = {
        "source": "discord", "user_id": "other-user", "chat_id": "other-chat",
        "session_key": e.key,
    }[foreign_field]
    e.db.create_session("foreign-hint", **foreign)
    assert e.store.rewrite_transcript("foreign-hint", HISTORY)
    e.db.enable_telegram_topic_mode(chat_id=incoming.chat_id, user_id=incoming.user_id)
    e.db.bind_telegram_topic(
        chat_id=incoming.chat_id, thread_id=incoming.thread_id, user_id=incoming.user_id,
        session_key=incoming_key, session_id="foreign-hint", managed_mode="restored",
    )
    assert e.store.lookup_by_session_key(incoming_key) is None
    db = SessionDB(db_path=e.tmp / "state.db")
    store = SessionStore(e.store.sessions_dir, e.store.config)
    store._db = db
    try:
        binding = db.get_telegram_topic_binding(chat_id=incoming.chat_id, thread_id=incoming.thread_id)
        assert binding is not None and binding["session_id"] == "foreign-hint"
        recovered = await asyncio.to_thread(
            store.get_or_create_session, incoming, recovery_session_id=binding["session_id"])
        assert recovered.session_id != "foreign-hint"
        assert not any(m.get("content") == "evidence" for m in store.load_transcript(recovered.session_id))
        foreign_row = db.get_session("foreign-hint")
        assert foreign_row is not None and foreign_row["session_key"] == foreign["session_key"]
    finally:
        store.close_all_db_handles()
        db.close()


@pytest.mark.asyncio
async def test_recovery_hint_restores_keyless_same_user_telegram_topic(env):
    e = env
    incoming = replace(e.source, chat_id="incoming-chat", thread_id="17585")
    e.db.create_session("restorable-topic", source="telegram", user_id=incoming.user_id)
    assert e.store.rewrite_transcript("restorable-topic", HISTORY)
    db = SessionDB(db_path=e.tmp / "state.db")
    store = SessionStore(e.store.sessions_dir, e.store.config)
    store._db = db
    try:
        recovered = await asyncio.to_thread(
            store.get_or_create_session, incoming, recovery_session_id="restorable-topic")
        assert recovered.session_id == "restorable-topic"
        assert any(m.get("content") == "evidence" for m in store.load_transcript(recovered.session_id))
    finally:
        store.close_all_db_handles()
        db.close()


@pytest.mark.asyncio
async def test_recovery_hint_rejects_sibling_profile_keyless_topic_in_shared_store(env, monkeypatch):
    e = env
    root = e.tmp / "profile-root"
    root.mkdir()
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    db = SessionDB(db_path=root / "state.db")
    incoming = replace(e.source, chat_type="dm", thread_id="17585")
    db.create_session("sibling-topic", source="telegram", user_id=incoming.user_id,
                      profile_name="sibling")
    store = SessionStore(e.tmp / "shared-routing", e.store.config)
    store._db = db
    try:
        assert db._own_profile_name() == "default"
        assert store.rewrite_transcript("sibling-topic", HISTORY)
        recovered = await asyncio.to_thread(
            store.get_or_create_session, incoming, recovery_session_id="sibling-topic")
        assert recovered.session_id != "sibling-topic"
        assert db.get_session("sibling-topic")["profile_name"] == "sibling"
        assert not any(m.get("content") == "evidence" for m in store.load_transcript(recovered.session_id))
    finally:
        store.close_all_db_handles()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("old_end_reason", [None, "agent_close"])
async def test_recovery_hint_cannot_cross_newer_reset_fence(env, old_end_reason):
    e = env
    old_id = e.entry.session_id
    if old_end_reason:
        e.db.end_session(old_id, old_end_reason)
    e.db.create_session(
        "newer-reset-fence", source=e.source.platform.value, session_key=e.key,
        user_id=e.source.user_id, chat_id=e.source.chat_id,
        chat_type=e.source.chat_type, thread_id=e.source.thread_id,
    )
    e.db.end_session("newer-reset-fence", "session_reset")
    peer = dict(source=e.source.platform.value, session_key=e.key,
                user_id=e.source.user_id, chat_id=e.source.chat_id,
                chat_type=e.source.chat_type, thread_id=e.source.thread_id)
    assert e.db.find_latest_gateway_session_for_peer(**peer) is None
    e.store._entries.pop(e.key)
    e.store._save_entries()
    db = SessionDB(db_path=e.tmp / "state.db")
    store = SessionStore(e.store.sessions_dir, e.store.config)
    store._db = db
    try:
        recovered = await asyncio.to_thread(
            store.get_or_create_session, e.source, recovery_session_id=old_id)
        assert recovered.session_id not in {old_id, "newer-reset-fence"}
        row = db.get_session(old_id)
        assert row is not None and row["end_reason"] == old_end_reason
    finally:
        store.close_all_db_handles()
        db.close()


@pytest.mark.asyncio
async def test_recovery_hint_cannot_revive_explicitly_reset_session(env):
    e = env
    old_id = e.entry.session_id
    replacement = e.store.reset_session(e.key, require_primary=True)
    assert replacement is not None and replacement.session_id != old_id
    old_row = e.db.get_session(old_id)
    assert old_row is not None and old_row["end_reason"] == "session_reset"
    e.store._entries.pop(e.key)
    e.store._save_entries()
    db = SessionDB(db_path=e.tmp / "state.db")
    store = SessionStore(e.store.sessions_dir, e.store.config)
    store._db = db
    try:
        recovered = await asyncio.to_thread(
            store.get_or_create_session, e.source, recovery_session_id=old_id)
        assert recovered.session_id == replacement.session_id
        old_row = db.get_session(old_id)
        assert old_row is not None and old_row["end_reason"] == "session_reset"
    finally:
        store.close_all_db_handles()
        db.close()


@pytest.fixture
def native_topic_env(native_env):
    e = native_env
    e.source = replace(e.source, thread_id="17585", chat_type="dm")
    e.entry = e.store.get_or_create_session(e.source)
    e.key = e.entry.session_key
    assert e.store.rewrite_transcript(e.entry.session_id, HISTORY)
    for name in ("_is_telegram_topic_lane", "_record_telegram_topic_binding",
                 "_sync_telegram_topic_binding"):
        delattr(e.runner, name)
    e.db.enable_telegram_topic_mode(chat_id=e.source.chat_id, user_id=e.source.user_id)
    e.runner._record_telegram_topic_binding(e.source, e.entry)
    return e


@pytest.mark.asyncio
async def test_stripped_thread_compress_uses_paused_topic_via_native_inbound(native_topic_env, monkeypatch):
    e = native_topic_env
    mark(e)
    parent_id = e.entry.session_id
    root_source = replace(e.source, thread_id=None)
    root_key = e.store._generate_session_key(root_source)
    assert root_key != e.key
    e.runner.__dict__.pop("_normalize_source_for_session_key")
    e.runner._recover_telegram_topic_thread_id = lambda source: (
        e.source.thread_id if source.thread_id is None else None)
    text = native_recovery_command(e, monkeypatch, "compress")
    reply = await e.runner._handle_message(MessageEvent(text=text, source=root_source))
    committed = reload_entry(e)
    assert reply and "not enough" not in str(reply).lower()
    assert committed.session_id != parent_id and not committed.compression_paused
    root_entry = e.store.lookup_by_session_key(root_key)
    assert root_entry is None or root_entry.session_id != committed.session_id
    e.runner._build_manual_compression_agent.assert_awaited_once()
    assert e.runner._build_manual_compression_agent.await_args.args[0] == parent_id
    e.runner._hmwa_prepare_turn.assert_not_awaited()
    e.runner._run_agent.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("reopen", [False, True])
@pytest.mark.parametrize("command", ["new", "resume", "compress", "policy_reset"])
async def test_committed_recovery_survives_stale_topic_mirror(native_topic_env, monkeypatch, command, reopen):
    e = native_topic_env
    mark(e)
    parent_id = e.entry.session_id
    before = e.db.get_messages(parent_id)
    text = native_recovery_command(e, monkeypatch, command)
    if command == "resume":
        e.db.append_message("authorized-target", "assistant", "authorized history")
    monkeypatch.setattr(SessionDB, "bind_telegram_topic", Mock(side_effect=OSError("mirror unavailable")))
    if command == "policy_reset":
        policy(e, "compression:\n  exhaustion_action: reset")
        text = "recover with explicit reset policy"
    reply = await native_message(e, text)
    committed = e.store.lookup_by_session_key(e.key)
    target_id = committed.session_id
    target_history = e.store.load_transcript(target_id)
    assert target_id != parent_id and not committed.compression_paused
    assert reply and "could not be persisted" not in str(reply)
    assert e.db.get_telegram_topic_binding(chat_id=e.source.chat_id, thread_id=e.source.thread_id)["session_id"] == parent_id
    # The stub compressor publishes a child but does not end its parent row. Preserve
    # that lifecycle state; reset/resume must leave their ended parent closed.
    parent_ended_at = e.db.get_session(parent_id)["ended_at"]
    if command != "compress":
        assert parent_ended_at is not None
    fresh = fresh_db = None
    if reopen:
        fresh_db = SessionDB(db_path=e.tmp / "state.db")
        fresh = SessionStore(e.store.sessions_dir, e.store.config)
        fresh._db = fresh_db
        e.runner.session_store = fresh
        e.runner._session_db = AsyncSessionDB(fresh_db)
    prepared = []
    async def prepare(event, source, entry, *args):
        prepared.append((entry.session_id, e.runner.session_store.load_transcript(entry.session_id)))
        return "prepared", None
    e.runner._hmwa_prepare_turn = AsyncMock(side_effect=prepare)
    try:
        assert await native_message(e, "next ordinary message") == "prepared"
        assert prepared == [(target_id, target_history)]
        assert e.runner.session_store.lookup_by_session_key(e.key).session_id == target_id
        assert e.db.get_messages(parent_id) == before
        assert e.db.get_session(parent_id)["ended_at"] == parent_ended_at
    finally:
        e.runner.session_store = e.store
        e.runner._session_db = AsyncSessionDB(e.db)
        if fresh:
            fresh.close_all_db_handles()
            fresh_db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["new", "resume", "policy_reset"])
async def test_primary_failure_keeps_topic_pause_before_next_message(native_topic_env, monkeypatch, command):
    e = native_topic_env
    mark(e)
    text = native_recovery_command(e, monkeypatch, command)
    if command == "policy_reset":
        policy(e, "compression:\n  exhaustion_action: reset")
        text = "recover"
    monkeypatch.setattr(e.db, "replace_gateway_routing_entries", Mock(side_effect=OSError("primary unavailable")))
    assert "could not be persisted" in str(await native_message(e, text))
    policy(e, "compression:\n  exhaustion_action: pause")
    assert "paused" in str(await native_message(e, "next ordinary message"))
    assert reload_entry(e).session_id == e.entry.session_id
    assert reload_entry(e).compression_paused
    e.runner._hmwa_prepare_turn.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["validation", "binding", "primary"])
async def test_topic_restore_holds_native_paused_admission(native_topic_env, monkeypatch, boundary):
    e = native_topic_env
    mark(e)
    e.db.create_session("topic-target", source="telegram", user_id=e.source.user_id, chat_id=e.source.chat_id)
    entered, release = threading.Event(), threading.Event()
    name = {"validation": "resolve_session_id", "binding": "_bind_telegram_topic_conn",
            "primary": "_replace_gateway_routing_entries_conn"}[boundary]
    original = getattr(e.db, name)
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(e.db, name, blocked)
    task = asyncio.create_task(e.runner._handle_topic_command(
        MessageEvent(text="/topic topic-target", source=e.source), "topic-target"))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        reply = await native_message(e, "ordinary message during restoration")
        assert "recovery is still settling" in str(reply)
        e.runner._hmwa_prepare_turn.assert_not_awaited()
        if boundary != "validation":
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            assert not task.done()
    finally:
        release.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
    assert reload_entry(e).session_id == "topic-target"
    assert not reload_entry(e).compression_paused
    assert not e.runner._is_session_running(e.key)


@pytest.mark.asyncio
@pytest.mark.parametrize("command,outcome", [
    ("new", "primary_failure"),
    ("compress", "primary_failure"),
    ("resume", "primary_failure"),
    ("resume", "missing_target"),
    ("retry", "paused_control"),
    ("new", "committed_control"),
])
async def test_paused_manual_command_preserves_accepted_adapter_queue_on_refusal(
    native_env, monkeypatch, command, outcome,
):
    from gateway.config import PlatformConfig
    from gateway.platforms.base import SendResult
    from tests.gateway.test_pending_drain_no_recursion import _StubAdapter

    e = native_env
    r = e.runner
    mark(e)
    policy(e, "compression:\n  exhaustion_action: pause\n")
    adapter = _StubAdapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="reply"))
    adapter._message_handler = r._primary_message_handler()
    r.adapters[Platform.TELEGRAM] = adapter
    pending = MessageEvent(text="accepted before recovery", source=e.source, message_id="pending")
    overflow = MessageEvent(text="/queue second in FIFO", source=e.source, message_id="overflow")
    adapter._active_sessions[e.key] = asyncio.Event()
    try:
        await adapter.handle_message(pending)
        assert pending._gateway_accepted and adapter._pending_messages[e.key] is pending
        await r._busy_queue_command(overflow, e.key, e.source)
    finally:
        adapter._active_sessions.pop(e.key, None)
    assert [item.text for item in r._overflow_queue(e.key)] == ["second in FIFO"]

    text = native_recovery_command(e, monkeypatch, command)
    if outcome == "missing_target":
        text = "/resume nonexistent-target"
    elif outcome == "primary_failure":
        monkeypatch.setattr(e.db, "replace_gateway_routing_entries",
                            Mock(side_effect=OSError("primary unavailable")))
    try:
        await adapter.handle_message(MessageEvent(text=text, source=e.source, message_id="command"))
        while adapter._background_tasks:
            await asyncio.gather(*tuple(adapter._background_tasks))
            await asyncio.sleep(0)
        current = reload_entry(e)
        assert current is not None
        if outcome == "committed_control":
            # A successful native /new deliberately clears its old queue (#111142).
            assert current.session_id != e.entry.session_id and not current.compression_paused
        else:
            assert current.session_id == e.entry.session_id and current.compression_paused
            assert pending._gateway_accepted
            assert adapter._pending_messages.get(e.key) is pending
            assert [item.text for item in r._overflow_queue(e.key)] == ["second in FIFO"]
            r._hmwa_prepare_turn.assert_not_awaited()
            r._run_agent.assert_not_awaited()
    finally:
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["reset", "pause", "reset_failure"])
async def test_terminal_model_pending_steer_survives_with_adapter_fifo(
    native_env, monkeypatch, action,
):
    from gateway.config import PlatformConfig
    from gateway.platforms.base import SendResult
    from tests.gateway.test_compression_failure_session_sync import (
        _CompressionThenFailureAgent, _install_compression_failure_agent,
    )
    from tests.gateway.test_pending_drain_no_recursion import _StubAdapter

    e = native_env
    r = e.runner
    del r._hmwa_prepare_turn
    del r._run_agent
    r._agent_cache = {}
    r._agent_cache_lock = threading.Lock()
    r._prefill_messages = []
    r._ephemeral_system_prompt = ""
    r._reasoning_config = r._service_tier = r._fallback_model = None
    r._provider_routing = {}
    r.hooks.loaded_hooks = []
    r._voice_mode = {}
    r._get_proxy_url = lambda: None
    r._resolve_session_agent_runtime = lambda **kwargs: (
        "gpt-5.4", {"provider": "custom", "api_key": "fixture-key",
                    "base_url": "https://model.invalid/v1", "api_mode": "chat_completions"},
    )
    adapter = _StubAdapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="reply"))
    adapter._message_handler = r._primary_message_handler()
    r.adapters[Platform.TELEGRAM] = adapter
    accepted = MessageEvent(text="adapter accepted first", source=e.source, message_id="accepted")
    overflow = MessageEvent(text="/queue overflow second", source=e.source, message_id="overflow")
    # The model's steer payload is user data, never authority for another slash command.
    steer = "/new literal user steer" if action == "reset" else "late native steer third"
    model_inputs = []
    loop = asyncio.get_running_loop()

    async def enqueue_waiting():
        await adapter.handle_message(accepted)
        assert accepted._gateway_accepted
        await r._busy_queue_command(overflow, e.key, e.source)
        if action == "reset_failure":
            # A paused route requires primary persistence; fresh resets retain JSON fallback.
            mark(e)
            monkeypatch.setattr(e.db, "replace_gateway_routing_entries",
                                Mock(side_effect=OSError("primary unavailable")))

    class ModelBoundary(_CompressionThenFailureAgent):
        def run_conversation(self, user_message, conversation_history=None, task_id=None, **kwargs):
            model_inputs.append(user_message)
            if len(model_inputs) == 1:
                asyncio.run_coroutine_threadsafe(enqueue_waiting(), loop).result(5)
            terminal = len(model_inputs) == 1
            response = "context full" if terminal else "completed"
            return {
                "final_response": response, "failed": terminal, "completed": not terminal,
                "compression_exhausted": terminal, "compression_deferred": False,
                "failure_reason": "context_overflow" if terminal else None,
                "pending_steer": steer if terminal else None,
                "messages": [*(conversation_history or []),
                             {"role": "user", "content": user_message},
                             {"role": "assistant", "content": response}],
                "api_calls": 1, "agent_persisted": False,
            }

    _install_compression_failure_agent(monkeypatch, ModelBoundary)
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda *_: "gpt-5.4")
    policy(e, f"compression:\n  exhaustion_action: {'pause' if action == 'pause' else 'reset'}\n")
    event = MessageEvent(text="first request", source=e.source, message_id="initial")
    try:
        await adapter.handle_message(event)
        while adapter._background_tasks:
            await asyncio.gather(*tuple(adapter._background_tasks))
            await asyncio.sleep(0)
        assert accepted._gateway_accepted
        if action == "reset":
            current = reload_entry(e)
            assert current is not None
            assert current.session_id != e.entry.session_id and not current.compression_paused
            assert "first request" in model_inputs[0]
            assert accepted.text in model_inputs[1]
            assert model_inputs[2:] == ["overflow second", steer]
            assert e.key not in adapter._pending_messages
            assert not r._overflow_queue(e.key)
        else:
            current = e.store.lookup_by_session_key(e.key)
            assert current.session_id == e.entry.session_id and current.compression_paused
            assert model_inputs == ["first request"]
            assert adapter._pending_messages.get(e.key) is accepted
            assert [item.text for item in r._overflow_queue(e.key)] == ["overflow second", steer]
    finally:
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["exhausted", "refused", "reset", "completed"])
async def test_fired_loop_tick_pauses_on_compression_without_rescheduling(
    native_env, monkeypatch, outcome,
):
    from hermes_cli.loops import LoopManager

    e = native_env
    monkeypatch.setenv("HERMES_HOME", str(e.tmp))  # LoopManager and SessionStore share the real state.db.
    del e.runner._post_turn_loop_completion
    del e.runner._run_post_turn_hooks
    manager = LoopManager(session_id=e.entry.session_id)
    manager.set("check the task", interval_seconds=60)
    assert manager.fire_tick() is not None
    assert manager.state is not None and manager.state.awaiting_response
    event = MessageEvent(text="loop wakeup", source=e.source)
    if outcome == "exhausted":
        policy(e, "compression:\n  exhaustion_action: pause\n")
        reply, _, event = await exhaust(e)
        assert "paused" in reply.lower() and getattr(event, "_gateway_skip_goal_continuation", False)
    elif outcome == "reset":
        policy(e, "compression:\n  exhaustion_action: reset\n")
        reply, _, event = await exhaust(e)
        reset_entry = reload_entry(e)
        assert reset_entry is not None
        assert "auto-reset" in reply and reset_entry.session_id != e.entry.session_id
    elif outcome == "refused":
        mark(e)
        policy(e, "compression:\n  exhaustion_action: pause\n")
        reply = await e.runner._handle_message(event)
        assert "paused" in str(reply).lower()
    else:
        reply = "completed the task"
    await e.runner._run_post_turn_hooks(agent_result=reply, source=e.source,
                                        is_internal=True, event=event)
    persisted = LoopManager(session_id=e.entry.session_id).state
    assert persisted is not None and persisted.ticks_fired == 1
    assert not persisted.awaiting_response
    if outcome == "completed":
        assert persisted.status == "active" and persisted.next_due_at > persisted.last_fired_at
    else:
        assert persisted.status == "paused"
        if outcome == "reset":
            reset_entry = reload_entry(e)
            assert reset_entry is not None
            assert LoopManager(session_id=reset_entry.session_id).state is None
        e.runner._post_turn_goal_continuation.assert_not_awaited()
