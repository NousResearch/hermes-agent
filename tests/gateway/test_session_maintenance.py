"""Gateway owner-loop maintenance against a real routing index and profile-scoped SQLite."""
import asyncio
import json
import os
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.control_socket import GatewayControlServer
from gateway.control_socket import _PipeControlProtocol
from gateway.run import GatewayRunner
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from gateway.session_maintenance import _BindingFence, maintain_existing_session, session_maintenance_verb
from gateway.turn_lease import SessionTurnLeaseRegistry
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def owner(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    token = set_hermes_home_override(str(home))
    config = GatewayConfig()
    store = SessionStore(home / "sessions", config)
    store._db = store._routing_db
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.session_store = store
    from hermes_state import AsyncSessionDB
    runner._session_db = AsyncSessionDB(store._db)
    runner._turn_leases = SessionTurnLeaseRegistry()
    runner._primary_profile_name = "default"
    runner._served_profile_homes = {"default": home}
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = {}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="user", chat_type="dm")
    yield runner, store, source
    store._db.close()
    reset_hermes_home_override(token)


def request(entry, action="inspect", **extra):
    return {"action": action, "profile": "default", "session_key": entry.session_key,
            "session_id": entry.session_id, "min_percent": 85, **extra}


@pytest.mark.asyncio
async def test_no_creation_busy_and_cas_usage_reset(owner):
    runner, store, source = owner
    key = store._generate_session_key(source)
    absent = await maintain_existing_session(runner, {"action": "inspect", "profile": "default",
        "session_key": key, "session_id": "missing", "min_percent": 85})
    assert absent == {"status": "stale_binding"}
    assert store.lookup_by_session_key(key) is None
    entry = store.get_or_create_session(source)
    store.update_session(key, last_prompt_tokens=1200, touch_activity=False)
    before = entry.updated_at
    store._db.update_session_model(entry.session_id, "test-model")
    assert (await maintain_existing_session(runner, request(entry)))["status"] == "unknown_usage"
    assert (await maintain_existing_session(runner, request(entry, session_id="wrong")))["status"] == "stale_binding"
    assert (await maintain_existing_session(runner, request(entry, profile="other")))["status"] == "profile_not_served"
    marker = store.mark_turn_active(key)
    assert (await maintain_existing_session(runner, request(entry)))["status"] == "busy"
    store.clear_turn_active(key, marker)
    lease = await runner._turn_leases.acquire(entry.session_id, owner_key="other", generation=1)
    assert (await maintain_existing_session(runner, request(entry)))["status"] == "busy"
    runner._turn_leases.release(lease)
    assert store.clear_prompt_usage_if_bound(key, "wrong") is False
    assert store.clear_prompt_usage_if_bound(key, entry.session_id) is True
    assert store.lookup_by_session_key(key).last_prompt_tokens == 0
    assert store.lookup_by_session_key(key).updated_at != before  # turn marker was real activity
    saved_at = store.lookup_by_session_key(key).updated_at
    assert store.clear_prompt_usage_if_bound(key, entry.session_id) is True
    assert store.lookup_by_session_key(key).updated_at == saved_at
    reopened = SessionStore(store.sessions_dir, runner.config)
    assert reopened.lookup_by_session_key(key).last_prompt_tokens == 0
    assert reopened.lookup_by_session_key(key).updated_at == saved_at


@pytest.mark.asyncio
async def test_threshold_codex_live_thread_and_socket_wire(owner, monkeypatch):
    runner, store, source = owner
    entry = store.get_or_create_session(source)
    sid, key = entry.session_id, entry.session_key
    store._db.update_session_model(sid, "test-model")
    store.update_session(key, last_prompt_tokens=900, touch_activity=False)
    initial_at = entry.updated_at
    ctx = SimpleNamespace(last_prompt_tokens=900, context_length=1000, compression_count=0)
    agent = SimpleNamespace(session_id=sid, model="test-model", context_compressor=ctx, _codex_session=object())
    def compact(*args, **kwargs):
        ctx.compression_count += 1
    agent._compress_context = compact
    runner._agent_cache[key] = (agent, "signature")
    runner._resolve_session_agent_runtime = lambda **kw: ("test-model", {"api_mode": "codex_app_server"})
    async def offload(fn):
        return await asyncio.to_thread(fn)
    runner._run_in_executor_with_context = offload
    below = await maintain_existing_session(runner, request(entry, min_percent=90))
    assert below["status"] == "below_threshold"
    assert below["used_tokens"] == 900
    assert (await maintain_existing_session(runner, request(entry)))["status"] == "above_threshold"
    agent._codex_session = None
    assert (await maintain_existing_session(runner, request(entry, "compact")))["status"] == "no_live_thread"
    agent._codex_session = object()
    compacted = await maintain_existing_session(runner, request(entry, "compact"))
    assert compacted["status"] == "compacted"
    assert ctx.compression_count == 1
    assert store.lookup_by_session_key(key).last_prompt_tokens == 0
    assert store.lookup_by_session_key(key).updated_at == initial_at
    assert store._db.get_session(sid)["id"] == sid
    # Socket handler is synchronous on an executor, but work runs on this owner loop.
    server = GatewayControlServer(home=store.sessions_dir.parent,
        verb_handlers={"session-maintenance": session_maintenance_verb(runner, asyncio.get_running_loop())})
    wire = await asyncio.to_thread(server.handle_request_line,
        json.dumps({"id": 1, "verb": "session-maintenance", "params": request(entry)}).encode())
    assert json.loads(wire)["result"]["status"] == "unknown_usage"


def test_fence_refuses_stale_tip_before_irreversible_commit(owner):
    _, store, source = owner
    entry = store.get_or_create_session(source)
    fence = _BindingFence(store, entry.session_key, entry.session_id, profile="default", topic_db=store._db)
    assert fence.begin_commit() is True
    fence.finish_commit()
    store.suspend_session(entry.session_key)
    assert fence.begin_commit() is False


@pytest.mark.asyncio
async def test_manual_in_place_path_uses_fence_and_keeps_routing_tip(owner, monkeypatch):
    from agent.conversation_compression_manual import CompressResult, CompressRequest
    import gateway.session_maintenance as maintenance

    runner, store, source = owner
    entry = store.get_or_create_session(source)
    sid, key = entry.session_id, entry.session_key
    for role in ("user", "assistant") * 5:
        store._db.append_message(sid, role=role, content="message")
    store._db.update_session_model(sid, "test-model")
    store.update_session(key, last_prompt_tokens=900, touch_activity=False)
    initial_at = entry.updated_at
    ctx = SimpleNamespace(last_prompt_tokens=900, context_length=1000, compression_count=0)
    live = SimpleNamespace(session_id=sid, model="test-model", context_compressor=ctx)
    runner._agent_cache[key] = (live, "signature")
    runner._resolve_session_agent_runtime = lambda **kw: ("test-model", {"api_key": "test-only"})
    runner._resolve_session_reasoning_config = lambda **kw: None
    tmp = SimpleNamespace(session_id=sid, _last_compaction_in_place=False,
                          context_compressor=SimpleNamespace())
    async def build(*args):
        return tmp
    async def cleanup(*args, **kwargs):
        pass
    runner._build_manual_compression_agent = build
    runner._cleanup_agent_resources_off_loop = cleanup
    runner._run_in_executor_with_context = lambda fn: asyncio.to_thread(fn)
    runner._evict_cached_agent = lambda key: runner._agent_cache.pop(key)
    def compress(agent, messages, request, **kwargs):
        assert agent.compression_in_place is True
        assert kwargs["commit_fence"].begin_commit() is True
        assert len(messages) >= 5
        agent._last_compaction_in_place = True
        # Test double for the compressor's durable active-row replacement.
        assert store.rewrite_transcript(sid, messages[:2])
        return CompressResult("compressed", messages, messages[:2], 900, 200, request)
    monkeypatch.setattr(maintenance, "compress_now", compress)
    result = await maintain_existing_session(runner, request(entry, "compact"))
    assert result["status"] == "compacted"
    assert store.lookup_by_session_key(key).session_id == sid
    assert store.lookup_by_session_key(key).updated_at == initial_at
    assert store.lookup_by_session_key(key).last_prompt_tokens == 0
    assert key not in runner._agent_cache


@pytest.mark.asyncio
async def test_route_switch_during_worker_does_not_clear_successor_usage(owner, monkeypatch):
    from agent.conversation_compression_manual import CompressResult
    import gateway.session_maintenance as maintenance

    runner, store, source = owner
    entry = store.get_or_create_session(source)
    old_sid, key = entry.session_id, entry.session_key
    for role in ("user", "assistant") * 5:
        store._db.append_message(old_sid, role=role, content="message")
    store._db.update_session_model(old_sid, "test-model")
    store.update_session(key, last_prompt_tokens=900, touch_activity=False)
    agent = SimpleNamespace(session_id=old_sid, model="test-model",
        context_compressor=SimpleNamespace(last_prompt_tokens=900, context_length=1000))
    runner._agent_cache[key] = (agent, "signature")
    runner._resolve_session_agent_runtime = lambda **kw: ("test-model", {"api_key": "test-only"})
    runner._resolve_session_reasoning_config = lambda **kw: None
    tmp = SimpleNamespace(session_id=old_sid, _last_compaction_in_place=False,
                          context_compressor=SimpleNamespace())
    async def build(*args):
        return tmp
    async def cleanup(*args, **kwargs):
        pass
    runner._build_manual_compression_agent = build
    runner._cleanup_agent_resources_off_loop = cleanup
    runner._run_in_executor_with_context = lambda fn: asyncio.to_thread(fn)
    def compress(agent, messages, req, **kwargs):
        # Simulate /resume repointing this key while summarization awaits a provider.
        successor = "successor"
        store._db.create_session(successor, source="telegram", model="test-model")
        assert store.switch_session(key, successor, expected_session_id=old_sid)
        store.update_session(key, last_prompt_tokens=777, touch_activity=False)
        assert kwargs["commit_fence"].begin_commit() is False
        return CompressResult("nothing_to_do", messages, messages, 900, 900, req)
    monkeypatch.setattr(maintenance, "compress_now", compress)
    assert (await maintain_existing_session(runner, request(entry, "compact")))["status"] == "not_compacted"
    assert store.lookup_by_session_key(key).session_id == "successor"
    assert store.lookup_by_session_key(key).last_prompt_tokens == 777


@pytest.mark.asyncio
async def test_pipe_protocol_keeps_owner_loop_alive_during_sync_verb():
    loop = asyncio.get_running_loop()
    server = GatewayControlServer(verb_handlers={"wait": lambda params:
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0, result={"done": True}), loop).result(timeout=3)})
    class Transport:
        def __init__(self):
            self.writes = []
            self.closed = asyncio.Event()
        def write(self, data):
            self.writes.append(data)
        def close(self):
            self.closed.set()
    transport = Transport()
    protocol = _PipeControlProtocol(server)
    protocol.connection_made(transport)
    protocol.data_received(b'{"verb":"wait","params":{},"id":1}\n')
    await asyncio.wait_for(transport.closed.wait(), 5)
    assert json.loads(transport.writes[0])["result"] == {"done": True}


@pytest.mark.asyncio
async def test_topic_row_change_without_route_change_refuses_commit(owner):
    runner, store, _ = owner
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="user",
                           chat_type="dm", thread_id="topic-1")
    entry = store.get_or_create_session(source)
    store._db.enable_telegram_topic_mode(chat_id="chat", user_id="user", profile_name="default")
    store._db.bind_telegram_topic(chat_id="chat", thread_id="topic-1", user_id="user",
                                  session_key=entry.session_key, session_id=entry.session_id)
    from gateway.session_maintenance import _topic_bound
    assert _topic_bound(entry, "default", store._db)
    assert store._routing_db.db_path == store._db.db_path
    assert (await maintain_existing_session(runner, request(entry)))["status"] == "unknown_usage"
    fence = _BindingFence(store, entry.session_key, entry.session_id, profile="default", topic_db=store._db)
    store._db.delete_telegram_topic_binding(chat_id="chat", thread_id="topic-1")
    assert store.lookup_by_session_key(entry.session_key).session_id == entry.session_id
    assert fence.begin_commit() is False
    assert (await maintain_existing_session(runner, request(entry)))["status"] == "stale_binding"


@pytest.mark.asyncio
async def test_topic_cross_database_route_fails_closed(owner, tmp_path):
    from hermes_state import AsyncSessionDB, SessionDB
    runner, store, _ = owner
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="user",
                           chat_type="dm", thread_id="topic-1")
    entry = store.get_or_create_session(source)
    other = SessionDB(tmp_path / "other.db")
    try:
        runner._session_db = AsyncSessionDB(other)
        assert (await maintain_existing_session(runner, request(entry)))["status"] == "stale_binding"
        assert store.lookup_by_session_key(entry.session_key).session_id == entry.session_id
    finally:
        other.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["route", "topic"])
async def test_codex_real_compressor_fence_blocks_stale_server_compact(owner, changed):
    from agent.conversation_compression import compress_context
    runner, store, source = owner
    if changed == "topic":
        source.thread_id = "topic-1"
    entry = store.get_or_create_session(source)
    sid, key = entry.session_id, entry.session_key
    if changed == "topic":
        store._db.enable_telegram_topic_mode(chat_id="chat", user_id="user")
        store._db.bind_telegram_topic(chat_id="chat", thread_id="topic-1", user_id="user",
                                      session_key=key, session_id=sid)
    store._db.update_session_model(sid, "test-model")
    store.update_session(key, last_prompt_tokens=900, touch_activity=False)
    compact_thread = Mock()
    ctx = SimpleNamespace(last_prompt_tokens=900, context_length=1000, compression_count=0)
    agent = SimpleNamespace(session_id=sid, model="test-model", api_mode="codex_app_server",
        context_compressor=ctx, _cached_system_prompt="cached",
        _codex_session=SimpleNamespace(compact_thread=compact_thread))
    def compact(messages, prompt, **kw):
        successor = "successor"
        store._db.create_session(successor, source="telegram", model="test-model")
        if changed == "route":
            assert store.switch_session(key, successor, expected_session_id=sid)
        else:
            store._db.bind_telegram_topic(chat_id="chat", thread_id="topic-1", user_id="user",
                                          session_key=key, session_id=successor)
            assert store.lookup_by_session_key(key).session_id == sid
        return compress_context(agent, messages, prompt, **kw)
    agent._compress_context = compact
    runner._agent_cache[key] = (agent, "signature")
    runner._resolve_session_agent_runtime = lambda **kw: ("test-model", {"api_mode": "codex_app_server"})
    runner._run_in_executor_with_context = lambda fn: asyncio.to_thread(fn)
    assert (await maintain_existing_session(runner, request(entry, "compact")))["status"] == "not_compacted"
    compact_thread.assert_not_called()
    assert store.lookup_by_session_key(key).session_id == ("successor" if changed == "route" else sid)


def _native_maintenance_agent(db, sid):
    """Real compressor and SQLite persistence, with only a fake key and no context-file reads."""
    from run_agent import AIAgent
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                        model="test/model", quiet_mode=True, session_db=db, session_id=sid,
                        skip_context_files=True, skip_memory=True)
    agent._compression_feasibility_checked = True
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("stale", [False, True], ids=["archive", "stale-before-commit"])
async def test_native_maintenance_archives_only_under_current_binding(owner, monkeypatch, stale):
    """Exercise the gateway's real compress_now -> AIAgent -> SQLite archive chain."""
    from hermes_state import SessionDB
    import agent.context_compressor as compressor_module
    import gateway.session_maintenance as maintenance

    runner, store, source = owner
    entry = store.get_or_create_session(source)
    sid, key = entry.session_id, entry.session_key
    store._db.update_session_model(sid, "test/model")
    original = []
    for i in range(10):
        for role, content in (("user", f"question {i} about fruit{i} " + "filler " * 40),
                              ("assistant", f"answer {i} " + "lorem " * 400)):
            content = content.strip()
            store._db.append_message(sid, role=role, content=content)
            original.append(content)
    store.update_session(key, last_prompt_tokens=900, touch_activity=False)
    initial_at = store.lookup_by_session_key(key).updated_at
    resident = _native_maintenance_agent(store._db, sid)
    resident.context_compressor.context_length = 1000
    resident.context_compressor.last_prompt_tokens = 900
    runner._agent_cache[key] = (resident, "signature")
    runner._resolve_session_agent_runtime = lambda **kw: ("test/model", {"api_key": "test-key"})
    runner._resolve_session_reasoning_config = lambda **kw: None
    runner._run_in_executor_with_context = lambda fn: asyncio.to_thread(fn)
    runner._evict_cached_agent = lambda key: runner._agent_cache.pop(key)
    runner._cleanup_agent_resources_off_loop = lambda *a, **kw: asyncio.sleep(0)
    built = []
    async def build(session_id, model, runtime):
        assert session_id == sid and model == "test/model"
        assert runtime["gateway_session_key"] == key
        agent = _native_maintenance_agent(store._db, sid)
        built.append(agent)
        return agent
    runner._build_manual_compression_agent = build
    adapter = Mock()
    runner.adapters = {Platform.TELEGRAM: adapter}

    summary = "## Goal\nNumbered fruit questions.\n## Progress\nEarly ones answered."
    calls = []
    def synthetic_llm(**kwargs):
        calls.append(kwargs)
        if stale:
            # The provider returns after the route changed but before durable admission.
            store._db.create_session("successor", source="telegram", model="test/model")
            assert store.switch_session(key, "successor", expected_session_id=sid)
            store.update_session(key, last_prompt_tokens=777, touch_activity=False)
        response = Mock()
        response.choices = [Mock(message=Mock(content=summary))]
        return response
    monkeypatch.setattr(compressor_module, "call_llm", synthetic_llm)
    admissions = []
    original_begin = maintenance._BindingFence.begin_commit
    def observed_begin(self, cancel_event=None):
        admitted = original_begin(self, cancel_event)
        admissions.append(admitted)
        return admitted
    monkeypatch.setattr(maintenance._BindingFence, "begin_commit", observed_begin)

    result = await maintain_existing_session(runner, request(entry, "compact"))
    assert len(calls) == 1  # a synthetic provider response, not a mocked compressor
    assert admissions == [not stale]  # refusal occurs at the real irreversible-commit gate
    assert len(built) == 1 and built[0].compression_in_place is True
    assert built[0].session_id == sid
    assert adapter.mock_calls == []  # no Telegram delivery, even on the successful branch

    # Reopen SQLite rather than trusting the gateway's in-memory transcript cache.
    reopened = SessionDB(db_path=store._db.db_path)
    reopened_route = SessionStore(store.sessions_dir, runner.config)
    try:
        rows = reopened._conn.execute(
            "SELECT content, active, compacted FROM messages WHERE session_id = ? ORDER BY id", (sid,)
        ).fetchall()
        active = reopened.get_messages_as_conversation(sid)
        resume, display = reopened.get_resume_conversations(sid)
        if stale:
            assert result["status"] == "not_compacted"
            assert [(row[0], row[1]) for row in rows] == [(content, 1) for content in original]
            assert [m["content"] for m in active] == original
            assert [m["content"] for m in resume] == original
            assert store.lookup_by_session_key(key).session_id == "successor"
            assert store.lookup_by_session_key(key).last_prompt_tokens == 777
            assert reopened_route.lookup_by_session_key(key).session_id == "successor"
        else:
            assert result["status"] == "compacted"
            assert all(any(row[0] == content and row[1] == 0 for row in rows)
                       for content in original)
            assert any(row[1] == 0 and row[2] == 1 for row in rows)
            assert any("Numbered fruit questions" in m["content"] for m in active)
            assert [m["content"] for m in resume] == [m["content"] for m in active]
            assert any(m["content"] == original[0] for m in display)
            assert len(active) < len(original)
            assert store.lookup_by_session_key(key).session_id == sid
            assert store.lookup_by_session_key(key).last_prompt_tokens == 0
            assert store.lookup_by_session_key(key).updated_at == initial_at
            assert reopened_route.lookup_by_session_key(key).session_id == sid
            assert reopened_route.lookup_by_session_key(key).updated_at == initial_at
            assert reopened.get_session(sid)["id"] == sid
    finally:
        reopened_route._db.close()
        reopened.close()
