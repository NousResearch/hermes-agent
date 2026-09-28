"""Gateway owner-loop maintenance against a real routing index and profile-scoped SQLite."""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.control_socket import GatewayControlServer
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
    fence = _BindingFence(store, entry.session_key, entry.session_id)
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
