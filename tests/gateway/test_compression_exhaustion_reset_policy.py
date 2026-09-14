"""Durable gateway pause: real routing DB, transcript and cancellation boundaries."""
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

KEY = "compression_exhausted"
HISTORY = [
    {"role": "user", "content": "inspect", "timestamp": 1700000000.0},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call-1", "type": "function", "function": {"name": "test", "arguments": "{}"}}]},
    {"role": "tool", "content": "evidence", "tool_call_id": "call-1"},
    {"role": "assistant", "content": "done"},
]


@pytest.fixture
def env(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("gateway.run._gateway_config_home", lambda: home)
    monkeypatch.setattr("hermes_cli.managed_scope.get_managed_dir", lambda: None)
    db = SessionDB(db_path=tmp_path / "state.db")
    config = GatewayConfig()
    store = SessionStore(tmp_path / "sessions", config)
    store._db = db
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="test-chat", user_id="test-user")
    entry = store.get_or_create_session(source)
    assert store.rewrite_transcript(entry.session_id, HISTORY)
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.session_store = store
    runner._session_db = AsyncSessionDB(db)
    runner._evict_cached_agent = Mock()
    runner._clear_conversation_scope = Mock()
    runner._sync_telegram_topic_binding = Mock()
    runner._recover_telegram_topic_thread_id = Mock(return_value=None)
    runner._is_telegram_topic_lane = Mock(return_value=False)
    runner._cache_session_source = Mock()
    runner._hmwa_prepare_turn = AsyncMock(return_value=("prepared", None))
    runner._run_agent = AsyncMock()
    runner._post_turn_goal_continuation = AsyncMock()
    runner._post_turn_loop_completion = AsyncMock()
    runner._normalize_source_for_session_key = lambda src: src
    runner._persist_active_agents = Mock()
    runner._release_running_agent_state = Mock(wraps=runner._release_running_agent_state)
    runner._invalidate_session_run_generation = Mock()
    runner._is_session_run_current = Mock(return_value=True)
    runner._cleanup_old_agent_for_reset = AsyncMock()
    runner._fire_session_reset_hooks = AsyncMock()
    runner._reset_notice_session_info = Mock(return_value="")
    runner._telegram_topic_new_header = Mock(return_value=None)
    runner._record_telegram_topic_binding = Mock()
    e = SimpleNamespace(runner=runner, store=store, db=db, source=source, entry=entry,
                        key=entry.session_key, home=home, tmp=tmp_path)
    yield e
    runner._shutdown_executor()
    store.close_all_db_handles()
    db.close()


def policy(e, value):
    path = e.home / "config.yaml"
    if value is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(value)


def mark(e):
    assert e.store.set_session_metadata(e.key, KEY, True, require_primary=True,
                                        expected_session_id=e.entry.session_id)


def reload_entry(e):
    db = SessionDB(db_path=e.tmp / "state.db")
    store = SessionStore(e.store.sessions_dir, e.store.config)
    store._db = db
    try:
        return store.lookup_by_session_key(e.key)
    finally:
        store.close_all_db_handles()
        db.close()


async def admit(e):
    event = MessageEvent(text="continue", source=e.source)
    reply = await e.runner._handle_message_with_agent(event, e.source, e.key, 1)
    return reply, event


async def exhaust(e, **flags):
    event = MessageEvent(text="work", source=e.source)
    result = {"compression_exhausted": True, **flags}
    reply, entry = await e.runner._hmwa_compression_exhaustion_reset(
        result, "context full", e.entry, e.key, e.source, event=event)
    return reply, entry, event


@pytest.mark.asyncio
async def test_pause_survives_reload_preserves_history_and_refuses_before_prepare(env):
    e = env
    policy(e, "compression:\n  exhaustion_action: pause\n")
    before = e.db.get_messages(e.entry.session_id)
    other = e.store.get_or_create_session(replace(e.source, chat_id="other"))
    reply, entry, event = await exhaust(e)
    assert entry is e.entry and "paused" in reply.lower()
    assert event._gateway_skip_goal_continuation is True
    assert reload_entry(e).compression_paused
    assert e.db.get_messages(entry.session_id) == before
    assert not other.compression_paused
    e.store._entries[e.key] = reload_entry(e)
    reply, _ = await admit(e)
    assert "paused" in reply.lower()
    e.runner._hmwa_prepare_turn.assert_not_awaited()
    e.runner._run_agent.assert_not_awaited()
    assert e.db.get_messages(entry.session_id) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [None, "{}", "compression: []", "compression:\n  exhaustion_action: invalid", "[bad", "compression:\n  exhaustion_action: reset"])
async def test_new_exhaustion_retains_native_reset_default(env, raw):
    policy(env, raw)
    reply, entry, _ = await exhaust(env)
    assert entry.session_id != env.entry.session_id
    assert "auto-reset" in reply
    assert reload_entry(env).session_id == entry.session_id


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [None, "{}", "compression: []", "compression:\n  exhaustion_action: invalid", "[bad", "compression:\n  exhaustion_action: pause"])
async def test_existing_pause_is_sticky_without_positive_reset(env, raw):
    mark(env)
    policy(env, raw)
    reply, _ = await admit(env)
    assert "paused" in reply.lower()
    assert reload_entry(env).compression_paused
    env.runner._hmwa_prepare_turn.assert_not_awaited()
    env.runner._run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_reset_recovers_without_running_triggering_input(env):
    mark(env)
    policy(env, "compression:\n  exhaustion_action: reset")
    reply, event = await admit(env)
    assert "reset" in reply.lower()
    assert reload_entry(env).session_id != env.entry.session_id
    assert not reload_entry(env).compression_paused
    assert event._gateway_skip_goal_continuation
    env.runner._hmwa_prepare_turn.assert_not_awaited()
    env.runner._run_agent.assert_not_awaited()
    assert (await admit(env))[0] == "prepared"


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["invalid", "read"])
async def test_managed_uncertainty_cannot_authorize_existing_pause_reset(env, monkeypatch, fault):
    mark(env)
    policy(env, "compression:\n  exhaustion_action: reset")
    managed = env.tmp / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text("[bad" if fault == "invalid" else "compression:\n  exhaustion_action: reset")
    monkeypatch.setattr("hermes_cli.managed_scope.get_managed_dir", lambda: managed)
    if fault == "read":
        original = type(managed).read_text
        def read(path, *args, **kwargs):
            if path == managed / "config.yaml":
                raise PermissionError("injected managed read failure")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(type(managed), "read_text", read)
    assert "paused" in (await admit(env))[0].lower()
    assert reload_entry(env).compression_paused


@pytest.mark.asyncio
async def test_policy_reads_source_profile_off_loop_and_managed_leaf_wins(env, monkeypatch):
    profile = env.tmp / "profile"
    profile.mkdir()
    (profile / "config.yaml").write_text("compression:\n  exhaustion_action: pause")
    policy(env, "compression:\n  exhaustion_action: reset")
    env.runner.config.multiplex_profiles = True
    env.runner._resolve_profile_home_for_source = Mock(return_value=profile)
    loop_thread = threading.get_ident()
    original = type(profile).read_text
    reads = []
    def read(path, *args, **kwargs):
        reads.append((path, threading.get_ident()))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(type(profile), "read_text", read)
    assert await env.runner._compression_exhaustion_action(env.source) == "pause"
    managed = env.tmp / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text("compression:\n  exhaustion_action: reset")
    monkeypatch.setattr("hermes_cli.managed_scope.get_managed_dir", lambda: managed)
    assert await env.runner._compression_exhaustion_action(env.source) == "reset"
    assert all(t != loop_thread for _, t in reads)
    assert all(p != env.home / "config.yaml" for p, _ in reads)


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_kind,ref", [
    ("single", "${POLICY_TEST_ACTION}"),
    ("secondary", "${POLICY_TEST_ACTION}"),
    ("primary", "${env:POLICY_TEST_ACTION}"),
])
@pytest.mark.parametrize("value,managed_leaf,paused", [
    ("reset", None, True), ("pause", None, False), ("pause", None, True),
    (None, None, True), (None, None, False), ("invalid", None, True),
    ("reset", "ref", True), ("pause", "ref", False), (None, "ref", True),
    ("pause", "reset", True), ("reset", "pause", True), ("pause", "sibling", False),
])
async def test_native_handler_policy_env_contract(
    native_env, monkeypatch, handler_kind, ref, value, managed_leaf, paused,
):
    import os
    from agent import secret_scope
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_constants import get_hermes_home
    import yaml

    e = native_env
    multiplex = handler_kind != "single"
    monkeypatch.setattr(type(e.home), "home", lambda: e.tmp)
    monkeypatch.setenv("HERMES_HOME", str(e.home))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", multiplex)
    e.runner.config.multiplex_profiles = multiplex
    profile = e.home / "profiles" / "policy-test" if multiplex else e.home
    profile.mkdir(parents=True, exist_ok=True)
    ambient = ("pause" if value == "reset" else "reset") if multiplex else value
    if ambient is None:
        monkeypatch.delenv("POLICY_TEST_ACTION", raising=False)
    else:
        monkeypatch.setenv("POLICY_TEST_ACTION", ambient)
    (profile / ".env").write_text("" if value is None else f"POLICY_TEST_ACTION={value}\n")
    user_leaf = ref if managed_leaf in {None, "sibling"} else "pause" if value == "reset" else "reset"
    (profile / "config.yaml").write_text(yaml.safe_dump({"compression": {"exhaustion_action": user_leaf}}))
    if multiplex:
        policy(e, "compression:\n  exhaustion_action: reset")
        e.source = replace(e.source, profile="policy-test")
        e.entry = e.store.get_or_create_session(e.source)
        e.key = e.entry.session_key
        assert e.store.rewrite_transcript(e.entry.session_id, HISTORY)
        assert e.runner._resolve_profile_home_for_source(e.source) == profile
    if managed_leaf is not None:
        managed = e.tmp / "managed"
        managed.mkdir()
        compression = ({"threshold": 0.5} if managed_leaf == "sibling" else
                       {"exhaustion_action": ref if managed_leaf == "ref" else managed_leaf})
        (managed / "config.yaml").write_text(yaml.safe_dump({"compression": compression}))
        monkeypatch.setattr("hermes_cli.managed_scope.get_managed_dir", lambda: managed)
    expected = managed_leaf if managed_leaf in {"reset", "pause"} else value
    expected = expected if expected in {"reset", "pause"} else None
    original = e.runner._compression_exhaustion_action
    reads = []

    async def checked_policy(source):
        # Exercise the real adapter-installed scope, not a prepared secret mapping.
        assert get_hermes_home() == profile
        if multiplex:
            scope = secret_scope.current_secret_scope()
            assert scope is not None and scope.get("POLICY_TEST_ACTION") == value
        native = await asyncio.to_thread(load_user_config_effective, profile / "config.yaml", fail_closed=True)
        native_choice = native["compression"]["exhaustion_action"]
        assert (native_choice if native_choice in {"reset", "pause"} else None) == expected
        actual = await original(source)
        reads.append(actual)
        assert actual == expected
        return actual

    e.runner._compression_exhaustion_action = checked_policy
    if paused:
        mark(e)
    else:
        # Supply exhaustion without running a model; ingress scope and policy/storage stay real.
        async def exhausted_turn(event, source, entry, key, quick_key, generation):
            reply, _ = await e.runner._hmwa_compression_exhaustion_reset(
                {"compression_exhausted": True}, "context full", entry, key, source,
                event=event, run_generation=generation)
            return reply, None
        e.runner._hmwa_prepare_turn.side_effect = exhausted_turn
    handler = (e.runner._make_profile_message_handler("policy-test") if handler_kind == "secondary"
               else e.runner._primary_message_handler())
    if handler_kind == "secondary":
        e.source.profile = None  # The native adapter handler must stamp this before routing.
    reply = await handler(MessageEvent(text="continue", source=e.source))
    assert reads == [expected]
    reset = expected == "reset" if paused else expected != "pause"
    persisted = reload_entry(e)
    assert persisted is not None
    assert (persisted.session_id != e.entry.session_id) is reset
    assert persisted.compression_paused is (not reset)
    assert ("Session paused:" in reply) is (not reset)
    assert e.db.get_messages(e.entry.session_id)
    assert os.environ.get("POLICY_TEST_ACTION") == ambient
    if paused:
        e.runner._hmwa_prepare_turn.assert_not_awaited()
    e.runner._run_agent.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("layer", ["user", "managed"])
@pytest.mark.parametrize("fault", ["missing", "syntax", "root", "section", "invalid", "read"])
async def test_env_policy_uncertainty_ignores_warm_config_cache(env, monkeypatch, layer, fault):
    import os
    from hermes_cli.config_effective import load_user_config_effective

    e = env
    mark(e)
    monkeypatch.setenv("POLICY_TEST_ACTION", "reset")
    target = e.home / "config.yaml"
    policy(e, "compression: {}")
    if layer == "managed":
        managed = e.tmp / "managed"
        managed.mkdir()
        target = managed / "config.yaml"
        monkeypatch.setattr("hermes_cli.managed_scope.get_managed_dir", lambda: managed)
    target.write_text("compression:\n  exhaustion_action: ${env:POLICY_TEST_ACTION}\n")
    native = await asyncio.to_thread(load_user_config_effective, e.home / "config.yaml", fail_closed=True)
    assert native["compression"]["exhaustion_action"] == "reset"
    before = target.stat()
    if fault == "missing":
        target.unlink()
    elif fault == "read":
        original = type(target).read_text
        def unreadable(path, *args, **kwargs):
            if path == target:
                raise PermissionError("injected policy read failure")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(type(target), "read_text", unreadable)
    else:
        raw = {"syntax": "[bad", "root": "[]", "section": "compression: []",
               "invalid": "compression:\n  exhaustion_action: invalid"}[fault]
        target.write_text(raw.ljust(before.st_size))
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert (target.stat().st_size, target.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    assert await e.runner._compression_exhaustion_action(e.source) is None
    assert "paused" in (await admit(e))[0].lower()
    persisted = reload_entry(e)
    assert persisted is not None and persisted.compression_paused
    e.runner._hmwa_prepare_turn.assert_not_awaited()
    e.runner._run_agent.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("exhausted", [False, True])
async def test_deferred_never_pauses_or_resets(env, exhausted):
    policy(env, "compression:\n  exhaustion_action: pause")
    before = env.entry.to_dict()
    _, entry, event = await exhaust(env, compression_exhausted=exhausted, compression_deferred=True)
    assert entry.to_dict() == before
    assert not getattr(event, "_gateway_skip_goal_continuation", False)


@pytest.mark.parametrize("operation", ["mark", "clear", "reset", "rotate", "switch", "force_new"])
def test_primary_failure_preserves_candidate_and_generation(env, monkeypatch, operation):
    e = env
    if operation != "mark":
        mark(e)
    before = copy.deepcopy(e.entry.to_dict())
    generation = e.store._persisted_routing_generation
    fast = copy.deepcopy(e.store._fast_persisted_entries)
    mirror = Mock()
    monkeypatch.setattr(e.store, "_save_sessions_json", mirror)
    monkeypatch.setattr(e.db, "replace_gateway_routing_entries", Mock(side_effect=OSError("primary")))
    with pytest.raises(OSError):
        if operation in {"mark", "clear"}:
            e.store.set_session_metadata(e.key, KEY, operation == "mark", require_primary=True)
        elif operation == "reset":
            e.store.reset_session(e.key)
        elif operation == "rotate":
            e.store.commit_manual_compression(e.key, e.entry.session_id, "child")
        elif operation == "switch":
            e.store.switch_session(e.key, "other")
        else:
            e.store.get_or_create_session(e.source, force_new=True)
    assert e.store._entries[e.key] is e.entry
    assert e.entry.to_dict() == before
    assert e.entry.compression_paused
    assert e.store._persisted_routing_generation == generation
    assert e.store._fast_persisted_entries == fast
    mirror.assert_not_called()


@pytest.mark.asyncio
async def test_failed_initial_mark_holds_process_without_claiming_durability(env, monkeypatch):
    policy(env, "compression:\n  exhaustion_action: pause")
    monkeypatch.setattr(env.db, "replace_gateway_routing_entries", Mock(side_effect=OSError("primary")))
    reply, _, _ = await exhaust(env)
    assert "persist" in reply.lower()
    assert env.entry.compression_paused
    assert not env.entry.metadata.get(KEY)
    assert not reload_entry(env).compression_paused
    await admit(env)
    env.runner._hmwa_prepare_turn.assert_not_awaited()


@pytest.mark.parametrize("operation", ["mark", "clear", "reset", "rotate", "switch"])
@pytest.mark.parametrize("mirror_enabled", [True, False])
def test_primary_commit_owns_truth_even_when_optional_mirror_fails(env, monkeypatch, operation, mirror_enabled):
    e = env
    mark(e)
    updated = e.entry.updated_at
    e.store._write_sessions_json = mirror_enabled
    mirror = Mock(side_effect=OSError("mirror"))
    monkeypatch.setattr(e.store, "_save_sessions_json", mirror)
    primary = Mock(wraps=e.db.replace_gateway_routing_entries)
    monkeypatch.setattr(e.db, "replace_gateway_routing_entries", primary)
    if operation in {"mark", "clear"}:
        e.store.set_session_metadata(e.key, KEY, operation == "mark", require_primary=True)
    elif operation == "reset":
        e.store.reset_session(e.key)
    else:
        e.db.create_session("child", source="telegram")
        if operation == "rotate":
            assert e.store.commit_manual_compression(e.key, e.entry.session_id, "child")
        else:
            e.store.switch_session(e.key, "child")
    current = e.store._entries[e.key]
    assert reload_entry(e).to_dict() == current.to_dict()
    assert current.compression_paused == (operation == "mark")
    assert primary.call_count == 1
    assert mirror.call_count == int(mirror_enabled)
    if operation in {"mark", "clear", "rotate"}:
        assert current.updated_at == updated


def test_strict_missing_stale_and_moved_writes_cannot_report_commit(env):
    e = env
    assert e.store.set_session_metadata("missing", KEY, True, require_primary=True) is False
    assert e.store.set_session_metadata(e.key, KEY, True, require_primary=True, expected_session_id="moved") is False
    assert e.store.commit_manual_compression(e.key, "moved", "child") is False
    with pytest.raises(RuntimeError):
        e.store._persist_routing_data({}, e.store._persisted_routing_generation, require_primary=True)


def test_strict_missing_primary_refuses_but_native_fallback_still_works(env):
    env.store._db = None
    with pytest.raises(RuntimeError):
        env.store.set_session_metadata(env.key, KEY, True, require_primary=True)
    env.store.set_session_metadata(env.key, "native", 7)
    assert json.loads((env.store.sessions_dir / "sessions.json").read_text())[env.key]["metadata"]["native"] == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("authority", ["marker", "event", "streamed"])
async def test_goal_hook_is_suppressed_but_loop_bookkeeping_runs(env, authority):
    event = MessageEvent(text="next", source=env.source)
    if authority == "marker":
        mark(env)
    else:
        event._gateway_skip_goal_continuation = True
    if authority == "streamed":
        event._streamed_final_response = "context full"
    await env.runner._run_post_turn_hooks(agent_result=None if authority == "streamed" else "context full",
                                          source=env.source, is_internal=False, event=event)
    env.runner._post_turn_goal_continuation.assert_not_awaited()
    env.runner._post_turn_loop_completion.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("deferred", [False, True, None])
async def test_terminal_exhaustion_does_not_drain_pending_input(env, deferred):
    r = env.runner
    result = None if deferred is None else {"compression_exhausted": True, "compression_deferred": deferred}
    ctx = SimpleNamespace(result_holder=[result], streaming_tts_consumer_holder=[None], stream_consumer_holder=[None])
    disp = SimpleNamespace(_native_slack_task_cards=False, needs_progress_queue=False, log_mode_enabled=False)
    r._get_proxy_url = lambda: None
    r._run_agent_display_settings = lambda source: disp
    r._run_agent_build_turn_context = Mock(return_value=(ctx, SimpleNamespace(run_sync=Mock()), None))
    r._run_agent_bind_turn_wiring = Mock()
    r._run_agent_start_streaming_tts = Mock()
    for name in ("stream_consumer_task", "track_agent", "monitor_for_interrupt", "notify_long_running", "finalize_streaming_tts", "mark_streamed_delivery"):
        setattr(r, "_run_agent_" + name, AsyncMock())
    r._run_agent_start_turn_worker = Mock(return_value=SimpleNamespace(executor_task=None))
    r._run_agent_await_turn_worker = AsyncMock(return_value=result)
    r._run_agent_evict_on_fallback = Mock()
    r._adapter_for_source = Mock(return_value=SimpleNamespace(_pending_messages={env.key: "queued"}))
    r._run_agent_drain_pending = AsyncMock(return_value=(None, None if deferred is None else "queued"))
    r._run_agent_queued_followup = AsyncMock(return_value={"followup": True})
    r._run_agent_schedule_bubble_cleanup = Mock()
    async def cleanup(ctx, **tasks):
        for task in tasks.values():
            if task is not None:
                await task
    r._run_agent_cleanup_turn_tasks = cleanup
    await r._run_agent_inner("work", "", [], env.source, env.entry.session_id, env.key)
    assert r._run_agent_drain_pending.await_count == int(deferred is not False)
    assert r._run_agent_queued_followup.await_count == int(deferred is True)
    assert r._adapter_for_source()._pending_messages[env.key] == "queued"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["rotate", "inplace", "noop", "rewrite_fail", "missing_db", "primary_fail", "moved"])
async def test_manual_commit_matrix_preserves_parent_and_clears_only_on_commit(env, monkeypatch, mode):
    e = env
    mark(e)
    before = e.db.get_messages(e.entry.session_id)
    old_id = e.entry.session_id
    e.entry.last_prompt_tokens = 987
    if mode == "inplace":
        # Simulate the compressor's real committed archive, rather than a truthy mock flag.
        e.db.archive_and_compact(old_id, [{"role": "user", "content": "summary"}])
        archived = e.db.get_messages(old_id, include_compacted=True)
    child = old_id if mode in {"inplace", "noop"} else "child"
    if child != old_id:
        e.db.create_session(child, source="telegram")
    agent = SimpleNamespace(session_id=child, _last_compaction_in_place=mode == "inplace")
    if mode == "rewrite_fail":
        monkeypatch.setattr(e.store, "rewrite_transcript", lambda *args: False)
    if mode == "missing_db":
        monkeypatch.setattr(e.store, "_db_for_session_id", lambda *args: None)
    if mode == "primary_fail":
        monkeypatch.setattr(e.db, "replace_gateway_routing_entries", Mock(side_effect=OSError("primary")))
    if mode == "moved":
        e.store.reset_session(e.key)
    if mode in {"rewrite_fail", "missing_db", "primary_fail"}:
        with pytest.raises((RuntimeError, OSError)):
            await e.runner._persist_manual_compression(agent, e.entry, e.source, HISTORY)
    else:
        committed = await e.runner._persist_manual_compression(agent, e.entry, e.source, HISTORY)
        assert committed is (mode in {"rotate", "inplace"})
    if mode in {"rotate", "inplace"}:
        assert not reload_entry(e).compression_paused
        assert e.entry.last_prompt_tokens == 0
    elif mode != "moved":
        assert reload_entry(e).compression_paused
        assert e.entry.last_prompt_tokens == 987
    if mode == "inplace":
        assert e.db.get_messages(old_id, include_compacted=True) == archived
    else:
        assert e.db.get_messages(old_id) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["success", "noop", "failure", "primary_fail"])
async def test_codex_count_receipt_controls_clear_without_eviction(env, monkeypatch, mode):
    mark(env)
    compressor = SimpleNamespace(compression_count=2)
    def compact(*args, **kwargs):
        if mode == "failure":
            raise RuntimeError("compress failed")
        if mode in {"success", "primary_fail"}:
            compressor.compression_count += 1
    agent = SimpleNamespace(_codex_session=object(), context_compressor=compressor, _compress_context=compact)
    env.runner._cached_agent_for = Mock(return_value=agent)
    if mode == "primary_fail":
        monkeypatch.setattr(env.db, "replace_gateway_routing_entries", Mock(side_effect=OSError("primary")))
    reply = await env.runner._compress_codex_app_server_session(env.key, env.entry.session_id)
    assert reload_entry(env).compression_paused == (mode != "success")
    assert ("thread compacted" in reply) == (mode == "success")
    env.runner._evict_cached_agent.assert_not_called()


@pytest.mark.asyncio
async def test_telegram_stale_binding_cannot_bypass_pause(env):
    mark(env)
    env.runner._session_db = SimpleNamespace(get_telegram_topic_binding=AsyncMock(return_value={"session_id": "other"}))
    assert await env.runner._hmwa_heal_telegram_topic_binding(env.source, env.entry, env.key) is env.entry
    env.runner._session_db.get_telegram_topic_binding.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["same", "allowed", "denied", "primary_fail"])
async def test_resume_retains_native_authorization_and_commits_other_target(env, monkeypatch, mode):
    e = env
    mark(e)
    target = e.entry.session_id if mode == "same" else "target-session"
    if mode != "same":
        other_source = e.source if mode != "denied" else replace(e.source, user_id="foreign", chat_id="foreign")
        e.db.create_session(target, source="telegram", user_id=other_source.user_id,
                            chat_id=other_source.chat_id, thread_id=other_source.thread_id)
    e.runner._resolve_resume_target = AsyncMock(return_value=(target, "target"))
    e.runner._resume_caller_is_admin = Mock(return_value=False)
    if mode == "primary_fail":
        monkeypatch.setattr(e.db, "replace_gateway_routing_entries", Mock(side_effect=OSError("primary")))
    reply = await e.runner._handle_resume_command(MessageEvent(text="/resume target", source=e.source))
    if mode == "allowed":
        assert reload_entry(e).session_id == target
        assert not reload_entry(e).compression_paused
    else:
        assert reload_entry(e).session_id == e.entry.session_id
        assert reload_entry(e).compression_paused
    assert bool(reply)


def test_native_suspension_and_force_new_remain_available(env):
    mark(env)
    old = env.entry.session_id
    env.store.suspend_session(env.key)
    entry = env.store.get_or_create_session(env.source)
    assert entry.session_id != old
    assert not entry.compression_paused
    env.entry = entry
    mark(env)
    assert env.store.get_or_create_session(env.source, force_new=True).session_id != entry.session_id


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["mark", "clear", "manual", "reset"])
@pytest.mark.parametrize("boundary", ["primary", "mirror"])
async def test_cancellation_waits_for_storage_and_binding_tail(env, monkeypatch, operation, boundary):
    e = env
    if operation != "mark":
        mark(e)
    entered, release = threading.Event(), threading.Event()
    original = e.db.replace_gateway_routing_entries if boundary == "primary" else e.store._save_sessions_json
    calls = []
    def blocked(*args, **kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        if boundary == "mirror":
            raise OSError("mirror after commit")
        return original(*args, **kwargs)
    monkeypatch.setattr(e.db if boundary == "primary" else e.store,
                        "replace_gateway_routing_entries" if boundary == "primary" else "_save_sessions_json", blocked)
    if operation in {"mark", "clear"}:
        work = e.runner._await_session_policy_commit(e.runner.async_session_store.set_session_metadata(
            e.key, KEY, operation == "mark", require_primary=True))
    elif operation == "manual":
        agent = SimpleNamespace(session_id=e.entry.session_id, _last_compaction_in_place=True)
        work = e.runner._persist_manual_compression(agent, e.entry, e.source, HISTORY)
    else:
        work = e.runner._reset_session_after_compression_exhaustion(e.key, e.entry, e.source)
    task = asyncio.create_task(work)
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "cancellation released admission with a storage worker still live"
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(calls) == 1
    assert reload_entry(e).compression_paused == (operation == "mark")
    if operation in {"manual", "reset"}:
        e.runner._sync_telegram_topic_binding.assert_called_once()

@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["success", "primary_fail", "missing"])
async def test_new_command_recovery_reports_only_committed_route(env, monkeypatch, mode):
    mark(env)
    if mode == "primary_fail":
        monkeypatch.setattr(env.db, "replace_gateway_routing_entries", Mock(side_effect=OSError("primary")))
    if mode == "missing":
        monkeypatch.setattr(env.store, "reset_session", Mock(return_value=None))
    with patch("gateway.slash_commands_session._reset_process_scoped_tool_state"), patch(
        "hermes_cli.lifecycle.invoke_hook"
    ), patch("tools.async_delegation.interrupt_for_session"):
        reply = await env.runner._handle_reset_command(MessageEvent(text="/new", source=env.source))
    if mode == "success":
        assert reload_entry(env).session_id != env.entry.session_id
        assert not reload_entry(env).compression_paused
        env.runner._sync_telegram_topic_binding.assert_called_once()
    else:
        assert reload_entry(env).compression_paused
        assert "persist" in str(reply).lower()
        env.runner._fire_session_reset_hooks.assert_not_awaited()
        env.runner._record_telegram_topic_binding.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["preview", "insufficient", "locked", "nothing", "exception", "noop", "success"])
async def test_manual_command_matrix_uses_native_compress_result(env, mode):
    from agent.conversation_compression_manual import CompressResult, CompressRequest
    mark(env)
    e = env
    before = e.db.get_messages(e.entry.session_id)
    agent = Mock()
    agent.session_id = e.entry.session_id
    agent._last_compaction_in_place = mode == "success"
    agent.context_compressor = SimpleNamespace(_last_summary_error=None, _last_aux_model_failure_model=None,
                                              _last_compress_aborted=False)
    e.runner._build_manual_compression_agent = AsyncMock(return_value=agent)
    e.runner._cleanup_agent_resources_off_loop = AsyncMock()
    e.runner._resolve_session_agent_runtime = Mock(return_value=("test", {"api_key": "test"}))
    if mode == "insufficient":
        e.store.rewrite_transcript(e.entry.session_id, HISTORY[:1])
    status = {"locked": "lock_skipped", "nothing": "nothing_to_do"}.get(mode, "compressed")
    result = CompressResult(status, HISTORY, HISTORY, 100, 90, CompressRequest(),
                            summary={"headline": "Compressed: test", "token_line": "90", "note": ""})
    with patch("agent.conversation_compression_manual.compress_now", return_value=result,
               side_effect=RuntimeError("compressor") if mode == "exception" else None) as compress, patch(
        "agent.conversation_compression.finalize_context_engine_compression_notification"
    ) as notify:
        reply = await e.runner._handle_compress_command(MessageEvent(
            text="/compress --preview" if mode == "preview" else "/compress", source=e.source))
    assert reload_entry(e).compression_paused == (mode != "success")
    commits = [c for c in notify.call_args_list if c.kwargs.get("committed") is True]
    assert len(commits) == int(mode == "success")
    if mode in {"preview", "insufficient"}:
        compress.assert_not_called()
    if mode != "insufficient":
        assert e.db.get_messages(e.entry.session_id) == before
    assert bool(reply)


@pytest.mark.asyncio
async def test_user_yaml_read_error_is_unknown_even_with_injected_default(env, monkeypatch):
    mark(env)
    policy(env, "compression:\n  exhaustion_action: reset")
    original = type(env.home).read_text
    def read(path, *args, **kwargs):
        if path == env.home / "config.yaml":
            raise PermissionError("user yaml")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(type(env.home), "read_text", read)
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda *args: {"compression": {"exhaustion_action": "reset"}})
    assert "paused" in (await admit(env))[0].lower()
    env.runner._hmwa_prepare_turn.assert_not_awaited()


def test_pending_first_mark_survives_recovered_primary_reconciliation(env, monkeypatch):
    e = env
    e.store._routing_db_loaded = False
    e.store._routing_fallback_baseline = copy.deepcopy(e.store._entries_as_dicts())
    loader = e.db.load_gateway_routing_entries
    monkeypatch.setattr(e.db, "load_gateway_routing_entries", Mock(side_effect=OSError("read unavailable")))
    with patch.object(e.db, "replace_gateway_routing_entries", side_effect=OSError("write unavailable")):
        with pytest.raises(OSError):
            mark(e)
    assert e.entry.compression_paused
    monkeypatch.setattr(e.db, "load_gateway_routing_entries", loader)
    assert e.store.lookup_by_session_key(e.key).compression_paused


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["new", "resume"])
@pytest.mark.parametrize("fail", [False, True])
async def test_command_cancellation_keeps_admission_until_primary_settles(env, monkeypatch, command, fail):
    e = env
    mark(e)
    target = "authorized-target"
    e.db.create_session(target, source="telegram", user_id=e.source.user_id, chat_id=e.source.chat_id)
    e.runner._resolve_resume_target = AsyncMock(return_value=(target, target))
    entered, release = threading.Event(), threading.Event()
    original = e.db.replace_gateway_routing_entries
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        if fail:
            raise OSError("primary unavailable")
        return original(*args, **kwargs)
    monkeypatch.setattr(e.db, "replace_gateway_routing_entries", blocked)
    fn = e.runner._handle_reset_command if command == "new" else e.runner._handle_resume_command
    with patch("gateway.slash_commands_session._reset_process_scoped_tool_state"), patch(
        "tools.async_delegation.interrupt_for_session"
    ):
        task = asyncio.create_task(fn(MessageEvent(text="/new" if command == "new" else "/resume target", source=e.source)))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            e.runner._release_running_agent_state.assert_not_called()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    e.runner._release_running_agent_state.assert_called_once()
    assert reload_entry(e).compression_paused is fail


@pytest.mark.asyncio
async def test_paused_new_cancelled_before_storage_releases_its_admission(env):
    mark(env)
    entered = asyncio.Event()
    async def cleanup(key):
        entered.set()
        await asyncio.Event().wait()
    env.runner._cleanup_old_agent_for_reset = cleanup
    task = asyncio.create_task(env.runner._handle_reset_command(MessageEvent(text="/new", source=env.source)))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    env.runner._release_running_agent_state.assert_called_once()
    assert reload_entry(env).compression_paused


@pytest.mark.asyncio
@pytest.mark.parametrize("paused", [False, True])
async def test_stale_generation_cannot_apply_policy_after_await(env, paused):
    e = env
    if paused:
        mark(e)
    policy(e, "compression:\n  exhaustion_action: reset")
    async def read(source):
        e.runner._is_session_run_current.return_value = False
        return "reset"
    e.runner._compression_exhaustion_action = read
    if paused:
        await admit(e)
    else:
        await e.runner._hmwa_compression_exhaustion_reset(
            {"compression_exhausted": True}, "", e.entry, e.key, e.source, run_generation=1)
    assert reload_entry(e).session_id == e.entry.session_id
    assert reload_entry(e).compression_paused is paused
    e.runner._hmwa_prepare_turn.assert_not_awaited()


def test_replaced_route_cannot_receive_old_binding_tail(env):
    old = env.entry
    env.store.reset_session(env.key)
    env.runner._sync_compression_recovery_binding(env.source, old, old.session_id, reason="test")
    env.runner._sync_telegram_topic_binding.assert_not_called()


@pytest.mark.asyncio
async def test_resume_binding_failure_does_not_reclassify_primary_success(env):
    mark(env)
    target = "authorized-target"
    env.db.create_session(target, source="telegram", user_id=env.source.user_id, chat_id=env.source.chat_id)
    env.runner._resolve_resume_target = AsyncMock(return_value=(target, target))
    env.runner._sync_telegram_topic_binding.side_effect = OSError("binding unavailable")
    reply = await env.runner._handle_resume_command(MessageEvent(text="/resume target", source=env.source))
    assert reload_entry(env).session_id == target
    assert not reload_entry(env).compression_paused
    assert "could not be persisted" not in reply


def test_native_tip_heal_retains_pause_metadata(env, monkeypatch):
    mark(env)
    env.db.create_session("tip", source="telegram")
    monkeypatch.setattr(env.store, "_compression_tip_for_session_id", lambda sid: "tip")
    entry = env.store.get_or_create_session(env.source)
    assert entry.session_id == "tip"
    assert entry.compression_paused


@pytest.mark.asyncio
async def test_new_default_reset_keeps_json_fallback(env):
    env.store._db = None
    policy(env, None)
    _, entry, _ = await exhaust(env)
    assert entry.session_id != env.entry.session_id
    assert json.loads((env.store.sessions_dir / "sessions.json").read_text())[env.key]["session_id"] == entry.session_id


@pytest.fixture
def native_env(env, monkeypatch):
    """Native dispatcher, state, lease, generation and storage; only transport/OS/model edges fake."""
    e = env
    r = e.runner
    for name in ("_release_running_agent_state", "_invalidate_session_run_generation",
                 "_is_session_run_current", "_clear_conversation_scope", "_evict_cached_agent"):
        delattr(r, name)
    r.adapters = {}
    r._draining = r._external_drain_active = False
    r._busy_input_mode = "queue"
    r._persist_active_agents = Mock()
    r._scale_to_zero_note_real_inbound = Mock()
    r._is_user_authorized_for_source = Mock(return_value=True)
    r._admit_bot_message_for_source = Mock(return_value=True)
    r._check_slash_access = Mock(return_value=None)
    r._is_telegram_topic_root_lobby = Mock(return_value=False)
    r._read_user_config = Mock(return_value={"approvals": {"destructive_slash_confirm": False}})
    r.hooks = SimpleNamespace(emit_collect=AsyncMock(return_value=[]), emit=AsyncMock())
    r._run_post_turn_hooks = AsyncMock()
    r._clear_durable_active_turn = AsyncMock()
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", Mock(return_value=[]))
    monkeypatch.setattr("hermes_cli.plugins.fire_pre_command_hook", Mock())
    monkeypatch.setattr("gateway.slash_commands_session._reset_process_scoped_tool_state", Mock())
    monkeypatch.setattr("tools.async_delegation.interrupt_for_session", Mock())
    yield e
    r._release_running_agent_state(e.key)


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
@pytest.mark.parametrize("boundary", ["primary", "binding"])
@pytest.mark.parametrize("command", ["new", "resume", "compress", "sessions"])
@pytest.mark.parametrize("fail", [False, True])
async def test_native_recovery_owns_admission_through_cancelled_commit(
    native_env, monkeypatch, command, boundary, fail,
):
    e = native_env
    mark(e)
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
    assert reload_entry(e).compression_paused is (fail and boundary == "primary")
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
async def test_native_new_early_cancellation_settles_own_admission(native_env, paused):
    e = native_env
    if paused:
        mark(e)
    entered = asyncio.Event()
    async def cleanup(*args):
        entered.set()
        await asyncio.Event().wait()
    e.runner._cleanup_old_agent_for_reset = cleanup
    task = asyncio.create_task(native_message(e, "/new"))
    lease = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        state = e.runner._peek_session_state(e.key)
        assert e.runner._is_session_running(e.key) is paused
        if paused:
            lease = state.turn.lease
            assert lease is not None and not lease.released
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
    assert not e.runner._is_session_running(e.key)
    if lease:
        assert lease.released
    assert reload_entry(e).compression_paused is paused



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
    name = {"validation": "resolve_session_id", "binding": "bind_telegram_topic",
            "primary": "replace_gateway_routing_entries"}[boundary]
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
