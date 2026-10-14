"""Reset must resolve receiving-account ownership before touching a legacy lane."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_first_native_reset_does_not_reset_other_bots_legacy_lane(tmp_path, monkeypatch):
    from dataclasses import replace
    from pathlib import Path
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.base import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import AsyncSessionStore, SessionSource, SessionStore
    from gateway.session_identity import RoutingIdentity

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))
    monkeypatch.setattr("gateway.slash_commands_session._reset_process_scoped_tool_state", lambda: None)
    monkeypatch.setattr("tools.async_delegation.interrupt_for_session", lambda **kwargs: None)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *args, **kwargs: None)
    config = GatewayConfig(multiplex_profiles=True)
    store = SessionStore(config.sessions_dir, config)
    old_source = SessionSource(Platform.MATRIX, "!room:example.org", chat_type="group", user_id="alice", profile="default")
    old_entry = store.get_or_create_session(old_source)
    old_entry.transport_profile = "shared-bot"
    store._save_entries()
    store.append_to_transcript(old_entry.session_id, {"role": "user", "content": "shared bot history"})
    native = replace(old_source)
    native._identity = RoutingIdentity("default", "default", tmp_path, tmp_path)
    runner = SimpleNamespace(
        session_store=store, async_session_store=AsyncSessionStore(store),
        _session_key_for_source=store._generate_session_key,
        _invalidate_session_run_generation=lambda *args, **kwargs: None,
        _release_running_agent_state=lambda *args, **kwargs: None,
        _cleanup_old_agent_for_reset=AsyncMock(),
        _evict_cached_agent=lambda *args, **kwargs: None,
        _clear_conversation_scope=lambda *args, **kwargs: None,
        _fire_session_reset_hooks=AsyncMock(),
        _reset_notice_session_info=lambda source: "",
        _telegram_topic_new_header=lambda source: None,
        _is_telegram_topic_lane=lambda source: False,
        _session_db=None,
    )
    try:
        await GatewayRunner._handle_reset_command(runner, MessageEvent(text="/new", source=native))
        shared = replace(old_source)
        shared._identity = RoutingIdentity("shared-bot", "default", tmp_path, tmp_path)
        resumed = store.get_or_create_session(shared)
        assert resumed.session_id == old_entry.session_id, "The native bot's /new reset the other receiver's transcript"
        assert any(m.get("content") == "shared bot history" for m in store.load_transcript(resumed.session_id))
    finally:
        store.close_all_db_handles()



@pytest.mark.parametrize("force_new", [False, True])
def test_native_creation_preserves_foreign_account_history(tmp_path, monkeypatch, force_new):
    from dataclasses import replace
    from pathlib import Path
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.session_identity import RoutingIdentity

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))
    store = SessionStore(tmp_path / "sessions", GatewayConfig(multiplex_profiles=True))
    legacy = SessionSource(Platform.MATRIX, "!room:example.org", chat_type="group", user_id="alice", profile="default")
    old = store.get_or_create_session(legacy)
    old.transport_profile = "shared-bot"
    store._save_entries()
    store.append_to_transcript(old.session_id, {"role": "user", "content": "preserved shared history"})
    native = replace(legacy)
    native._identity = RoutingIdentity("default", "default", tmp_path, tmp_path)
    try:
        own = store.get_or_create_session(native, force_new=force_new)
        assert own.session_id != old.session_id
        shared = replace(legacy)
        shared._identity = RoutingIdentity("shared-bot", "default", tmp_path, tmp_path)
        resumed = store.get_or_create_session(shared)
        assert resumed.session_id == old.session_id
        assert any(m.get("content") == "preserved shared history" for m in store.load_transcript(resumed.session_id))
    finally:
        store.close_all_db_handles()
