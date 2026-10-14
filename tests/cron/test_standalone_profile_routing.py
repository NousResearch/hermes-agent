"""Standalone legacy keys do not erase the runtime profile owning cron delivery."""

from unittest.mock import AsyncMock, patch
from pathlib import Path

import pytest

from cron.scheduler_delivery import _deliver_result
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.conversation import conversation_origin, existing_delivery_session
from gateway.session import SessionSource, SessionStore
from gateway.session_identity import RoutingIdentity


@pytest.mark.parametrize("stored_profile", [None, "ops"])
def test_standalone_mirror_resolves_the_owner_of_a_legacy_main_key(tmp_path, monkeypatch, stored_profile):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "ops")
    config = GatewayConfig(multiplex_profiles=False)
    source = SessionSource(Platform.MATRIX, "!room:example.org", chat_type="dm", profile=stored_profile)
    source._identity = RoutingIdentity("ops", "ops", tmp_path, tmp_path, multiplexed=False)
    store = SessionStore(config.sessions_dir, config)
    entry = store.get_or_create_session(source)
    origin = {**source.to_dict(), "conversation": conversation_origin(source, thread_id=None)}
    # Durable state has no pinned identity; the owning profile is the bound runtime.
    restored = SessionStore(config.sessions_dir, config)
    matched = existing_delivery_session(restored, "matrix", source.chat_id, None, origin)
    assert matched is not None
    assert matched.session_id == entry.session_id
    origin["conversation"]["address"]["profile"] = "other"
    assert existing_delivery_session(restored, "matrix", source.chat_id, None, origin) is None


@pytest.mark.parametrize("origin_profile", ["ops", "default", "other"])
def test_standalone_send_requires_the_origin_runtime_profile(tmp_path, monkeypatch, origin_profile):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "ops")
    source = SessionSource(Platform.MATRIX, "!room:example.org", chat_type="dm", profile=origin_profile)
    job = {"id": "digest", "deliver": "origin", "attach_to_session": False,
           "origin": {**source.to_dict(), "conversation": conversation_origin(source, thread_id=None)}}
    config = GatewayConfig(platforms={Platform.MATRIX: PlatformConfig(enabled=True, token="test-token")})
    sender = AsyncMock(return_value={"success": True})
    with patch("gateway.config.load_gateway_config", return_value=config), \
            patch("tools.send_message_tool._send_to_platform", sender):
        error = _deliver_result(job, "private brief")
    if origin_profile == "ops":
        assert error is None
        sender.assert_awaited_once()
    else:
        assert error
        sender.assert_not_awaited()


def test_standalone_mirror_reads_runtime_rows_without_central_index(tmp_path, monkeypatch):
    import hermes_state
    from gateway.config import GatewayConfig, Platform
    from gateway.conversation import conversation_origin
    from gateway.session import SessionSource, SessionStore
    from gateway.session_identity import RoutingIdentity
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from cron.scheduler_delivery import _maybe_mirror_cron_delivery

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    home = tmp_path / "isolated"
    runtime = home / "profiles" / "ops"
    runtime.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: runtime)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "ops")
    config = GatewayConfig(multiplex_profiles=True)
    store = SessionStore(config.sessions_dir, config)
    source = SessionSource(Platform.MATRIX, "!room:example.org", chat_type="dm", user_id="alice", profile="ops")
    source._identity = RoutingIdentity("ops", "ops", runtime, runtime)
    entry = store.get_or_create_session(source)
    origin = {**source.to_dict(), "conversation": conversation_origin(source, thread_id=None)}
    token = set_hermes_home_override(str(runtime))
    try:
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: GatewayConfig(multiplex_profiles=True))
        _maybe_mirror_cron_delivery({"id": "digest", "origin": origin}, "matrix", source.chat_id, "delivered brief", user_id="alice", enabled=True)
        assert any("delivered brief" in m["content"] for m in store.load_transcript(entry.session_id))
    finally:
        reset_hermes_home_override(token)
        store.close_all_db_handles()


def test_standalone_mirror_cannot_reuse_an_old_private_key_after_policy_relaxes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))
    config = GatewayConfig(thread_sessions_per_user=True)
    store = SessionStore(config.sessions_dir, config)
    source = SessionSource(Platform.MATRIX, "!room:example.org", chat_type="group", user_id="alice", thread_id="$root")
    entry = store.get_or_create_session(source)
    config.thread_sessions_per_user = False
    origin = {**source.to_dict(), "user_id": "bob"}
    assert existing_delivery_session(store, "matrix", source.chat_id, "$root", origin) is None
    assert store.lookup_by_session_key(entry.session_key).session_id == entry.session_id


def test_standalone_discord_kind_aliases_match_the_actual_thread_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(Platform.DISCORD, "456", chat_type="thread", parent_chat_id="123",
                           thread_id="456", scope_id="789", user_id="alice", conversation_kind="channel")
    entry = store.get_or_create_session(source)
    origin = {**source.to_dict(), "conversation": conversation_origin(source, thread_id="456")}
    origin["conversation"]["kind"] = "group"
    matched = existing_delivery_session(store, "discord", "123", "456", origin)
    assert matched is not None and matched.session_id == entry.session_id
    store.close_all_db_handles()


def test_standalone_non_dm_kind_must_still_match_the_requested_key_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(Platform.FEISHU, "oc_chat", chat_type="forum", thread_id="om_root", user_id="alice")
    store.get_or_create_session(source)
    origin = {**source.to_dict(), "chat_type": "group"}
    assert existing_delivery_session(store, "feishu", source.chat_id, source.thread_id, origin) is None
    store.close_all_db_handles()
