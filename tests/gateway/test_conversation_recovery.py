"""A stricter conversation policy must not revive a formerly shared transcript."""

from dataclasses import replace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


def test_enabling_thread_isolation_does_not_adopt_the_old_shared_lane(tmp_path):
    config = GatewayConfig(group_sessions_per_user=False, thread_sessions_per_user=False)
    store = SessionStore(tmp_path / "sessions", config)
    source = SessionSource(Platform.MATRIX, "!room:example.org", chat_type="group",
                           thread_id="$root", user_id="alice")
    shared = store.get_or_create_session(source)
    store.append_to_transcript(shared.session_id, {"role": "user", "content": "shared history"})
    config.thread_sessions_per_user = True
    isolated = store.get_or_create_session(source)
    assert isolated.session_id != shared.session_id
    assert not store.load_transcript(isolated.session_id)
    bob = store.get_or_create_session(replace(source, user_id="bob"))
    assert bob.session_id not in {shared.session_id, isolated.session_id}


def test_conversation_kind_cannot_override_a_group_into_a_dm():
    with pytest.raises(ValueError):
        SessionSource(Platform.MATRIX, "!room:example.org", chat_type="group", conversation_kind="dm")


def test_account_tag_is_not_reconstructed_as_a_thread_or_participant():
    from gateway.conversation import _parse_session_key
    from gateway.platforms.qqbot.adapter import QQAdapter

    assert _parse_session_key("agent:work:telegram:dm:123:account~:default") is None
    parsed = QQAdapter._parse_gateway_session_key("agent:work:qqbot:group:room:alice:account~:default")
    assert parsed["user_id"] == "alice"
    assert "user_id" not in QQAdapter._parse_gateway_session_key("agent:work:qqbot:dm:alice:account~:default")


@pytest.mark.parametrize("cached", [False, True])
def test_account_scoped_notification_requires_a_proven_source(cached):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from gateway.run import GatewayRunner

    source = SessionSource(Platform.TELEGRAM, "123", user_id="alice", profile="work")
    runner = SimpleNamespace(
        session_store=MagicMock(_entries={}),
        _get_cached_session_source=lambda key: source if cached else None,
    )
    result = GatewayRunner._build_process_event_source(runner, {
        "session_key": "agent:work:telegram:dm:123:account~:default",
        "platform": "telegram", "chat_type": "dm", "chat_id": "456", "user_id": "bob",
    })
    assert result is source if cached else result is None


def test_native_bot_cannot_claim_a_legacy_lane_owned_by_a_shared_bot(tmp_path):
    from gateway.session_identity import RoutingIdentity

    store = SessionStore(tmp_path / "sessions", GatewayConfig(multiplex_profiles=True))
    source = SessionSource(Platform.MATRIX, "!room:example.org", chat_type="group",
                           user_id="alice", profile="default")
    legacy = store.get_or_create_session(source)
    legacy.transport_profile = "shared-bot"
    store._save_entries()
    store.append_to_transcript(legacy.session_id, {"role": "user", "content": "other bot history"})
    native = replace(source)
    native._identity = RoutingIdentity("default", "default", tmp_path, tmp_path)
    entry = store.get_or_create_session(native)
    assert entry.session_id != legacy.session_id
    assert not store.load_transcript(entry.session_id)
    shared = replace(source)
    shared._identity = RoutingIdentity("shared-bot", "default", tmp_path, tmp_path)
    restored = store.get_or_create_session(shared)
    assert restored.session_id == legacy.session_id
    assert store.load_transcript(restored.session_id)[0]["content"] == "other bot history"


@pytest.mark.parametrize("thread", [None, "$root"])
def test_relaxing_isolation_does_not_publish_a_private_history(tmp_path, monkeypatch, thread):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    config = GatewayConfig(group_sessions_per_user=True, thread_sessions_per_user=True)
    store = SessionStore(tmp_path / "sessions", config)
    source = SessionSource(Platform.MATRIX, "!room:example.org", chat_type="group",
                           thread_id=thread, user_id="alice")
    private = store.get_or_create_session(source)
    store.append_to_transcript(private.session_id, {"role": "user", "content": "private history"})
    config.group_sessions_per_user = False
    config.thread_sessions_per_user = False
    shared = store.get_or_create_session(source)
    assert shared.session_id != private.session_id
    assert not store.load_transcript(shared.session_id)
