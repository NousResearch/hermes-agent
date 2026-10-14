"""Conversation addresses preserve account, scope and explicit participant isolation."""

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.conversation import ConversationAddress, address_of, existing_delivery_session, resolve_delivery_source, source_for_address
from gateway.session import SessionSource, SessionStore, build_session_key
from gateway.session_identity import RoutingIdentity


@pytest.mark.parametrize("group_per_user", [False, True])
def test_thread_isolation_is_independent_of_flat_group_isolation(group_per_user):
    alice = SessionSource(Platform.MATRIX, "!room:example.org", chat_type="group",
                          thread_id="$root", user_id="@alice:example.org")
    bob = replace(alice, user_id="@bob:example.org")
    assert build_session_key(alice, group_per_user, True) != build_session_key(bob, group_per_user, True)
    assert build_session_key(alice, group_per_user, False) == build_session_key(bob, group_per_user, False)


@pytest.mark.parametrize("legacy_owner", [None, "bot-a", "bot-b"])
def test_account_migration_preserves_only_a_proven_owner(tmp_path, legacy_owner):
    store = SessionStore(tmp_path / "sessions", GatewayConfig(multiplex_profiles=True))
    old_source = SessionSource(Platform.MATRIX, "!room:example.org", chat_type="group", user_id="alice", profile="default")
    old = store.get_or_create_session(old_source)
    old.transport_profile = legacy_owner
    store._save_entries()
    store.append_to_transcript(old.session_id, {"role": "user", "content": "legacy history"})

    def source(account):
        source = replace(old_source)
        source._identity = RoutingIdentity(account, "default", tmp_path, tmp_path)
        return source

    first = store.get_or_create_session(source("bot-a"))
    second = store.get_or_create_session(source("bot-b"))
    assert first.session_id != second.session_id
    assert (first.session_id == old.session_id) == (legacy_owner == "bot-a")
    assert (second.session_id == old.session_id) == (legacy_owner == "bot-b")
    native = store.get_or_create_session(source("default"))
    assert native.session_id not in {first.session_id, second.session_id}


@pytest.mark.asyncio
async def test_unrelated_delivery_cannot_inherit_origin_kind_or_scope():
    adapter = MagicMock()
    adapter.get_chat_info = AsyncMock(return_value={"type": "group"})
    source = await resolve_delivery_source(adapter, "matrix", "!other:example.org", "$other", {
        "platform": "matrix", "chat_id": "!dm:example.org", "chat_type": "dm",
        "scope_id": "private", "user_id": "alice",
    })
    assert source.chat_type == "group"
    assert source.scope_id is None
    assert source.user_id is None


@pytest.mark.parametrize("platform,kind", [("matrix", "dm"), ("matrix", "group"), ("slack", "group"), ("discord", "channel")])
def test_kind_and_address_round_trip_separately(platform, kind):
    address = ConversationAddress(platform, "room", "thread", "scope", "default")
    source = source_for_address(address, kind, user_id="alice")
    restored = SessionSource.from_dict(source.to_dict())
    assert address_of(restored) == address
    assert restored.conversation_kind == kind
    assert build_session_key(restored) == build_session_key(source)


def test_standalone_mirror_requires_exact_scope_and_participant(tmp_path):
    store = SessionStore(tmp_path / "sessions", GatewayConfig(thread_sessions_per_user=True))
    source = SessionSource(Platform.SLACK, "C1", chat_type="group", thread_id="root",
                           user_id="alice", scope_id="workspace-a")
    entry = store.get_or_create_session(source)
    origin = source.to_dict()
    assert existing_delivery_session(store, "slack", "C1", "root", origin).session_id == entry.session_id
    for changed in [{"scope_id": "workspace-b"}, {"user_id": "bob"}, {"user_id": None}]:
        assert existing_delivery_session(store, "slack", "C1", "root", {**origin, **changed}) is None
    assert existing_delivery_session(store, "slack", "C1", None, origin) is None


def test_child_origin_recovers_account_from_the_exact_durable_lane(tmp_path, monkeypatch):
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.cronjob_job_args import _origin_from_env

    config = GatewayConfig(multiplex_profiles=True)
    source = SessionSource(Platform.MATRIX, "!room:example.org", chat_type="group", user_id="alice", profile="default")
    source._identity = RoutingIdentity("bot-a", "default", tmp_path, tmp_path)
    store = SessionStore(config.sessions_dir, config)
    entry = store.get_or_create_session(source)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    tokens = set_session_vars(platform="matrix", chat_id=source.chat_id, chat_type="group",
                              user_id="alice", session_key=entry.session_key, profile="default")
    try:
        origin = _origin_from_env()
        assert origin["conversation"]["address"]["account"] == "bot-a"
        assert origin["conversation"]["kind"] == "group"
    finally:
        clear_session_vars(tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [None, "workspace"])
async def test_legacy_slack_origin_needs_authoritative_workspace_scope(scope):
    adapter = MagicMock()
    adapter.get_chat_info = AsyncMock(return_value={"type": "group", "scope_id": scope})
    origin = {"platform": "slack", "chat_id": "C1", "user_id": "alice"}
    if scope is None:
        with pytest.raises(ValueError, match="workspace scope"):
            await resolve_delivery_source(adapter, "slack", "C1", "root", origin)
    else:
        source = await resolve_delivery_source(adapter, "slack", "C1", "root", origin)
        reply = SessionSource(Platform.SLACK, "C1", chat_type="group", user_id="alice",
                              thread_id="root", scope_id=scope)
        assert build_session_key(source) == build_session_key(reply)


@pytest.mark.asyncio
@pytest.mark.parametrize("rejection", ["version", "address", "account", "profile", "unserved"])
async def test_captured_origin_requires_matching_receiving_identity(tmp_path, rejection):
    from gateway.conversation import conversation_origin

    source = SessionSource(Platform.MATRIX, "!room:example.org", chat_type="dm", profile="runtime")
    source._identity = RoutingIdentity("receiver", "runtime", tmp_path, tmp_path)
    origin = {"platform": "matrix", "chat_id": source.chat_id,
              "conversation": conversation_origin(source, thread_id=None)}
    if rejection == "version":
        origin["conversation"]["version"] = 99
    elif rejection == "address":
        origin["conversation"]["address"]["chat_id"] = "!other:example.org"
    elif rejection in {"account", "profile"}:
        origin["conversation"]["address"][rejection] = "other"

    def canonicalize(resolved):
        resolved._identity = source._identity
        resolved.profile_route_rejected = rejection == "unserved"

    adapter = MagicMock()
    adapter._canonicalize.side_effect = canonicalize
    with pytest.raises(PermissionError):
        await resolve_delivery_source(adapter, "matrix", source.chat_id, None, origin)


@pytest.mark.parametrize("stored_profile", [None, "ops", "default"])
def test_child_captures_standalone_runtime_profile(tmp_path, monkeypatch, stored_profile):
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.cronjob_job_args import _origin_from_env

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    config = GatewayConfig(multiplex_profiles=False)
    source = SessionSource(Platform.MATRIX, "!room:example.org", chat_type="dm", profile=stored_profile)
    entry = SessionStore(config.sessions_dir, config).get_or_create_session(source)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    tokens = set_session_vars(platform="matrix", chat_id=source.chat_id, chat_type="dm",
                              session_key=entry.session_key, profile="ops")
    try:
        origin = _origin_from_env()
        assert origin["conversation"]["address"]["profile"] == "ops"
    finally:
        clear_session_vars(tokens)
