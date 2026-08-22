"""End-to-end routing invariants for webhook session handoff processing."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.profile_routing import ProfileRoute
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.run import GatewayRunner
from gateway.session import (
    AsyncSessionStore,
    SessionSource,
    SessionStore,
    build_session_key,
)
from hermes_state import AsyncSessionDB, SessionDB


def _discord_config(
    tmp_path,
    *,
    thread_sessions_per_user=False,
    home_user_id=None,
    home_scope_id="guild-1",
    multiplex_profiles=False,
    profile_routes=None,
):
    return GatewayConfig(
        sessions_dir=tmp_path / "sessions",
        thread_sessions_per_user=thread_sessions_per_user,
        multiplex_profiles=multiplex_profiles,
        profile_routes=profile_routes or [],
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                token="test-token",
                home_channel=HomeChannel(
                    platform=Platform.DISCORD,
                    chat_id="parent-1",
                    name="Hermes Home",
                    user_id=home_user_id,
                    scope_id=home_scope_id,
                ),
            )
        },
    )


def _runner_with_store(config, store, db):
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_db = AsyncSessionDB(db)

    adapter = SimpleNamespace(
        create_handoff_thread=AsyncMock(return_value="thread-42"),
        get_chat_info=AsyncMock(
            return_value={
                "name": "Hermes Home",
                "type": "channel",
                "guild_id": "guild-1",
            }
        ),
        send=AsyncMock(return_value=SimpleNamespace(success=True)),
    )
    runner.adapters = {Platform.DISCORD: adapter}
    runner._evict_cached_agent = MagicMock()
    runner._release_running_agent_state = MagicMock()
    runner._handle_message = AsyncMock(return_value="Ready in the handoff thread.")
    return runner, adapter


@pytest.mark.asyncio
async def test_webhook_handoff_moves_exact_session_and_next_reply_reuses_it(tmp_path):
    """The source key disappears and an organic thread event sees the same ID."""
    config = _discord_config(tmp_path)
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=config.sessions_dir, config=config)
    db = SessionDB(db_path=tmp_path / "state.db")
    store._db = db
    store._loaded = True

    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="webhook:build-finished:delivery-7",
        chat_name="build-finished",
        chat_type="dm",
        user_id="build-finished",
    )
    source_entry = store.get_or_create_session(source)
    source_key = source_entry.session_key
    session_id = source_entry.session_id
    db.append_message(session_id, "user", "Build 7 finished successfully")
    db.append_message(
        session_id,
        "assistant",
        "",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }
        ],
    )
    db.append_message(
        session_id,
        "tool",
        "checks: all green",
        tool_name="terminal",
        tool_call_id="call-1",
    )
    db.append_message(
        session_id,
        "assistant",
        "I verified the release artifacts.",
    )

    runner, adapter = _runner_with_store(config, store, db)
    row = db.get_session(session_id)
    row.update(
        {
            "handoff_platform": "discord",
            "source": "webhook",
            "session_key": source_key,
            "title": "Build 7",
        }
    )

    await runner._process_handoff(row)

    destination_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-42",
        chat_name="Hermes Home",
        chat_type="thread",
        user_id="someone-replying",
        thread_id="thread-42",
        scope_id="guild-1",
        parent_chat_id="parent-1",
    )
    destination_key = build_session_key(destination_source)

    assert store.lookup_by_session_key(source_key) is None
    moved = store.lookup_by_session_key(destination_key)
    assert moved is not None
    assert moved.session_id == session_id
    assert moved.origin is not None
    assert moved.origin.platform == Platform.DISCORD
    assert moved.origin.chat_id == "thread-42"
    assert moved.origin.thread_id == "thread-42"
    assert moved.origin.parent_chat_id == "parent-1"

    # This is the exact source shape a subsequent Discord thread event uses.
    next_turn = store.get_or_create_session(destination_source)
    assert next_turn.session_id == session_id

    transcript = store.load_transcript(session_id)
    assert [(message["role"], message["content"]) for message in transcript] == [
        ("user", "Build 7 finished successfully"),
        ("assistant", ""),
        ("tool", "checks: all green"),
        ("assistant", "I verified the release artifacts."),
    ]

    durable = db.get_session(session_id)
    assert durable["source"] == "discord"
    assert durable["session_key"] == destination_key
    assert durable["chat_id"] == "thread-42"
    assert durable["chat_type"] == "thread"
    assert durable["thread_id"] == "thread-42"

    synthetic_event = runner._handle_message.await_args.args[0]
    assert synthetic_event.source == moved.origin
    assert "from CLI" not in synthetic_event.text
    adapter.send.assert_awaited_once_with(
        "parent-1",
        "Ready in the handoff thread.",
        metadata={"thread_id": "thread-42"},
    )
    db.close()


@pytest.mark.asyncio
async def test_thread_per_user_handoff_keys_to_authenticated_home_user(tmp_path):
    """The production global setting keys handoff and next reply identically."""
    config = _discord_config(
        tmp_path,
        thread_sessions_per_user=True,
        home_user_id="discord-user-7",
    )
    assert "thread_sessions_per_user" not in config.platforms[Platform.DISCORD].extra
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=config.sessions_dir, config=config)
    db = SessionDB(db_path=tmp_path / "state.db")
    store._db = db
    store._loaded = True

    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="webhook:alerts:per-user-delivery",
        chat_type="webhook",
        user_id="webhook:alerts",
    )
    entry = store.get_or_create_session(source)
    runner, _adapter = _runner_with_store(config, store, db)
    row = db.get_session(entry.session_id)
    row.update(
        {
            "handoff_platform": "discord",
            "source": "webhook",
            "session_key": entry.session_key,
        }
    )

    await runner._process_handoff(row)

    next_reply_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-42",
        chat_name="Hermes Home",
        chat_type="thread",
        user_id="discord-user-7",
        thread_id="thread-42",
        scope_id="guild-1",
        parent_chat_id="parent-1",
    )
    next_reply_key = runner._session_key_for_source(next_reply_source)
    moved = store.lookup_by_session_key(next_reply_key)

    assert moved is not None
    assert moved.session_id == entry.session_id
    assert moved.origin is not None
    assert moved.origin.user_id == "discord-user-7"
    assert store.get_or_create_session(next_reply_source).session_id == entry.session_id
    assert next_reply_key.endswith(":discord-user-7")
    db.close()


@pytest.mark.asyncio
async def test_multiplex_default_handoff_matches_unprofiled_organic_reply(
    tmp_path,
):
    """An unprefixed default webhook and Discord reply share one namespace."""
    config = _discord_config(tmp_path, multiplex_profiles=True)
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=config.sessions_dir, config=config)
    db = SessionDB(db_path=tmp_path / "state.db")
    store._db = db
    store._loaded = True

    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="webhook:alerts:default-profile",
        chat_type="webhook",
        user_id="webhook:alerts",
    )
    with patch(
        "hermes_cli.profiles.get_active_profile_name",
        return_value="default",
    ):
        entry = store.get_or_create_session(source)
        runner, _adapter = _runner_with_store(config, store, db)
        row = db.get_session(entry.session_id)
        row.update(
            {
                "handoff_platform": "discord",
                "source": "webhook",
                "session_key": entry.session_key,
            }
        )

        await runner._process_handoff(row)

        next_reply = SessionSource(
            platform=Platform.DISCORD,
            chat_id="thread-42",
            chat_name="Hermes Home",
            chat_type="thread",
            user_id="discord-user",
            thread_id="thread-42",
            scope_id="guild-1",
            guild_id="guild-1",
            parent_chat_id="parent-1",
        )
        destination_key = runner._session_key_for_source(next_reply)
        resumed = store.get_or_create_session(next_reply)

    assert entry.session_key.startswith("agent:main:")
    assert destination_key.startswith("agent:main:")
    assert resumed.session_id == entry.session_id
    assert store.lookup_by_session_key(entry.session_key) is None
    db.close()


@pytest.mark.asyncio
async def test_named_destination_profile_route_is_rejected_before_thread(
    tmp_path,
):
    """A named-profile Discord home cannot receive a default-profile handoff."""
    config = _discord_config(
        tmp_path,
        home_scope_id=None,
        multiplex_profiles=True,
        profile_routes=[
            ProfileRoute(
                name="work-guild",
                platform="discord",
                guild_id="guild-1",
                profile="work",
            )
        ],
    )
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=config.sessions_dir, config=config)
    db = SessionDB(db_path=tmp_path / "state.db")
    store._db = db
    store._loaded = True

    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="webhook:alerts:named-destination",
        chat_type="webhook",
        user_id="webhook:alerts",
        profile="default",
    )
    entry = store.get_or_create_session(source)
    runner, adapter = _runner_with_store(config, store, db)
    row = db.get_session(entry.session_id)
    row.update(
        {
            "handoff_platform": "discord",
            "source": "webhook",
            "session_key": entry.session_key,
        }
    )

    with pytest.raises(RuntimeError, match="named profile route 'work-guild'"):
        await runner._process_handoff(row)

    assert store.peek_session_id(entry.session_key) == entry.session_id
    adapter.get_chat_info.assert_awaited_once_with("parent-1")
    adapter.create_handoff_thread.assert_not_awaited()
    runner._handle_message.assert_not_awaited()
    adapter.send.assert_not_awaited()
    db.close()


@pytest.mark.asyncio
async def test_relay_handoff_reuses_persisted_scope_for_profile_routing(
    tmp_path,
):
    """Relay homes use /sethome provenance without an unsupported info probe."""
    config = GatewayConfig(
        sessions_dir=tmp_path / "sessions",
        multiplex_profiles=True,
        profile_routes=[
            ProfileRoute(
                name="default-guild",
                platform="discord",
                guild_id="guild-1",
                profile="default",
            )
        ],
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=False,
                home_channel=HomeChannel(
                    platform=Platform.DISCORD,
                    chat_id="parent-1",
                    name="Relay Discord Home",
                    user_id="discord-user-7",
                    scope_id="guild-1",
                ),
            ),
            Platform.RELAY: PlatformConfig(enabled=True),
        },
    )
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=config.sessions_dir, config=config)
    db = SessionDB(db_path=tmp_path / "state.db")
    store._db = db
    store._loaded = True

    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="webhook:alerts:relay-profile-route",
        chat_type="webhook",
        user_id="webhook:alerts",
        profile="default",
    )
    entry = store.get_or_create_session(source)
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_db = AsyncSessionDB(db)
    descriptor = CapabilityDescriptor(
        contract_version=CONTRACT_VERSION,
        platform="discord",
        label="Discord",
        max_message_length=2000,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="discord",
        len_unit="chars",
        supported_ops=("send", "thread_create"),
    )
    sent = []

    class _ColdRelayTransport:
        _identities = [("discord", "bot-1")]

        async def send_outbound(self, action, *, platform=None):
            sent.append((action, platform))
            if action.get("op") == "thread_create":
                return {"success": True, "thread_id": "thread-42"}
            return {"success": True, "message_id": "message-1"}

    relay = RelayAdapter(
        config.platforms[Platform.RELAY],
        descriptor,
        _ColdRelayTransport(),
    )
    runner.adapters = {Platform.RELAY: relay}
    runner._evict_cached_agent = MagicMock()
    runner._release_running_agent_state = MagicMock()
    runner._handle_message = AsyncMock(
        return_value="Ready in the relay handoff thread."
    )
    row = db.get_session(entry.session_id)
    row.update(
        {
            "handoff_platform": "discord",
            "source": "webhook",
            "session_key": entry.session_key,
        }
    )

    await runner._process_handoff(row)

    next_reply = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-42",
        chat_type="thread",
        user_id="discord-user-7",
        thread_id="thread-42",
        scope_id="guild-1",
        guild_id="guild-1",
        parent_chat_id="parent-1",
    )
    destination_key = runner._session_key_for_source(next_reply)
    assert store.peek_session_id(entry.session_key) is None
    assert store.peek_session_id(destination_key) == entry.session_id
    assert [action["op"] for action, _platform in sent] == [
        "thread_create",
        "send",
    ]
    for action, logical_platform in sent:
        assert logical_platform == "discord"
        assert action["chat_id"] == "parent-1"
        assert action["metadata"]["scope_id"] == "guild-1"
        assert action["metadata"]["user_id"] == "discord-user-7"
    assert sent[1][0]["metadata"]["thread_id"] == "thread-42"
    db.close()


@pytest.mark.asyncio
async def test_destination_occupied_during_thread_creation_is_not_stolen(tmp_path):
    """A Discord reply racing thread publication keeps its newly-created owner."""
    config = _discord_config(tmp_path)
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=config.sessions_dir, config=config)
    db = SessionDB(db_path=tmp_path / "state.db")
    store._db = db
    store._loaded = True

    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="webhook:alerts:racing-delivery",
        chat_type="webhook",
        user_id="webhook:alerts",
    )
    source_entry = store.get_or_create_session(source)
    runner, adapter = _runner_with_store(config, store, db)
    destination_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-42",
        chat_name="Hermes Home",
        chat_type="thread",
        user_id="fast-replier",
        thread_id="thread-42",
        scope_id="guild-1",
        parent_chat_id="parent-1",
    )
    occupant = None

    async def _create_then_receive_reply(_parent_chat_id, _name):
        nonlocal occupant
        occupant = store.get_or_create_session(destination_source)
        return "thread-42"

    adapter.create_handoff_thread.side_effect = _create_then_receive_reply
    row = db.get_session(source_entry.session_id)
    row.update(
        {
            "handoff_platform": "discord",
            "source": "webhook",
            "session_key": source_entry.session_key,
        }
    )

    with pytest.raises(RuntimeError, match="could not route session key"):
        await runner._process_handoff(row)

    destination_key = runner._session_key_for_source(destination_source)
    assert occupant is not None
    assert store.peek_session_id(source_entry.session_key) == source_entry.session_id
    assert store.peek_session_id(destination_key) == occupant.session_id
    assert occupant.session_id != source_entry.session_id
    runner._handle_message.assert_not_awaited()
    adapter.send.assert_not_awaited()
    db.close()


@pytest.mark.asyncio
async def test_interactive_handoff_keeps_legacy_discord_destination_shape(tmp_path):
    """CLI/TUI handoff does not adopt webhook-only provenance or home identity."""
    config = _discord_config(
        tmp_path,
        thread_sessions_per_user=True,
        home_user_id="discord-home-user",
    )
    # Interactive handoffs historically read the platform extra. Set it only
    # in this direct CLI/TUI regression instead of masking webhook tests in the
    # shared production-shape fixture.
    config.platforms[Platform.DISCORD].extra["thread_sessions_per_user"] = True
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=config.sessions_dir, config=config)
    db = SessionDB(db_path=tmp_path / "state.db")
    store._db = db
    store._loaded = True
    db.create_session("interactive-session", source="cli")
    runner, _adapter = _runner_with_store(config, store, db)

    await runner._process_handoff(
        {
            "id": "interactive-session",
            "source": "cli",
            "handoff_platform": "discord",
            "title": "Existing CLI session",
        }
    )

    synthetic_source = runner._handle_message.await_args.args[0].source
    assert synthetic_source == SessionSource(
        platform=Platform.DISCORD,
        chat_id="thread-42",
        chat_name="Hermes Home",
        chat_type="thread",
        user_id="system:handoff",
        user_name="Handoff",
        thread_id="thread-42",
    )
    destination_key = runner._session_key_for_source(synthetic_source)
    assert destination_key.endswith(":system:handoff")
    assert store.peek_session_id(destination_key) == "interactive-session"
    assert synthetic_source.user_id != config.get_home_channel(Platform.DISCORD).user_id
    db.close()


@pytest.mark.asyncio
async def test_webhook_handoff_requires_a_destination_thread(tmp_path):
    """Webhook mode never falls back to the legacy parent-channel lane."""
    config = _discord_config(tmp_path)
    store = MagicMock()
    store.move_session_route.side_effect = AssertionError(
        "routing must not move when thread creation failed"
    )
    db = MagicMock()
    runner, adapter = _runner_with_store(config, store, db)
    adapter.create_handoff_thread.return_value = None

    with pytest.raises(RuntimeError, match="could not create a handoff thread"):
        await runner._process_handoff(
            {
                "id": "webhook-session",
                "source": "webhook",
                "session_key": "agent:main:webhook:dm:route:delivery",
                "handoff_platform": "discord",
            }
        )

    adapter.send.assert_not_awaited()
    runner._handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_webhook_handoff_removes_route_and_ends_exact_session(tmp_path):
    config = _discord_config(tmp_path)
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=config.sessions_dir, config=config)
    db = SessionDB(db_path=tmp_path / "state.db")
    store._db = db
    store._loaded = True

    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="webhook:route:failed-delivery",
        chat_type="dm",
    )
    entry = store.get_or_create_session(source)
    assert db.request_handoff_once(entry.session_id, "discord") is True
    assert db.claim_handoff(entry.session_id) is True
    runner, _adapter = _runner_with_store(config, store, db)
    row = {
        "id": entry.session_id,
        "source": "webhook",
        "session_key": entry.session_key,
    }

    await runner._finalize_failed_webhook_handoff(row, "test failure")

    assert store.lookup_by_session_key(entry.session_key) is None
    durable = db.get_session(entry.session_id)
    assert durable["ended_at"] is not None
    assert durable["end_reason"] == "webhook_handoff_failed"
    db.close()


@pytest.mark.asyncio
async def test_post_move_send_failure_cleans_compressed_destination(
    tmp_path, monkeypatch
):
    """Cleanup follows a synthetic-turn compression child, not the stale root ID."""
    config = _discord_config(tmp_path)
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=config.sessions_dir, config=config)
    db = SessionDB(db_path=tmp_path / "state.db")
    store._db = db
    store._loaded = True

    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="webhook:route:compress-before-send",
        chat_type="webhook",
        user_id="webhook:route",
    )
    entry = store.get_or_create_session(source)
    assert db.request_handoff_once(entry.session_id, "discord") is True
    runner, adapter = _runner_with_store(config, store, db)
    compressed_session_id = "handoff-compression-tip"

    async def _compress_during_synthetic_turn(event):
        destination_key = runner._session_key_for_source(event.source)
        db.end_session(entry.session_id, "compression")
        db.create_session(
            compressed_session_id,
            source="discord",
            parent_session_id=entry.session_id,
        )
        advanced = store.advance_compression_session(
            destination_key,
            entry.session_id,
            compressed_session_id,
        )
        assert advanced is not None
        return "Compressed, but delivery will fail."

    runner._handle_message = AsyncMock(side_effect=_compress_during_synthetic_turn)
    adapter.send.return_value = SimpleNamespace(
        success=False,
        error="destination rejected message",
    )
    states = iter([True, False])

    class _Running:
        def __bool__(self):
            try:
                return next(states)
            except StopIteration:
                return False

    runner._running = _Running()

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr("gateway.run.asyncio.sleep", _no_sleep)
    await GatewayRunner._handoff_watcher(runner, interval=0)

    destination_source = runner._handle_message.await_args.args[0].source
    destination_key = runner._session_key_for_source(destination_source)
    assert store.lookup_by_session_key(entry.session_key) is None
    assert store.lookup_by_session_key(destination_key) is None
    assert db.get_session(entry.session_id)["end_reason"] == "compression"
    compressed = db.get_session(compressed_session_id)
    assert compressed["ended_at"] is not None
    assert compressed["end_reason"] == "webhook_handoff_failed"
    assert db.get_session(entry.session_id)["handoff_state"] == "failed"
    adapter.send.assert_awaited_once()
    db.close()


@pytest.mark.asyncio
async def test_post_move_cancellation_cleans_destination(tmp_path, monkeypatch):
    """Cancellation after ownership moves cannot leave either routing alias live."""
    config = _discord_config(tmp_path)
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=config.sessions_dir, config=config)
    db = SessionDB(db_path=tmp_path / "state.db")
    store._db = db
    store._loaded = True

    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="webhook:route:cancel-after-move",
        chat_type="webhook",
        user_id="webhook:route",
    )
    entry = store.get_or_create_session(source)
    assert db.request_handoff_once(entry.session_id, "discord") is True
    runner, adapter = _runner_with_store(config, store, db)
    runner._handle_message = AsyncMock(side_effect=asyncio.CancelledError())
    runner._running = True

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr("gateway.run.asyncio.sleep", _no_sleep)
    with pytest.raises(asyncio.CancelledError):
        await GatewayRunner._handoff_watcher(runner, interval=0)

    destination_source = runner._handle_message.await_args.args[0].source
    destination_key = runner._session_key_for_source(destination_source)
    assert store.lookup_by_session_key(entry.session_key) is None
    assert store.lookup_by_session_key(destination_key) is None
    durable = db.get_session(entry.session_id)
    assert durable["ended_at"] is not None
    assert durable["end_reason"] == "webhook_handoff_failed"
    assert durable["handoff_state"] == "failed"
    adapter.send.assert_not_awaited()
    db.close()


@pytest.mark.asyncio
async def test_claim_cancellation_reconciles_running_webhook_handoff(
    tmp_path, monkeypatch
):
    """Cancellation cannot strand a committed claim outside the pending scan."""
    config = _discord_config(tmp_path)
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=config.sessions_dir, config=config)
    db = SessionDB(db_path=tmp_path / "state.db")
    store._db = db
    store._loaded = True

    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="webhook:route:cancel-during-claim",
        chat_type="webhook",
        user_id="webhook:route",
    )
    entry = store.get_or_create_session(source)
    assert db.request_handoff_once(entry.session_id, "discord") is True
    runner, adapter = _runner_with_store(config, store, db)
    runner._running = True
    claim_started = asyncio.Event()
    release_claim = asyncio.Event()

    async def _claim_after_release(session_id):
        claim_started.set()
        await release_claim.wait()
        return db.claim_handoff(session_id)

    runner._session_db.claim_handoff = AsyncMock(side_effect=_claim_after_release)

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr("gateway.run.asyncio.sleep", _no_sleep)
    watcher_task = asyncio.create_task(
        GatewayRunner._handoff_watcher(runner, interval=0)
    )
    await claim_started.wait()
    watcher_task.cancel()
    release_claim.set()
    with pytest.raises(asyncio.CancelledError):
        await watcher_task

    assert store.lookup_by_session_key(entry.session_key) is None
    durable = db.get_session(entry.session_id)
    assert durable["handoff_state"] == "failed"
    assert durable["handoff_error"] == "handoff claim was cancelled"
    assert durable["ended_at"] is not None
    assert durable["end_reason"] == "webhook_handoff_failed"
    adapter.create_handoff_thread.assert_not_awaited()
    runner._handle_message.assert_not_awaited()
    adapter.send.assert_not_awaited()
    db.close()


@pytest.mark.asyncio
async def test_synthetic_agent_failure_cleans_moved_destination(
    tmp_path, monkeypatch
):
    """A normalized agent error response cannot complete a webhook handoff."""
    config = _discord_config(tmp_path)
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=config.sessions_dir, config=config)
    db = SessionDB(db_path=tmp_path / "state.db")
    store._db = db
    store._loaded = True

    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="webhook:route:synthetic-agent-failed",
        chat_type="webhook",
        user_id="webhook:route",
    )
    entry = store.get_or_create_session(source)
    assert db.request_handoff_once(entry.session_id, "discord") is True
    runner, adapter = _runner_with_store(config, store, db)

    async def _failed_synthetic_turn(event):
        event._agent_run_failed = True
        return "Sorry, I encountered an unexpected error."

    runner._handle_message = AsyncMock(side_effect=_failed_synthetic_turn)
    states = iter([True, False])

    class _Running:
        def __bool__(self):
            try:
                return next(states)
            except StopIteration:
                return False

    runner._running = _Running()

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr("gateway.run.asyncio.sleep", _no_sleep)
    await GatewayRunner._handoff_watcher(runner, interval=0)

    destination_source = runner._handle_message.await_args.args[0].source
    destination_key = runner._session_key_for_source(destination_source)
    assert store.lookup_by_session_key(entry.session_key) is None
    assert store.lookup_by_session_key(destination_key) is None
    durable = db.get_session(entry.session_id)
    assert durable["handoff_state"] == "failed"
    assert durable["handoff_error"] == "synthetic destination agent run failed"
    assert durable["ended_at"] is not None
    assert durable["end_reason"] == "webhook_handoff_failed"
    adapter.send.assert_not_awaited()
    db.close()


@pytest.mark.asyncio
async def test_pending_handoff_recovers_after_restart_and_missing_home_fails_cleanly(
    tmp_path, monkeypatch
):
    """A persisted request is claimed after restart; pre-move failure leaves no ghost."""
    config = GatewayConfig(
        sessions_dir=tmp_path / "sessions",
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                token="test-token",
                home_channel=None,
            )
        },
    )
    with patch("gateway.session.SessionStore._ensure_loaded"):
        original_store = SessionStore(sessions_dir=config.sessions_dir, config=config)
    db = SessionDB(db_path=tmp_path / "state.db")
    original_store._db = db
    original_store._loaded = True
    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="webhook:route:pending-before-restart",
        chat_type="dm",
    )
    entry = original_store.get_or_create_session(source)
    assert db.request_handoff_once(entry.session_id, "discord") is True

    # New store instance models a gateway restart loading the durable routing
    # index and pending handoff from state.db.
    restarted_store = SessionStore(sessions_dir=config.sessions_dir, config=config)
    restarted_store._db = db
    runner, adapter = _runner_with_store(config, restarted_store, db)
    states = iter([True, False])

    class _Running:
        def __bool__(self):
            try:
                return next(states)
            except StopIteration:
                return False

    runner._running = _Running()

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr("gateway.run.asyncio.sleep", _no_sleep)
    await GatewayRunner._handoff_watcher(runner, interval=0)

    assert restarted_store.lookup_by_session_key(entry.session_key) is None
    durable = db.get_session(entry.session_id)
    assert durable["handoff_state"] == "failed"
    assert "no home channel configured" in durable["handoff_error"]
    assert durable["ended_at"] is not None
    assert durable["end_reason"] == "webhook_handoff_failed"
    adapter.create_handoff_thread.assert_not_awaited()
    adapter.send.assert_not_awaited()
    db.close()
