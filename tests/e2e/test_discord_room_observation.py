"""Composed Discord room-observation tests with real durable session state.

These tests stop at the agent boundary but exercise the actual adapter, base
adapter routing, gateway authorization/disposition path, and SessionStore.
"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import hermes_state
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.room_observation import load_observed_room_context, room_scoped_source
from gateway.session import SessionSource, SessionStore
from tests.e2e.conftest import (
    BOT_USER_ID,
    CHANNEL_ID,
    _make_discord_adapter_wired,
    make_discord_message,
    make_fake_text_channel,
    make_fake_thread,
    make_runner,
)

pytestmark = pytest.mark.asyncio
_REAL_SESSION_DB = hermes_state.SessionDB


def _author(user_id: int, name: str):
    return SimpleNamespace(
        id=user_id,
        name=name.lower(),
        display_name=name,
        bot=False,
    )


def _source_for(author, channel) -> SessionSource:
    parent_id = getattr(channel, "parent_id", None)
    is_thread = parent_id is not None
    return SessionSource(
        platform=Platform.DISCORD,
        scope_id=str(channel.guild.id),
        chat_id=str(channel.id),
        chat_name=f"{channel.guild.name} / #{channel.name}",
        chat_type="thread" if is_thread else "group",
        user_id=str(author.id),
        user_name=author.display_name,
        thread_id=str(channel.id) if is_thread else None,
        parent_chat_id=str(parent_id) if parent_id is not None else None,
    )


def _new_store(*, sessions_dir, db_path, config) -> SessionStore:
    with patch.object(
        hermes_state,
        "SessionDB",
        side_effect=lambda: _REAL_SESSION_DB(db_path),
    ):
        return SessionStore(sessions_dir=sessions_dir, config=config)


async def _drain_adapter(adapter) -> None:
    """Wait until all BasePlatformAdapter processing tasks have completed."""

    for _ in range(20):
        tasks = list(adapter._background_tasks)
        if tasks:
            await asyncio.gather(*tasks)
        await asyncio.sleep(0)
        if not adapter._background_tasks:
            return
    raise AssertionError("adapter background tasks did not drain")


async def _route_authorized_message(stack, message) -> bool:
    """Mirror the authorized portion of Discord's on_message callback."""

    observed = await stack.adapter._maybe_observe_room_message(message)
    if not observed and stack.adapter._discord_multi_agent_message_allowed(message):
        await stack.adapter._handle_message(message)
    await _drain_adapter(stack.adapter)
    return observed


def _room_context(stack, author, channel) -> str | None:
    return load_observed_room_context(
        stack.store,
        _source_for(author, channel),
        now=datetime.now(timezone.utc),
    )


@pytest.fixture()
def presence_stack(tmp_path, monkeypatch):
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_HISTORY_BACKFILL", "false")
    monkeypatch.setenv("HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS", "0")
    monkeypatch.setenv("HERMES_DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS", "0")
    monkeypatch.delenv("DISCORD_ALLOWED_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_IGNORED_CHANNELS", raising=False)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])

    sessions_dir = tmp_path / "sessions"
    db_path = tmp_path / "state.db"
    platform_config = PlatformConfig(
        enabled=True,
        token="e2e-test-token",
        extra={
            "require_mention": True,
            "observe_unmentioned_group_messages": True,
            "observed_channels": [str(CHANNEL_ID)],
            "free_response_channels": [],
        },
    )
    config = GatewayConfig(platforms={Platform.DISCORD: platform_config})
    config.sessions_dir = sessions_dir

    runner = make_runner(Platform.DISCORD)
    runner.config = config
    runner._is_user_authorized = MagicMock(return_value=True)
    store = _new_store(
        sessions_dir=sessions_dir,
        db_path=db_path,
        config=config,
    )
    runner.session_store = store

    adapter, _ = _make_discord_adapter_wired(runner)
    adapter.config = platform_config

    agent_calls = []

    async def fake_agent_boundary(event, source, session_key, run_generation):
        entry = store.get_or_create_session(source)
        context = load_observed_room_context(
            store,
            source,
            now=datetime.now(timezone.utc),
        )
        agent_calls.append(
            SimpleNamespace(
                event=event,
                source=source,
                session_key=session_key,
                run_generation=run_generation,
                session_id=entry.session_id,
                room_context=context,
            )
        )
        return None

    runner._handle_message_with_agent = AsyncMock(side_effect=fake_agent_boundary)
    stack = SimpleNamespace(
        adapter=adapter,
        runner=runner,
        store=store,
        sessions_dir=sessions_dir,
        db_path=db_path,
        config=config,
        agent_calls=agent_calls,
    )
    yield stack
    db = store._db
    if db is not None:
        db.close()


async def test_alice_ambient_persists_once_without_agent_reply_or_typing(presence_stack):
    alice = _author(101, "Alice")
    channel = make_fake_text_channel()
    message = make_discord_message(
        content="The launch moved to Tuesday.",
        author=alice,
        channel=channel,
    )
    message.add_reaction = AsyncMock()
    message.remove_reaction = AsyncMock()

    observed = await _route_authorized_message(presence_stack, message)

    assert observed is True
    assert _room_context(presence_stack, alice, channel) == (
        "[Alice|101]\nThe launch moved to Tuesday."
    )
    presence_stack.runner._handle_message_with_agent.assert_not_awaited()
    presence_stack.adapter.send.assert_not_awaited()
    presence_stack.adapter.send_typing.assert_not_awaited()
    message.add_reaction.assert_not_awaited()
    message.remove_reaction.assert_not_awaited()


async def test_bob_and_alice_share_room_stream_with_stable_attribution(presence_stack):
    alice = _author(101, "Alice")
    bob = _author(202, "Bob")
    channel = make_fake_text_channel()

    await _route_authorized_message(
        presence_stack,
        make_discord_message(content="Tuesday works.", author=alice, channel=channel),
    )
    await _route_authorized_message(
        presence_stack,
        make_discord_message(content="I will update the brief.", author=bob, channel=channel),
    )

    expected = (
        "[Alice|101]\nTuesday works.\n"
        "[Bob|202]\nI will update the brief."
    )
    assert _room_context(presence_stack, alice, channel) == expected
    assert _room_context(presence_stack, bob, channel) == expected
    room_entry = presence_stack.store.get_existing_session(
        room_scoped_source(_source_for(alice, channel))
    )
    assert room_entry is not None
    rows = presence_stack.store.load_transcript(room_entry.session_id)
    assert [row["content"] for row in rows] == [
        "[Alice|101]\nTuesday works.",
        "[Bob|202]\nI will update the brief.",
    ]
    presence_stack.runner._handle_message_with_agent.assert_not_awaited()


async def test_alice_mention_uses_alice_session_and_shared_room_context(presence_stack):
    alice = _author(101, "Alice")
    channel = make_fake_text_channel()
    await _route_authorized_message(
        presence_stack,
        make_discord_message(content="The launch moved.", author=alice, channel=channel),
    )

    observed = await _route_authorized_message(
        presence_stack,
        make_discord_message(
            content=f"<@{BOT_USER_ID}> when is the launch?",
            author=alice,
            channel=channel,
            mentions=[presence_stack.adapter._client.user],
        ),
    )

    assert observed is False
    presence_stack.runner._handle_message_with_agent.assert_awaited_once()
    call = presence_stack.agent_calls[0]
    assert call.source.user_id == "101"
    assert call.room_context == "[Alice|101]\nThe launch moved."
    assert call.session_key.endswith(":101")


async def test_shared_thread_observation_session_is_distinct_from_conversation(
    presence_stack,
):
    alice = _author(101, "Alice")
    parent = make_fake_text_channel()
    thread = make_fake_thread(parent=parent)
    presence_stack.adapter.config.extra["observed_channels"].append(str(thread.id))
    presence_stack.config.thread_sessions_per_user = False

    assert await _route_authorized_message(
        presence_stack,
        make_discord_message(
            content="Ambient thread fact.",
            author=alice,
            channel=thread,
        ),
    ) is True

    room_entry = presence_stack.store.get_existing_session(
        room_scoped_source(_source_for(alice, thread))
    )
    assert room_entry is not None

    assert await _route_authorized_message(
        presence_stack,
        make_discord_message(
            content=f"<@{BOT_USER_ID}> what was the ambient fact?",
            author=alice,
            channel=thread,
            mentions=[presence_stack.adapter._client.user],
        ),
    ) is False

    call = presence_stack.agent_calls[0]
    assert call.room_context == "[Alice|101]\nAmbient thread fact."
    assert call.session_id != room_entry.session_id
    assert call.session_key != room_entry.session_key
    assert presence_stack.store.load_transcript(room_entry.session_id)[0]["observed"] is True


async def test_managed_self_role_dispatches_without_delayed_room_observation(
    presence_stack,
):
    lucy = _author(451517281223180310, "LucyferOS")
    channel = make_fake_text_channel()
    self_role = SimpleNamespace(
        id=1501871552705204237,
        tags=SimpleNamespace(bot_id=BOT_USER_ID),
        managed=True,
    )
    role_question = make_discord_message(
        content=(
            f"<@&{self_role.id}> what did ciacon and Destruxus "
            "contribute to our conversation?"
        ),
        author=lucy,
        channel=channel,
    )
    role_question.role_mentions = [self_role]

    observed = await _route_authorized_message(presence_stack, role_question)

    assert observed is False
    assert len(presence_stack.agent_calls) == 1
    first_call = presence_stack.agent_calls[0]
    assert first_call.event.text == (
        "what did ciacon and Destruxus contribute to our conversation?"
    )
    assert first_call.event.addressing.direct_mention is True
    assert first_call.room_context is None
    assert _room_context(presence_stack, lucy, channel) is None

    observed = await _route_authorized_message(
        presence_stack,
        make_discord_message(
            content=f"<@{BOT_USER_ID}> are you there?",
            author=lucy,
            channel=channel,
            mentions=[presence_stack.adapter._client.user],
        ),
    )

    assert observed is False
    assert len(presence_stack.agent_calls) == 2
    second_call = presence_stack.agent_calls[1]
    assert second_call.event.text == "are you there?"
    assert second_call.room_context is None
    assert _room_context(presence_stack, lucy, channel) is None


async def test_bob_mention_gets_separate_session_and_same_room_context(presence_stack):
    alice = _author(101, "Alice")
    bob = _author(202, "Bob")
    channel = make_fake_text_channel()
    await _route_authorized_message(
        presence_stack,
        make_discord_message(content="The launch moved.", author=alice, channel=channel),
    )

    for author, question in (
        (alice, "when is it?"),
        (bob, "what changed?"),
    ):
        await _route_authorized_message(
            presence_stack,
            make_discord_message(
                content=f"<@{BOT_USER_ID}> {question}",
                author=author,
                channel=channel,
                mentions=[presence_stack.adapter._client.user],
            ),
        )

    alice_call, bob_call = presence_stack.agent_calls
    assert alice_call.session_key != bob_call.session_key
    assert alice_call.session_id != bob_call.session_id
    assert alice_call.room_context == bob_call.room_context
    assert bob_call.room_context == "[Alice|101]\nThe launch moved."


async def test_other_room_and_thread_are_absent_from_observed_context(presence_stack):
    alice = _author(101, "Alice")
    observed_channel = make_fake_text_channel()
    other_channel = make_fake_text_channel(channel_id=CHANNEL_ID + 1, name="other")
    observed_thread = make_fake_thread(parent=observed_channel)
    presence_stack.adapter.config.extra["observed_channels"].append(
        str(observed_thread.id)
    )

    await _route_authorized_message(
        presence_stack,
        make_discord_message(
            content="Visible only elsewhere.",
            author=alice,
            channel=other_channel,
        ),
    )
    await _route_authorized_message(
        presence_stack,
        make_discord_message(
            content="Visible here.",
            author=alice,
            channel=observed_channel,
        ),
    )
    await _route_authorized_message(
        presence_stack,
        make_discord_message(
            content="Visible only in the thread.",
            author=alice,
            channel=observed_thread,
        ),
    )

    context = _room_context(presence_stack, alice, observed_channel)
    assert context == "[Alice|101]\nVisible here."
    assert "elsewhere" not in context
    assert "thread" not in context
    assert _room_context(presence_stack, alice, other_channel) is None
    assert _room_context(presence_stack, alice, observed_thread) == (
        "[Alice|101]\nVisible only in the thread."
    )


async def test_message_to_another_bot_observes_but_never_dispatches(presence_stack):
    alice = _author(101, "Alice")
    other_bot = SimpleNamespace(id=303, name="OtherBot", bot=True)
    channel = make_fake_text_channel()

    observed = await _route_authorized_message(
        presence_stack,
        make_discord_message(
            content="<@303> please update the ticket",
            author=alice,
            channel=channel,
            mentions=[other_bot],
        ),
    )

    assert observed is True
    context = _room_context(
        presence_stack,
        alice,
        channel,
    )
    assert context is not None
    assert "please update the ticket" in context
    presence_stack.runner._handle_message_with_agent.assert_not_awaited()
    presence_stack.adapter.send.assert_not_awaited()


async def test_feature_disabled_preserves_legacy_ambient_drop(presence_stack):
    presence_stack.adapter.config.extra["observe_unmentioned_group_messages"] = False
    alice = _author(101, "Alice")
    channel = make_fake_text_channel()
    message = make_discord_message(
        content="ambient legacy traffic",
        author=alice,
        channel=channel,
    )
    before = message.content

    observed = await _route_authorized_message(presence_stack, message)

    assert observed is False
    assert message.content == before
    assert presence_stack.store._entries == {}
    presence_stack.runner._handle_message_with_agent.assert_not_awaited()
    presence_stack.adapter.send.assert_not_awaited()
    presence_stack.adapter.send_typing.assert_not_awaited()


async def test_restart_reloads_observations_without_creating_pending_work(presence_stack):
    alice = _author(101, "Alice")
    channel = make_fake_text_channel()
    await _route_authorized_message(
        presence_stack,
        make_discord_message(
            content="Durable room fact.",
            author=alice,
            channel=channel,
        ),
    )
    original_db = presence_stack.store._db
    assert original_db is not None
    original_db.close()
    presence_stack.store._db = None

    reloaded = _new_store(
        sessions_dir=presence_stack.sessions_dir,
        db_path=presence_stack.db_path,
        config=presence_stack.config,
    )
    try:
        assert load_observed_room_context(
            reloaded,
            _source_for(alice, channel),
            now=datetime.now(timezone.utc),
        ) == "[Alice|101]\nDurable room fact."

        fresh_runner = make_runner(Platform.DISCORD)
        fresh_runner.config = presence_stack.config
        fresh_runner.session_store = reloaded
        fresh_runner._handle_message_with_agent = AsyncMock()
        fresh_adapter, _ = _make_discord_adapter_wired(fresh_runner)
        await asyncio.sleep(0)

        fresh_runner._handle_message_with_agent.assert_not_awaited()
        assert fresh_adapter._background_tasks == set()
        assert fresh_adapter._pending_messages == {}
        cast(AsyncMock, fresh_adapter.send).assert_not_awaited()
    finally:
        reloaded_db = reloaded._db
        if reloaded_db is not None:
            reloaded_db.close()
