"""Focused tests for immediate Telegram reaction-event routing."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource, build_session_key


def _make_reaction_adapter(
    monkeypatch,
    tmp_path,
    *,
    authorized=True,
    thread_id: str | None = "77",
    sender_id="111",
):
    from gateway import rich_sent_store
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setattr(
        rich_sent_store,
        "_store_path",
        lambda: str(tmp_path / "state" / "rich_sent_index.json"),
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: tmp_path / "base",
    )
    rich_sent_store.record(
        "-100",
        "900",
        "A bot-authored answer",
        thread_id=thread_id,
        sender_id=sender_id,
    )

    origin = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100",
        chat_type="forum",
        user_id="1000",
        user_name="Hermes user",
        thread_id=thread_id,
        profile="default",
    )
    session_entry = SimpleNamespace(origin=origin)
    runner = SimpleNamespace(
        session_store=SimpleNamespace(_entries={"telegram-session": session_entry}),
        _is_user_authorized=Mock(return_value=authorized),
        _session_key_for_source=Mock(
            side_effect=lambda source: f"telegram:{source.user_id}:{source.thread_id or 'root'}"
        ),
        _set_pending_turn_sidecar_notes=Mock(),
        _peek_session_state=lambda key: None,
        _profile_name_for_source=lambda source: "default",
        _profile_adapters={},
    )
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.gateway_runner = runner
    adapter._bot = SimpleNamespace(id=111)
    adapter.handle_message = AsyncMock()
    return adapter, runner


def _update(*, old, new, user_id="42", is_bot=False, chat_id="-100", update_id=123):
    return SimpleNamespace(
        update_id=update_id,
        message_reaction=SimpleNamespace(
            chat=SimpleNamespace(id=chat_id, type="supergroup", is_forum=True),
            message_id=900,
            user=SimpleNamespace(
                id=user_id,
                username="authorized-user",
                full_name="Authorized User",
                is_bot=is_bot,
            ),
            old_reaction=[SimpleNamespace(emoji=value) for value in old],
            new_reaction=[SimpleNamespace(emoji=value) for value in new],
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("old", "new", "added", "removed"),
    [
        ([], ["❤️"], "added ❤️", None),
        (["👍"], ["❤️"], "added ❤️", "removed 👍"),
        (["❤️"], [], None, "removed ❤️"),
    ],
)
async def test_reaction_delta_starts_immediate_authenticated_turn(
    monkeypatch, tmp_path, old, new, added, removed,
):
    adapter, runner = _make_reaction_adapter(monkeypatch, tmp_path)

    await adapter._handle_message_reaction(_update(old=old, new=new))

    runner._is_user_authorized.assert_called_once()
    source = runner._is_user_authorized.call_args.args[0]
    assert source.chat_id == "-100"
    assert source.chat_type == "group"
    assert source.thread_id == "77"
    assert source.user_id == "42"
    runner._session_key_for_source.assert_called_once_with(source)
    runner._set_pending_turn_sidecar_notes.assert_not_called()
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source is source
    assert event.user_id == "42"
    assert event.user_name == "authorized-user"
    assert event.message_id is None
    assert event.platform_update_id == 123
    assert event.reply_to_message_id == "900"
    assert event.reply_to_text == "A bot-authored answer"
    assert event.reply_to_is_own_message is True
    assert event.internal is False
    assert event.metadata["telegram_reaction_event"] is True
    assert event.metadata["deferred_followup_event"] is True
    assert event.metadata["telegram_reaction_target_message_id"] == "900"
    assert event.metadata["telegram_reaction_session_key"] == "telegram:42:77"
    note = event.text
    if added:
        assert added in note
    if removed:
        assert removed in note
    assert '"A bot-authored answer"' in note
    assert "return exactly NO_REPLY" in event.channel_prompt
    assert "consequential or risky action" in event.channel_prompt


@pytest.mark.asyncio
async def test_reaction_authorization_and_bot_actor_filtering(monkeypatch, tmp_path):
    adapter, runner = _make_reaction_adapter(monkeypatch, tmp_path, authorized=False)

    await adapter._handle_message_reaction(_update(old=[], new=["👍"]))
    runner._set_pending_turn_sidecar_notes.assert_not_called()
    assert adapter.handle_message.await_count == 0

    adapter, runner = _make_reaction_adapter(monkeypatch, tmp_path, authorized=True)
    await adapter._handle_message_reaction(
        _update(old=[], new=["👍"], is_bot=True)
    )
    runner._is_user_authorized.assert_not_called()
    runner._set_pending_turn_sidecar_notes.assert_not_called()
    assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
async def test_unknown_or_unrelated_target_is_ignored(monkeypatch, tmp_path):
    adapter, runner = _make_reaction_adapter(monkeypatch, tmp_path)

    await adapter._handle_message_reaction(
        _update(old=[], new=["👍"], chat_id="-200")
    )

    runner._is_user_authorized.assert_not_called()
    runner._set_pending_turn_sidecar_notes.assert_not_called()
    assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
async def test_forum_target_without_thread_identity_is_ignored(monkeypatch, tmp_path):
    adapter, runner = _make_reaction_adapter(
        monkeypatch, tmp_path, thread_id=None,
    )

    await adapter._handle_message_reaction(_update(old=[], new=["👍"]))

    runner._is_user_authorized.assert_not_called()
    runner._session_key_for_source.assert_not_called()
    assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
async def test_target_not_sent_by_hermes_is_ignored(monkeypatch, tmp_path):
    adapter, runner = _make_reaction_adapter(monkeypatch, tmp_path)

    update = _update(old=[], new=["👍"])
    update.message_reaction.message_id = 901
    await adapter._handle_message_reaction(update)

    runner._is_user_authorized.assert_not_called()
    runner._set_pending_turn_sidecar_notes.assert_not_called()
    assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
async def test_target_sent_by_different_telegram_bot_is_ignored(monkeypatch, tmp_path):
    adapter, runner = _make_reaction_adapter(monkeypatch, tmp_path)
    adapter._bot = SimpleNamespace(id=222)

    await adapter._handle_message_reaction(_update(old=[], new=["👍"]))

    runner._is_user_authorized.assert_not_called()
    assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
async def test_legacy_unstamped_target_is_ignored_with_multiple_bots(
    monkeypatch, tmp_path,
):
    adapter, runner = _make_reaction_adapter(
        monkeypatch, tmp_path, sender_id=None,
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {
        "family": {Platform.TELEGRAM: object()},
    }

    await adapter._handle_message_reaction(_update(old=[], new=["👍"]))

    runner._is_user_authorized.assert_not_called()
    assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
async def test_no_delta_is_ignored(monkeypatch, tmp_path):
    adapter, runner = _make_reaction_adapter(monkeypatch, tmp_path)

    await adapter._handle_message_reaction(_update(old=["👍"], new=["👍"]))

    runner._is_user_authorized.assert_not_called()
    runner._set_pending_turn_sidecar_notes.assert_not_called()
    assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
async def test_duplicate_reaction_update_id_starts_only_one_turn(monkeypatch, tmp_path):
    adapter, runner = _make_reaction_adapter(monkeypatch, tmp_path)
    update = _update(old=[], new=["👍"], update_id=456)

    await adapter._handle_message_reaction(update)
    await adapter._handle_message_reaction(update)

    assert adapter.handle_message.await_count == 1
    assert runner._is_user_authorized.call_count == 2


@pytest.mark.asyncio
async def test_reaction_lookup_finds_newest_matching_entry_in_secondary_profile(
    monkeypatch, tmp_path,
):
    from gateway import rich_sent_store
    from plugins.platforms.telegram.adapter import TelegramAdapter

    current_path = tmp_path / "current" / "state" / "rich_sent_index.json"
    base_home = tmp_path / "base"
    secondary_path = base_home / "profiles" / "secondary" / "state" / "rich_sent_index.json"
    monkeypatch.setattr(rich_sent_store, "_store_path", lambda: str(current_path))
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: base_home,
    )
    current_path.parent.mkdir(parents=True)
    current_path.write_text(
        json.dumps({"-100:900": {"t": "older answer", "ts": 100, "thread_id": "77"}}),
        encoding="utf-8",
    )
    secondary_path.parent.mkdir(parents=True)
    secondary_path.write_text(
        json.dumps({"-100:900": {"t": "secondary answer", "ts": 200, "thread_id": "77"}}),
        encoding="utf-8",
    )

    captured = {}
    runner = SimpleNamespace(
        _is_user_authorized=Mock(return_value=True),
        _session_key_for_source=Mock(
            side_effect=lambda current: captured.setdefault("source", current)
            and "actor-session"
        ),
        _set_pending_turn_sidecar_notes=Mock(),
        _peek_session_state=lambda key: None,
        _profile_name_for_source=lambda current: "secondary",
        _profile_adapters={},
    )
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.gateway_runner = runner
    adapter.handle_message = AsyncMock()

    await adapter._handle_message_reaction(_update(old=[], new=["👍"]))

    assert captured["source"].user_id == "42"
    assert captured["source"].thread_id == "77"
    assert captured["source"].profile == "secondary"
    runner._set_pending_turn_sidecar_notes.assert_not_called()
    event = adapter.handle_message.await_args.args[0]
    assert event.metadata["telegram_reaction_session_key"] == "actor-session"
    assert "secondary answer" in event.text


def test_conflicting_profile_index_routes_are_ignored(monkeypatch, tmp_path):
    from gateway import rich_sent_store

    current_path = tmp_path / "current" / "state" / "rich_sent_index.json"
    base_home = tmp_path / "base"
    secondary_path = (
        base_home / "profiles" / "secondary" / "state" / "rich_sent_index.json"
    )
    monkeypatch.setattr(rich_sent_store, "_store_path", lambda: str(current_path))
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: base_home,
    )
    current_path.parent.mkdir(parents=True)
    current_path.write_text(
        json.dumps({"-100:900": {"t": "one", "ts": 100, "thread_id": "77"}}),
        encoding="utf-8",
    )
    secondary_path.parent.mkdir(parents=True)
    secondary_path.write_text(
        json.dumps({"-100:900": {"t": "two", "ts": 200, "thread_id": "88"}}),
        encoding="utf-8",
    )

    assert rich_sent_store.lookup("-100", "900") == "one"
    local_entry = rich_sent_store.lookup_entry("-100", "900")
    assert local_entry is not None
    assert local_entry["thread_id"] == "77"
    assert rich_sent_store.lookup_entry("-100", "900", all_profiles=True) is None


@pytest.mark.asyncio
async def test_reaction_turn_is_actor_specific_and_does_not_touch_sidecar_notes(
    monkeypatch, tmp_path,
):
    from gateway import rich_sent_store
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setattr(
        rich_sent_store,
        "_store_path",
        lambda: str(tmp_path / "state" / "rich_sent_index.json"),
    )
    rich_sent_store.record("-100", "900", "answer", thread_id="77")
    captured = {}
    runner = SimpleNamespace(
        _is_user_authorized=Mock(return_value=True),
        _session_key_for_source=Mock(
            side_effect=lambda source: captured.setdefault("source", source)
            and "actor-specific-session"
        ),
        _set_pending_turn_sidecar_notes=Mock(),
        _peek_session_state=lambda key: None,
    )
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.gateway_runner = runner
    adapter.handle_message = AsyncMock()

    await adapter._handle_message_reaction(_update(old=[], new=["👍"], user_id="777"))

    assert captured["source"].user_id == "777"
    runner._set_pending_turn_sidecar_notes.assert_not_called()
    event = adapter.handle_message.await_args.args[0]
    assert event.metadata["telegram_reaction_session_key"] == "actor-specific-session"
    assert "added 👍" in event.text
    assert captured["source"] is runner._session_key_for_source.call_args.args[0]


@pytest.mark.asyncio
async def test_forum_reaction_uses_same_real_session_key_as_forum_message(
    monkeypatch, tmp_path,
):
    from gateway import rich_sent_store
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setattr(
        rich_sent_store,
        "_store_path",
        lambda: str(tmp_path / "state" / "rich_sent_index.json"),
    )
    rich_sent_store.record("-100", "900", "answer", thread_id="77")
    routed_sources = []

    def real_session_key(source):
        routed_sources.append(source)
        return build_session_key(
            source,
            group_sessions_per_user=True,
            thread_sessions_per_user=False,
        )

    runner = SimpleNamespace(
        _is_user_authorized=Mock(return_value=True),
        _session_key_for_source=real_session_key,
        _set_pending_turn_sidecar_notes=Mock(),
        _peek_session_state=lambda key: None,
        _profile_adapters={},
    )
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.gateway_runner = runner
    adapter.handle_message = AsyncMock()

    await adapter._handle_message_reaction(_update(old=[], new=["👍"]))

    expected_message_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100",
        chat_type="group",
        user_id="42",
        thread_id="77",
    )
    expected_key = build_session_key(
        expected_message_source,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    event = adapter.handle_message.await_args.args[0]
    assert routed_sources[0].chat_type == "group"
    assert routed_sources[0].thread_id == "77"
    assert event.metadata["telegram_reaction_session_key"] == expected_key
    runner._set_pending_turn_sidecar_notes.assert_not_called()


def test_reaction_turn_reply_anchor_targets_reacted_root_group_message():
    from gateway.platforms.base import MessageEvent, _reply_anchor_for_event

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100",
        chat_type="group",
        user_id="42",
        thread_id=None,
    )
    event = MessageEvent(
        text="[Telegram reaction: added 👍]",
        source=source,
        reply_to_message_id="900",
        metadata={"telegram_reaction_target_message_id": "900"},
    )

    assert _reply_anchor_for_event(event) == "900"


def test_reaction_turn_in_forum_routes_by_topic_not_reply_anchor():
    from gateway.platforms.base import MessageEvent, _reply_anchor_for_event

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100",
        chat_type="group",
        user_id="42",
        thread_id="77",
    )
    event = MessageEvent(
        text="[Telegram reaction: added 👍]",
        source=source,
        reply_to_message_id="900",
        metadata={"telegram_reaction_target_message_id": "900"},
    )

    assert _reply_anchor_for_event(event) is None


def test_sent_result_ignores_non_forum_reply_anchor_thread_id():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter._record_sent_message = Mock()
    result = SimpleNamespace(
        message_id=900,
        message_thread_id=123,
        is_topic_message=False,
        chat=SimpleNamespace(is_forum=False),
        text="answer",
        caption=None,
    )

    adapter._record_sent_result(
        "-100",
        result,
        effective_thread_id=None,
    )

    assert adapter._record_sent_message.call_args.kwargs["effective_thread_id"] is None


@pytest.mark.asyncio
async def test_live_general_topic_final_send_then_reaction_routes_immediate_turn(
    monkeypatch, tmp_path
):
    """End-to-end regression for the live General-topic reaction failure.

    Drive the production ``send()`` path exactly as the gateway does for a
    final reply in a forum's General topic (logical thread id "1"): the
    transport request must omit ``message_thread_id`` and Telegram's success
    response omits it as well. The sent index must still record the logical
    topic so the user's reaction — which carries no topic identity — routes
    to an immediate MessageEvent in thread "1" instead of being fail-closed
    dropped.
    """
    from gateway import rich_sent_store
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setattr(
        rich_sent_store,
        "_store_path",
        lambda: str(tmp_path / "state" / "rich_sent_index.json"),
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: tmp_path / "base",
    )

    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="fake-token",
            extra={"rich_messages": False, "inbound_reactions": True},
        )
    )
    bot = MagicMock()
    bot.id = 111
    bot.send_message = AsyncMock(
        return_value=SimpleNamespace(message_id=700, message_thread_id=None)
    )
    bot.send_chat_action = AsyncMock()
    adapter._bot = bot

    send_result = await adapter.send(
        "-100",
        "The final answer.",
        metadata={"thread_id": "1", "notify": True},
    )

    assert send_result.success is True
    # Live transport constraint: the send really omitted the General topic id.
    assert bot.send_message.await_args.kwargs["message_thread_id"] is None
    entry = rich_sent_store.lookup_entry("-100", "700")
    assert entry is not None
    assert entry["thread_id"] == "1"
    assert entry["sender_id"] == "111"

    runner = SimpleNamespace(
        _is_user_authorized=Mock(return_value=True),
        _session_key_for_source=Mock(return_value="telegram:42:1"),
        _profile_adapters={},
    )
    adapter.gateway_runner = runner
    adapter.handle_message = AsyncMock()

    update = _update(old=[], new=["👍"])
    update.message_reaction.message_id = 700
    await adapter._handle_message_reaction(update)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.thread_id == "1"
    assert event.reply_to_message_id == "700"
    assert event.metadata["telegram_reaction_event"] is True
    assert event.metadata["telegram_reaction_target_message_id"] == "700"
    assert event.metadata["telegram_reaction_session_key"] == "telegram:42:1"
    routed_source = runner._session_key_for_source.call_args.args[0]
    assert routed_source.thread_id == "1"
