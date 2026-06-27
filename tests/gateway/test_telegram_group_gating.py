import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from gateway.config import Platform, PlatformConfig, load_gateway_config
from gateway.platforms.base import MessageType
from gateway.session import SessionSource


def _make_adapter(
    require_mention=None,
    free_response_chats=None,
    mention_patterns=None,
    exclusive_bot_mentions=None,
    ignored_threads=None,
    allowed_topics=None,
    allow_from=None,
    group_allow_from=None,
    allowed_chats=None,
    group_allowed_chats=None,
    guest_mode=None,
    observe_unmentioned_group_messages=None,
    bot_username="hermes_bot",
):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    extra = {}
    if require_mention is not None:
        extra["require_mention"] = require_mention
    if free_response_chats is not None:
        extra["free_response_chats"] = free_response_chats
    if mention_patterns is not None:
        extra["mention_patterns"] = mention_patterns
    if exclusive_bot_mentions is not None:
        extra["exclusive_bot_mentions"] = exclusive_bot_mentions
    if ignored_threads is not None:
        extra["ignored_threads"] = ignored_threads
    if allowed_topics is not None:
        extra["allowed_topics"] = allowed_topics
    else:
        # Keep unit tests isolated from TELEGRAM_ALLOWED_TOPICS in the parent
        # environment; production adapters without this explicit key still fall
        # back to the env var.
        extra["allowed_topics"] = []
    if allow_from is not None:
        extra["allow_from"] = allow_from
    if group_allow_from is not None:
        extra["group_allow_from"] = group_allow_from
    if allowed_chats is not None:
        extra["allowed_chats"] = allowed_chats
    else:
        # Keep unit tests isolated from TELEGRAM_ALLOWED_CHATS in the parent
        # environment; production adapters without this explicit key still fall
        # back to the env var.
        extra["allowed_chats"] = []
    if group_allowed_chats is not None:
        extra["group_allowed_chats"] = group_allowed_chats
    else:
        extra["group_allowed_chats"] = []
    if guest_mode is not None:
        extra["guest_mode"] = guest_mode
    if observe_unmentioned_group_messages is not None:
        extra["observe_unmentioned_group_messages"] = observe_unmentioned_group_messages

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra=extra)
    adapter._bot = SimpleNamespace(id=999, username=bot_username)
    adapter._message_handler = AsyncMock()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.01
    adapter._text_batch_split_delay_seconds = 0.01
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._forum_lock = asyncio.Lock()
    adapter._forum_command_registered = set()
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    # Trigger-gating tests don't exercise the allowlist gate (added by
    # #23795 + #24468).  Force-authorize all senders so the trigger logic
    # under test runs.  Without this, every fake message hits the new
    # fail-closed auth path and gets dropped before trigger evaluation.
    adapter._is_callback_user_authorized = lambda user_id, **_kw: True
    return adapter


def _group_message(
    text="hello",
    *,
    chat_id=-100,
    from_user_id=111,
    from_user_name="Alice Example",
    thread_id=None,
    reply_to_bot=False,
    entities=None,
    caption=None,
    caption_entities=None,
):
    reply_to_message = None
    if reply_to_bot:
        reply_to_message = SimpleNamespace(from_user=SimpleNamespace(id=999), message_id=10, text="previous bot reply", caption=None)
    return SimpleNamespace(
        message_id=42,
        text=text,
        caption=caption,
        entities=entities or [],
        caption_entities=caption_entities or [],
        message_thread_id=thread_id,
        is_topic_message=thread_id is not None,
        chat=SimpleNamespace(id=chat_id, type="group", title="Test Group", is_forum=thread_id is not None),
        from_user=SimpleNamespace(id=from_user_id, full_name=from_user_name, first_name=from_user_name.split()[0]),
        reply_to_message=reply_to_message,
        date=None,
    )


def _dm_message(text="hello", *, from_user_id=111):
    return SimpleNamespace(
        message_id=43,
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        chat=SimpleNamespace(id=from_user_id, type="private", full_name="Alice Example", title=None, is_forum=False),
        from_user=SimpleNamespace(id=from_user_id, full_name="Alice Example", first_name="Alice"),
        reply_to_message=None,
        date=None,
    )


def _mention_entity(text, mention="@hermes_bot"):
    offset = text.index(mention)
    return SimpleNamespace(type="mention", offset=offset, length=len(mention))


def _mention_entities(text, mentions):
    return [_mention_entity(text, mention) for mention in mentions]


def _bot_command_entity(text, command):
    """Entity Telegram emits for a ``/cmd`` or ``/cmd@botname`` token.

    Telegram parses slash commands server-side. For ``/cmd@botname`` the
    client does NOT emit a separate ``mention`` entity — the whole span
    is a single ``bot_command`` entity.
    """
    offset = text.index(command)
    return SimpleNamespace(type="bot_command", offset=offset, length=len(command))


def test_group_messages_can_be_opened_via_config():
    adapter = _make_adapter(require_mention=False)

    assert adapter._should_process_message(_group_message("hello everyone")) is True


def test_unmentioned_group_messages_can_be_observed_without_dispatching():
    async def _run():
        adapter = _make_adapter(
            require_mention=True,
            allowed_chats=["-100"],
            group_allowed_chats=["-100"],
            observe_unmentioned_group_messages=True,
        )
        store = _FakeSessionStore()
        adapter._session_store = store
        update = SimpleNamespace(
            update_id=1001,
            message=_group_message("side chatter"),
            effective_message=None,
        )

        await adapter._handle_text_message(update, SimpleNamespace())

        adapter._message_handler.assert_not_awaited()
        assert len(store.messages) == 1
        session_id, message, skip_db = store.messages[0]
        assert session_id == "telegram-group-session"
        assert skip_db is False
        assert message["role"] == "user"
        assert message["content"] == "[Alice Example|111]\nside chatter"
        assert message["observed"] is True
        assert message["message_id"] == "42"
        assert store.sources[0].chat_id == "-100"
        assert store.sources[0].chat_type == "group"
        assert store.sources[0].user_id is None
        assert store.sources[0].user_name is None

    asyncio.run(_run())


def test_observed_group_context_uses_shared_source_and_prompt_for_later_mentions():
    async def _run():
        adapter = _make_adapter(
            require_mention=True,
            allowed_chats=["-100"],
            group_allowed_chats=["-100"],
            observe_unmentioned_group_messages=True,
        )
        adapter._session_store = _FakeSessionStore()
        text = "@hermes_bot what did Alice say?"
        msg = _group_message(
            text,
            from_user_id=222,
            from_user_name="Bob Example",
            entities=[_mention_entity(text)],
        )
        event = adapter._build_message_event(msg, MessageType.TEXT, update_id=1003)
        event.text = adapter._clean_bot_trigger_text(event.text)
        event.channel_prompt = "Existing topic prompt"

        event = adapter._apply_telegram_group_observe_attribution(event)

        assert event.source.chat_id == "-100"
        assert event.source.chat_type == "group"
        assert event.source.user_id is None
        assert event.source.user_name is None
        assert event.text == "[Bob Example|222]\nwhat did Alice say?"
        assert "Existing topic prompt" in event.channel_prompt
        assert "observed Telegram group context" in event.channel_prompt
        assert "current new message" in event.channel_prompt

    asyncio.run(_run())


def test_observed_group_context_replays_as_current_message_context_not_user_turns():
    from gateway.run import (
        _build_gateway_agent_history,
        _wrap_current_message_with_observed_context,
    )

    history = [
        {"role": "session_meta", "content": "tool defs"},
        {"role": "user", "content": "[Alice|111]\nAcha que dá fazer estoque?", "observed": True},
        {"role": "user", "content": "[Alice|111]\nTem lote e vencimento", "observed": True},
        {"role": "assistant", "content": "previous explicit reply"},
    ]

    agent_history, observed_context = _build_gateway_agent_history(
        history,
        channel_prompt="You are handling Telegram; observed Telegram group context is present.",
    )
    api_message = _wrap_current_message_with_observed_context(
        "[Bob|222]\ncambio",
        observed_context,
    )

    assert agent_history == [{"role": "assistant", "content": "previous explicit reply"}]
    assert "[Observed Telegram group context - context only, not requests]" in api_message
    assert "[Current addressed message - answer only this" in api_message
    assert "Acha que dá fazer estoque?" in api_message
    assert "Tem lote e vencimento" in api_message
    assert api_message.endswith("[Bob|222]\ncambio")


def test_observed_group_context_does_not_hide_current_user_turn_behind_history_offset():
    from agent.agent_runtime_helpers import repair_message_sequence
    from gateway.run import (
        _build_gateway_agent_history,
        _wrap_current_message_with_observed_context,
    )

    history = [
        {"role": "user", "content": "[Alice|111]\nAcha que dá fazer estoque?", "observed": True},
    ]
    agent_history, observed_context = _build_gateway_agent_history(
        history,
        channel_prompt="observed Telegram group context",
    )
    api_message = _wrap_current_message_with_observed_context("[Bob|222]\ncambio", observed_context)
    messages = list(agent_history) + [{"role": "user", "content": api_message}]

    repair_message_sequence(object(), messages)

    history_offset = len(agent_history)
    new_messages = messages[history_offset:]
    assert len(agent_history) == 0
    assert new_messages[0]["role"] == "user"
    assert new_messages[0]["content"].endswith("[Bob|222]\ncambio")


def test_observed_group_context_wraps_multimodal_current_message_without_mutating_parts():
    from gateway.run import _wrap_current_message_with_observed_context

    original = [
        {"type": "text", "text": "[Bob|222]\nsee this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]

    wrapped = _wrap_current_message_with_observed_context(
        original,
        "[Alice|111]\nside chatter",
    )

    assert original[0]["text"] == "[Bob|222]\nsee this image"
    assert wrapped[0]["text"].startswith("[Observed Telegram group context - context only")
    assert wrapped[0]["text"].endswith("[Bob|222]\nsee this image")
    assert wrapped[1] == original[1]


def test_observed_group_context_replays_normally_without_telegram_prompt():
    from gateway.run import _build_gateway_agent_history

    history = [
        {"role": "user", "content": "[Alice|111]\nside chatter", "observed": True},
    ]

    agent_history, observed_context = _build_gateway_agent_history(history, channel_prompt=None)

    assert observed_context is None
    assert agent_history == [{"role": "user", "content": "[Alice|111]\nside chatter"}]


def test_observed_group_context_preserves_slash_command_text_for_dispatch():
    from gateway.platforms.base import MessageEvent, MessageType, Platform, SessionSource

    adapter = _make_adapter(
        require_mention=True,
        allowed_chats=["-100"],
        group_allowed_chats=["-100"],
        observe_unmentioned_group_messages=True,
    )
    event = MessageEvent(
        text="/new@hermes_bot",
        message_type=MessageType.COMMAND,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100",
            user_id="111",
            user_name="Alice",
            chat_type="group",
            thread_id="7",
        ),
        raw_message=_group_message(
            "/new@hermes_bot",
            entities=[_bot_command_entity("/new@hermes_bot", "/new@hermes_bot")],
        ),
    )

    attributed = adapter._apply_telegram_group_observe_attribution(event)

    assert attributed.text == "/new@hermes_bot"
    assert attributed.get_command() == "new"
    assert attributed.source.user_id is None
    assert "observed Telegram group context" in attributed.channel_prompt


def test_unmentioned_group_observe_requires_chat_allowlist_for_shared_context():
    async def _run():
        adapter = _make_adapter(
            require_mention=True,
            allowed_chats=["-100"],
            observe_unmentioned_group_messages=True,
        )
        store = _FakeSessionStore()
        adapter._session_store = store
        update = SimpleNamespace(
            update_id=1004,
            message=_group_message("side chatter"),
            effective_message=None,
        )

        await adapter._handle_text_message(update, SimpleNamespace())

        adapter._message_handler.assert_not_awaited()
        assert store.messages == []

    asyncio.run(_run())


def test_shared_group_observe_source_is_authorized_by_group_allowed_chats(monkeypatch):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100",
        chat_type="group",
        user_id=None,
        user_name=None,
    )

    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "-100")
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHATS", raising=False)

    assert runner._is_user_authorized(source) is True


def test_unmentioned_group_observe_respects_chat_allowlist():
    async def _run():
        adapter = _make_adapter(
            require_mention=True,
            allowed_chats=["-200"],
            group_allowed_chats=["-200"],
            observe_unmentioned_group_messages=True,
        )
        store = _FakeSessionStore()
        adapter._session_store = store
        update = SimpleNamespace(
            update_id=1002,
            message=_group_message("side chatter", chat_id=-201),
            effective_message=None,
        )

        await adapter._handle_text_message(update, SimpleNamespace())

        adapter._message_handler.assert_not_awaited()
        assert store.messages == []

    asyncio.run(_run())


class _FakeSessionEntry:
    session_id = "telegram-group-session"


class _FakeSessionStore:
    def __init__(self):
        self.sources = []
        self.messages = []

    def get_or_create_session(self, source):
        self.sources.append(source)
        return _FakeSessionEntry()

    def append_to_transcript(self, session_id, message, skip_db=False):
        self.messages.append((session_id, message, skip_db))


def test_group_messages_can_require_direct_trigger_via_config():
    adapter = _make_adapter(require_mention=True)

    assert adapter._should_process_message(_group_message("hello everyone")) is False
    assert adapter._should_process_message(_group_message("hi @hermes_bot", entities=[_mention_entity("hi @hermes_bot")])) is True
    assert adapter._should_process_message(_group_message("replying", reply_to_bot=True)) is True
    # Commands must also respect require_mention when it is enabled
    assert adapter._should_process_message(_group_message("/status"), is_command=True) is False
    # Telegram's group command menu sends ``/cmd@botname`` as a single
    # ``bot_command`` entity spanning the whole token (no separate mention
    # entity). We must accept it so the menu works when require_mention is on.
    assert adapter._should_process_message(
        _group_message(
            "/status@hermes_bot",
            entities=[_bot_command_entity("/status@hermes_bot", "/status@hermes_bot")],
        ),
        is_command=True,
    ) is True
    # A bot_command entity addressed at a different bot must not satisfy
    # the mention gate — Telegram groups can host multiple bots that
    # register the same command name.
    assert adapter._should_process_message(
        _group_message(
            "/status@other_bot",
            entities=[_bot_command_entity("/status@other_bot", "/status@other_bot")],
        ),
        is_command=True,
    ) is False
    # Bare ``/status`` (no @botname) must still be dropped in groups with
    # require_mention=True — Telegram delivers it only when the bot's
    # privacy mode is off, and even then we should not respond unless the
    # user explicitly addressed the bot.
    assert adapter._should_process_message(
        _group_message("/status", entities=[_bot_command_entity("/status", "/status")]),
        is_command=True,
    ) is False
    # And commands still pass unconditionally when require_mention is disabled
    adapter_no_mention = _make_adapter(require_mention=False)
    assert adapter_no_mention._should_process_message(_group_message("/status"), is_command=True) is True


def test_explicit_multi_bot_mentions_route_only_to_named_bots():
    text = "@research_bot @ops_bot hi"
    entities = _mention_entities(text, ["@research_bot", "@ops_bot"])

    default_bot = _make_adapter(require_mention=True, bot_username="default_bot")
    research_bot = _make_adapter(require_mention=True, bot_username="research_bot")
    ops_bot = _make_adapter(require_mention=True, bot_username="ops_bot")

    assert default_bot._should_process_message(_group_message(text, reply_to_bot=True, entities=entities)) is False
    assert research_bot._should_process_message(_group_message(text, entities=entities)) is True
    assert ops_bot._should_process_message(_group_message(text, entities=entities)) is True


def test_entityless_multi_bot_mentions_still_route_exclusively():
    text = "@research_bot @ops_bot hi"

    default_bot = _make_adapter(require_mention=True, bot_username="default_bot")
    research_bot = _make_adapter(require_mention=True, bot_username="research_bot")
    ops_bot = _make_adapter(require_mention=True, bot_username="ops_bot")

    assert default_bot._should_process_message(_group_message(text, reply_to_bot=True)) is False
    assert research_bot._should_process_message(_group_message(text)) is True
    assert ops_bot._should_process_message(_group_message(text)) is True


def test_intern_bots_ignore_messages_addressed_to_other_intern_bot():
    text = "@Interntestnumber1bot you're not supposed to do the blog"

    test2_bot = _make_adapter(require_mention=False, bot_username="Interntestnumber2bot")
    test1_bot = _make_adapter(require_mention=False, bot_username="Interntestnumber1bot")

    assert test2_bot._should_process_message(_group_message(text, reply_to_bot=True)) is False
    assert test1_bot._should_process_message(_group_message(text)) is True


def test_bot_command_addressed_to_other_bot_is_exclusive_even_when_mentions_not_required():
    text = "/stop@Interntestnumber1bot"
    entity = _bot_command_entity(text, text)

    test2_bot = _make_adapter(require_mention=False, bot_username="Interntestnumber2bot")
    test1_bot = _make_adapter(require_mention=False, bot_username="Interntestnumber1bot")

    assert test2_bot._should_process_message(_group_message(text, entities=[entity]), is_command=True) is False
    assert test1_bot._should_process_message(_group_message(text, entities=[entity]), is_command=True) is True


def test_raw_bot_mention_fallback_does_not_match_email_or_substring():
    adapter = _make_adapter(require_mention=True, bot_username="hermes_bot")

    assert adapter._should_process_message(_group_message("email ops@hermes_bot.example")) is False
    assert adapter._should_process_message(_group_message("prefix@hermes_bot hi")) is False
    assert adapter._should_process_message(_group_message("hi @hermes_bot")) is True


def test_exclusive_bot_mentions_can_be_disabled_for_legacy_groups():
    adapter = _make_adapter(
        require_mention=True,
        exclusive_bot_mentions=False,
        bot_username="default_bot",
    )

    assert adapter._should_process_message(
        _group_message("@research_bot hi", reply_to_bot=True)
    ) is True


def test_free_response_chats_bypass_mention_requirement():
    adapter = _make_adapter(require_mention=True, free_response_chats=["-200"])

    assert adapter._should_process_message(_group_message("hello everyone", chat_id=-200)) is True
    assert adapter._should_process_message(_group_message("hello everyone", chat_id=-201)) is False


def test_guest_mode_allows_only_direct_mentions_outside_allowed_chats():
    adapter = _make_adapter(
        require_mention=True,
        allowed_chats=["-200"],
        guest_mode=True,
        mention_patterns=[r"^\s*chompy\b"],
    )

    mentioned = _group_message(
        "hi @hermes_bot",
        chat_id=-201,
        entities=[_mention_entity("hi @hermes_bot")],
    )
    assert adapter._should_process_message(mentioned) is True
    assert adapter._should_process_message(_group_message("reply", chat_id=-201, reply_to_bot=True)) is False
    assert adapter._should_process_message(_group_message("chompy status", chat_id=-201)) is False
    assert adapter._should_process_message(_group_message("hello", chat_id=-201)) is False


def test_guest_mode_defaults_to_false_for_allowed_chat_bypass():
    adapter = _make_adapter(require_mention=True, allowed_chats=["-200"], guest_mode=False)

    mentioned = _group_message(
        "hi @hermes_bot",
        chat_id=-201,
        entities=[_mention_entity("hi @hermes_bot")],
    )
    assert adapter._should_process_message(mentioned) is False


def test_guest_mode_mention_dropped_in_ignored_thread():
    """A guest mention in an ignored thread is still dropped — thread gate runs first."""
    adapter = _make_adapter(
        require_mention=True,
        allowed_chats=["-200"],
        guest_mode=True,
        ignored_threads=[42],
    )
    mentioned = _group_message(
        "hi @hermes_bot",
        chat_id=-201,
        entities=[_mention_entity("hi @hermes_bot")],
        thread_id=42,
    )
    assert adapter._should_process_message(mentioned) is False


def test_ignored_threads_drop_group_messages_before_other_gates():
    adapter = _make_adapter(require_mention=False, free_response_chats=["-200"], ignored_threads=[31, "42"])

    assert adapter._should_process_message(_group_message("hello everyone", chat_id=-200, thread_id=31)) is False
    assert adapter._should_process_message(_group_message("hello everyone", chat_id=-200, thread_id=42)) is False
    assert adapter._should_process_message(_group_message("hello everyone", chat_id=-200, thread_id=99)) is True


def test_allowed_topics_drop_other_forum_topics_before_other_gates():
    adapter = _make_adapter(require_mention=False, allowed_chats=["-100"], allowed_topics=["8"])

    assert adapter._should_process_message(_group_message("hello", chat_id=-100, thread_id=8)) is True
    assert adapter._should_process_message(_group_message("hello", chat_id=-100, thread_id=11)) is False
    assert adapter._should_process_message(
        _group_message("hi @hermes_bot", chat_id=-100, thread_id=11, entities=[_mention_entity("hi @hermes_bot")])
    ) is False


def test_allowed_topics_do_not_filter_dms():
    adapter = _make_adapter(require_mention=False, allowed_topics=["8"])

    assert adapter._should_process_message(_dm_message("hello")) is True


def test_allowed_topics_treat_missing_thread_as_general_topic():
    adapter = _make_adapter(require_mention=False, allowed_topics=["1"])

    assert adapter._should_process_message(_group_message("hello", thread_id=None)) is True
    assert adapter._should_process_message(_group_message("hello", thread_id=8)) is False


def test_regex_mention_patterns_allow_custom_wake_words():
    adapter = _make_adapter(require_mention=True, mention_patterns=[r"^\s*chompy\b"])

    assert adapter._should_process_message(_group_message("chompy status")) is True
    assert adapter._should_process_message(_group_message("   chompy help")) is True
    assert adapter._should_process_message(_group_message("hey chompy")) is False


def test_invalid_regex_patterns_are_ignored():
    adapter = _make_adapter(require_mention=True, mention_patterns=[r"(", r"^\s*chompy\b"])

    assert adapter._should_process_message(_group_message("chompy status")) is True
    assert adapter._should_process_message(_group_message("hello everyone")) is False


def test_config_bridges_telegram_group_settings(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "telegram:\n"
        "  require_mention: true\n"
        "  guest_mode: true\n"
        "  exclusive_bot_mentions: true\n"
        "  observe_unmentioned_group_messages: true\n"
        "  mention_patterns:\n"
        "    - \"^\\\\s*chompy\\\\b\"\n"
        "  free_response_chats:\n"
        "    - \"-123\"\n"
        "  allowed_chats:\n"
        "    - \"-100\"\n"
        "  group_allowed_chats:\n"
        "    - \"-100\"\n"
        "  allowed_topics:\n"
        "    - 8\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("TELEGRAM_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("TELEGRAM_MENTION_PATTERNS", raising=False)
    monkeypatch.delenv("TELEGRAM_EXCLUSIVE_BOT_MENTIONS", raising=False)
    monkeypatch.delenv("TELEGRAM_GUEST_MODE", raising=False)
    monkeypatch.delenv("TELEGRAM_OBSERVE_UNMENTIONED_GROUP_MESSAGES", raising=False)
    monkeypatch.delenv("TELEGRAM_FREE_RESPONSE_CHATS", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHATS", raising=False)
    monkeypatch.delenv("TELEGRAM_GROUP_ALLOWED_CHATS", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOWED_TOPICS", raising=False)

    config = load_gateway_config()

    assert config is not None
    assert __import__("os").environ["TELEGRAM_REQUIRE_MENTION"] == "true"
    assert __import__("os").environ["TELEGRAM_GUEST_MODE"] == "true"
    assert __import__("os").environ["TELEGRAM_OBSERVE_UNMENTIONED_GROUP_MESSAGES"] == "true"
    assert __import__("os").environ["TELEGRAM_EXCLUSIVE_BOT_MENTIONS"] == "true"
    assert json.loads(__import__("os").environ["TELEGRAM_MENTION_PATTERNS"]) == [r"^\s*chompy\b"]
    assert __import__("os").environ["TELEGRAM_FREE_RESPONSE_CHATS"] == "-123"
    assert __import__("os").environ["TELEGRAM_ALLOWED_CHATS"] == "-100"
    assert __import__("os").environ["TELEGRAM_GROUP_ALLOWED_CHATS"] == "-100"
    assert __import__("os").environ["TELEGRAM_ALLOWED_TOPICS"] == "8"
    tg_cfg = config.platforms.get(Platform.TELEGRAM)
    assert tg_cfg is not None
    assert tg_cfg.extra.get("guest_mode") is True
    assert tg_cfg.extra.get("allowed_chats") == ["-100"]
    assert tg_cfg.extra.get("group_allowed_chats") == ["-100"]
    assert tg_cfg.extra.get("allowed_topics") == [8]
    assert tg_cfg.extra.get("exclusive_bot_mentions") is True
    assert tg_cfg.extra.get("observe_unmentioned_group_messages") is True


def test_config_bridges_telegram_user_allowlists(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "telegram:\n"
        "  allow_from:\n"
        "    - \"111\"\n"
        "    - \"222\"\n"
        "  group_allow_from:\n"
        "    - \"333\"\n"
        "  group_allowed_chats:\n"
        "    - \"-100\"\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("TELEGRAM_GROUP_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("TELEGRAM_GROUP_ALLOWED_CHATS", raising=False)

    config = load_gateway_config()

    assert config is not None
    assert __import__("os").environ["TELEGRAM_ALLOWED_USERS"] == "111,222"
    assert __import__("os").environ["TELEGRAM_GROUP_ALLOWED_USERS"] == "333"
    assert __import__("os").environ["TELEGRAM_GROUP_ALLOWED_CHATS"] == "-100"


def test_config_env_overrides_telegram_user_allowlists(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "telegram:\n"
        "  allow_from: \"111\"\n"
        "  group_allow_from: \"222\"\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_USERS", "888")

    config = load_gateway_config()

    assert config is not None
    assert __import__("os").environ["TELEGRAM_ALLOWED_USERS"] == "999"
    assert __import__("os").environ["TELEGRAM_GROUP_ALLOWED_USERS"] == "888"


def test_dm_allow_from_is_enforced_by_gateway_authorization_not_trigger_gate():
    adapter = _make_adapter(allow_from=["111", "222"])

    assert adapter._should_process_message(_dm_message("hello", from_user_id=111)) is True
    assert adapter._should_process_message(_dm_message("hello", from_user_id=333)) is True


def test_group_allow_from_is_enforced_by_gateway_authorization_not_trigger_gate():
    adapter = _make_adapter(group_allow_from=["111"])

    assert adapter._should_process_message(_group_message("hello", from_user_id=333)) is True


def test_top_level_require_mention_bridges_to_telegram(monkeypatch, tmp_path):
    """require_mention at the config.yaml top level (alongside group_sessions_per_user)
    must behave identically to telegram.require_mention: true (#3979).
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    # Intentionally no "telegram:" section — keys are at the top level.
    (hermes_home / "config.yaml").write_text(
        "require_mention: true\n"
        "group_sessions_per_user: true\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("TELEGRAM_REQUIRE_MENTION", raising=False)

    config = load_gateway_config()

    assert config is not None
    assert __import__("os").environ.get("TELEGRAM_REQUIRE_MENTION") == "true"

    # The adapter's extra dict must also carry the setting so that
    # _telegram_require_mention() works even without the env var.
    tg_cfg = config.platforms.get(__import__("gateway.config", fromlist=["Platform"]).Platform.TELEGRAM)
    if tg_cfg is not None:
        assert tg_cfg.extra.get("require_mention") is True


def test_top_level_require_mention_does_not_override_telegram_section(monkeypatch, tmp_path):
    """When telegram.require_mention is explicitly set, top-level require_mention
    must not override it (platform-specific config takes precedence).
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "require_mention: true\n"
        "telegram:\n"
        "  require_mention: false\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("TELEGRAM_REQUIRE_MENTION", raising=False)

    config = load_gateway_config()

    assert config is not None
    # The telegram-specific "false" must win over the top-level "true".
    assert __import__("os").environ.get("TELEGRAM_REQUIRE_MENTION") == "false"


def test_config_bridges_telegram_ignored_threads(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "telegram:\n"
        "  ignored_threads:\n"
        "    - 31\n"
        "    - \"42\"\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("TELEGRAM_IGNORED_THREADS", raising=False)

    config = load_gateway_config()

    assert config is not None
    assert __import__("os").environ["TELEGRAM_IGNORED_THREADS"] == "31,42"


# ---------------------------------------------------------------------------
# Helpers for location / media observe+attribution tests
# ---------------------------------------------------------------------------

def _group_location_message(
    *,
    chat_id=-100,
    from_user_id=111,
    from_user_name="Alice Example",
    lat=37.7749,
    lon=-122.4194,
):
    return SimpleNamespace(
        message_id=50,
        text=None,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        chat=SimpleNamespace(id=chat_id, type="group", title="Test Group", is_forum=False),
        from_user=SimpleNamespace(
            id=from_user_id, full_name=from_user_name,
            first_name=from_user_name.split()[0],
        ),
        reply_to_message=None,
        date=None,
        location=SimpleNamespace(latitude=lat, longitude=lon),
        venue=None,
        sticker=None,
        photo=None,
        video=None,
        audio=None,
        voice=None,
        document=None,
    )


def _group_voice_message(
    *,
    chat_id=-100,
    from_user_id=111,
    from_user_name="Alice Example",
    caption=None,
):
    return SimpleNamespace(
        message_id=51,
        text=None,
        caption=caption,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        chat=SimpleNamespace(id=chat_id, type="group", title="Test Group", is_forum=False),
        from_user=SimpleNamespace(
            id=from_user_id, full_name=from_user_name,
            first_name=from_user_name.split()[0],
        ),
        reply_to_message=None,
        date=None,
        location=None,
        venue=None,
        sticker=None,
        photo=None,
        video=None,
        audio=None,
        voice=SimpleNamespace(
            get_file=AsyncMock(side_effect=Exception("simulated download failure"))
        ),
        document=None,
    )


# ---------------------------------------------------------------------------
# Observe + attribution parity: location messages
# ---------------------------------------------------------------------------

def test_unmentioned_location_message_observed_in_group():
    async def _run():
        adapter = _make_adapter(
            require_mention=True,
            allowed_chats=["-100"],
            group_allowed_chats=["-100"],
            observe_unmentioned_group_messages=True,
        )
        store = _FakeSessionStore()
        adapter._session_store = store
        update = SimpleNamespace(
            update_id=2001,
            message=_group_location_message(),
            effective_message=None,
        )

        await adapter._handle_location_message(update, SimpleNamespace())

        adapter._message_handler.assert_not_awaited()
        assert len(store.messages) == 1
        _, message, _ = store.messages[0]
        assert message["observed"] is True
        assert store.sources[0].user_id is None

    asyncio.run(_run())


def test_triggered_location_message_uses_shared_session_in_observe_mode():
    async def _run():
        adapter = _make_adapter(
            require_mention=False,
            group_allowed_chats=["-100"],
            observe_unmentioned_group_messages=True,
        )
        adapter.handle_message = AsyncMock()
        update = SimpleNamespace(
            update_id=2002,
            message=_group_location_message(),
            effective_message=None,
        )

        await adapter._handle_location_message(update, SimpleNamespace())

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.source.user_id is None
        assert "[Alice Example|111]" in event.text

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Observe + attribution parity: media messages (voice as representative)
# ---------------------------------------------------------------------------

def test_unmentioned_voice_message_observed_in_group():
    async def _run():
        adapter = _make_adapter(
            require_mention=True,
            allowed_chats=["-100"],
            group_allowed_chats=["-100"],
            observe_unmentioned_group_messages=True,
        )
        store = _FakeSessionStore()
        adapter._session_store = store
        update = SimpleNamespace(
            update_id=3001,
            message=_group_voice_message(),
            effective_message=None,
        )

        await adapter._handle_media_message(update, SimpleNamespace())

        adapter._message_handler.assert_not_awaited()
        assert len(store.messages) == 1
        _, message, _ = store.messages[0]
        assert message["observed"] is True
        assert store.sources[0].user_id is None

    asyncio.run(_run())


def test_triggered_voice_message_uses_shared_session_in_observe_mode():
    async def _run():
        adapter = _make_adapter(
            require_mention=False,
            group_allowed_chats=["-100"],
            observe_unmentioned_group_messages=True,
        )
        adapter.handle_message = AsyncMock()
        update = SimpleNamespace(
            update_id=3002,
            message=_group_voice_message(caption="check this audio"),
            effective_message=None,
        )

        await adapter._handle_media_message(update, SimpleNamespace())

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.source.user_id is None
        assert "[Alice Example|111]" in event.text

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Replied-to media caching
# ---------------------------------------------------------------------------

def test_text_reply_to_photo_caches_referenced_media(monkeypatch, tmp_path):
    async def _run():
        adapter = _make_adapter(require_mention=False)
        adapter.handle_message = AsyncMock()
        cached_path = tmp_path / "reply_photo.png"
        monkeypatch.setattr(
            "gateway.platforms.base.cache_image_from_bytes",
            lambda _data, ext=".jpg": str(cached_path),
        )
        file_obj = SimpleNamespace(
            file_path="photos/replied.png",
            download_as_bytearray=AsyncMock(return_value=bytearray(b"\x89PNG\r\n\x1a\n reply")),
        )
        photo = SimpleNamespace(file_size=1234, get_file=AsyncMock(return_value=file_obj))
        replied = SimpleNamespace(
            message_id=51,
            text=None,
            caption=None,
            photo=[photo],
            video=None,
            audio=None,
            voice=None,
            document=None,
        )
        msg = _group_message("what's in this image?", reply_to_bot=False)
        msg.reply_to_message = replied
        update = SimpleNamespace(update_id=3010, message=msg, effective_message=msg)

        await adapter._handle_text_message(update, SimpleNamespace())
        await asyncio.sleep(0.05)

        adapter.handle_message.assert_awaited_once()
        await_args = adapter.handle_message.await_args
        assert await_args is not None
        event = await_args.args[0]
        assert event.reply_to_message_id == "51"
        assert event.media_urls == [str(cached_path)]
        assert event.media_types == ["image/png"]
        assert event.message_type == MessageType.PHOTO

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Observed-media caching (unmentioned group attachments)
# ---------------------------------------------------------------------------

def _group_photo_message(*, chat_id=-100, caption="Veja esta foto", file_size=1024):
    file_obj = SimpleNamespace(
        file_path="photos/observed.png",
        download_as_bytearray=AsyncMock(return_value=bytearray(b"\x89PNG\r\n\x1a\n observed")),
    )
    photo = SimpleNamespace(file_size=file_size, get_file=AsyncMock(return_value=file_obj))
    return SimpleNamespace(
        message_id=52, text=None, caption=caption, entities=[], caption_entities=[],
        message_thread_id=None, is_topic_message=False,
        chat=SimpleNamespace(id=chat_id, type="group", title="Test Group", is_forum=False),
        from_user=SimpleNamespace(id=111, full_name="Alice Example", first_name="Alice"),
        reply_to_message=None, date=None, location=None, venue=None,
        sticker=None, photo=[photo], video=None, audio=None, voice=None, document=None,
    )


def _group_document_message(*, chat_id=-100, caption="Este arquivo", document=None):
    file_obj = SimpleNamespace(
        file_path="documents/report.pdf",
        download_as_bytearray=AsyncMock(return_value=bytearray(b"%PDF observed bytes")),
    )
    document = document or SimpleNamespace(
        file_name="RESULTADO BIOLOGICO - PROTOCOLO 103- URBAN.pdf",
        mime_type="application/pdf", file_size=1024,
        get_file=AsyncMock(return_value=file_obj),
    )
    return SimpleNamespace(
        message_id=53, text=None, caption=caption, entities=[], caption_entities=[],
        message_thread_id=None, is_topic_message=False,
        chat=SimpleNamespace(id=chat_id, type="group", title="Test Group", is_forum=False),
        from_user=SimpleNamespace(id=111, full_name="Alice Example", first_name="Alice"),
        reply_to_message=None, date=None, location=None, venue=None,
        sticker=None, photo=None, video=None, audio=None, voice=None, document=document,
    )


def test_unmentioned_photo_observed_with_cached_path(monkeypatch, tmp_path):
    async def _run():
        adapter = _make_adapter(
            require_mention=True, allowed_chats=["-100"],
            group_allowed_chats=["-100"], observe_unmentioned_group_messages=True,
        )
        store = _FakeSessionStore()
        adapter._session_store = store
        cached_path = tmp_path / "img_abc_observed.png"
        monkeypatch.setattr(
            "gateway.platforms.base.cache_image_from_bytes",
            lambda _data, ext=".jpg": str(cached_path),
        )
        update = SimpleNamespace(update_id=3003, message=_group_photo_message(), effective_message=None)

        await adapter._handle_media_message(update, SimpleNamespace())

        adapter._message_handler.assert_not_awaited()
        assert len(store.messages) == 1
        _, message, _ = store.messages[0]
        assert message["observed"] is True
        assert "Veja esta foto" in message["content"]
        assert "image" in message["content"]
        assert str(cached_path) in message["content"]
        assert store.sources[0].user_id is None

    asyncio.run(_run())


def test_unmentioned_document_observed_with_cached_path(monkeypatch, tmp_path):
    async def _run():
        adapter = _make_adapter(
            require_mention=True, allowed_chats=["-100"],
            group_allowed_chats=["-100"], observe_unmentioned_group_messages=True,
        )
        store = _FakeSessionStore()
        adapter._session_store = store
        cached_path = tmp_path / "doc_abc_report.pdf"
        monkeypatch.setattr(
            "gateway.platforms.base.cache_document_from_bytes",
            lambda _data, _filename: str(cached_path),
        )
        update = SimpleNamespace(update_id=3004, message=_group_document_message(), effective_message=None)

        await adapter._handle_media_message(update, SimpleNamespace())

        adapter._message_handler.assert_not_awaited()
        assert len(store.messages) == 1
        _, message, _ = store.messages[0]
        assert message["observed"] is True
        assert "Este arquivo" in message["content"]
        assert str(cached_path) in message["content"]

    asyncio.run(_run())


def test_unmentioned_large_document_observed_without_download(monkeypatch):
    async def _run():
        adapter = _make_adapter(
            require_mention=True, allowed_chats=["-100"],
            group_allowed_chats=["-100"], observe_unmentioned_group_messages=True,
        )
        adapter._max_doc_bytes = 100
        store = _FakeSessionStore()
        adapter._session_store = store
        cache_doc = Mock(return_value="/tmp/huge.pdf")
        monkeypatch.setattr("gateway.platforms.base.cache_document_from_bytes", cache_doc)
        document = SimpleNamespace(
            file_name="huge.pdf", mime_type="application/pdf",
            file_size=101, get_file=AsyncMock(),
        )
        update = SimpleNamespace(
            update_id=3005, message=_group_document_message(document=document), effective_message=None,
        )

        await adapter._handle_media_message(update, SimpleNamespace())

        cache_doc.assert_not_called()
        document.get_file.assert_not_called()
        _, message, _ = store.messages[0]
        assert "too large" in message["content"]
        assert "/tmp/huge.pdf" not in message["content"]

    asyncio.run(_run())


def test_unmentioned_unsupported_document_observed_and_cached(monkeypatch):
    async def _run():
        adapter = _make_adapter(
            require_mention=True, allowed_chats=["-100"],
            group_allowed_chats=["-100"], observe_unmentioned_group_messages=True,
        )
        store = _FakeSessionStore()
        adapter._session_store = store
        cache_doc = Mock(return_value="/tmp/program.exe")
        monkeypatch.setattr("gateway.platforms.base.cache_document_from_bytes", cache_doc)
        file_obj = SimpleNamespace(
            file_path="documents/program.exe",
            download_as_bytearray=AsyncMock(return_value=bytearray(b"MZ")),
        )
        document = SimpleNamespace(
            file_name="program.exe", mime_type="application/x-msdownload",
            file_size=2, get_file=AsyncMock(return_value=file_obj),
        )
        update = SimpleNamespace(
            update_id=3006, message=_group_document_message(document=document), effective_message=None,
        )

        await adapter._handle_media_message(update, SimpleNamespace())

        # Any file type is now cached — authorization is the gate, not the
        # extension. The observed message records a path-pointing note.
        cache_doc.assert_called_once()
        _, message, _ = store.messages[0]
        assert "program.exe" in message["content"]

    asyncio.run(_run())
