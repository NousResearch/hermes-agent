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
    free_response_topics=None,
    mention_patterns=None,
    exclusive_bot_mentions=None,
    ignored_threads=None,
    allowed_topics=None,
    require_explicit_mention_topics=None,
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
    if free_response_topics is not None:
        extra["free_response_topics"] = free_response_topics
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
    if require_explicit_mention_topics is not None:
        extra["require_explicit_mention_topics"] = require_explicit_mention_topics
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
    # Commands preserve sender identity for slash-access control (#67816).
    assert attributed.source.user_id == "111"
    assert attributed.source.user_name == "Alice"
    assert "observed Telegram group context" in attributed.channel_prompt


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


def test_free_response_topic_messages_are_dispatched_not_observed():
    """A free-response topic message must go to the dispatcher, not the observe path."""
    adapter = _make_adapter(
        require_mention=True,
        allowed_chats=["-200"],
        group_allowed_chats=["-200"],
        observe_unmentioned_group_messages=True,
        free_response_topics=["-200:31"],
    )

    in_topic = _group_message("hello everyone", chat_id=-200, thread_id=31)
    assert adapter._should_process_message(in_topic) is True
    assert adapter._should_observe_unmentioned_group_message(in_topic) is False

    # Same chat, different topic: not dispatched, but still observable.
    other_topic = _group_message("side chatter", chat_id=-200, thread_id=32)
    assert adapter._should_process_message(other_topic) is False
    assert adapter._should_observe_unmentioned_group_message(other_topic) is True


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


def test_allowed_topics_drop_other_forum_topics_before_other_gates():
    adapter = _make_adapter(require_mention=False, allowed_chats=["-100"], allowed_topics=["8"])

    assert adapter._should_process_message(_group_message("hello", chat_id=-100, thread_id=8)) is True
    assert adapter._should_process_message(_group_message("hello", chat_id=-100, thread_id=11)) is False
    assert adapter._should_process_message(
        _group_message("hi @hermes_bot", chat_id=-100, thread_id=11, entities=[_mention_entity("hi @hermes_bot")])
    ) is False


def test_allowed_topics_can_be_scoped_to_one_chat_topic_pair():
    adapter = _make_adapter(
        require_mention=False,
        allowed_chats=["-100", "-200"],
        allowed_topics=["-100:8"],
    )

    assert adapter._should_process_message(
        _group_message("hello", chat_id=-100, thread_id=8)
    ) is True
    assert adapter._should_process_message(
        _group_message("hello", chat_id=-200, thread_id=8)
    ) is False


def test_observed_group_context_honors_chat_scoped_topic_gate():
    adapter = _make_adapter(
        require_mention=True,
        allowed_chats=["-100", "-200"],
        group_allowed_chats=["-100", "-200"],
        allowed_topics=["-100:8"],
        observe_unmentioned_group_messages=True,
    )

    assert adapter._should_observe_unmentioned_group_message(
        _group_message("context", chat_id=-100, thread_id=8)
    ) is True
    assert adapter._should_observe_unmentioned_group_message(
        _group_message("context", chat_id=-200, thread_id=8)
    ) is False


def test_explicit_mention_topic_does_not_treat_reply_as_a_tag():
    adapter = _make_adapter(
        require_mention=True,
        allowed_chats=["-100"],
        group_allowed_chats=["-100"],
        require_explicit_mention_topics=["-100:8"],
        observe_unmentioned_group_messages=True,
    )

    reply = _group_message("continuing", chat_id=-100, thread_id=8, reply_to_bot=True)
    assert adapter._should_process_message(reply) is False
    assert adapter._should_observe_unmentioned_group_message(reply) is True

    tagged_reply = _group_message(
        "@hermes_bot continuing",
        chat_id=-100,
        thread_id=8,
        reply_to_bot=True,
    )
    assert adapter._should_process_message(tagged_reply) is True
    assert adapter._should_observe_unmentioned_group_message(tagged_reply) is False

    # The setting is route-scoped: ordinary Telegram reply semantics remain
    # unchanged in another topic served by the same profile.
    other_topic_reply = _group_message(
        "continuing", chat_id=-100, thread_id=9, reply_to_bot=True
    )
    assert adapter._should_process_message(other_topic_reply) is True


def _forum_message(*, chat_id, thread_id, is_topic_message, is_forum, chat_type="supergroup"):
    """Build a message with independently-controlled topic/forum flags.

    The shared ``_group_message`` fixture couples ``is_topic_message`` and
    ``is_forum`` to ``thread_id is not None``, which cannot express a plain
    reply-UI anchor (``message_thread_id`` set, ``is_topic_message=False``,
    ``is_forum=False``). This helper decouples them for gating regressions.
    """
    return SimpleNamespace(
        message_id=42,
        text="hello",
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=thread_id,
        is_topic_message=is_topic_message,
        chat=SimpleNamespace(id=chat_id, type=chat_type, title="T", is_forum=is_forum),
        from_user=SimpleNamespace(id=111, full_name="Alice", first_name="Alice"),
        reply_to_message=None,
        date=None,
    )


def test_gating_ignores_non_forum_reply_anchor_thread_id():
    """A plain group reply's ``message_thread_id`` is a UI anchor, not a topic.

    Before the shared ``_effective_message_thread_id`` normalizer, gating read
    the raw ``message_thread_id`` — so a non-forum group reply whose anchor id
    happened to match an ``ignored_threads`` entry was wrongly dropped, and its
    anchor id was treated as a routable topic under ``allowed_topics``. The
    normalizer drops reply anchors (non-forum, ``is_topic_message=False``), so
    such a reply gates as the General topic instead.
    """
    # ignored_threads: reply anchor 55 must NOT be treated as thread 55.
    adapter = _make_adapter(require_mention=False, free_response_chats=["-200"], ignored_threads=[55])
    reply_anchor = _forum_message(
        chat_id=-200, thread_id=55, is_topic_message=False, is_forum=False, chat_type="group"
    )
    assert adapter._should_process_message(reply_anchor) is True

    # allowed_topics: reply anchor 55 normalizes to General ("1"), so a group
    # that only allows topic "1" still processes the reply.
    adapter2 = _make_adapter(require_mention=False, allowed_chats=["-200"], allowed_topics=["1"])
    assert adapter2._should_process_message(reply_anchor) is True


def test_gating_forum_general_topic_normalizes_to_one():
    """Forum General-topic messages (thread_id=None) gate as topic "1"."""
    adapter = _make_adapter(require_mention=False, allowed_chats=["-100"], allowed_topics=["1"])
    general = _forum_message(chat_id=-100, thread_id=None, is_topic_message=False, is_forum=True)
    assert adapter._should_process_message(general) is True

    adapter2 = _make_adapter(require_mention=False, allowed_chats=["-100"], allowed_topics=["8"])
    assert adapter2._should_process_message(general) is False


def test_bot_self_messages_are_ignored_in_dm_and_group():
    """Bot-authored messages must not re-enter as fresh user turns (issue #11905).

    Telegram echoes the bot's own outbound messages back through getUpdates.
    Without a self-author guard, those echoes — including
    ``[SYSTEM: Background process ...]`` watcher notifications — get ingested
    as new inbound turns, producing the "haunted topic" loop. The guard keys
    on ``from_user.id == self._bot.id`` (bot id is 999 in ``_make_adapter``).
    """
    adapter = _make_adapter(require_mention=False)

    # Control: a real user in the same group IS processed.
    assert adapter._should_process_message(_group_message("hi", chat_id=-100)) is True

    # The exact reported symptom: a bot-authored DM-topic watcher echo.
    self_dm = _group_message(
        "[SYSTEM: Background process matched watch pattern ...]",
        chat_id=555,
        from_user_id=999,
    )
    self_dm.chat.type = "private"
    assert adapter._should_process_message(self_dm) is False

    # Same guard applies in groups/supergroups.
    self_group = _group_message("status tick", chat_id=-100, from_user_id=999)
    assert adapter._should_process_message(self_group) is False


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
        "    - 8\n"
        '    - "-100:9"\n',
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # Clear the TELEGRAM_* vars this test exercises so a developer's ambient
    # shell/.env values don't pre-empt the YAML→env bridge (env-over-YAML
    # precedence, adapter.py::_apply_yaml_config). The authoritative assertions
    # below read the returned config object, which is immune to env pollution
    # from third-party import-time load_dotenv calls; see the note at the asserts.
    for _var in (
        "TELEGRAM_REQUIRE_MENTION",
        "TELEGRAM_MENTION_PATTERNS",
        "TELEGRAM_EXCLUSIVE_BOT_MENTIONS",
        "TELEGRAM_GUEST_MODE",
        "TELEGRAM_OBSERVE_UNMENTIONED_GROUP_MESSAGES",
        "TELEGRAM_FREE_RESPONSE_CHATS",
        "TELEGRAM_ALLOWED_CHATS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "TELEGRAM_ALLOWED_TOPICS",
    ):
        monkeypatch.delenv(_var, raising=False)

    config = load_gateway_config()

    # Assert against the returned config object — the authoritative result of the
    # bridge. We deliberately do NOT assert on os.environ here: a third-party
    # import (microsoft_teams/apps/app.py) runs load_dotenv(find_dotenv(usecwd=True))
    # at import time, which walks up from cwd and can repopulate TELEGRAM_* vars
    # from a developer's real ~/.hermes/.env, defeating the env-over-YAML bridge
    # for any key present there. The PlatformConfig.extra values below are parsed
    # straight from the test's config.yaml and are immune to that ambient leak.
    assert config is not None
    tg_cfg = config.platforms.get(Platform.TELEGRAM)
    assert tg_cfg is not None
    assert tg_cfg.extra.get("require_mention") is True
    assert tg_cfg.extra.get("guest_mode") is True
    assert tg_cfg.extra.get("exclusive_bot_mentions") is True
    assert tg_cfg.extra.get("observe_unmentioned_group_messages") is True
    assert tg_cfg.extra.get("mention_patterns") == [r"^\s*chompy\b"]
    assert tg_cfg.extra.get("allowed_chats") == ["-100"]
    assert tg_cfg.extra.get("group_allowed_chats") == ["-100"]
    assert tg_cfg.extra.get("allowed_topics") == [8, "-100:9"]
    # free_response_chats is bridged to the env var only (not PlatformConfig.extra).
    # TELEGRAM_FREE_RESPONSE_CHATS is not a key that appears in developer .env
    # files, so asserting it via os.environ stays deterministic.
    assert __import__("os").environ["TELEGRAM_FREE_RESPONSE_CHATS"] == "-123"


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


# ---------------------------------------------------------------------------
# Replied-to media caching
# ---------------------------------------------------------------------------


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


# ── Bot identity: renames and non-"bot"-suffixed handles ────────────────────
# Two failure modes fixed together (both break the mention gate):
#   1. PTB caches getMe() in Bot._bot_user and only rewrites it inside
#      get_me(). After a BotFather rename the adapter compares against the OLD
#      handle, so the exclusive-mention gate reads a message addressed to us as
#      one addressed to a different bot and silently drops it.
#   2. The bot-handle pattern assumed every bot username ends in "bot".
#      Collectible (Fragment) usernames can be assigned to bots and don't
#      (@jarvis, @pic), making such a bot unable to recognise its own handle.


class _IdentityBot:
    """Stand-in for PTB's Bot: ``.username`` only changes when get_me() runs."""

    def __init__(self, bot_id=999, cached="hermes_bot", server=None):
        self.id = bot_id
        self._cached = cached
        self._server = server if server is not None else cached
        self.get_me_calls = 0

    @property
    def username(self):
        return self._cached

    async def get_me(self):
        self.get_me_calls += 1
        self._cached = self._server
        return SimpleNamespace(id=self.id, username=self._server)


def _reply_to_bot_message(text, *, entities=None, bot_username, bot_id=999):
    """Group message replying to one of our messages.

    Telegram stamps the bot's CURRENT username on ``reply_to_message.from_user``,
    which is how a rename becomes observable without an extra API call.
    """
    message = _group_message(text, entities=entities)
    message.reply_to_message = SimpleNamespace(
        from_user=SimpleNamespace(id=bot_id, username=bot_username),
        message_id=10, text="previous bot reply", caption=None,
    )
    return message


def test_stale_username_schedules_background_identity_recheck():
    """A drop caused by a stale handle self-corrects via a TTL-guarded getMe."""
    async def _run():
        adapter = _make_adapter(require_mention=True, exclusive_bot_mentions=True)
        adapter._bot = _IdentityBot(cached="old_helper_bot", server="new_helper_bot")
        adapter._background_tasks = set()
        text = "@new_helper_bot what's the weather"
        message = _group_message(text, entities=_mention_entities(text, ["@new_helper_bot"]))

        # First message is lost — nothing has revealed the new handle yet.
        assert adapter._should_process_message(message) is False
        await asyncio.gather(*list(adapter._background_tasks))

        assert adapter._bot.get_me_calls == 1
        assert adapter._current_bot_username() == "new_helper_bot"
        # Recovered without a gateway restart.
        assert adapter._should_process_message(message) is True

    asyncio.run(_run())


def test_bot_never_adopts_another_accounts_username():
    """Only a user id matching this bot may update our own handle."""
    adapter = _make_adapter(require_mention=True)
    adapter._bot = _IdentityBot(cached="hermes_bot")
    message = _group_message("hello")
    message.from_user = SimpleNamespace(id=555, username="impostor_bot", full_name="Impostor", first_name="Impostor")

    adapter._observe_bot_identity_from_message(message)

    assert adapter._current_bot_username() == "hermes_bot"


def test_collectible_username_not_suppressed_by_other_bot_mention():
    """@jarvis + @other_bot in one message must still reach @jarvis."""
    adapter = _make_adapter(
        require_mention=True, exclusive_bot_mentions=True, bot_username="jarvis",
    )
    text = "@jarvis ask @other_helper_bot for the log"
    message = _group_message(
        text, entities=_mention_entities(text, ["@jarvis", "@other_helper_bot"]),
    )

    assert adapter._explicit_bot_mentions_exclude_self(message) is False
    assert adapter._should_process_message(message) is True


def test_identity_freshness_does_not_depend_on_host_uptime(monkeypatch):
    """A never-checked identity is stale even when monotonic() is near zero.

    time.monotonic() has an arbitrary epoch: on a freshly-booted host (CI
    runners, containers) it starts near 0. A 0.0 "never checked" sentinel
    therefore reads as "checked just now" for the first TTL seconds of
    uptime, suppressing the very first identity refresh — so the stale-handle
    recovery silently did nothing on exactly the machines most likely to be
    freshly booted. Caught by CI, invisible on a long-lived dev box.
    """
    adapter = _make_adapter(require_mention=True)
    adapter._bot = _IdentityBot(cached="old_helper_bot", server="new_helper_bot")

    # Simulate a host that booted 12 seconds ago.
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.time.monotonic", lambda: 12.0
    )

    assert adapter._bot_identity_is_fresh() is False

    adapter._note_bot_username("new_helper_bot")
    assert adapter._bot_identity_is_fresh() is True
