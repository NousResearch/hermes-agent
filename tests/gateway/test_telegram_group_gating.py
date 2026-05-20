import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig, load_gateway_config


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
    guest_mode=None,
    bot_username="hermes_bot",
):
    from gateway.platforms.telegram import TelegramAdapter

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
    if guest_mode is not None:
        extra["guest_mode"] = guest_mode

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra=extra)
    adapter._bot = SimpleNamespace(id=999, username=bot_username)
    adapter._message_handler = AsyncMock()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.01
    adapter._mention_patterns = adapter._compile_mention_patterns()
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
    thread_id=None,
    reply_to_bot=False,
    entities=None,
    caption=None,
    caption_entities=None,
):
    reply_to_message = None
    if reply_to_bot:
        reply_to_message = SimpleNamespace(from_user=SimpleNamespace(id=999))
    return SimpleNamespace(
        text=text,
        caption=caption,
        entities=entities or [],
        caption_entities=caption_entities or [],
        message_thread_id=thread_id,
        chat=SimpleNamespace(id=chat_id, type="group"),
        from_user=SimpleNamespace(id=from_user_id),
        reply_to_message=reply_to_message,
    )


def _dm_message(text="hello", *, from_user_id=111):
    return SimpleNamespace(
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        chat=SimpleNamespace(id=from_user_id, type="private"),
        from_user=SimpleNamespace(id=from_user_id),
        reply_to_message=None,
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
        "  mention_patterns:\n"
        "    - \"^\\\\s*chompy\\\\b\"\n"
        "  free_response_chats:\n"
        "    - \"-123\"\n"
        "  allowed_chats:\n"
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
    monkeypatch.delenv("TELEGRAM_FREE_RESPONSE_CHATS", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHATS", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOWED_TOPICS", raising=False)

    config = load_gateway_config()

    assert config is not None
    assert __import__("os").environ["TELEGRAM_REQUIRE_MENTION"] == "true"
    assert __import__("os").environ["TELEGRAM_GUEST_MODE"] == "true"
    assert __import__("os").environ["TELEGRAM_EXCLUSIVE_BOT_MENTIONS"] == "true"
    assert json.loads(__import__("os").environ["TELEGRAM_MENTION_PATTERNS"]) == [r"^\s*chompy\b"]
    assert __import__("os").environ["TELEGRAM_FREE_RESPONSE_CHATS"] == "-123"
    assert __import__("os").environ["TELEGRAM_ALLOWED_CHATS"] == "-100"
    assert __import__("os").environ["TELEGRAM_ALLOWED_TOPICS"] == "8"
    tg_cfg = config.platforms.get(Platform.TELEGRAM)
    assert tg_cfg is not None
    assert tg_cfg.extra.get("guest_mode") is True
    assert tg_cfg.extra.get("allowed_chats") == ["-100"]
    assert tg_cfg.extra.get("allowed_topics") == [8]
    assert tg_cfg.extra.get("exclusive_bot_mentions") is True


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
