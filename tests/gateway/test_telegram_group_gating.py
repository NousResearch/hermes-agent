import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig, load_gateway_config


def _make_adapter(
    require_mention=None,
    free_response_chats=None,
    mention_patterns=None,
    ignored_threads=None,
    allow_from=None,
    group_allow_from=None,
    allowed_chats=None,
    guest_mode=None,
    group_topics=None,
):
    from gateway.platforms.telegram import TelegramAdapter

    extra = {}
    if require_mention is not None:
        extra["require_mention"] = require_mention
    if free_response_chats is not None:
        extra["free_response_chats"] = free_response_chats
    if mention_patterns is not None:
        extra["mention_patterns"] = mention_patterns
    if ignored_threads is not None:
        extra["ignored_threads"] = ignored_threads
    if allow_from is not None:
        extra["allow_from"] = allow_from
    if group_allow_from is not None:
        extra["group_allow_from"] = group_allow_from
    if allowed_chats is not None:
        extra["allowed_chats"] = allowed_chats
    if guest_mode is not None:
        extra["guest_mode"] = guest_mode
    if group_topics is not None:
        extra["group_topics"] = group_topics

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra=extra)
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    adapter._message_handler = AsyncMock()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.01
    adapter._mention_patterns = adapter._compile_mention_patterns()
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
        "  mention_patterns:\n"
        "    - \"^\\\\s*chompy\\\\b\"\n"
        "  free_response_chats:\n"
        "    - \"-123\"\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("TELEGRAM_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("TELEGRAM_MENTION_PATTERNS", raising=False)
    monkeypatch.delenv("TELEGRAM_GUEST_MODE", raising=False)
    monkeypatch.delenv("TELEGRAM_FREE_RESPONSE_CHATS", raising=False)

    config = load_gateway_config()

    assert config is not None
    assert __import__("os").environ["TELEGRAM_REQUIRE_MENTION"] == "true"
    assert __import__("os").environ["TELEGRAM_GUEST_MODE"] == "true"
    assert json.loads(__import__("os").environ["TELEGRAM_MENTION_PATTERNS"]) == [r"^\s*chompy\b"]
    assert __import__("os").environ["TELEGRAM_FREE_RESPONSE_CHATS"] == "-123"
    tg_cfg = config.platforms.get(Platform.TELEGRAM)
    assert tg_cfg is not None
    assert tg_cfg.extra.get("guest_mode") is True


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


# ── resolve_topic_allowlist (helper-level) ──────────────────────────────────


class TestResolveTopicAllowlist:
    """Direct tests for the resolve_topic_allowlist helper."""

    def test_returns_none_when_extra_is_not_a_dict(self):
        from gateway.platforms.telegram import resolve_topic_allowlist

        assert resolve_topic_allowlist(None, "-100") is None
        assert resolve_topic_allowlist("not a dict", "-100") is None

    def test_returns_none_when_no_group_topics(self):
        from gateway.platforms.telegram import resolve_topic_allowlist

        assert resolve_topic_allowlist({}, "-100") is None
        assert resolve_topic_allowlist({"group_topics": None}, "-100") is None

    def test_returns_none_when_chat_id_not_found(self):
        from gateway.platforms.telegram import resolve_topic_allowlist

        extra = {"group_topics": [{"chat_id": -200, "strict": True, "topics": [{"thread_id": 1}]}]}
        assert resolve_topic_allowlist(extra, "-100") is None

    def test_returns_none_when_strict_missing(self):
        from gateway.platforms.telegram import resolve_topic_allowlist

        extra = {"group_topics": [{"chat_id": -100, "topics": [{"thread_id": 5}]}]}
        assert resolve_topic_allowlist(extra, "-100") is None

    def test_returns_none_when_strict_false(self):
        from gateway.platforms.telegram import resolve_topic_allowlist

        extra = {"group_topics": [{"chat_id": -100, "strict": False, "topics": [{"thread_id": 5}]}]}
        assert resolve_topic_allowlist(extra, "-100") is None

    def test_returns_thread_ids_as_strings_when_strict(self):
        from gateway.platforms.telegram import resolve_topic_allowlist

        extra = {
            "group_topics": [
                {
                    "chat_id": -100,
                    "strict": True,
                    "topics": [
                        {"name": "General", "thread_id": 1},
                        {"name": "ops", "thread_id": "7"},
                    ],
                }
            ]
        }
        assert resolve_topic_allowlist(extra, "-100") == {"1", "7"}

    def test_returns_empty_set_when_strict_with_no_topics(self):
        """Empty set vs None matters: empty means 'strict policy, nothing allowed' (rejects all)."""
        from gateway.platforms.telegram import resolve_topic_allowlist

        # Explicit empty list
        extra_empty = {"group_topics": [{"chat_id": -100, "strict": True, "topics": []}]}
        assert resolve_topic_allowlist(extra_empty, "-100") == set()

        # `topics` key entirely absent — same outcome (strict policy, nothing allowed).
        # Documents the behavior so a future refactor doesn't quietly switch to None.
        # User foot-gun: writing `{chat_id: -100, strict: true}` rejects ALL threads.
        extra_absent = {"group_topics": [{"chat_id": -100, "strict": True}]}
        assert resolve_topic_allowlist(extra_absent, "-100") == set()

    def test_chat_id_string_int_coercion(self):
        from gateway.platforms.telegram import resolve_topic_allowlist

        # Integer in config, string lookup
        extra_int = {"group_topics": [{"chat_id": -100, "strict": True, "topics": [{"thread_id": 5}]}]}
        assert resolve_topic_allowlist(extra_int, "-100") == {"5"}

        # String in config, integer-string lookup
        extra_str = {"group_topics": [{"chat_id": "-100", "strict": True, "topics": [{"thread_id": 5}]}]}
        assert resolve_topic_allowlist(extra_str, "-100") == {"5"}

    def test_skips_malformed_entries_gracefully(self):
        from gateway.platforms.telegram import resolve_topic_allowlist

        extra = {
            "group_topics": [
                "not a dict",
                {"chat_id": -100, "strict": True, "topics": [
                    {"thread_id": 7},
                    "not a dict",
                    {"name": "no thread_id"},
                    {"thread_id": None},
                ]},
            ]
        }
        assert resolve_topic_allowlist(extra, "-100") == {"7"}


# ── _should_process_message wired with strict allowlist ─────────────────────


class TestStrictAllowlistGating:
    """Integration tests via _should_process_message — the actual gating path."""

    def test_strict_false_or_missing_accepts_all_threads(self):
        # Backward compat: existing configs without `strict` keep current behavior.
        gt = [{"chat_id": -100, "topics": [{"thread_id": 7}]}]
        adapter = _make_adapter(require_mention=False, group_topics=gt)
        assert adapter._should_process_message(_group_message(chat_id=-100, thread_id=99)) is True
        assert adapter._should_process_message(_group_message(chat_id=-100, thread_id=7)) is True

    def test_strict_true_rejects_unlisted_threads(self):
        gt = [{"chat_id": -100, "strict": True, "topics": [{"thread_id": 7}, {"thread_id": 8}]}]
        adapter = _make_adapter(require_mention=False, group_topics=gt)
        assert adapter._should_process_message(_group_message(chat_id=-100, thread_id=7)) is True
        assert adapter._should_process_message(_group_message(chat_id=-100, thread_id=8)) is True
        assert adapter._should_process_message(_group_message(chat_id=-100, thread_id=9)) is False

    def test_strict_only_applies_to_its_chat(self):
        # foreman-style multi-chat profile: strict in 7stars, lax in ai-villa.
        gt = [
            {"chat_id": -100, "strict": True, "topics": [{"thread_id": 7}]},
            {"chat_id": -200, "topics": [{"thread_id": 5}]},
        ]
        adapter = _make_adapter(require_mention=False, group_topics=gt)
        # Strict chat
        assert adapter._should_process_message(_group_message(chat_id=-100, thread_id=99)) is False
        # Non-strict chat — accept anything
        assert adapter._should_process_message(_group_message(chat_id=-200, thread_id=99)) is True

    def test_strict_does_not_apply_to_messages_without_thread_id(self):
        # Non-forum messages or General-without-thread bypass strict (no thread to check).
        gt = [{"chat_id": -100, "strict": True, "topics": [{"thread_id": 7}]}]
        adapter = _make_adapter(require_mention=False, group_topics=gt)
        assert adapter._should_process_message(_group_message(chat_id=-100, thread_id=None)) is True

    def test_strict_combines_with_ignored_threads(self):
        # Both denylist (global) and allowlist (per-chat) — either rejects.
        gt = [{"chat_id": -100, "strict": True, "topics": [{"thread_id": 7}, {"thread_id": 8}]}]
        adapter = _make_adapter(require_mention=False, ignored_threads=[7], group_topics=gt)
        assert adapter._should_process_message(_group_message(chat_id=-100, thread_id=7)) is False  # denylist wins
        assert adapter._should_process_message(_group_message(chat_id=-100, thread_id=8)) is True   # allowlisted, not denied
        assert adapter._should_process_message(_group_message(chat_id=-100, thread_id=99)) is False # not allowlisted

    def test_strict_with_require_mention_true_still_filters_threads(self):
        # Strict happens before mention check — unlisted thread rejected even with mention.
        gt = [{"chat_id": -100, "strict": True, "topics": [{"thread_id": 7}]}]
        adapter = _make_adapter(require_mention=True, group_topics=gt)
        msg = _group_message("hi @hermes_bot", chat_id=-100, thread_id=99,
                             entities=[_mention_entity("hi @hermes_bot")])
        assert adapter._should_process_message(msg) is False
        # Listed thread + mention → accepted as expected
        msg_ok = _group_message("hi @hermes_bot", chat_id=-100, thread_id=7,
                                entities=[_mention_entity("hi @hermes_bot")])
        assert adapter._should_process_message(msg_ok) is True

    def test_strict_with_free_response_chats_still_filters(self):
        # free_response_chats bypasses require_mention but NOT strict allowlist
        # (otherwise strict would be useless against the very chats that need it).
        gt = [{"chat_id": -100, "strict": True, "topics": [{"thread_id": 7}]}]
        adapter = _make_adapter(require_mention=True, free_response_chats=["-100"], group_topics=gt)
        assert adapter._should_process_message(_group_message(chat_id=-100, thread_id=99)) is False
        assert adapter._should_process_message(_group_message(chat_id=-100, thread_id=7)) is True

    def test_strict_with_empty_topics_rejects_all_threads(self):
        # Edge case: strict policy declared but no topics listed = bot accepts no thread.
        gt = [{"chat_id": -100, "strict": True, "topics": []}]
        adapter = _make_adapter(require_mention=False, group_topics=gt)
        assert adapter._should_process_message(_group_message(chat_id=-100, thread_id=1)) is False
        assert adapter._should_process_message(_group_message(chat_id=-100, thread_id=99)) is False
