import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig, load_gateway_config


def _make_adapter(require_mention=None, free_response_chats=None, mention_patterns=None, ignored_threads=None, topic_id=None, chat_id=None):
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
    if topic_id is not None:
        extra["topic_id"] = topic_id
    if chat_id is not None:
        extra["chat_id"] = chat_id

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
        reply_to_message=reply_to_message,
    )


def _mention_entity(text, mention="@hermes_bot"):
    offset = text.index(mention)
    return SimpleNamespace(type="mention", offset=offset, length=len(mention))


def test_group_messages_can_be_opened_via_config():
    adapter = _make_adapter(require_mention=False)

    assert adapter._should_process_message(_group_message("hello everyone")) is True


def test_group_messages_can_require_direct_trigger_via_config():
    adapter = _make_adapter(require_mention=True)

    assert adapter._should_process_message(_group_message("hello everyone")) is False
    assert adapter._should_process_message(_group_message("hi @hermes_bot", entities=[_mention_entity("hi @hermes_bot")])) is True
    assert adapter._should_process_message(_group_message("replying", reply_to_bot=True)) is True
    assert adapter._should_process_message(_group_message("/status"), is_command=True) is True


def test_free_response_chats_bypass_mention_requirement():
    adapter = _make_adapter(require_mention=True, free_response_chats=["-200"])

    assert adapter._should_process_message(_group_message("hello everyone", chat_id=-200)) is True
    assert adapter._should_process_message(_group_message("hello everyone", chat_id=-201)) is False


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
        "  mention_patterns:\n"
        "    - \"^\\\\s*chompy\\\\b\"\n"
        "  free_response_chats:\n"
        "    - \"-123\"\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("TELEGRAM_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("TELEGRAM_MENTION_PATTERNS", raising=False)
    monkeypatch.delenv("TELEGRAM_FREE_RESPONSE_CHATS", raising=False)

    config = load_gateway_config()

    assert config is not None
    assert __import__("os").environ["TELEGRAM_REQUIRE_MENTION"] == "true"
    assert json.loads(__import__("os").environ["TELEGRAM_MENTION_PATTERNS"]) == [r"^\s*chompy\b"]
    assert __import__("os").environ["TELEGRAM_FREE_RESPONSE_CHATS"] == "-123"


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


def test_topic_id_filters_messages_from_other_topics():
    """When topic_id is set, only messages from that topic pass through."""
    adapter = _make_adapter(topic_id=100, require_mention=False)

    # Correct topic — accepted
    assert adapter._should_process_message(_group_message("hello", thread_id=100)) is True

    # Wrong topic — rejected regardless of other gates
    assert adapter._should_process_message(_group_message("hello", thread_id=200)) is False
    assert adapter._should_process_message(_group_message("hello", thread_id=99)) is False

    # No topic (group-level message) — rejected
    assert adapter._should_process_message(_group_message("hello")) is False


def test_topic_id_does_not_affect_dm():
    """DMs always pass regardless of topic_id config."""
    adapter = _make_adapter(topic_id=100, require_mention=False)

    dm_msg = SimpleNamespace(
        text="hello",
        caption=None,
        entities=[],
        caption_entities=[],
        chat=SimpleNamespace(id=500, type="private"),
        reply_to_message=None,
    )
    assert adapter._should_process_message(dm_msg) is True


def test_topic_id_allows_commands_in_correct_topic():
    """Commands in the configured topic pass through."""
    adapter = _make_adapter(topic_id=100, require_mention=True)

    # Command in correct topic
    assert adapter._should_process_message(_group_message("/status", thread_id=100), is_command=True) is True

    # Command in wrong topic — still rejected (topic filter runs first)
    assert adapter._should_process_message(_group_message("/status", thread_id=200), is_command=True) is False


def test_topic_id_overrides_free_response_chats():
    """topic_id filter runs before free_response_chats, so even allowlisted
    chats are filtered by topic."""
    adapter = _make_adapter(topic_id=100, free_response_chats=["-200"])

    # Message from free_response_chats but wrong topic — rejected
    assert adapter._should_process_message(_group_message("hello", chat_id=-200, thread_id=200)) is False

    # Message from free_response_chats AND correct topic — accepted
    assert adapter._should_process_message(_group_message("hello", chat_id=-200, thread_id=100)) is True


def test_no_topic_id_all_topics_accepted():
    """When topic_id is not configured, behavior is unchanged."""
    adapter = _make_adapter(require_mention=False)

    assert adapter._should_process_message(_group_message("hello", thread_id=100)) is True
    assert adapter._should_process_message(_group_message("hello", thread_id=200)) is True
    assert adapter._should_process_message(_group_message("hello")) is True


def test_config_bridges_telegram_topic_id(monkeypatch, tmp_path):
    """telegram.topic_id in config.yaml bridges to TELEGRAM_TOPIC_ID env var."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "telegram:\n"
        "  topic_id: 12345\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("TELEGRAM_TOPIC_ID", raising=False)

    config = load_gateway_config()

    assert config is not None
    assert __import__("os").environ["TELEGRAM_TOPIC_ID"] == "12345"


def test_chat_id_filters_messages_from_other_groups():
    """When chat_id is set, only messages from that group pass through."""
    adapter = _make_adapter(chat_id=-100, require_mention=False)

    # Correct group — accepted
    assert adapter._should_process_message(_group_message("hello", chat_id=-100)) is True

    # Wrong group — rejected
    assert adapter._should_process_message(_group_message("hello", chat_id=-200)) is False
    assert adapter._should_process_message(_group_message("hello", chat_id=-99)) is False


def test_chat_id_does_not_affect_dm():
    """DMs always pass regardless of chat_id config."""
    adapter = _make_adapter(chat_id=-100, require_mention=False)

    dm_msg = SimpleNamespace(
        text="hello",
        caption=None,
        entities=[],
        caption_entities=[],
        chat=SimpleNamespace(id=500, type="private"),
        reply_to_message=None,
    )
    assert adapter._should_process_message(dm_msg) is True


def test_chat_id_and_topic_id_combined():
    """Both chat_id and topic_id must match for a message to be accepted."""
    adapter = _make_adapter(chat_id=-100, topic_id=7, require_mention=False)

    # Correct group + correct topic — accepted
    assert adapter._should_process_message(_group_message("hello", chat_id=-100, thread_id=7)) is True

    # Correct group + wrong topic — rejected
    assert adapter._should_process_message(_group_message("hello", chat_id=-100, thread_id=99)) is False

    # Wrong group + correct topic — rejected (chat_id filter runs first)
    assert adapter._should_process_message(_group_message("hello", chat_id=-200, thread_id=7)) is False

    # Wrong group + wrong topic — rejected
    assert adapter._should_process_message(_group_message("hello", chat_id=-200, thread_id=99)) is False


def test_chat_id_overrides_free_response_chats():
    """chat_id filter runs before free_response_chats, so even allowlisted
    chats from a different group are filtered."""
    adapter = _make_adapter(chat_id=-100, free_response_chats=["-200"], require_mention=True)

    # free_response_chats but wrong group — rejected
    assert adapter._should_process_message(_group_message("hello", chat_id=-200)) is False

    # Correct group + free_response_chats — accepted
    adapter2 = _make_adapter(chat_id=-200, free_response_chats=["-200"], require_mention=True)
    assert adapter2._should_process_message(_group_message("hello", chat_id=-200)) is True


def test_no_chat_id_all_groups_accepted():
    """When chat_id is not configured, behavior is unchanged."""
    adapter = _make_adapter(require_mention=False)

    assert adapter._should_process_message(_group_message("hello", chat_id=-100)) is True
    assert adapter._should_process_message(_group_message("hello", chat_id=-200)) is True
