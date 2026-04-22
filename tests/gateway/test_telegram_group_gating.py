import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig, load_gateway_config


def _make_adapter(require_mention=None, free_response_chats=None, mention_patterns=None, ignored_threads=None):
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
    # Commands must also respect require_mention when it is enabled
    assert adapter._should_process_message(_group_message("/status"), is_command=True) is False
    # But commands with @mention still pass (Telegram emits a MENTION entity
    # for /cmd@botname — the bot menu and python-telegram-bot's CommandHandler
    # rely on this same mechanism)
    assert adapter._should_process_message(
        _group_message("/status@hermes_bot", entities=[_mention_entity("/status@hermes_bot")])
    ) is True
    # And commands still pass unconditionally when require_mention is disabled
    adapter_no_mention = _make_adapter(require_mention=False)
    assert adapter_no_mention._should_process_message(_group_message("/status"), is_command=True) is True


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
