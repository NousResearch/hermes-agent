"""Regression tests for Telegram /command@BotName argument parsing (#56337)."""

from types import SimpleNamespace

from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter_with_bot(username: str = "LynxBot") -> TelegramAdapter:
    adapter = object.__new__(TelegramAdapter)
    adapter._bot = SimpleNamespace(username=username)
    return adapter


def test_slash_command_preserves_args_after_bot_mention():
    adapter = _adapter_with_bot()
    assert adapter._clean_bot_trigger_text("/reasoning@LynxBot medium") == "/reasoning medium"


def test_slash_command_strips_punctuation_after_bot_mention():
    adapter = _adapter_with_bot()
    assert (
        adapter._clean_bot_trigger_text("/reasoning@LynxBot: high --global")
        == "/reasoning high --global"
    )


def test_bare_slash_command_mention_stays_compact():
    adapter = _adapter_with_bot()
    assert adapter._clean_bot_trigger_text("/new@LynxBot") == "/new"


def test_leading_mention_trigger_still_strips_bot_prefix():
    adapter = _adapter_with_bot("hermes_bot")
    assert adapter._clean_bot_trigger_text("@hermes_bot what did Alice say?") == (
        "what did Alice say?"
    )


def test_other_bot_mention_in_slash_command_is_unchanged():
    adapter = _adapter_with_bot("LynxBot")
    assert adapter._clean_bot_trigger_text("/reasoning@OtherBot medium") == (
        "/reasoning@OtherBot medium"
    )
