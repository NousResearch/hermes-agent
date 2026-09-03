"""Tests for hermes_cli/chat_commands.py — chat command helpers."""


def test_chat_history_limit_default():
    from hermes_cli.chat_commands import CHAT_HISTORY_LIMIT
    assert isinstance(CHAT_HISTORY_LIMIT, int)
    assert CHAT_HISTORY_LIMIT > 0
