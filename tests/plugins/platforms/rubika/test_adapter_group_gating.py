import pytest
from unittest.mock import AsyncMock

from gateway.config import PlatformConfig
from plugins.platforms.rubika.adapter import RubikaAdapter
from plugins.platforms.rubika.inbound import ParsedMessage


def _adapter(**extra) -> RubikaAdapter:
    cfg = PlatformConfig()
    cfg.extra = {"token": "TESTTOKEN", **extra}
    return RubikaAdapter(cfg)


def test_dm_always_processed():
    adapter = _adapter(require_mention="true", bot_username="mybot")
    parsed = ParsedMessage(chat_id="c1", sender_id="u1", text="hi", message_id="m1", is_group=False)
    assert adapter._should_process_message(parsed) is True


def test_group_without_require_mention_processed():
    adapter = _adapter(require_mention="false")
    parsed = ParsedMessage(chat_id="g1", sender_id="u1", text="hi", message_id="m1", is_group=True)
    assert adapter._should_process_message(parsed) is True


def test_group_with_require_mention_and_no_mention_dropped():
    adapter = _adapter(require_mention="true", bot_username="mybot")
    parsed = ParsedMessage(chat_id="g1", sender_id="u1", text="hi everyone", message_id="m1", is_group=True)
    assert adapter._should_process_message(parsed) is False


def test_group_with_require_mention_and_mention_processed():
    adapter = _adapter(require_mention="true", bot_username="mybot")
    parsed = ParsedMessage(chat_id="g1", sender_id="u1", text="@mybot hi", message_id="m1", is_group=True)
    assert adapter._should_process_message(parsed) is True
