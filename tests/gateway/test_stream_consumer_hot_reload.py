"""Hot-reload tolerance (#72628): new consumer code paired with an old in-memory adapter."""
from unittest.mock import MagicMock

from gateway.platforms.base import BasePlatformAdapter
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _stale_adapter(legacy_len_fn):
    adapter = MagicMock(spec=BasePlatformAdapter)
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.max_message_length_for_chat.return_value = 4096
    adapter.message_len_fn_for_chat.side_effect = AttributeError("message_len_fn_for_chat")
    adapter.message_len_fn = legacy_len_fn
    return adapter


def test_length_budget_falls_back_to_legacy_message_len_fn():
    def utf16ish(s):
        return 2 * len(s)

    consumer = GatewayStreamConsumer(_stale_adapter(utf16ish), "chat-1", StreamConsumerConfig())
    len_fn, safe_limit = consumer._resolve_length_budget()
    assert len_fn is utf16ish
    assert safe_limit >= 500


def test_length_budget_falls_back_to_len_when_no_length_api():
    consumer = GatewayStreamConsumer(_stale_adapter(None), "chat-1", StreamConsumerConfig())
    len_fn, _ = consumer._resolve_length_budget()
    assert len_fn is len
