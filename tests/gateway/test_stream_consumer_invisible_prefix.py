"""Regression tests for the invisible-stream prefix-dedup skip (#95753).

``_continuation_text`` strips the already-streamed prefix from the final
text — correct only when a human could SEE that preview. A2A (and any
future headless adapter) persists exactly the text it receives, so the
strip silently truncated the reply by the preview length ("Peter, good
copy..." → "ter, good copy..."). Adapters declaring
``HAS_VISIBLE_STREAM = False`` must get the complete final text.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _make_consumer(*, has_visible_stream: bool) -> GatewayStreamConsumer:
    adapter = MagicMock()
    adapter.REQUIRES_EDIT_FINALIZE = False
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
    adapter.HAS_VISIBLE_STREAM = has_visible_stream
    return GatewayStreamConsumer(
        adapter=adapter,
        chat_id="chat",
        config=StreamConsumerConfig(fresh_final_after_seconds=0.0),
    )


def test_visible_adapter_still_gets_continuation_text():
    consumer = _make_consumer(has_visible_stream=True)
    consumer._last_sent_text = "Pe"
    assert consumer._continuation_text("Peter, good copy") == "ter, good copy"


def test_invisible_adapter_gets_full_final_text():
    consumer = _make_consumer(has_visible_stream=False)
    consumer._last_sent_text = "Pe"
    assert consumer._continuation_text("Peter, good copy") == "Peter, good copy"


def test_fallback_prefix_also_skipped_for_invisible_adapter():
    consumer = _make_consumer(has_visible_stream=False)
    consumer._fallback_prefix = "Maryjane"
    assert consumer._continuation_text("Maryjane — comms confirmed") == "Maryjane — comms confirmed"


def test_magicmock_default_does_not_flip_the_flag():
    """Consumers built on plain MagicMock adapters (the test-suite norm) keep
    the visible-stream path: the attribute access returns a Mock, which is
    not ``is False``."""
    consumer = GatewayStreamConsumer(
        adapter=MagicMock(),
        chat_id="chat",
        config=StreamConsumerConfig(fresh_final_after_seconds=0.0),
    )
    consumer._last_sent_text = "Pe"
    assert consumer._continuation_text("Peter, good copy") == "ter, good copy"
