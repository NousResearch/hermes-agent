"""Regression tests for ntfy adapter streaming behavior."""
import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.stream_consumer import StreamConsumerConfig
from plugins.platforms.ntfy.adapter import NtfyAdapter


class _NtfySource:
    """Minimal SessionSource stand-in for the config builder."""

    platform = Platform("ntfy")
    chat_id = "hermes-out"
    chat_type = "dm"


def _build_config(on_missing_cursor: str):
    """Build the stream-consumer config the gateway would build for ntfy.

    ``_build_stream_consumer_config`` never touches ``self``, so it is
    called unbound and handed the adapter *class* -- no runner, no HTTP
    client and no network are needed.
    """
    return GatewayRunner._build_stream_consumer_config(
        None,
        _NtfySource(),
        StreamConsumerConfig(cursor=" ▉"),
        NtfyAdapter,
        on_missing_cursor=on_missing_cursor,
    )


def test_ntfy_adapter_does_not_support_message_editing() -> None:
    """NtfyAdapter.SUPPORTS_MESSAGE_EDITING must be False.

    ntfy publishes immutable notifications -- there is no edit API for an
    already-published message, so a streamed preview IS the final message.
    """
    assert NtfyAdapter.SUPPORTS_MESSAGE_EDITING is False


def test_in_process_path_skips_streaming_for_ntfy() -> None:
    """The in-process agent path must skip streaming for ntfy.

    Without edit support the consumer sends a partial first message it can
    never update, so a single reply fragments into several notifications
    (#83352). The builder signals this by raising, and the call site's
    ``except`` then skips streaming and delivers one final message.
    """
    with pytest.raises(RuntimeError, match="skip streaming"):
        _build_config("raise")


def test_proxy_path_suppresses_streaming_cursor_for_ntfy() -> None:
    """The proxy path must stream with an empty cursor.

    Otherwise the cursor (▉ U+2589) is appended to a preview that can
    never be edited, stranding a tofu square in the delivered text (#83352).
    """
    cfg, _ = _build_config("fallback")
    assert cfg.cursor == ""
