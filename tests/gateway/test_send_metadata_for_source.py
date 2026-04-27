from types import SimpleNamespace

from gateway.platforms.base import Platform, SessionSource
from gateway.run import GatewayRunner


def _source(thread_id):
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-1",
        chat_id="chat-1",
        thread_id=thread_id,
    )


def test_send_metadata_for_source_preserves_source_thread_without_event_metadata():
    event = SimpleNamespace()

    metadata = GatewayRunner._send_metadata_for_source(event, _source("17585"))

    assert metadata == {"thread_id": "17585"}


def test_send_metadata_for_source_keeps_existing_metadata_and_overrides_thread():
    event = SimpleNamespace(metadata={"thread_id": "stale", "parse_mode": "Markdown"})

    metadata = GatewayRunner._send_metadata_for_source(event, _source("17585"))

    assert metadata == {"thread_id": "17585", "parse_mode": "Markdown"}


def test_send_metadata_for_source_preserves_existing_metadata_without_thread():
    event = SimpleNamespace(metadata={"parse_mode": "Markdown"})

    metadata = GatewayRunner._send_metadata_for_source(event, _source(None))

    assert metadata == {"parse_mode": "Markdown"}


def test_send_metadata_for_source_returns_none_without_metadata_or_thread():
    event = SimpleNamespace()

    metadata = GatewayRunner._send_metadata_for_source(event, _source(None))

    assert metadata is None
