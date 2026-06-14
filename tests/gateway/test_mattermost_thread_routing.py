from __future__ import annotations

from types import SimpleNamespace

from gateway.config import Platform
from gateway.run import (
    _progress_thread_target_for_source,
    _status_thread_metadata_for_progress,
)


def test_mattermost_top_level_progress_uses_event_message_as_thread_anchor():
    """Top-level Mattermost channel prompts should keep progress/status in a thread."""
    source = SimpleNamespace(
        platform=Platform.MATTERMOST,
        thread_id=None,
    )

    thread_id, reply_to = _progress_thread_target_for_source(
        source,
        event_message_id="trigger-post-123",
    )

    assert thread_id == "trigger-post-123"
    assert reply_to == "trigger-post-123"


def test_slack_top_level_progress_keeps_existing_thread_anchor_behavior():
    source = SimpleNamespace(
        platform=Platform.SLACK,
        thread_id=None,
    )

    thread_id, reply_to = _progress_thread_target_for_source(
        source,
        event_message_id="trigger-post-123",
    )

    assert thread_id == "trigger-post-123"
    assert reply_to is None


def test_mattermost_top_level_status_reuses_progress_fallback_metadata():
    """Compression/status notices should thread like progress bubbles."""
    source = SimpleNamespace(
        platform=Platform.MATTERMOST,
        thread_id=None,
    )
    progress_thread_id, _ = _progress_thread_target_for_source(
        source,
        event_message_id="trigger-post-123",
    )
    progress_metadata = {"thread_id": progress_thread_id}

    def base_metadata_builder(_source, _event_message_id):  # pragma: no cover - guard
        raise AssertionError("top-level Mattermost status must not fall back to source.thread_id metadata")

    metadata = _status_thread_metadata_for_progress(
        source,
        "trigger-post-123",
        progress_thread_id,
        progress_metadata,
        base_metadata_builder,
    )

    assert metadata == {"thread_id": "trigger-post-123"}
