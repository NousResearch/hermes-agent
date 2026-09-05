"""Tests for opt-in Matrix ``process_edits``: forwarding an ``m.replace`` edit of a user's
own message as a new agent turn (issue: incoming user edits are silently skipped by default).
"""

import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig


def _make_adapter(process_edits=None):
    """Create a MatrixAdapter with mocked config; process_edits set via config.extra when given."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    extra = {"homeserver": "https://matrix.example.org", "user_id": "@hermes:example.org"}
    if process_edits is not None:
        extra["process_edits"] = process_edits
    config = PlatformConfig(enabled=True, token="syt_test_token", extra=extra)
    adapter = MatrixAdapter(config)
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    adapter._startup_ts = time.time() - 10  # avoid startup grace filter
    return adapter


def _make_edit_event(
    new_body,
    target_event_id="$original",
    sender="@alice:example.org",
    event_id="$edit1",
    room_id="!room1:example.org",
    new_relates_to=None,
):
    """Build a fake ``m.replace`` edit event: fallback body plus ``m.new_content``."""
    new_content = {"msgtype": "m.text", "body": new_body}
    if new_relates_to:
        new_content["m.relates_to"] = new_relates_to
    content = {
        "msgtype": "m.text",
        "body": f"* {new_body}",
        "m.new_content": new_content,
        "m.relates_to": {"rel_type": "m.replace", "event_id": target_event_id},
    }
    return SimpleNamespace(
        sender=sender, event_id=event_id, room_id=room_id,
        timestamp=int(time.time() * 1000), content=content)


@pytest.mark.asyncio
async def test_process_edits_default_disabled_ignores_edit(monkeypatch):
    """Default (opt-in not set): edits are silently ignored — today's behavior is unchanged."""
    monkeypatch.delenv("MATRIX_PROCESS_EDITS", raising=False)
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")

    adapter = _make_adapter()
    event = _make_edit_event("corrected prompt")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_edits_enabled_via_env_forwards_corrected_body(monkeypatch):
    """Enabled via MATRIX_PROCESS_EDITS: the corrected m.new_content body becomes a new turn,
    tagged with the original event id as metadata (not rewriting any prior turn)."""
    monkeypatch.setenv("MATRIX_PROCESS_EDITS", "true")
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")

    adapter = _make_adapter()
    event = _make_edit_event("actually send it to bob", target_event_id="$original1", event_id="$edit1")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.text == "actually send it to bob"
    assert msg.metadata.get("edited_message") is True
    assert msg.metadata.get("edited_message_original_id") == "$original1"


@pytest.mark.asyncio
async def test_process_edits_enabled_via_config_extra(monkeypatch):
    """``_parse_process_edits`` prefers a directly-set ``config.extra["process_edits"]`` over the
    env var. This does not by itself prove config.yaml's ``matrix.process_edits: true`` works —
    see ``test_apply_yaml_config_bridges_process_edits`` for the real YAML→env bridge."""
    monkeypatch.delenv("MATRIX_PROCESS_EDITS", raising=False)
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")

    adapter = _make_adapter(process_edits=True)
    event = _make_edit_event("corrected via config")

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()


def test_apply_yaml_config_bridges_process_edits(monkeypatch):
    """config.yaml's ``matrix.process_edits: true`` must reach ``MATRIX_PROCESS_EDITS`` via the
    real ``apply_yaml_config_fn`` hook (``_apply_yaml_config``) — the same bridge that already
    carries ``process_notices`` — not just via a hand-constructed ``PlatformConfig(extra=...)``."""
    from plugins.platforms.matrix.adapter import _apply_yaml_config

    monkeypatch.delenv("MATRIX_PROCESS_EDITS", raising=False)
    seeded = _apply_yaml_config({}, {"process_edits": True})

    assert seeded is None  # _apply_yaml_config always returns None; everything flows through env
    assert os.environ["MATRIX_PROCESS_EDITS"] == "true"


@pytest.mark.asyncio
async def test_process_edits_preserves_thread_from_new_content(monkeypatch):
    """A threaded edit mirrors the thread relation into m.new_content (MSC2676); the top-level
    relates_to on an edit is exclusively the m.replace pointer, so thread routing must read
    m.new_content's own m.relates_to, not the edit's."""
    monkeypatch.setenv("MATRIX_PROCESS_EDITS", "true")
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")

    adapter = _make_adapter()
    adapter._threads.mark("$thread_root")
    event = _make_edit_event(
        "corrected in-thread", new_relates_to={"rel_type": "m.thread", "event_id": "$thread_root"})

    await adapter._on_room_message(event)
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.source.thread_id == "$thread_root"
