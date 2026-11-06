"""Matrix ``channel_prompts``: per-room ephemeral system prompts.

Discord, Slack, Telegram and Mattermost have resolved ``channel_prompts`` into
``MessageEvent.channel_prompt`` for a while; the Matrix adapter never did, so a
``matrix.channel_prompts`` block was silently ignored. These tests pin the new
behaviour and the config surface, so an adapter refactor that drops the field
fails loudly instead of quietly losing room instructions.
"""

import asyncio

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource

ROOM = "!lounge:example.org"
OTHER_ROOM = "!elsewhere:example.org"


def _make_adapter(extra=None):
    from gateway.config import PlatformConfig
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = object.__new__(MatrixAdapter)
    adapter.config = PlatformConfig(enabled=True, extra=dict(extra or {}))
    return adapter


def _source(room_id=ROOM):
    return SessionSource(platform=Platform.MATRIX, chat_id=room_id, chat_type="group")


def test_resolve_channel_prompt_reads_the_room_entry():
    adapter = _make_adapter({"channel_prompts": {ROOM: "  keep it terse  "}})
    assert adapter._resolve_channel_prompt(ROOM) == "keep it terse"


def test_resolve_channel_prompt_absent_room_and_blank_entry_are_none():
    adapter = _make_adapter({"channel_prompts": {ROOM: "rules", OTHER_ROOM: "   "}})
    assert adapter._resolve_channel_prompt("!unlisted:example.org") is None
    assert adapter._resolve_channel_prompt(OTHER_ROOM) is None


def test_resolve_channel_prompt_tolerates_missing_or_odd_config():
    assert _make_adapter({})._resolve_channel_prompt(ROOM) is None
    assert _make_adapter({"channel_prompts": ["not", "a", "dict"]})._resolve_channel_prompt(ROOM) is None


def test_build_inbound_event_carries_the_channel_prompt(monkeypatch):
    """The room's prompt must land on the MessageEvent — that is the field the
    gateway turns into the ephemeral system prompt."""
    adapter = _make_adapter({"channel_prompts": {ROOM: "this room is agent-only"}})

    async def _no_reply(room_id, body, relates_to):
        return body, None, None, None, None

    monkeypatch.setattr(adapter, "_extract_reply_context", _no_reply)
    ctx = ("hello", False, "group", None, "think", _source())

    event = asyncio.run(adapter._build_inbound_event(
        ROOM, "@think:example.org", "$evt", "hello", {"body": "hello"}, {}, ctx=ctx))

    assert isinstance(event, MessageEvent)
    assert event.channel_prompt == "this room is agent-only"


def test_build_inbound_event_without_config_has_no_channel_prompt(monkeypatch):
    adapter = _make_adapter({})

    async def _no_reply(room_id, body, relates_to):
        return body, None, None, None, None

    monkeypatch.setattr(adapter, "_extract_reply_context", _no_reply)
    ctx = ("hi", False, "group", None, "think", _source(OTHER_ROOM))

    event = asyncio.run(adapter._build_inbound_event(
        OTHER_ROOM, "@think:example.org", "$evt2", "hi", {"body": "hi"}, {}, ctx=ctx))

    assert event.channel_prompt is None


def test_matrix_config_defaults_advertise_channel_prompts():
    """The key has to exist in the platform defaults, or nobody discovers it and
    ``hermes config set matrix.channel_prompts`` has no documented home."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    matrix = DEFAULT_CONFIG["matrix"]
    assert "channel_prompts" in matrix
    assert matrix["channel_prompts"] == {}
