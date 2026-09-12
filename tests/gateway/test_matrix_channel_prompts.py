"""Matrix per-room prompts + skill bindings (channel_prompts / channel_skill_bindings).

Parity with discord/telegram: ``_build_inbound_event`` resolves an ephemeral
``channel_prompt`` and ``auto_skill`` for (thread_id, room_id) and passes both
into the MessageEvent. Empty config => neither field (no behavior change).
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _make_adapter(extra=None):
    from plugins.platforms.matrix.adapter import MatrixAdapter
    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
            "require_mention": False,
            **(extra or {}),
        },
    )
    adapter = MatrixAdapter(config)
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._resolve_room_identity = AsyncMock(
        return_value=MagicMock(display_name="Project Room", room_topic=None, server_name="example.org"))
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._background_read_receipt = MagicMock()
    return adapter


async def _build(adapter, room_id="!room:example.org", event_id="$evt1",
                 body="hello", relates_to=None):
    return await adapter._build_inbound_event(
        room_id, "@alice:example.org", event_id, body,
        {"body": body}, relates_to or {})


FITNESS = "!l4ov9AshdbJPMzyydE:example.org"
CFO = "!0DtNoREf7ZeiroFObR:example.org"
_EXTRA = {
    "channel_prompts": {
        FITNESS: "You are Greg the coach. DB-first, evidence over vibes.",
        CFO: "You are Greg the CFO. Terse, numbers, deltas never balances.",
    },
    "channel_skill_bindings": [
        {"id": FITNESS, "skills": ["fitness-coach"]},
        {"id": CFO, "skill": "finance"},
    ],
}


class TestMatrixChannelResolutionEmpty:
    @pytest.mark.asyncio
    async def test_no_config_gets_neither_field(self):
        event = await _build(_make_adapter())
        assert event is not None
        assert event.channel_prompt is None
        assert event.auto_skill is None

    @pytest.mark.asyncio
    async def test_empty_containers_get_neither_field(self):
        adapter = _make_adapter({"channel_prompts": {}, "channel_skill_bindings": []})
        event = await _build(adapter)
        assert event is not None
        assert event.channel_prompt is None
        assert event.auto_skill is None

    @pytest.mark.asyncio
    async def test_blank_prompt_counts_as_absent(self):
        adapter = _make_adapter({"channel_prompts": {"!room:example.org": "   "}})
        event = await _build(adapter)
        assert event is not None
        assert event.channel_prompt is None


class TestMatrixChannelResolutionConfigured:
    @pytest.mark.asyncio
    async def test_room_prompt_and_skill(self):
        event = await _build(_make_adapter(_EXTRA), room_id=FITNESS)
        assert event is not None
        assert event.channel_prompt == "You are Greg the coach. DB-first, evidence over vibes."
        assert event.auto_skill == ["fitness-coach"]

    @pytest.mark.asyncio
    async def test_second_room_resolves_own_pair(self):
        event = await _build(_make_adapter(_EXTRA), room_id=CFO)
        assert event is not None
        assert event.channel_prompt == "You are Greg the CFO. Terse, numbers, deltas never balances."
        assert event.auto_skill == ["finance"]

    @pytest.mark.asyncio
    async def test_thread_inherits_parent_room(self):
        event = await _build(
            _make_adapter(_EXTRA), room_id=FITNESS, event_id="$reply1",
            relates_to={"rel_type": "m.thread", "event_id": "$threadroot1"})
        assert event is not None
        assert event.channel_prompt == "You are Greg the coach. DB-first, evidence over vibes."
        assert event.auto_skill == ["fitness-coach"]

    @pytest.mark.asyncio
    async def test_exact_thread_entry_wins_over_parent(self):
        extra = dict(_EXTRA, channel_prompts={
            **_EXTRA["channel_prompts"], "$threadroot1": "Thread-specific prompt."})
        event = await _build(
            _make_adapter(extra), room_id=FITNESS, event_id="$reply1",
            relates_to={"rel_type": "m.thread", "event_id": "$threadroot1"})
        assert event is not None
        assert event.channel_prompt == "Thread-specific prompt."

    @pytest.mark.asyncio
    async def test_unbound_room_gets_neither(self):
        event = await _build(_make_adapter(_EXTRA), room_id="!other:example.org")
        assert event is not None
        assert event.channel_prompt is None
        assert event.auto_skill is None
