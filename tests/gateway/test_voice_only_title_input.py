"""A voice-only Discord turn titles from its transcript, not the empty-content placeholder."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.run_inbound import discord_triggering_note
from gateway.session import SessionEntry, SessionSource

PLACEHOLDER = "(The user sent a message with no text content)"
TRANSCRIPT = '"Check whether SSH to the media box still works"'


def _runner(enriched_text: str) -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    runner._hmwa_open_session = AsyncMock(return_value=(False, True))
    runner._set_session_env = lambda context: {}
    runner._pinned_session_context_prompt = lambda *args, **kwargs: ""
    runner._hmwa_acquire_turn_lease = AsyncMock()
    runner._mark_durable_active_turn = AsyncMock()
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(_store=runner.session_store, load_transcript=AsyncMock(return_value=[]))
    runner._hmwa_run_session_hygiene = AsyncMock(return_value=[])
    runner._hmwa_first_contact_notes = AsyncMock()
    runner._voice_channel_sidecar_note = lambda *args: None
    # Stands in for STT enrichment plus the Discord routing note the inbound path prepends.
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value=enriched_text)
    runner._bind_adapter_run_generation = lambda *args: None
    return runner


async def _prepared_title(monkeypatch, event_text: str, enriched_text: str):
    monkeypatch.setattr("gateway.run._load_gateway_config", dict)
    source = SessionSource(platform=Platform.DISCORD, chat_id="thread", chat_type="thread",
                           thread_id="thread", user_id="user", user_name="Example")
    event = MessageEvent(text=event_text, source=source, message_id="1559627533203538044")
    entry = SessionEntry(session_id="voice-title", session_key="discord:voice-title",
                         created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 1))
    runner = _runner(f"{discord_triggering_note(event.message_id)}\n\n{enriched_text}")
    prepared, _tokens = await runner._hmwa_prepare_turn(event, source, entry, entry.session_key, entry.session_key, 1)
    return prepared.title_user_message


@pytest.mark.asyncio
async def test_voice_only_turn_titles_from_transcript(monkeypatch):
    title = await _prepared_title(monkeypatch, PLACEHOLDER, TRANSCRIPT)
    assert title == TRANSCRIPT


@pytest.mark.asyncio
async def test_typed_turn_still_titles_from_typed_text(monkeypatch):
    typed = "Does SSH to the media box still work?"
    title = await _prepared_title(monkeypatch, typed, typed)
    assert title == typed
