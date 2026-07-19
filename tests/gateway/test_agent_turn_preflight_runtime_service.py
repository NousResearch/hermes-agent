"""Production-path unit tests for promoted agent_turn_preflight service."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.agent_turn_preflight_runtime_service import (
    GatewayPreparedAgentTurnPreflight,
    prepare_gateway_agent_turn_preflight,
)
from gateway.config import Platform
from gateway.session import SessionSource


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _session_entry(**overrides):
    base = dict(
        session_key="sk",
        session_id="sid",
        last_prompt_tokens=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_preflight_prepares_message_and_stages_sidecar_notes():
    staged = {}

    def _stage(session_key, notes):
        staged[session_key] = list(notes)

    runner = SimpleNamespace(
        _voice_channel_sidecar_note=MagicMock(return_value="[VC note]"),
        _prepare_profile_scoped_inbound_message_text=AsyncMock(
            return_value="hello world"
        ),
        _set_pending_turn_sidecar_notes=_stage,
        async_session_store=SimpleNamespace(
            has_any_sessions=AsyncMock(return_value=True),
        ),
        config=SimpleNamespace(get_home_channel=MagicMock(return_value="set")),
    )
    event = SimpleNamespace(text="hello world", timestamp=None, media_urls=[])
    notes = ["[System note: prior]"]

    preflight = await prepare_gateway_agent_turn_preflight(
        runner=runner,
        event=event,
        source=_source(),
        session_entry=_session_entry(),
        session_key="sk",
        history=[],  # skip hygiene (< 4 msgs)
        turn_sidecar_notes=notes,
        quick_key="qk",
        run_generation=1,
        logger=MagicMock(),
    )

    assert isinstance(preflight, GatewayPreparedAgentTurnPreflight)
    assert preflight.aborted is False
    assert preflight.message_text == "hello world"
    assert "[System note: prior]" in preflight.turn_sidecar_notes
    assert "[VC note]" in preflight.turn_sidecar_notes
    assert staged["sk"] == preflight.turn_sidecar_notes
    runner._prepare_profile_scoped_inbound_message_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_preflight_aborts_without_staging_when_message_is_none():
    staged = {}

    def _stage(session_key, notes):
        staged[session_key] = list(notes)

    runner = SimpleNamespace(
        _voice_channel_sidecar_note=MagicMock(return_value=None),
        _prepare_profile_scoped_inbound_message_text=AsyncMock(return_value=None),
        _set_pending_turn_sidecar_notes=_stage,
        async_session_store=SimpleNamespace(
            has_any_sessions=AsyncMock(return_value=True),
        ),
        config=SimpleNamespace(get_home_channel=MagicMock(return_value="set")),
    )
    event = SimpleNamespace(text="", timestamp=None, media_urls=[])

    preflight = await prepare_gateway_agent_turn_preflight(
        runner=runner,
        event=event,
        source=_source(),
        session_entry=_session_entry(),
        session_key="sk",
        history=[],
        turn_sidecar_notes=["should-not-stage"],
        quick_key="qk",
        run_generation=1,
        logger=MagicMock(),
    )

    assert preflight.aborted is True
    assert preflight.message_text is None
    assert staged == {}


@pytest.mark.asyncio
async def test_preflight_skips_hygiene_for_short_history(monkeypatch):
    """History with < 4 messages must not touch compression helpers."""
    called = {"context": False}

    async def _boom(*_a, **_k):
        called["context"] = True
        raise AssertionError("hygiene should not run")

    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length_async",
        _boom,
    )

    runner = SimpleNamespace(
        _voice_channel_sidecar_note=MagicMock(return_value=None),
        _prepare_profile_scoped_inbound_message_text=AsyncMock(return_value="hi"),
        _set_pending_turn_sidecar_notes=MagicMock(),
        async_session_store=SimpleNamespace(
            has_any_sessions=AsyncMock(return_value=True),
        ),
        config=SimpleNamespace(get_home_channel=MagicMock(return_value="set")),
    )
    history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]

    preflight = await prepare_gateway_agent_turn_preflight(
        runner=runner,
        event=SimpleNamespace(text="hi", timestamp=None, media_urls=[]),
        source=_source(),
        session_entry=_session_entry(),
        session_key="sk",
        history=history,
        turn_sidecar_notes=[],
        quick_key="qk",
        run_generation=1,
        logger=MagicMock(),
    )

    assert preflight.aborted is False
    assert preflight.message_text == "hi"
    assert called["context"] is False
    assert len(preflight.history) == 3
