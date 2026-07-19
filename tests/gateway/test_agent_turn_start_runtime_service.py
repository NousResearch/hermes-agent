"""Production-path unit tests for promoted agent_turn_start bootstrap."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.agent_turn_start_runtime_service import (
    GatewayPreparedAgentTurnStart,
    prepare_gateway_agent_turn_start,
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


@pytest.mark.asyncio
async def test_prepare_returns_pinned_context_and_empty_sidecar_for_plain_dm(monkeypatch):
    session_entry = SimpleNamespace(
        session_key="sk",
        session_id="sid",
        created_at=1,
        updated_at=2,
        was_auto_reset=False,
        is_fresh_reset=False,
        auto_reset_reason=None,
    )
    runner = SimpleNamespace(
        async_session_store=SimpleNamespace(
            get_or_create_session=AsyncMock(return_value=session_entry),
            load_transcript=AsyncMock(return_value=[]),
        ),
        _session_db=None,
        _recover_telegram_topic_thread_id=MagicMock(return_value=None),
        _is_telegram_topic_lane=MagicMock(return_value=False),
        _cache_session_source=MagicMock(),
        _configured_admin_user_ids=MagicMock(return_value=None),
        _is_admin_user=MagicMock(return_value=None),
        _set_session_env=MagicMock(return_value={}),
        _pinned_session_context_prompt=MagicMock(return_value="PINNED_CONTEXT"),
        _clear_conversation_scope=MagicMock(),
        _evict_cached_agent=MagicMock(),
        _maybe_auto_background_turn=MagicMock(return_value=None),
        hooks=SimpleNamespace(emit=AsyncMock()),
        _turn_leases=None,
        config=SimpleNamespace(
            get_connected_platforms=MagicMock(return_value=[]),
            get_home_channel=MagicMock(return_value=None),
        ),
        adapters={},
    )
    event = SimpleNamespace(text="hi", metadata={}, auto_skill=None, source=_source())
    monkeypatch.setattr(
        "gateway.agent_turn_start_runtime_service.try_handle_direct_gateway_shortcuts",
        lambda *a, **k: None,
    )
    # also patch the late re-import site used inside prepare
    monkeypatch.setattr(
        "gateway.direct_shortcut_runtime_service.try_handle_direct_gateway_shortcuts",
        lambda *a, **k: None,
    )

    prepared = await prepare_gateway_agent_turn_start(
        runner=runner,
        event=event,
        source=_source(),
        quick_key="qk",
        run_generation=1,
        logger=MagicMock(),
    )

    assert isinstance(prepared, GatewayPreparedAgentTurnStart)
    assert prepared.aborted is False
    assert prepared.immediate_response is None
    assert prepared.context_prompt == "PINNED_CONTEXT"
    assert prepared.session_key == "sk"
    assert prepared.turn_sidecar_notes == []
    runner._pinned_session_context_prompt.assert_called_once()
