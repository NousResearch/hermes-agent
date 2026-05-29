"""End-to-end coverage for MCP passthrough delivery into gateway replay history."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import hermes_state
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.mcp_passthrough import forward_mcp_passthrough_notification
from gateway.run import _build_gateway_agent_history
from gateway.session import SessionSource, SessionStore


@pytest.mark.asyncio
async def test_mcp_passthrough_is_replayed_in_future_gateway_history(tmp_path, monkeypatch):
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="e2e-token",
                home_channel=HomeChannel(
                    platform=Platform.TELEGRAM,
                    chat_id="home-chat",
                    name="Ops Home",
                ),
            )
        },
        group_sessions_per_user=False,
    )
    session_store = SessionStore(sessions_dir=tmp_path / "sessions", config=config)
    entry = session_store.get_or_create_session(
        SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="home-chat",
            chat_type="dm",
            user_id="home-chat",
            user_name="Ops Home",
        )
    )
    adapter = SimpleNamespace(
        send=AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg-1"))
    )
    runner = SimpleNamespace(
        config=config,
        adapters={Platform.TELEGRAM: adapter},
        session_store=session_store,
    )

    with patch("gateway.mcp_passthrough._get_live_runner", return_value=runner):
        await forward_mcp_passthrough_notification(
            server_name="notif_srv",
            payload_json='{"level": "warn", "text": "hello"}',
            targets=["telegram"],
        )

    adapter.send.assert_awaited_once_with(
        chat_id="home-chat",
        content='{"level": "warn", "text": "hello"}',
        metadata=None,
    )

    history = session_store.load_transcript(entry.session_id)
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == '[MCP passthrough from notif_srv] {"level": "warn", "text": "hello"}'

    agent_history, observed_context = _build_gateway_agent_history(history)
    assert observed_context is None
    assert agent_history[-1] == {
        "role": "assistant",
        "content": '[MCP passthrough from notif_srv] {"level": "warn", "text": "hello"}',
    }