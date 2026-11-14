"""Quiet failed turns preserve actual partial answers and never send gateway failure copy."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_ingress import deliver_response


def _quiet_source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id="quiet-chat", user_id="user")


def _failed_partial_result() -> dict:
    return {
        "failed": True,
        "partial": True,
        "final_response": "The completed first step is ready.",
        "error": "HTTP 429: retry budget exhausted",
        "history_offset": 1,
        "messages": [
            {"role": "user", "content": "previous question"},
            {"role": "user", "content": "do the work"},
            {"role": "assistant", "content": "The completed first step is ready."},
        ],
    }


def test_quiet_failed_turn_preserves_only_current_turn_assistant_content(monkeypatch):
    """The notice is removed, but model text produced before the failure remains visible."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"display": {"failed_turn_notice": False}})
    runner = object.__new__(GatewayRunner)
    result = _failed_partial_result()

    response = runner._hmwa_apply_failed_turn_notice(
        result["final_response"], result, _quiet_source(), True, False,
    )

    assert response == "The completed first step is ready."


@pytest.mark.asyncio
async def test_quiet_system_only_failed_turn_skips_adapter_send(monkeypatch):
    """A quiet provider failure reaches the normal no-response delivery path without retries."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"display": {"failed_turn_notice": False}})
    runner = object.__new__(GatewayRunner)
    result = {
        "failed": True,
        "final_response": "API call failed after 3 retries: 429 Too Many Requests",
        "error": "429 Too Many Requests",
        "messages": [],
    }
    response = runner._hmwa_apply_failed_turn_notice(
        "Hermes could not complete the request.", result, _quiet_source(), True, False,
    )
    adapter = SimpleNamespace(
        _unwrap_ephemeral=MagicMock(return_value=(response, 0)),
        _send_final_text=AsyncMock(),
    )

    delivered = await deliver_response(
        adapter, MessageEvent(text="do the work", source=_quiet_source()), "session", response,
    )

    assert response is None
    assert delivered is True
    adapter._send_final_text.assert_not_awaited()
