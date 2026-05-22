import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner, _format_modelroute_result, _parse_modelroute_args
from gateway.session import SessionSource


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            chat_type="group",
            user_id="user-1",
            user_name="operator",
        ),
        message_id="msg-1",
    )


def test_parse_modelroute_args_requires_task_and_message():
    assert _parse_modelroute_args("free_utility_general hello") == (
        "free_utility_general",
        "hello",
    )
    with pytest.raises(ValueError, match="Usage"):
        _parse_modelroute_args("")
    with pytest.raises(ValueError, match="Usage"):
        _parse_modelroute_args("free_utility_general")


def test_format_modelroute_result_includes_telemetry_without_raw_message():
    payload = {
        "ok": True,
        "selected_model": "free/mistral-small-4-119b",
        "transport": "direct_clawrouter",
        "text": "telegram model route ok",
        "telemetry": {
            "task_class": "free_utility_general",
            "elapsed_sec": 0.51,
            "fallback_used": False,
            "evidence": "evidence/runtime_health/model_routes/proof.json",
        },
    }
    rendered = _format_modelroute_result(payload)
    assert "selected_model=free/mistral-small-4-119b" in rendered
    assert "transport=direct_clawrouter" in rendered
    assert "elapsed_sec=0.51" in rendered
    assert "fallback_used=false" in rendered
    assert "telegram model route ok" in rendered
    assert "raw_message" not in rendered


@pytest.mark.asyncio
async def test_handle_modelroute_command_dispatches_selected_route(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    dispatch_result = {
        "ok": True,
        "selected_model": "free/mistral-small-4-119b",
        "transport": "direct_clawrouter",
        "text": "telegram model route ok",
        "telemetry": {"task_class": "free_utility_general", "elapsed_sec": 0.4, "fallback_used": False},
    }

    async def fake_dispatch(*, task_class, message):
        assert task_class == "free_utility_general"
        assert message == "Reply exactly: telegram model route ok"
        return dispatch_result

    monkeypatch.setattr(runner, "_dispatch_modelroute", fake_dispatch)
    reply = await runner._handle_modelroute_command(
        _event("/modelroute free_utility_general Reply exactly: telegram model route ok")
    )
    assert "selected_model=free/mistral-small-4-119b" in reply
    assert "telegram model route ok" in reply
