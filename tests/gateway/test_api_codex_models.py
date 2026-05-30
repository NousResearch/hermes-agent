"""API-server Codex model catalog and session reasoning endpoints."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def api_server():
    from gateway.platforms.api_server import APIServerAdapter

    server = object.__new__(APIServerAdapter)
    server._session_model_overrides = {}
    server._session_reasoning_overrides = {}
    server._check_auth = lambda request: None
    server._MAX_SESSION_HEADER_LEN = 256
    return server


def test_codex_models_endpoint_returns_catalog(api_server):
    sample = [
        {
            "id": "gpt-5.5",
            "display_name": "GPT-5.5",
            "description": "Frontier model",
            "context_length": 272000,
            "reasoning_efforts": [{"effort": "medium", "description": "Balanced"}],
            "default_reasoning_effort": "medium",
        }
    ]

    request = MagicMock()
    request.rel_url.query.get.side_effect = lambda key, default="": {
        "q": "",
        "limit": "",
    }.get(key, default)

    with patch(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        return_value={"api_key": "codex-token"},
    ):
        with patch(
            "hermes_cli.codex_models.list_codex_picker_models",
            return_value=sample,
        ):
            response = asyncio.run(api_server._handle_codex_models(request))

    payload = json.loads(response.body)

    assert payload["provider"] == "codex"
    assert payload["authenticated"] is True
    assert payload["data"][0]["id"] == "gpt-5.5"
    assert payload["data"][0]["default_reasoning_effort"] == "medium"


def test_set_session_model_accepts_reasoning_effort(api_server):
    switch_result = MagicMock(
        success=True,
        new_model="gpt-5.3-codex",
        target_provider="openai-codex",
        api_key="key",
        base_url="https://example.test",
        api_mode="chat_completions",
        warning_message="",
    )

    request = MagicMock()
    request.match_info.get.return_value = "session-1"
    request.json = MagicMock(
        return_value={
            "provider": "openai-codex",
            "model": "gpt-5.3-codex",
            "reasoning_effort": "high",
        }
    )

    with patch("hermes_cli.model_switch.switch_model", return_value=switch_result):
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={"provider": "openai-codex", "api_key": "key", "base_url": ""},
        ):
            with patch("gateway.run._resolve_gateway_model", return_value="gpt-5.4"):
                with patch("hermes_cli.config.load_config", return_value={}):
                    with patch("hermes_cli.config.get_compatible_custom_providers", return_value=[]):
                        response = asyncio.run(api_server._handle_set_session_model(request))

    payload = json.loads(response.body)

    assert payload["model"] == "gpt-5.3-codex"
    assert payload["reasoning_effort"] == "high"
    assert api_server._session_reasoning_overrides["session-1"]["effort"] == "high"
