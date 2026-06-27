"""Regression tests for config-backed auxiliary vision routing.

A configured ``auxiliary.vision.base_url`` must not be re-sent as an
explicit resolver override.  The resolver intentionally treats explicit
``base_url`` as a custom endpoint, so forwarding the config-derived value a
second time rewrites providers such as ``openai-codex`` to ``custom`` and drops
their provider-specific auth/header handling.

"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


CODEX_PROVIDER = "openai-codex"
CODEX_MODEL = "gpt-5"
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_API_KEY = "config-derived-token"


def _response(content: str = "ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_call_llm_vision_does_not_forward_config_derived_base_url_as_explicit_override():
    captured = {}
    client = MagicMock()
    client.chat.completions.create.return_value = _response()

    def fake_resolve_vision_provider_client(**kwargs):
        captured.update(kwargs)
        return CODEX_PROVIDER, client, CODEX_MODEL

    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=(CODEX_PROVIDER, CODEX_MODEL, CODEX_BASE_URL, CODEX_API_KEY, "codex_responses"),
    ), patch(
        "agent.auxiliary_client.resolve_vision_provider_client",
        side_effect=fake_resolve_vision_provider_client,
    ), patch(
        "agent.auxiliary_client._get_task_extra_body",
        return_value={},
    ):
        from agent.auxiliary_client import call_llm

        call_llm(task="vision", messages=[{"role": "user", "content": "describe"}])

    assert captured["provider"] == CODEX_PROVIDER
    assert captured["model"] == CODEX_MODEL
    assert captured["base_url"] is None
    assert captured["api_key"] is None
    assert captured["async_mode"] is False


@pytest.mark.asyncio
async def test_async_call_llm_vision_does_not_forward_config_derived_base_url_as_explicit_override():
    captured = {}

    class AsyncCompletions:
        async def create(self, **kwargs):
            return _response()

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=AsyncCompletions()),
        base_url=CODEX_BASE_URL,
    )

    def fake_resolve_vision_provider_client(**kwargs):
        captured.update(kwargs)
        return CODEX_PROVIDER, client, CODEX_MODEL

    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=(CODEX_PROVIDER, CODEX_MODEL, CODEX_BASE_URL, CODEX_API_KEY, "codex_responses"),
    ), patch(
        "agent.auxiliary_client.resolve_vision_provider_client",
        side_effect=fake_resolve_vision_provider_client,
    ), patch(
        "agent.auxiliary_client._get_task_extra_body",
        return_value={},
    ):
        from agent.auxiliary_client import async_call_llm

        await async_call_llm(task="vision", messages=[{"role": "user", "content": "describe"}])

    assert captured["provider"] == CODEX_PROVIDER
    assert captured["model"] == CODEX_MODEL
    assert captured["base_url"] is None
    assert captured["api_key"] is None
    assert captured["async_mode"] is True