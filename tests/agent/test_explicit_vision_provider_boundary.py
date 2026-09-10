"""Explicit vision routes must never escape to auto provider discovery."""

import asyncio
from unittest.mock import patch

import pytest


_EXPLICIT_ROUTE = ("openai-codex", "gpt-5.6-terra", None, None, None)


def _unavailable_resolver(*, provider, **_kwargs):
    assert provider == "openai-codex"
    return provider, None, None


def test_sync_explicit_vision_provider_does_not_fall_back_to_auto():
    """A failed explicitly configured provider remains a hard routing failure."""
    from agent.auxiliary_client import call_llm

    with (
        patch("agent.auxiliary_client._resolve_task_provider_model", return_value=_EXPLICIT_ROUTE),
        patch("agent.auxiliary_client.resolve_vision_provider_client", side_effect=_unavailable_resolver) as resolver,
        pytest.raises(RuntimeError, match="provider=openai-codex"),
    ):
        call_llm(task="vision", messages=[])

    assert resolver.call_count == 1


def test_async_explicit_vision_provider_does_not_fall_back_to_auto():
    """The asynchronous vision path preserves the same routing boundary."""
    from agent.auxiliary_client import async_call_llm

    async def invoke():
        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=_EXPLICIT_ROUTE),
            patch("agent.auxiliary_client.resolve_vision_provider_client", side_effect=_unavailable_resolver) as resolver,
            pytest.raises(RuntimeError, match="provider=openai-codex"),
        ):
            await async_call_llm(task="vision", messages=[])
        assert resolver.call_count == 1

    asyncio.run(invoke())
