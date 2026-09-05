"""Per-model client-cache isolation: two concurrent auxiliary calls to the same
provider/base_url/key but DIFFERENT models (e.g. MoA advisors) never share one cache entry.
"""

from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

def test_model_participates_in_client_cache_key():
    """Same provider/base_url/key, different model -> different cache key.

    This is what stops two concurrent advisors from sharing (and racing on)
    one cached client entry."""
    from agent.auxiliary_client import _client_cache_key

    k_opus = _client_cache_key(
        "openrouter", async_mode=False, base_url="https://openrouter.ai/api/v1",
        api_key="K", model="anthropic/claude-opus-4.8",
    )
    k_gpt = _client_cache_key(
        "openrouter", async_mode=False, base_url="https://openrouter.ai/api/v1",
        api_key="K", model="openai/gpt-5.5",
    )
    assert k_opus != k_gpt
    # Same model still collides (cache still works for reuse).
    k_opus2 = _client_cache_key(
        "openrouter", async_mode=False, base_url="https://openrouter.ai/api/v1",
        api_key="K", model="anthropic/claude-opus-4.8",
    )
    assert k_opus == k_opus2


def test_missing_model_key_is_stable():
    """Omitting model (legacy callers) is still a valid, stable key."""
    from agent.auxiliary_client import _client_cache_key

    a = _client_cache_key("openrouter", async_mode=False, base_url="u", api_key="k")
    b = _client_cache_key("openrouter", async_mode=False, base_url="u", api_key="k")
    assert a == b


@pytest.mark.parametrize("async_mode", [False, True])
def test_call_can_pin_provider_and_disable_all_fallbacks(async_mode):
    """A policy-sensitive caller can keep request data on one endpoint."""
    from agent import auxiliary_client as ac

    mock_type = AsyncMock if async_mode else MagicMock
    create = mock_type(side_effect=ConnectionError("local endpoint unavailable"))
    client = types.SimpleNamespace(
        base_url="http://192.0.2.10:8000/v1",
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create),
        ),
    )

    with (
        patch.object(
            ac,
            "_resolve_task_provider_model",
            return_value=(
                "custom",
                "local-model",
                "http://192.0.2.10:8000/v1",
                "test-key",
                "chat_completions",
            ),
        ),
        patch.object(ac, "_get_cached_client", return_value=(client, "local-model")),
        patch.object(ac, "_transient_retry_count", return_value=0),
        patch.object(ac, "_try_configured_fallback_chain") as configured_fallback,
        patch.object(ac, "_try_main_agent_model_fallback") as main_fallback,
        patch.object(ac, "_try_payment_fallback") as payment_fallback,
    ):
        with pytest.raises(ConnectionError, match="local endpoint unavailable"):
            call = ac.async_call_llm if async_mode else ac.call_llm
            result = call(
                task="approval",
                provider="custom",
                model="local-model",
                base_url="http://192.0.2.10:8000/v1",
                api_key="test-key",
                messages=[{"role": "user", "content": "redacted command"}],
                allow_provider_fallback=False,
            )
            if async_mode:
                asyncio.run(result)

    configured_fallback.assert_not_called()
    main_fallback.assert_not_called()
    payment_fallback.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True])
def test_vision_call_respects_provider_fallback_pin(async_mode):
    """The public route-pinning option also covers the vision entry path."""
    from agent import auxiliary_client as ac

    with (
        patch.object(
            ac,
            "_resolve_task_provider_model",
            return_value=("openai", "vision-model", None, "test-key", None),
        ),
        patch.object(
            ac,
            "resolve_vision_provider_client",
            return_value=("openai", None, "vision-model"),
        ) as resolve_vision,
    ):
        with pytest.raises(RuntimeError, match="No LLM provider configured"):
            call = ac.async_call_llm if async_mode else ac.call_llm
            result = call(
                task="vision",
                provider="openai",
                model="vision-model",
                api_key="test-key",
                messages=[{"role": "user", "content": "redacted image request"}],
                allow_provider_fallback=False,
            )
            if async_mode:
                asyncio.run(result)

    resolve_vision.assert_called_once()
