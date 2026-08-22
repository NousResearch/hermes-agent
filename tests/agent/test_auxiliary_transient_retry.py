"""Transient-transport retry count + per-model client-cache isolation.

Two related hardening behaviors for auxiliary calls (which include MoA
reference advisors, a pinned-model path where provider fallback is not a
meaningful recovery):

1. A transient transport blip (connection reset / timeout / 5xx) is retried
   on the SAME provider several times with backoff before giving up — a single
   upstream blip should not silently lose a pinned auxiliary call (root of the
   run2 double-advisor "Connection error" collapse).
2. Two auxiliary calls to the same provider/base_url/key but DIFFERENT models
   get DISTINCT client-cache keys, so a concurrent fan-out (e.g. opus + gpt-5.5
   advisors) never shares one client entry.
"""

from __future__ import annotations

import os
import types
from unittest.mock import MagicMock, patch

import pytest




def test_transient_retry_count_default(monkeypatch):
    from agent import auxiliary_client as ac

    # No config value -> default.
    monkeypatch.setattr(ac, "load_config", lambda: {}, raising=False)
    with patch("hermes_cli.config.load_config", return_value={}), \
         patch("hermes_cli.config.cfg_get", return_value=None):
        assert ac._transient_retry_count() == ac._DEFAULT_TRANSIENT_RETRIES




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


def test_title_generation_timeout_does_not_retry_or_fallback(monkeypatch):
    """A cosmetic title timeout should keep the derived title, not add load."""
    from agent.auxiliary_client import call_llm

    client = MagicMock()
    client.base_url = "http://localhost:13305/v1"
    client.chat.completions.create.side_effect = TimeoutError("request timed out")

    monkeypatch.setattr("agent.auxiliary_client._TRANSIENT_RETRY_BACKOFF_BASE", 0)

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("custom", "tiny-title-model", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "tiny-title-model"),
        ),
        patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"timeout": 1},
        ),
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
        ) as configured_fallback,
        patch(
            "agent.auxiliary_client._try_main_agent_model_fallback",
        ) as main_model_fallback,
        patch("agent.auxiliary_client._try_payment_fallback") as payment_fallback,
        pytest.raises(TimeoutError, match="request timed out"),
    ):
        call_llm(
            task="title_generation",
            messages=[{"role": "user", "content": "fix title timeouts"}],
        )

    assert client.chat.completions.create.call_count == 1
    configured_fallback.assert_not_called()
    main_model_fallback.assert_not_called()
    payment_fallback.assert_not_called()


def test_title_generation_forwards_output_cap():
    """The title task must stay bounded on OpenAI-compatible local servers."""
    from agent.auxiliary_client import call_llm

    client = MagicMock()
    client.base_url = "http://localhost:13305/v1"
    client.chat.completions.create.return_value = MagicMock()

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("custom", "tiny-title-model", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "tiny-title-model"),
        ),
        patch(
            "agent.auxiliary_client._validate_llm_response",
            side_effect=lambda response, _task, **_kwargs: response,
        ),
    ):
        call_llm(
            task="title_generation",
            messages=[{"role": "user", "content": "fix title timeouts"}],
            max_tokens=64,
        )

    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 64
