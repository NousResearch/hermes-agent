"""Regression tests for ``agent.auxiliary_client._build_call_kwargs`` and
the explicit-caller-supplied ``max_tokens`` propagation fix from #59763.

The fixture-stance match for the issue body:

- ``max_tokens=None`` (the default): the key MUST be absent from the built
  kwargs — we do **not** cap output by default, providers that reject the
  parameter (ZAI vision 1210, GitHub Copilot, GPT-5 needing
  ``max_completion_tokens``) must continue to work without changes.
- ``max_tokens=<int>`` (caller explicitly supplies): the key MUST be
  present and equal to that int, ON ALL wires — OpenAI-compat,
  OpenRouter, custom, Anthropic-Messages, NVIDIA NIM. The retry ladder
  further down the file (6288-6294) already strips a 400-rejected
  ``max_tokens`` parameter, so the wire-rejected safety argument
  survives this change.

The auxiliary_client.py:5871-5890 comment documented the old behaviour
as "do NOT cap output by default. … omitting it sidesteps all wire-format
quirks at once" — that clause still holds for ``None``; the fix applies
only when the caller passes an int.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _scrub_openrouter_key(monkeypatch):
    """Make sure no test accidentally inherits OPENROUTER_API_KEY from
    the developer's environment — the OpenRouter branch needs to drive the
    detection deterministically off the mocked URL, not the env var."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def _build(provider: str, model: str = "glm-5.1", base_url: str | None = None,
           max_tokens=None, base_url_value: str = ""):
    """Tiny wrapper that builds kwargs through ``_build_call_kwargs`` with
    the custom-base-url stubbed so provider detection only depends on the
    inputs we care about."""
    from agent.auxiliary_client import _build_call_kwargs

    if base_url_value:
        with patch(
            "agent.auxiliary_client._current_custom_base_url",
            return_value=base_url_value,
        ):
            return _build_call_kwargs(
                provider=provider,
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=max_tokens,
                base_url=base_url,
            )
    return _build_call_kwargs(
        provider=provider,
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=max_tokens,
        base_url=base_url,
    )


class TestBuildCallKwargsMaxTokensPropagation:
    """Regression for #59763: explicit caller cap must reach every wire."""

    def test_default_omits_max_tokens(self):
        """Default behaviour unchanged — None means "let the model decide"."""
        from agent.auxiliary_client import _build_call_kwargs

        with patch(
            "agent.auxiliary_client._current_custom_base_url",
            return_value="https://openrouter.ai/api/v1",
        ):
            kwargs = _build_call_kwargs(
                provider="openrouter",
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=None,
            )
        assert "max_tokens" not in kwargs
        assert "max_completion_tokens" not in kwargs

    def test_caller_supplied_cap_reaches_openrouter(self):
        """Regression #59763: caller-supplied 512 must reach OpenRouter."""
        kwargs = _build(
            provider="openrouter",
            model="openai/gpt-4o-mini",
            base_url_value="https://openrouter.ai/api/v1",
            max_tokens=512,
        )
        assert kwargs.get("max_tokens") == 512

    def test_caller_supplied_cap_reaches_openai_compat(self):
        """Custom OpenAI-compatible endpoints (e.g. Ollama, vLLM) must also
        receive the caller cap. The previous branch silently dropped it.
        """
        kwargs = _build(
            provider="custom",
            model="my-model",
            base_url_value="http://localhost:11434/v1",
            max_tokens=256,
        )
        assert kwargs.get("max_tokens") == 256

    def test_caller_supplied_cap_reaches_anthropic_compat(self):
        """Anthropic-Messages wire is mandatory-on-ground: must always carry.
        (This was already working before #59763 — pinned here so the new
        default-forwarding branch doesn't break it.)
        """
        kwargs = _build(
            provider="anthropic",
            model="claude-opus-4-8",
            base_url_value="https://api.anthropic.com",
            max_tokens=1024,
        )
        assert kwargs.get("max_tokens") == 1024

    def test_caller_supplied_cap_reaches_nvidia_nim(self):
        """NVIDIA NIM wire: must always carry the cap (empty-choices
        workaround)."""
        kwargs = _build(
            provider="nvidia",
            model="meta/llama-3-70b",
            base_url_value="https://integrate.api.nvidia.com/v1",
            max_tokens=2048,
        )
        assert kwargs.get("max_tokens") == 2048
