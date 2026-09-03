"""Tests for the Ollama-cloud Kimi top_p contract override.

Ollama-cloud-proxied Kimi models (``kimi-k2.6:cloud`` etc., served through a
local Ollama daemon with ``provider: custom``) reject requests that omit
``top_p``: the Ollama server substitutes its own default of 1 and the cloud
endpoint fails with::

    400 Bad Request: Invalid value for 'top_p': 1.
    This endpoint requires top_p=0.95.

``_fixed_top_p_for_model`` must pin 0.95 for exactly that model family
(Kimi AND the ``:cloud`` tag) and never touch top_p for anything else.
An explicit caller/request-override top_p must always win.
"""

from __future__ import annotations

import pytest

from agent.auxiliary_client import (
    _fixed_top_p_for_model,
    _is_ollama_cloud_model,
)
from agent.transports.chat_completions import ChatCompletionsTransport


@pytest.mark.parametrize(
    "model",
    [
        "kimi-k2.6:cloud",
        "kimi-k2:cloud",
        "Kimi-K2.6:CLOUD",  # case-insensitive
        "  kimi-k2.6:cloud  ",  # whitespace tolerant
        "ollama/kimi-k2.6:cloud",  # prefixed form
    ],
)
def test_fixed_top_p_for_ollama_cloud_kimi(model: str) -> None:
    assert _fixed_top_p_for_model(model) == 0.95


@pytest.mark.parametrize(
    "model",
    [
        None,
        "",
        "kimi-k2.6",  # Kimi but not Ollama-cloud
        "kimi-k2-turbo-preview",  # direct Moonshot API
        "llama3.2",  # local Ollama, not cloud
        "qwen3:8b",  # local Ollama tag, not :cloud
        "nemotron-3-nano:30b",  # ollama-cloud provider but no :cloud tag
        "gpt-oss:120b-cloud",  # dash suffix, not the :cloud tag
        "moonshot-v1:cloud",  # Moonshot is not the verified Kimi contract
        "claude-sonnet-4.6",
        "trinity-large-thinking",
    ],
)
def test_fixed_top_p_none_for_other_models(model) -> None:
    assert _fixed_top_p_for_model(model) is None


def test_is_ollama_cloud_model() -> None:
    assert _is_ollama_cloud_model("kimi-k2.6:cloud") is True
    assert _is_ollama_cloud_model("glm-4.6:cloud") is True
    assert _is_ollama_cloud_model("kimi-k2.6") is False
    assert _is_ollama_cloud_model("qwen3:8b") is False
    assert _is_ollama_cloud_model(None) is False


# ── Transport threading ──────────────────────────────────────────────────


def _messages() -> list[dict]:
    return [{"role": "user", "content": "hi"}]


def test_transport_applies_fixed_top_p_on_profile_path() -> None:
    from providers import get_provider_profile

    profile = get_provider_profile("custom")
    assert profile is not None
    kwargs = ChatCompletionsTransport().build_kwargs(
        model="kimi-k2.6:cloud",
        messages=_messages(),
        provider_profile=profile,
        fixed_top_p=0.95,
    )
    assert kwargs["top_p"] == 0.95


def test_transport_request_override_beats_fixed_top_p_profile_path() -> None:
    from providers import get_provider_profile

    profile = get_provider_profile("custom")
    kwargs = ChatCompletionsTransport().build_kwargs(
        model="kimi-k2.6:cloud",
        messages=_messages(),
        provider_profile=profile,
        fixed_top_p=0.95,
        request_overrides={"top_p": 0.8},
    )
    assert kwargs["top_p"] == 0.8


def test_transport_applies_fixed_top_p_on_legacy_path() -> None:
    kwargs = ChatCompletionsTransport().build_kwargs(
        model="kimi-k2.6:cloud",
        messages=_messages(),
        fixed_top_p=0.95,
    )
    assert kwargs["top_p"] == 0.95


def test_transport_request_override_beats_fixed_top_p_legacy_path() -> None:
    kwargs = ChatCompletionsTransport().build_kwargs(
        model="kimi-k2.6:cloud",
        messages=_messages(),
        fixed_top_p=0.95,
        request_overrides={"top_p": 0.8},
    )
    assert kwargs["top_p"] == 0.8


def test_transport_no_top_p_without_override() -> None:
    # No fixed_top_p → the param must not appear at all (other providers keep
    # their server-side defaults).
    kwargs = ChatCompletionsTransport().build_kwargs(
        model="deepseek-chat",
        messages=_messages(),
    )
    assert "top_p" not in kwargs
