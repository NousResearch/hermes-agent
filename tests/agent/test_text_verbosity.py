from __future__ import annotations

import pytest

from agent.codex_responses_adapter import _preflight_codex_api_kwargs
from agent.text_verbosity import merge_text_verbosity_override, parse_text_verbosity
from agent.transports.codex import ResponsesApiTransport


def test_parse_text_verbosity_accepts_only_openai_values():
    assert parse_text_verbosity("") is None
    assert parse_text_verbosity(" LOW ") == "low"
    assert parse_text_verbosity("medium") == "medium"
    assert parse_text_verbosity("high") == "high"
    assert parse_text_verbosity("verbose") is None


def test_merge_text_verbosity_only_for_openai_responses_runtime():
    merged = merge_text_verbosity_override(
        {"service_tier": "priority"},
        "low",
        provider="openai",
        api_mode="codex_responses",
        base_url="https://api.openai.com/v1",
    )
    assert merged == {"service_tier": "priority", "text": {"verbosity": "low"}}

    unsupported = merge_text_verbosity_override(
        {"service_tier": "priority"},
        "high",
        provider="xai",
        api_mode="codex_responses",
        base_url="https://api.x.ai/v1",
    )
    assert unsupported == {"service_tier": "priority"}


def test_codex_transport_keeps_top_level_text_verbosity_override():
    kwargs = ResponsesApiTransport().build_kwargs(
        model="gpt-5.1",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        request_overrides={"text": {"verbosity": "high"}},
    )

    assert kwargs["text"] == {"verbosity": "high"}


def test_codex_preflight_allows_and_normalizes_text_verbosity():
    payload = _preflight_codex_api_kwargs(
        {
            "model": "gpt-5.1",
            "instructions": "system",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
            ],
            "store": False,
            "text": {"verbosity": "HIGH"},
        }
    )

    assert payload["text"] == {"verbosity": "high"}


def test_codex_preflight_rejects_invalid_text_verbosity():
    with pytest.raises(ValueError, match="text.verbosity"):
        _preflight_codex_api_kwargs(
            {
                "model": "gpt-5.1",
                "instructions": "system",
                "input": [
                    {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
                ],
                "store": False,
                "text": {"verbosity": "verbose"},
            }
        )
