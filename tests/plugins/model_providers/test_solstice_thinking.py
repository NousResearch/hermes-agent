"""Solstice sends Gemini's native ``thinking_config``: its client reads only that key, so the generic
``extra_body.reasoning`` fallback left every effort (and thinking-off for titles) off the wire."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

import providers
from agent.auxiliary_client import _build_call_kwargs
from agent.title_generator import TITLE_MAX_TOKENS

_BASE = "https://generativelanguage.googleapis.com/v1alpha"


@pytest.mark.parametrize(
    ("reasoning_config", "expected"),
    [
        ({"enabled": False}, {"thinking_config": {"includeThoughts": False, "thinkingLevel": "low"}}),
        ({"enabled": True, "effort": "medium"}, {"thinking_config": {"includeThoughts": True, "thinkingLevel": "medium"}}),
        ({"enabled": True, "effort": "xhigh"}, {"thinking_config": {"includeThoughts": True, "thinkingLevel": "xhigh"}}),
        (None, {}),  # unset effort keeps the model's own default
    ],
)
def test_solstice_profile_translates_reasoning_to_native_thinking_config(reasoning_config, expected):
    profile = providers.get_provider_profile("solstice")

    assert profile is not None
    assert profile.build_extra_body(model="gemini-4-neon", reasoning_config=reasoning_config, base_url=_BASE) == expected


def test_solstice_title_call_reaches_the_wire_with_thinking_bounded():
    """The title pass's thinking-off must arrive as ``thinking_config`` (not the ignored ``reasoning``)."""
    kwargs = _build_call_kwargs(
        "solstice", "gemini-4-neon", [{"role": "user", "content": "hi"}], max_tokens=512,
        reasoning_config={"enabled": False}, base_url=_BASE, task="title_generation",
    )

    assert kwargs["extra_body"]["thinking_config"] == {"includeThoughts": False, "thinkingLevel": "low"}
    assert "reasoning" not in kwargs["extra_body"]


_REPLY = {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}]}


def _generation_config(**create_kwargs) -> dict:
    """Send *create_kwargs* through the real Solstice client; return the posted ``generationConfig``."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_REPLY)

    profile = providers.get_provider_profile("solstice")
    client = profile.create_client(api_key="token", base_url=_BASE, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.chat.completions.create(**create_kwargs)
    (request,) = seen
    assert request.url.path.endswith(":generateContentPerUserQuota")
    return json.loads(request.content)["generationConfig"]


def test_solstice_title_request_bounds_thinking_and_keeps_the_title_cap():
    kwargs = _build_call_kwargs(
        "solstice", "gemini-4-neon", [{"role": "user", "content": "hi"}], max_tokens=TITLE_MAX_TOKENS,
        reasoning_config={"enabled": False}, base_url=_BASE, task="title_generation",
    )

    config = _generation_config(**kwargs)

    assert config["thinkingConfig"] == {"includeThoughts": False, "thinkingLevel": "low"}
    assert config["maxOutputTokens"] == TITLE_MAX_TOKENS


def _main_turn_kwargs(reasoning_config: dict) -> dict:
    """The request kwargs a Solstice agent turn builds under *reasoning_config*."""
    from agent.turn_api_request import build_api_request
    from agent.turn_context import _PER_TURN_RESET_STATE
    from run_agent import AIAgent

    with patch("model_tools.get_tool_definitions", return_value=[]), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(provider="solstice", model="gemini-4-neon", base_url=_BASE, api_key="token", quiet_mode=True,
                        skip_context_files=True, skip_memory=True, session_id="solstice-thinking")
    for name, value in _PER_TURN_RESET_STATE:
        setattr(agent, name, value)
    agent.reasoning_config = reasoning_config
    messages = [{"role": "system", "content": "You are a test agent."}, {"role": "user", "content": "hi"}]
    return build_api_request(
        agent, api_messages=messages, _moa_prepared_request=None, tools_for_api=None,
        system_message=messages[0]["content"], messages=messages, original_user_message="hi", approx_tokens=0,
        total_chars=0, retry_count=0, api_call_count=1, api_request_id="req-1", api_start_time=0.0,
        effective_task_id="task-1", turn_id="turn-1",
    ).api_kwargs


@pytest.mark.parametrize(("effort", "level"), [("medium", "medium"), ("high", "high"), ("xhigh", "xhigh")])
def test_solstice_main_turn_effort_reaches_the_wire(effort, level):
    config = _generation_config(**_main_turn_kwargs({"enabled": True, "effort": effort}))

    assert config["thinkingConfig"] == {"includeThoughts": True, "thinkingLevel": level}
    # No caller cap on a main turn: the model's own ceiling, so thinking cannot starve the answer.
    assert config["maxOutputTokens"] == 65535
