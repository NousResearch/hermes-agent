"""Solstice sends Gemini's native ``thinking_config``: its client reads only that key, so the generic
``extra_body.reasoning`` fallback left every effort (and thinking-off for titles) off the wire. Each
listed model takes its own levels (any other one is HTTP 400) and, with no effort set,
its default level with thoughts on (the summaries stream only when requested)."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

import providers
from agent.auxiliary_client import _build_call_kwargs
from agent.title_generator import TITLE_MAX_TOKENS

_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _thinking(model: str, reasoning_config) -> dict:
    profile = providers.get_provider_profile("solstice")
    assert profile is not None
    return profile.build_extra_body(model=model, reasoning_config=reasoning_config, base_url=_BASE)


def _on(level: str) -> dict:
    return {"thinking_config": {"includeThoughts": True, "thinkingLevel": level}}


def _off(level: str) -> dict:
    return {"thinking_config": {"includeThoughts": False, "thinkingLevel": level}}


@pytest.mark.parametrize(
    ("reasoning_config", "expected"),
    [
        ({"enabled": False}, _off("low")),
        ({"enabled": True, "effort": "medium"}, _on("medium")),
        ({"enabled": True, "effort": "xhigh"}, _on("xhigh")),
        (None, _on("high")),  # unset effort: Neon's default, thoughts on
    ],
)
def test_solstice_profile_translates_reasoning_to_native_thinking_config(reasoning_config, expected):
    assert _thinking("gemini-4-neon", reasoning_config) == expected


@pytest.mark.parametrize(
    ("model", "level"),
    [("gemini-4-hydrogen", "high"), ("gemini-4-neon", "high"), ("gemini-4-carbon", "medium"),
     ("gemini-3.5-flash-lite", "low"), ("gemini-flash-lite-latest", "low")],
)
@pytest.mark.parametrize("reasoning_config", [None, {}, {"enabled": True}])
def test_unset_effort_sends_each_models_default_with_thoughts(model, level, reasoning_config):
    assert _thinking(model, reasoning_config) == _on(level)


_GEMINI4_XHIGH = {"minimal": "low", "low": "low", "medium": "medium", "high": "high",
                  "xhigh": "xhigh", "max": "xhigh", "ultra": "xhigh"}
_CARBON = {**_GEMINI4_XHIGH, "xhigh": "high", "max": "high", "ultra": "high"}
_FLASH_LITE = {**_CARBON, "minimal": "minimal"}


@pytest.mark.parametrize(
    ("model", "table"),
    [("gemini-4-hydrogen", _GEMINI4_XHIGH), ("gemini-4-neon", _GEMINI4_XHIGH), ("gemini-4-carbon", _CARBON),
     ("gemini-3.5-flash-lite", _FLASH_LITE), ("gemini-flash-lite-latest", _FLASH_LITE)],
)
def test_every_effort_clamps_onto_the_models_allowed_levels(model, table):
    """Never ``minimal`` to Gemini 4, never ``xhigh`` to Carbon or Flash-Lite: both are HTTP 400."""
    assert {effort: _thinking(model, {"enabled": True, "effort": effort}) for effort in table} == {
        effort: _on(level) for effort, level in table.items()
    }


@pytest.mark.parametrize(
    ("model", "floor"),
    [("gemini-4-hydrogen", "low"), ("gemini-4-neon", "low"), ("gemini-4-carbon", "low"),
     ("gemini-3.5-flash-lite", "minimal"), ("gemini-flash-lite-latest", "minimal")],
)
@pytest.mark.parametrize("reasoning_config", [{"enabled": False}, {"enabled": True, "effort": "none"},
                                              {"enabled": True, "effort": "disabled"}])
def test_thinking_off_hides_thoughts_at_the_models_lowest_level(model, floor, reasoning_config):
    assert _thinking(model, reasoning_config) == _off(floor)


def test_an_effort_off_the_ladder_gets_the_models_default():
    assert _thinking("gemini-4-carbon", {"enabled": True, "effort": "turbo"}) == _on("medium")


def test_an_unlisted_model_keeps_the_shared_gemini_mapping():
    """A user-typed id outside the list gets what the gemini provider would send."""
    from agent.reasoning_effort import gemini_top_thinking_level
    from agent.transports.chat_completions import _build_gemini_thinking_config

    for config in (None, {"enabled": False}, {"enabled": True, "effort": "ultra"}):
        raw = _build_gemini_thinking_config("gemini-4-xenon", config)
        expected = {"thinking_config": gemini_top_thinking_level("gemini-4-xenon", config, raw)} if raw else {}
        assert _thinking("gemini-4-xenon", config) == expected
    assert _thinking("gemini-4-xenon", None) == {}


@pytest.mark.parametrize("model", ["gemini-4-hydrogen", "gemini-4-carbon", "gemini-flash-lite-latest"])
@pytest.mark.parametrize("reasoning_config", [None, {"enabled": False}, {"enabled": True, "effort": "ultra"}])
def test_no_output_budget_rides_in_extra_body(model, reasoning_config):
    """A long-decode budget needs stored interactions, which the per-user-quota endpoint rejects."""
    body = _thinking(model, reasoning_config)
    assert set(body) == {"thinking_config"}
    assert "max_output_tokens" not in str(body) and "maxOutputTokens" not in str(body)


@pytest.mark.parametrize(("model", "floor"), [("gemini-4-neon", "low"), ("gemini-flash-lite-latest", "minimal")])
def test_solstice_title_call_reaches_the_wire_with_thinking_bounded(model, floor):
    """The title pass's thinking-off must arrive as ``thinking_config`` (not the ignored ``reasoning``)."""
    kwargs = _build_call_kwargs(
        "solstice", model, [{"role": "user", "content": "hi"}], max_tokens=512,
        reasoning_config={"enabled": False}, base_url=_BASE, task="title_generation",
    )

    assert kwargs["extra_body"]["thinking_config"] == {"includeThoughts": False, "thinkingLevel": floor}
    assert "reasoning" not in kwargs["extra_body"]


_REPLY = {"id": "i1", "status": "completed", "steps": [{"type": "model_output", "content": [{"type": "text", "text": "ok"}]}]}


def _generation_config(**create_kwargs) -> dict:
    """Send *create_kwargs* through the real Solstice client; return the posted ``generation_config``."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_REPLY)

    profile = providers.get_provider_profile("solstice")
    client = profile.create_client(api_key="token", base_url=_BASE, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.chat.completions.create(**create_kwargs)
    (request,) = seen
    assert request.url.path.endswith("/interactions:createPerUserQuota")
    return json.loads(request.content)["generation_config"]


def test_solstice_title_request_bounds_thinking_and_keeps_the_title_cap():
    kwargs = _build_call_kwargs(
        "solstice", "gemini-4-neon", [{"role": "user", "content": "hi"}], max_tokens=TITLE_MAX_TOKENS,
        reasoning_config={"enabled": False}, base_url=_BASE, task="title_generation",
    )

    config = _generation_config(**kwargs)

    assert (config["thinking_level"], config["thinking_summaries"]) == ("low", "none")
    assert config["max_output_tokens"] == TITLE_MAX_TOKENS


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

    assert (config["thinking_level"], config["thinking_summaries"]) == (level, "auto")
    # No caller cap on a main turn: the model's own ceiling, so thinking cannot starve the answer.
    assert config["max_output_tokens"] == 65535
