"""Responses (codex_responses) truncation continuation must not retry at the cap that truncated.

``continue_codex_incomplete`` recovers a ``status=incomplete`` turn by asking the model to
continue. Its sibling truncation retries — the text continuation and ``_retry_truncated_tool_call``
— both escalate ``max_output_tokens`` through ``agent._ephemeral_max_output_tokens`` so the retry
never re-sends the budget that truncated. This continuation used to re-send the cap unchanged, so
a recognized ``incomplete_details.reason = max_output_tokens`` turn spent all three attempts at the
same ceiling and ended on "Codex response remained incomplete after 3 continuation attempts".

These run against a self-hosted Responses endpoint: the ChatGPT Codex backend never receives
``max_output_tokens``, so the budget is only observable on the wire for endpoints that accept it.
"""

import sys
import types
from types import SimpleNamespace

import pytest


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent
from agent import conversation_loop


SELF_HOSTED_RESPONSES_URL = "http://127.0.0.1:8080/v1"
BASE_OUTPUT_CAP = 4096


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    import time as _time

    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **k: 0.0)
    monkeypatch.setattr(_time, "sleep", lambda *a, **k: None)


def _patch_agent_bootstrap(monkeypatch):
    monkeypatch.setattr(
        "model_tools.get_tool_definitions",
        lambda **kwargs: [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Run shell commands.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})


def _build_agent(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model="gpt-5-codex",
        provider="custom",
        api_mode="codex_responses",
        base_url=SELF_HOSTED_RESPONSES_URL,
        api_key="local-token",
        max_tokens=BASE_OUTPUT_CAP,
        quiet_mode=True,
        max_iterations=4,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    return agent


def _incomplete_text_response(text):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                status="incomplete",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=BASE_OUTPUT_CAP, total_tokens=BASE_OUTPUT_CAP + 10),
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        model="gpt-5-codex",
    )


def _incomplete_function_call_response():
    """Canonical truncated function call: the cap ran out mid-arguments."""
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="terminal",
                arguments='{"command": "echo partial',
                status="incomplete",
            )
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=BASE_OUTPUT_CAP, total_tokens=BASE_OUTPUT_CAP + 10),
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        model="gpt-5-codex",
    )


def _completed_response(text):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                status="completed",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        usage=SimpleNamespace(input_tokens=20, output_tokens=5, total_tokens=25),
        status="completed",
        model="gpt-5-codex",
    )


def _run_turn(agent, monkeypatch, responses):
    """Drive one turn against canned Responses payloads, recording each request's kwargs."""
    requests = []
    monkeypatch.setattr(
        agent, "_interruptible_api_call",
        lambda api_kwargs: requests.append(dict(api_kwargs)) or responses.pop(0),
    )
    agent.run_conversation("say hello")
    return requests


def test_output_cap_truncated_text_continuation_raises_the_budget(monkeypatch):
    agent = _build_agent(monkeypatch)
    requests = _run_turn(
        agent, monkeypatch, [_incomplete_text_response("partial answer"), _completed_response("Hello.")]
    )

    caps = [r.get("max_output_tokens") for r in requests]
    assert len(requests) == 2, f"expected one continuation request, got {len(requests)}"
    assert caps[0] == BASE_OUTPUT_CAP
    assert caps[1] == BASE_OUTPUT_CAP * 2, (
        f"continuation re-sent the output cap that truncated: {caps}"
    )


def test_truncated_function_call_is_not_executed_and_retry_raises_the_budget(monkeypatch):
    agent = _build_agent(monkeypatch)
    tool_rounds = []
    monkeypatch.setattr(conversation_loop, "run_tool_round", lambda *a, **k: tool_rounds.append(a))
    requests = _run_turn(
        agent, monkeypatch,
        [_incomplete_function_call_response(), _completed_response("Hello.")],
    )

    caps = [r.get("max_output_tokens") for r in requests]
    assert tool_rounds == [], "a function call truncated mid-arguments must never be executed"
    assert len(requests) == 2, f"expected one continuation request, got {len(requests)}"
    assert caps[1] == BASE_OUTPUT_CAP * 2, (
        f"continuation re-sent the output cap that truncated: {caps}"
    )
