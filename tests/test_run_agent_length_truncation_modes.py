"""Regression: truncation continuation must not be chat-modes-only.

The truncation continuation branches in agent/conversation_loop.py are
gated on ``api_mode in {"chat_completions", "bedrock_converse",
"anthropic_messages"}``. Responses-API modes (codex_responses / responses /
codex) therefore never request a continuation, never stitch the partial,
never consult the content-filter fallback ladder, and never budget-boost
retry a truncated tool call.

Run: uv run --with pytest python -m pytest tests/test_run_agent_length_truncation_modes.py

Covers both wire shapes: omitted incomplete_details (self-hosted servers) and the canonical Responses ``incomplete_details.reason == "max_output_tokens"`` shape routed through the shared length-continuation budget-boost dispatcher.
"""
import sys
import types
from types import SimpleNamespace

import pytest


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent


def _patch_agent_bootstrap(monkeypatch):
    monkeypatch.setattr(
        "agent.context_compressor.get_model_context_length",
        lambda *args, **kwargs: 131072,
    )
    monkeypatch.setattr(
        run_agent,
        "get_tool_definitions",
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
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})


def _build_agent(monkeypatch, *, api_mode):
    _patch_agent_bootstrap(monkeypatch)

    if api_mode == "chat_completions":
        agent = run_agent.AIAgent(
            model="gpt-4.1-mini",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            api_mode="chat_completions",
            quiet_mode=True,
            max_iterations=4,
            skip_context_files=True,
            skip_memory=True,
        )
    else:
        agent = run_agent.AIAgent(
            model="gpt-5-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="codex-token",
            api_mode=api_mode,
            quiet_mode=True,
            max_iterations=4,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    agent._save_session_log = lambda messages: None
    # Force the non-streaming response path so the truncation-continuation
    # guard logic (which lives in the shared finish-reason handling) is
    # exercised deterministically instead of the provider's stream consumer.
    agent._disable_streaming = True
    return agent


def _responses_partial_response(text, *, incomplete_details=None):
    """Responses-API wire cut short by the output cap.

    ``incomplete_details=None`` mirrors self-hosted /v1/responses servers
    (e.g. llama.cpp server-chat.cpp) that stop at the cap without
    populating incomplete_details.
    """
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        usage=SimpleNamespace(input_tokens=5, output_tokens=3, total_tokens=8),
        status="incomplete",
        incomplete_details=incomplete_details,
        model="gpt-5-codex",
    )


def _responses_complete_response(text):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        usage=SimpleNamespace(input_tokens=5, output_tokens=3, total_tokens=8),
        status="completed",
        incomplete_details=None,
        model="gpt-5-codex",
    )


def _chat_truncated_response(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None),
            finish_reason="length",
        )],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


def _chat_complete_response(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


def test_chat_completions_truncation_requests_continuation(monkeypatch):
    """Control: chat modes DO continue a truncated turn (should pass)."""
    agent = _build_agent(monkeypatch, api_mode="chat_completions")
    responses = [
        _chat_truncated_response("Partial text"),
        _chat_complete_response("Partial text — and the rest."),
    ]
    monkeypatch.setattr(
        agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    result = agent.run_conversation("Write a very long essay")

    assert result["completed"] is True
    assert "and the rest" in (result["final_response"] or "")


def test_codex_responses_truncation_requests_continuation(monkeypatch):
    """A Responses-family text truncation must still recover the partial.

    Behavior-pinning contract for a /v1/responses backend that hits the
    output cap without populating incomplete_details: the turn must not
    hard-fail or lose the partial — a continuation is issued and the
    stitched content becomes the final response, however the routing is
    implemented internally.
    """
    agent = _build_agent(monkeypatch, api_mode="codex_responses")
    responses = [
        _responses_partial_response("Partial text", incomplete_details=None),
        _responses_complete_response("Partial text — and the rest."),
    ]
    monkeypatch.setattr(
        agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    result = agent.run_conversation("Write a very long essay")

    assert result["completed"] is True
    assert "and the rest" in (result["final_response"] or "")


def test_codex_responses_truncated_tool_call_retried_before_execution(monkeypatch):
    """A length-truncated tool call must be retried, not executed as-is.

    Chat modes re-issue the API call with a boosted max_tokens budget
    (truncated_tool_call_retries). Responses modes skip that branch, so
    the broken call executes immediately. Encode the expected contract:
    the truncated call is retried before any tool executes.
    """
    agent = _build_agent(monkeypatch, api_mode="codex_responses")
    truncated_tool = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="terminal",
                arguments='{"command": "rm -rf ',
            )
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=4, total_tokens=16),
        status="incomplete",
        incomplete_details=None,
        model="gpt-5-codex",
    )
    responses = [truncated_tool, _responses_complete_response("Done.")]
    monkeypatch.setattr(
        agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))
    executed = []
    monkeypatch.setattr(
        agent,
        "_execute_tool_calls",
        lambda assistant_message, messages, effective_task_id: executed.append(
            assistant_message),
    )

    result = agent.run_conversation("Clean up /tmp leftovers")

    # The truncated JSON was retried with the boosted budget (guard 2's
    # 'continue' re-runs the inner attempt loop), so the turn completes with
    # the retried answer instead of the broken call executing.
    assert result["completed"] is True
    assert not executed                            # truncated call NOT executed
    assert (result["final_response"] or "") == "Done."   # retry result won


def _responses_max_output_tokens_tool_call():
    """Canonical OpenAI Responses truncation: status='incomplete' with
    incomplete_details.reason='max_output_tokens' and a truncated tool call.

    This is the standard wire shape for a function call cut off by the
    output cap — the shape the old routing (finish_reason='incomplete')
    retried at an unchanged budget via the Codex-specific branch.
    """
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="terminal",
                arguments='{"command": "rm -rf ',
            )
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=4, total_tokens=16),
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        model="gpt-5-codex",
    )


def test_codex_responses_max_output_tokens_tool_call_budget_boosted(monkeypatch):
    """Standard Responses truncation must share the length-continuation
    budget boost, not burn Codex-incomplete retries at an unchanged cap.

    Regression for the review finding on #91779: ``status == "incomplete"``
    with ``incomplete_details.reason == "max_output_tokens"`` was routed to
    the Codex-specific incomplete branch (bounded 3× replay, no
    ``_ephemeral_max_output_tokens`` boost), so a normal truncated function
    call was retried at the same insufficient budget and could terminate
    with "Codex response remained incomplete".  The recognized-reason
    shape must now flow through the same length-continuation dispatcher as
    the chat modes: the next request's output-token budget increases and
    the truncated tool call is never executed.
    """
    agent = _build_agent(monkeypatch, api_mode="codex_responses")
    responses = [
        _responses_max_output_tokens_tool_call(),
        _responses_complete_response("Done."),
    ]
    # _ephemeral_max_output_tokens is set by the length-continuation
    # tool-call retry (guard) and consumed by build_kwargs on the next
    # request.  Capture it at build time — right before the real
    # build_kwargs consumes it — so the budget boost is observable.
    boosts_seen = []

    real_build = agent._build_api_kwargs

    def build_wrapper(api_messages, tools_for_api=None):
        ephemeral = getattr(agent, "_ephemeral_max_output_tokens", None)
        if ephemeral is not None:
            boosts_seen.append(ephemeral)
        return real_build(api_messages, tools_for_api=tools_for_api)

    monkeypatch.setattr(agent, "_build_api_kwargs", build_wrapper)

    def fake_api_call(api_kwargs):
        return responses.pop(0)

    monkeypatch.setattr(agent, "_interruptible_api_call", fake_api_call)
    executed = []
    monkeypatch.setattr(
        agent,
        "_execute_tool_calls",
        lambda assistant_message, messages, effective_task_id: executed.append(
            assistant_message),
    )

    result = agent.run_conversation("Clean up /tmp leftovers")

    assert result["completed"] is True
    assert not executed                            # truncated call NOT executed
    assert (result["final_response"] or "") == "Done."   # retry result won
    # The retry carried a boosted output-token budget via
    # _ephemeral_max_output_tokens (2^retries × base), proving the standard
    # max_output_tokens shape reached the length-continuation dispatcher
    # instead of the unchanged-budget Codex-incomplete path.
    assert len(boosts_seen) >= 1, (
        f"expected a budget boost on the retry, saw none "
        f"(boosts_seen={boosts_seen!r})"
    )


def test_codex_responses_max_output_tokens_text_continuation(monkeypatch):
    """A recognized-reason (max_output_tokens) text truncation must still be
    recovered by a continuation — the other half of the review finding.

    Before the fix, this shape took the Codex-incomplete replay branch:
    it appended the interim and nudged, which works for text too, but at an
    unchanged budget.  After routing through length-continuation the partial
    is stitched and the turn completes with the full response.
    """
    agent = _build_agent(monkeypatch, api_mode="codex_responses")
    responses = [
        _responses_partial_response(
            "Partial text",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        ),
        _responses_complete_response("Partial text — and the rest."),
    ]
    monkeypatch.setattr(
        agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    result = agent.run_conversation("Write a very long essay")

    assert result["completed"] is True
    assert "and the rest" in (result["final_response"] or "")