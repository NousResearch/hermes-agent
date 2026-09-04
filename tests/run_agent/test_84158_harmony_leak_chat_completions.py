# Regression tests for #84158: Harmony-format tool calls leak as raw text
# for self-hosted gpt-oss models served through generic OpenAI-compatible
# chat-completions endpoints (Ollama, vLLM, ...).
#
# The ChatGPT Codex Responses adapter already recovers leaked
# `to=functions.<name>` markup (agent/codex_responses_adapter.py), but that
# path only runs for api_mode=codex_responses. gpt-oss models emit Harmony
# syntax natively regardless of backend, so the conversation loop now detects
# the leak at finalization for Harmony-format models on ANY transport and
# re-prompts the model to emit a structured tool call instead of surfacing
# raw markup as a confident-looking answer.
import sys
import types
from types import SimpleNamespace

import pytest


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent  # noqa: E402


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Short-circuit retry backoff so recovery tests don't block on real
    wall-clock waits (jittered_backoff base delay + tight time.sleep loop)."""
    import time as _time

    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)


def _patch_agent_bootstrap(monkeypatch):
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


def _build_agent(monkeypatch, *, model="gpt-oss:20b"):
    _patch_agent_bootstrap(monkeypatch)

    agent = run_agent.AIAgent(
        model=model,
        provider="custom",
        base_url="http://localhost:11434/v1",
        api_key="ollama-key",
        quiet_mode=True,
        max_iterations=4,
        skip_context_files=True,
        skip_memory=True,
    )
    # Force the non-streaming call path so _interruptible_api_call mocks
    # below are actually consulted (mirrors the codex test suite).
    setattr(agent, "_disable_streaming", True)
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    return agent


# Leaked Harmony/Codex tool-call serialization, matching the shape the
# codex adapter's _TOOL_CALL_LEAK_PATTERN detects.
_LEAKED_MARKUP = 'assistant to=functions.terminal {"cmd": "echo hi"}'


def _chat_leak_response(text: str = _LEAKED_MARKUP):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=text,
                    tool_calls=None,
                    refusal=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        model="gpt-oss:20b",
    )


def _chat_text_response(text: str = "Done."):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=text,
                    tool_calls=None,
                    refusal=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        model="gpt-oss:20b",
    )


def _assert_no_leaked_markup(result):
    assert "to=functions" not in (result.get("final_response") or "")
    for msg in result.get("messages", []):
        if msg.get("role") == "assistant":
            assert "to=functions" not in (msg.get("content") or "")


# ---------------------------------------------------------------------------
# Model-family gate
# ---------------------------------------------------------------------------


def test_model_uses_harmony_format():
    assert run_agent.AIAgent._model_uses_harmony_format("gpt-oss:20b") is True
    assert run_agent.AIAgent._model_uses_harmony_format("gpt-oss-120b") is True
    assert run_agent.AIAgent._model_uses_harmony_format("ollama/gpt-oss:20b") is True
    assert run_agent.AIAgent._model_uses_harmony_format("GPT-OSS:20b") is True
    assert run_agent.AIAgent._model_uses_harmony_format("deepseek-v4-flash") is False
    assert run_agent.AIAgent._model_uses_harmony_format("gpt-5.4") is False
    assert run_agent.AIAgent._model_uses_harmony_format("") is False


# ---------------------------------------------------------------------------
# Recovery on generic chat-completions endpoints
# ---------------------------------------------------------------------------


def test_harmony_leak_recovered_on_chat_completions(monkeypatch):
    """A gpt-oss leak on a chat_completions endpoint is re-prompted away and
    never surfaces as visible content."""
    agent = _build_agent(monkeypatch)
    calls = {"n": 0}

    def _fake_api_call(api_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _chat_leak_response()
        return _chat_text_response("The command ran.")

    monkeypatch.setattr(agent, "_interruptible_api_call", _fake_api_call)

    result = agent.run_conversation("Run a command")

    # Leak detected → nudge → clean response on the second API call.
    assert calls["n"] == 2
    assert result["completed"] is True
    assert result["final_response"] == "The command ran."
    _assert_no_leaked_markup(result)


def test_harmony_leak_gives_up_with_clear_failure_after_3(monkeypatch):
    """A persistently-degenerating gpt-oss model is re-prompted at most 3
    times, then the turn ends with an explicit failure instead of the raw
    markup."""
    agent = _build_agent(monkeypatch)
    calls = {"n": 0}

    def _fake_api_call(api_kwargs):
        calls["n"] += 1
        return _chat_leak_response()

    monkeypatch.setattr(agent, "_interruptible_api_call", _fake_api_call)

    result = agent.run_conversation("Run a command")

    # 1 original leak + 3 re-prompt retries, then give up.
    assert calls["n"] == 4
    assert "repeatedly emitted tool calls as raw text" in result["final_response"]
    _assert_no_leaked_markup(result)


def test_harmony_leak_unchanged_for_non_gpt_oss_model(monkeypatch):
    """Non-Harmony models keep today's behavior: leaked-looking text is
    delivered verbatim (no re-prompt), so the gate is model-specific."""
    agent = _build_agent(monkeypatch, model="deepseek-v4-flash")
    calls = {"n": 0}

    def _fake_api_call(api_kwargs):
        calls["n"] += 1
        return _chat_leak_response()

    monkeypatch.setattr(agent, "_interruptible_api_call", _fake_api_call)

    result = agent.run_conversation("Run a command")

    assert calls["n"] == 1
    assert result["completed"] is True
    assert "to=functions" in result["final_response"]
