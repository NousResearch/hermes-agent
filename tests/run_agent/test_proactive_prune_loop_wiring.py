"""Behavioral tests for the post-tool proactive tool-result prune wiring.

The conversation loop's post-tool gate now has a prune arm inside the
``elif agent.compression_enabled`` branch: when full compression does NOT
fire (the usual case on a large-window model), the deterministic no-LLM
prune gets one shot per tool iteration, committing only when the engine
returns a NEW list object with a non-zero prune count.

These tests drive ``run_conversation()`` through real tool iterations and pin:
- the prune is consulted when compression stands down;
- a committed prune replaces ``messages`` for subsequent iterations;
- a no-op (input object returned) commits nothing;
- a compressor WITHOUT the method (plugin engine predating the hook /
  SimpleNamespace test double) does not raise — getattr-guarded;
- a raising prune is swallowed (debug log), never fails the turn.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _tool_call(i: int):
    return SimpleNamespace(
        id=f"call_{i}",
        type="function",
        function=SimpleNamespace(name="web_search", arguments='{"query": "x"}'),
    )


def _tool_response(i: int):
    msg = SimpleNamespace(
        content=None,
        reasoning_content=None,
        reasoning=None,
        tool_calls=[_tool_call(i)],
    )
    choice = SimpleNamespace(message=msg, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _stop_response():
    msg = SimpleNamespace(
        content="done",
        reasoning_content=None,
        reasoning=None,
        tool_calls=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _quiet_compressor() -> MagicMock:
    """A compressor that never demands full compression.

    ``should_compress`` False routes the post-tool gate into the ``elif``
    branch where the proactive prune arm lives.  ``should_compress_info``
    reports unblocked (no block reason) so the overflow warning stays quiet.
    """
    compressor = MagicMock()
    compressor.protect_first_n = 3
    compressor.protect_last_n = 20
    compressor.threshold_tokens = 500_000
    compressor.context_length = 1_000_000
    compressor.last_prompt_tokens = 120_000
    compressor.should_compress.return_value = False
    compressor.should_compress_info.return_value = (False, None)
    compressor.should_defer_preflight_to_real_usage.return_value = True
    compressor.get_active_compression_failure_cooldown.return_value = None
    return compressor


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=10,
        )
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a._disable_streaming = True
    a.tool_delay = 0
    a.save_trajectories = False
    a.compression_enabled = True
    a.context_compressor = _quiet_compressor()
    return a


def _run_tool_loop(agent, n_tool_iterations: int):
    responses = [_tool_response(i) for i in range(n_tool_iterations)]
    responses.append(_stop_response())
    agent.client.chat.completions.create.side_effect = responses

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch(
            "run_agent.handle_function_call",
            lambda name, args, task_id=None, **kwargs: json.dumps({"ok": True}),
        ),
    ):
        result = agent.run_conversation("do a lot of tool work")

    return result


class TestProactivePruneLoopWiring:
    def test_prune_consulted_when_compression_stands_down(self, agent):
        calls = []

        def _prune(messages, current_tokens=None):
            calls.append(current_tokens)
            return messages, 0  # no-op contract: input object back

        agent.context_compressor.prune_tool_results_only = _prune
        result = _run_tool_loop(agent, n_tool_iterations=3)
        assert result["completed"] is True
        assert len(calls) == 3  # one shot per tool iteration
        assert all(t == 120_000 for t in calls)  # fed the real usage reading

    def test_committed_prune_replaces_messages(self, agent):
        marker = "[old tool output pruned]"

        def _prune(messages, current_tokens=None):
            pruned = [dict(m) for m in messages]
            changed = 0
            for m in pruned:
                if m.get("role") == "tool" and m.get("content") != marker:
                    m["content"] = marker
                    changed += 1
            if not changed:
                return messages, 0
            return pruned, changed

        agent.context_compressor.prune_tool_results_only = _prune
        result = _run_tool_loop(agent, n_tool_iterations=2)
        assert result["completed"] is True
        tool_rows = [m for m in result["messages"] if m.get("role") == "tool"]
        assert tool_rows, "expected tool rows in the final transcript"
        assert all(m["content"] == marker for m in tool_rows)

    def test_noop_input_object_commits_nothing(self, agent):
        """Engine returns the INPUT object with a (bogus) non-zero count —
        the caller's ``result is not input`` gate must refuse the commit."""
        def _prune(messages, current_tokens=None):
            return messages, 5  # lies about count but returns input object

        agent.context_compressor.prune_tool_results_only = _prune
        result = _run_tool_loop(agent, n_tool_iterations=2)
        assert result["completed"] is True
        tool_rows = [m for m in result["messages"] if m.get("role") == "tool"]
        # tool output may be wrapped in an untrusted_tool_result envelope —
        # assert the original payload survived un-pruned.
        assert all('"ok": true' in m["content"] for m in tool_rows)

    def test_engine_without_method_does_not_raise(self, agent):
        """Plugin engines predating the hook / minimal doubles lack the
        method entirely — the getattr guard treats absence as a no-op."""
        compressor = SimpleNamespace(
            protect_first_n=3,
            protect_last_n=20,
            threshold_tokens=500_000,
            context_length=1_000_000,
            last_prompt_tokens=120_000,
            should_compress=lambda _t: False,
            should_defer_preflight_to_real_usage=lambda _t: True,
            get_active_compression_failure_cooldown=lambda: None,
        )
        agent.context_compressor = compressor
        result = _run_tool_loop(agent, n_tool_iterations=2)
        assert result["completed"] is True

    def test_raising_prune_is_swallowed(self, agent):
        def _prune(messages, current_tokens=None):
            raise RuntimeError("boom")

        agent.context_compressor.prune_tool_results_only = _prune
        result = _run_tool_loop(agent, n_tool_iterations=2)
        assert result["completed"] is True
        tool_rows = [m for m in result["messages"] if m.get("role") == "tool"]
        # tool output may be wrapped in an untrusted_tool_result envelope —
        # assert the original payload survived un-pruned.
        assert all('"ok": true' in m["content"] for m in tool_rows)
