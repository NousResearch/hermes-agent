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
    compressor.awaiting_real_usage_after_compression = False
    compressor.should_compress.return_value = False
    compressor.should_compress_info.return_value = (False, None)
    compressor.should_defer_preflight_to_real_usage.return_value = True
    compressor.get_active_compression_failure_cooldown.return_value = None
    return compressor


@pytest.fixture()
def agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
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


def _run_tool_loop(agent, n_tool_iterations: int, task_id=None):
    responses = [_tool_response(i) for i in range(n_tool_iterations)]
    responses.append(_stop_response())
    agent.client.chat.completions.create.side_effect = responses

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch(
            "model_tools.handle_function_call",
            lambda name, args, task_id=None, **kwargs: json.dumps({"ok": True}),
        ),
    ):
        result = agent.run_conversation("do a lot of tool work", task_id=task_id)

    return result


class TestProactivePruneLoopWiring:
    def test_pending_checkpoint_waits_without_warning_or_pruning(self, agent):
        from agent.turn_preflight import compress_after_tool_results

        compressor = agent.context_compressor
        compressor.awaiting_real_usage_after_compression = True
        compressor.last_prompt_tokens = 255_933
        compressor.threshold_tokens = 231_200
        compressor.should_compress.return_value = True
        compressor.should_compress_info.return_value = (True, None)
        compressor.prune_tool_results_only.side_effect = lambda messages, **kw: (messages, 0)
        messages = [{"role": "user", "content": "continue"}]
        with patch.object(agent, "_warn_context_overflow_blocked") as warn:
            verdict = compress_after_tool_results(
                agent, messages=messages, system_message="system", user_message="continue",
                active_system_prompt="system", conversation_history=[],
                compression_attempts=0, max_compression_attempts=3,
                effective_task_id=None, final_response="", turn_exit_reason=None,
                current_turn_user_idx=0,
            )
        assert verdict.messages is messages
        assert not verdict.end_turn
        warn.assert_not_called()
        compressor.prune_tool_results_only.assert_not_called()

    def test_full_compression_preempts_proactive_prune(self, agent):
        agent.context_compressor.should_compress.return_value = True

        def _compress(messages, system_message, **_kwargs):
            return [dict(m) for m in messages], system_message

        with (
            patch.object(agent, "_compress_context", side_effect=_compress) as compress,
            patch(
                "agent.conversation_compression.conversation_history_after_compression",
                return_value=[],
            ),
        ):
            result = _run_tool_loop(agent, n_tool_iterations=1)

        assert result["completed"] is True
        compress.assert_called_once()
        agent.context_compressor.prune_tool_results_only.assert_not_called()

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

    def test_should_compress_true_but_skipped_is_warned(self, agent):
        """``should_compress_info`` says RUN (``(True, None)``) yet this branch
        was taken — the per-turn compression budget is spent. Over threshold
        with no reclamation running must not be swallowed silently (#101889).

        Faithful to the real engine: ``should_compress()`` is
        ``should_compress_info()[0]``, so the only way into this branch with
        ``(True, None)`` is an exhausted per-turn budget."""
        agent.max_compression_attempts = 0  # budget already spent this turn
        agent.context_compressor.should_compress.return_value = True
        agent.context_compressor.should_compress_info.return_value = (True, None)
        agent.context_compressor.prune_tool_results_only = (
            lambda messages, current_tokens=None: (messages, 0)
        )
        warned = []
        with patch.object(
            agent,
            "_warn_context_overflow_blocked",
            side_effect=lambda reason, tokens, threshold: warned.append(reason),
        ):
            result = _run_tool_loop(agent, n_tool_iterations=1)

        assert result["completed"] is True
        assert warned, "over-threshold turn with no compaction ran silently"
        assert all(r.startswith("attempts_exhausted") for r in warned)

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


class TestCommittedPruneIsDedupBoundary:
    """A committed proactive prune demotes old skill_view / read_file bodies to one-line markers
    without a compaction boundary; the repeat-read dedup must stop answering "unchanged" for them
    or the reload the marker asks for is refused (#112763)."""

    @staticmethod
    def _seed_dedup(tmp_path, task_id):
        from tools.file_tools_read_tracking import _read_tracker, _read_tracker_lock, _task_data
        from tools.skills_tool_dedup import _record_skill_view, reset_skill_view_dedup

        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("# s\n", encoding="utf-8")
        reset_skill_view_dedup(task_id)
        _record_skill_view(task_id, "bigskill", None, {"name": "bigskill", "_source_path": str(skill_md)})
        with _read_tracker_lock:
            _read_tracker.pop(task_id, None)
            td = _task_data(task_id)
            td["dedup"][("/x/big.txt", 1, 2000)] = 1.0
            td["dedup_generation_reads"].add(("/x/big.txt", 1, 2000))
        return skill_md

    @staticmethod
    def _dedup_state(task_id):
        from tools.file_tools_read_tracking import _read_tracker
        from tools.skills_tool_dedup import _check_skill_view_dedup

        skill_stubbed = _check_skill_view_dedup(task_id, "bigskill", None) is not None
        file_in_generation = ("/x/big.txt", 1, 2000) in _read_tracker[task_id]["dedup_generation_reads"]
        return skill_stubbed, file_in_generation

    def test_committed_prune_releases_skill_and_file_dedup(self, agent, tmp_path):
        task_id = "prune-boundary-task"
        self._seed_dedup(tmp_path, task_id)
        assert self._dedup_state(task_id) == (True, True)

        def _prune(messages, current_tokens=None):
            pruned = [dict(m) for m in messages]
            changed = 0
            for m in pruned:
                if m.get("role") == "tool" and m.get("content") != "[pruned]":
                    m["content"] = "[pruned]"
                    changed += 1
            return (pruned, changed) if changed else (messages, 0)

        agent.context_compressor.prune_tool_results_only = _prune
        assert _run_tool_loop(agent, n_tool_iterations=1, task_id=task_id)["completed"] is True
        # Skill: next view serves full content again. File: the generation-read set is cleared so the
        # first unchanged re-read serves content; the mtime map itself is preserved (later reads stub).
        assert self._dedup_state(task_id) == (False, False)

    def test_noop_prune_keeps_dedup(self, agent, tmp_path):
        task_id = "prune-noop-task"
        self._seed_dedup(tmp_path, task_id)
        agent.context_compressor.prune_tool_results_only = lambda messages, current_tokens=None: (messages, 0)
        assert _run_tool_loop(agent, n_tool_iterations=1, task_id=task_id)["completed"] is True
        assert self._dedup_state(task_id) == (True, True)
