"""Regression tests for degenerate-final recovery.

Provider-side collapse: a turn executes all of its tool work correctly, then ends on
``finish_reason=stop`` with an ENTIRE visible answer that is a fragment — a stray token, a
wrong-script word, or a truncated non-sentence. Before the fix the loop accepted the fragment as
the answer, the turn reported ``completed``, and an unattended job silently abandoned the task.

The fix keys on the collapse itself (a short fragment, no tool calls, after real tool work in the
same turn), re-prompts ONCE, and keeps the fragment as the fallback so a second collapse still
ends the turn with the previous behaviour. Scoping lives in ``_degenerate_final_guard_mode`` and
is exercised separately; the loop tests set the explicit override so they do not depend on which
transport the mocked client talks.

Imports of the new module symbol are LAZY (``_tfr()``) so this file still imports against pre-fix
code: on base the loop tests must fail on their own assertions, not at collection.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _tfr():
    """The module under test, imported at call time (see the module docstring)."""
    from agent import turn_final_response
    return turn_final_response


@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client (mirrors test_dropped_tool_call_recovery)."""
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        # Explicit on: the loop tests exercise the guard mechanism, not the auto scope.
        agent._degenerate_final_guard = True
        return agent


def _run(agent, stages):
    from tests.agent.test_run_agent import _mock_response  # noqa: F401  (side-effect free import check)
    agent.client.chat.completions.create.side_effect = stages
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation("build the workbook")


def _tool_round(call_id):
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call
    return _mock_response(
        content="",
        finish_reason="tool_calls",
        tool_calls=[_mock_tool_call(name="web_search", arguments="{}", call_id=call_id)],
    )


def _last_user_content(agent, call_index=-1):
    call = agent.client.chat.completions.create.call_args_list[call_index]
    msgs = call.kwargs.get("messages") or call.args[0].get("messages")
    return next((m.get("content") or "" for m in reversed(msgs) if m.get("role") == "user"), "")


class TestDegenerateFinalPredicate:
    """The predicate must catch every reported collapse and no ordinary answer."""

    @pytest.mark.parametrize("fragment", [
        "пар", "парень", "паршиво", "пароля", "парсер", "измерение",          # Cyrillic
        "éclair", "ếch", "өдөр", "たくさん", "غذایی", "剧情", "？（warming",  # other scripts
        "Geschmacklose", "impl sheduling", "orme_skill_view paganize",        # ASCII corruption
    ])
    def test_reported_collapse_fragments_match(self, fragment):
        assert _tfr()._looks_like_degenerate_final(fragment) is True

    @pytest.mark.parametrize("answer", [
        "Done.", "Fixed.", "Yes", "Ok", "Approved", "Noted",
        "All checks pass. Approved.",
        "The workbook is built and verified; 26 live formulas, worst deviation 1.8e-15.",
        "Protection is gone from both sheets. Confirming the logo survived.",
        "",
        "   ",
    ])
    def test_ordinary_answers_do_not_match(self, answer):
        assert _tfr()._looks_like_degenerate_final(answer) is False

    def test_allowlist_entries_never_match(self):
        for entry in _tfr()._DEGENERATE_FINAL_ALLOWLIST:
            assert _tfr()._looks_like_degenerate_final(entry) is False
            assert _tfr()._looks_like_degenerate_final(entry.capitalize()) is False

    def test_long_fragment_above_ceiling_is_a_real_answer(self):
        assert _tfr()._looks_like_degenerate_final("A" * 25) is False


class TestToolResultCounter:
    def test_counts_only_after_the_last_user_row(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "{}"},
            {"role": "user", "content": "again"},
            {"role": "tool", "tool_call_id": "c2", "content": "{}"},
            {"role": "tool", "tool_call_id": "c3", "content": "{}"},
        ]
        assert _tfr()._tool_results_since_last_user(messages) == 2

    def test_chat_only_turn_has_none(self):
        assert _tfr()._tool_results_since_last_user([{"role": "user", "content": "hi"}]) == 0


class TestGuardScope:
    def _agent(self, **kw):
        base = {"model": "deepseek-flash", "api_mode": "chat_completions"}
        base.update(kw)
        return SimpleNamespace(**base)

    def test_auto_on_for_responses_wire(self):
        assert _tfr()._degenerate_final_guard_mode(self._agent(api_mode="codex_responses")) == "all"

    def test_auto_on_for_reported_model_family(self):
        assert _tfr()._degenerate_final_guard_mode(
            self._agent(model="muse-spark-1.3-contributor")
        ) == "all"

    def test_auto_off_for_unrelated_chat_model(self):
        assert _tfr()._degenerate_final_guard_mode(self._agent()) == "off"

    def test_explicit_override_wins(self):
        assert _tfr()._degenerate_final_guard_mode(
            self._agent(_degenerate_final_guard=False)
        ) == "off"
        assert _tfr()._degenerate_final_guard_mode(
            self._agent(api_mode="codex_responses", _degenerate_final_guard=False)
        ) == "off"
        assert _tfr()._degenerate_final_guard_mode(
            self._agent(_degenerate_final_guard=True)
        ) == "all"
        assert _tfr()._degenerate_final_guard_mode(
            self._agent(_degenerate_final_guard=["muse-spark"])
        ) == "off"
        assert _tfr()._degenerate_final_guard_mode(
            self._agent(model="muse-spark-1.3-contributor", _degenerate_final_guard=["muse-spark"])
        ) == "all"


class TestDegenerateFinalRecovery:
    def test_fragment_after_tool_work_reprompts_instead_of_exiting(self, loop_agent):
        from tests.agent.test_run_agent import _mock_response

        result = _run(loop_agent, [
            _tool_round("c1"),
            _tool_round("c2"),
            _mock_response(content="пар", finish_reason="stop"),
            _mock_response(content="The workbook is built and verified.", finish_reason="stop"),
        ])

        assert loop_agent.client.chat.completions.create.call_count == 4, (
            "A degenerate fragment after tool work must trigger one re-prompt (fourth call), "
            "not exit the loop on the fragment."
        )
        assert "built and verified" in result["final_response"]
        assert "fragment" in _last_user_content(loop_agent).lower(), (
            "The nudge must tell the model its answer collapsed and to finish the task."
        )

    def test_second_fragment_is_bounded_and_kept(self, loop_agent):
        from tests.agent.test_run_agent import _mock_response

        result = _run(loop_agent, [
            _tool_round("c1"),
            _tool_round("c2"),
            _mock_response(content="пар", finish_reason="stop"),
            _mock_response(content="пар", finish_reason="stop"),
        ])

        assert loop_agent.client.chat.completions.create.call_count == 4, (
            "The re-prompt must be bounded to one per turn, not loop on a collapsing provider."
        )
        assert result["final_response"].strip() == "пар", (
            "After the bound is spent the fragment is the answer, matching prior behaviour."
        )

    def test_terse_real_answer_after_tool_work_is_unaffected(self, loop_agent):
        from tests.agent.test_run_agent import _mock_response

        result = _run(loop_agent, [
            _tool_round("c1"),
            _tool_round("c2"),
            _mock_response(content="All checks pass. Approved.", finish_reason="stop"),
        ])

        assert loop_agent.client.chat.completions.create.call_count == 3
        assert "Approved" in result["final_response"]

    def test_chat_only_short_answer_is_unaffected(self, loop_agent):
        from tests.agent.test_run_agent import _mock_response

        result = _run(loop_agent, [_mock_response(content="Yes", finish_reason="stop")])

        assert loop_agent.client.chat.completions.create.call_count == 1, (
            "A tool-free turn may answer in one word; only a mid-task collapse is re-prompted."
        )
        assert result["final_response"].strip() == "Yes"

    def test_recovery_disabled_when_stall_guards_off(self, loop_agent):
        from tests.agent.test_run_agent import _mock_response

        loop_agent._stall_guards = False
        result = _run(loop_agent, [
            _tool_round("c1"),
            _tool_round("c2"),
            _mock_response(content="пар", finish_reason="stop"),
        ])

        assert loop_agent.client.chat.completions.create.call_count == 3
        assert result["final_response"].strip() == "пар"

    def test_nudge_pair_is_ephemeral_scaffolding(self, loop_agent):
        from agent.session_persistence import _is_ephemeral_scaffolding
        from tests.agent.test_run_agent import _mock_response

        result = _run(loop_agent, [
            _tool_round("c1"),
            _tool_round("c2"),
            _mock_response(content="пар", finish_reason="stop"),
            _mock_response(content="The workbook is built and verified.", finish_reason="stop"),
        ])

        assert result["completed"] is True
        leftover = [
            m for m in result["messages"]
            if isinstance(m, dict) and m.get("_degenerate_final_nudge")
        ]
        assert not leftover, (
            "The re-prompt pair must be stripped at finalization, not kept in the transcript."
        )
        assert _is_ephemeral_scaffolding(
            {"role": "user", "content": "nudge", "_degenerate_final_nudge": True}
        ), "The flag must classify as ephemeral so a mid-turn flush cannot persist it."
