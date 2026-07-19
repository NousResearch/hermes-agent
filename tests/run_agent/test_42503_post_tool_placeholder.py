"""Tests for #42503 — progress placeholder after tool calls mistaken for a final answer.

The failure mode: after executing tool calls, the model replies with a short
progress note ("Working on it...", "I'll now update the file…") and no tool
calls.  The conversation loop treats any non-empty no-tool-call response as
the final answer, so the turn ends silently with the task unfinished.

The fix nudges the model to continue (capped at 2 per tool round) and, if the
nudges are exhausted, appends a visible "Incomplete turn" explanation instead
of ending the turn silently.
"""

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Fixtures / helpers (mirrors tests/run_agent/test_run_agent.py)
# ---------------------------------------------------------------------------


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


@pytest.fixture()
def agent():
    """Minimal AIAgent with mocked OpenAI client and tool loading."""
    with (
        patch(
            "run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = False
        a.save_trajectories = False
        return a


def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _run(agent, prompt="do the task"):
    with (
        patch("run_agent.handle_function_call", return_value="tool result"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(prompt)


# ---------------------------------------------------------------------------
# Detector unit tests
# ---------------------------------------------------------------------------


class TestPlaceholderDetector:
    """The detector decides whether a no-tool-call reply is a progress
    placeholder.  A false negative silently ends the turn with the task
    unfinished (the #42503 bug); a false positive costs one nudge round-trip.
    """

    def _detect(self, agent, text):
        from agent.agent_runtime_helpers import (
            looks_like_post_tool_progress_placeholder,
        )
        return looks_like_post_tool_progress_placeholder(agent, text)

    @pytest.mark.parametrize(
        "text",
        [
            # Explicit progress phrases — the exact strings reported in
            # #42503 as being mistaken for final answers.
            "Working on it...",
            "Still working on the report.",
            "Please wait, processing the files.",
            "Hang tight!",
            "One moment.",
            "I'll continue with the next step.",
            # Future commitment with an open ending: announces work but
            # delivers nothing.
            "I'll now analyze the search results...",
            "Let me update the config file:",
            "Now I'm going to check the logs…",
        ],
    )
    def test_progress_placeholders_detected(self, agent, text):
        assert self._detect(agent, text) is True

    @pytest.mark.parametrize(
        "text",
        [
            # Real short answers must end the turn normally — nudging a
            # legitimate "Done." would loop the model for no reason.
            "Done.",
            "The file has been updated and all tests pass.",
            "The search returned 3 results: A, B and C.",
            # Questions are legitimate stops: the model needs user input.
            "Should I also update the README?",
            # A future-tense sentence that terminates normally is a real
            # answer ("I'll be available if you need more help.").
            "I'll be here if you need anything else.",
            # Empty content is handled by the empty-response path, not here.
            "",
            "<think>internal reasoning only</think>",
        ],
    )
    def test_real_answers_not_detected(self, agent, text):
        assert self._detect(agent, text) is False

    def test_long_responses_never_detected(self, agent):
        # Length guard: a substantive multi-sentence reply that happens to
        # contain "working on it" is an answer, not a placeholder.
        text = (
            "I finished working on it. Here is the summary of everything "
            "that was changed: the config loader now validates the schema "
            "before use, the retry loop caps at three attempts with jittered "
            "backoff, and the test suite covers both the success and failure "
            "paths end to end. All 42 tests pass."
        )
        assert len(text) > 240  # the guard under test is the length cap
        assert self._detect(agent, text) is False


# ---------------------------------------------------------------------------
# Conversation-loop integration tests
# ---------------------------------------------------------------------------


class TestPostToolPlaceholderNudge:
    def test_placeholder_after_tools_is_nudged_to_completion(self, agent):
        """A placeholder after tool calls must NOT end the turn — the model
        gets nudged and its real answer becomes the final response (#42503)."""
        tc = _mock_tool_call(call_id="c1")
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc]),
            _mock_response(content="Working on it..."),
            _mock_response(content="Task finished: 3 files were updated."),
        ]
        result = _run(agent)
        assert result["completed"] is True
        assert result["final_response"] == "Task finished: 3 files were updated."
        assert result["api_calls"] == 3  # tools + placeholder + nudged answer

    def test_nudge_scaffolding_not_persisted(self, agent):
        """The synthetic placeholder/nudge pair exists only to drive the next
        API call — persisting it would replay fake turns on session resume."""
        tc = _mock_tool_call(call_id="c1")
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc]),
            _mock_response(content="Working on it..."),
            _mock_response(content="All done."),
        ]
        result = _run(agent)
        assert all(
            not m.get("_post_tool_placeholder_synthetic")
            for m in result["messages"]
            if isinstance(m, dict)
        )
        # No consecutive assistant messages left behind by the popped nudge.
        roles = [m.get("role") for m in result["messages"] if isinstance(m, dict)]
        assert not any(
            a == b == "assistant" for a, b in zip(roles, roles[1:])
        )

    def test_tool_call_after_nudge_leaves_no_buried_scaffolding(self, agent):
        """When the nudge succeeds via ANOTHER tool call, the synthetic pair
        would be buried under the new tool turns — the terminal pop only
        strips a trailing suffix, so without the tool-call-path pop the fake
        placeholder/nudge rows stay in the live context forever and leak
        into everything assembled from the message list (PR #57610 review)."""
        tc1 = _mock_tool_call(call_id="c1")
        tc2 = _mock_tool_call(call_id="c2")
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc1]),
            _mock_response(content="Working on it..."),
            _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc2]),
            _mock_response(content="Task finished: both lookups done."),
        ]
        result = _run(agent)
        assert result["completed"] is True
        assert result["final_response"] == "Task finished: both lookups done."
        # The scaffold pair must be gone even though it is no longer a
        # trailing suffix — this is the buried-scaffold regression.
        assert all(
            not m.get("_post_tool_placeholder_synthetic")
            for m in result["messages"]
            if isinstance(m, dict)
        )
        # The placeholder note itself must not survive as transcript context.
        assert all(
            "Working on it..." != m.get("content")
            for m in result["messages"]
            if isinstance(m, dict)
        )

    def test_interleaved_prefill_and_placeholder_scaffolds_all_popped(self, agent):
        """tool round → thinking-only (prefill row) → placeholder (nudge pair)
        → tool calls leaves [prefill, placeholder, nudge] stacked.  The
        tool-call-path cleanup must pop across BOTH flags in one loop —
        sequential single-flag loops strand the prefill row beneath the
        popped nudge pair, leaving a fake assistant turn (and consecutive
        assistant messages) in the live context forever."""
        agent.client.chat.completions.create.side_effect = [
            _mock_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_mock_tool_call(call_id="c1")],
            ),
            # In-content <think> routes to the prefill recovery, not the
            # empty-response nudge (see _has_inline_thinking in the loop).
            _mock_response(content="<think>planning the next step</think>"),
            _mock_response(content="Working on it..."),
            _mock_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_mock_tool_call(call_id="c2")],
            ),
            _mock_response(content="Done: everything checked."),
        ]
        result = _run(agent)
        assert result["completed"] is True
        assert result["final_response"] == "Done: everything checked."
        leftovers = [
            m
            for m in result["messages"]
            if isinstance(m, dict)
            and (
                m.get("_thinking_prefill")
                or m.get("_post_tool_placeholder_synthetic")
            )
        ]
        assert leftovers == []
        # No consecutive assistant rows left by a stranded scaffold.
        roles = [m.get("role") for m in result["messages"] if isinstance(m, dict)]
        assert not any(a == b == "assistant" for a, b in zip(roles, roles[1:]))

    def test_trajectory_conversion_drops_buried_scaffolding(self, agent):
        """Trajectory saving converts the raw message list; a scaffold pair
        buried mid-list (nudge followed by more tool calls) must be filtered
        there too, or save_trajectories=True records the synthetic turns as
        if the model had really said them (PR #57610 review)."""
        nudge_text = (
            "[System: Your previous message was a progress update, not a "
            "final answer. Continue the task now.]"
        )
        messages = [
            {"role": "user", "content": "do the task"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "web_search", "content": "r1"},
            # Buried scaffold pair: nudge fired, then the model continued
            # with another tool call, so these rows are interior.
            {
                "role": "assistant",
                "content": "Working on it...",
                "_post_tool_placeholder_synthetic": True,
            },
            {
                "role": "user",
                "content": nudge_text,
                "_post_tool_placeholder_synthetic": True,
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c2", "name": "web_search", "content": "r2"},
            {"role": "assistant", "content": "All done."},
        ]
        trajectory = agent._convert_to_trajectory_format(
            messages, "do the task", completed=True
        )
        values = [t["value"] for t in trajectory]
        assert not any(nudge_text in v for v in values)
        assert not any("Working on it..." in v for v in values)
        # The real turns survive the sweep.
        assert any("All done." in v for v in values)

    def test_exhausted_nudges_surface_incomplete_turn_explanation(self, agent):
        """If the model keeps emitting placeholders after 2 nudges, the turn
        must end VISIBLY incomplete — never silently (#42503's core complaint:
        no error, no hint, task just stops)."""
        tc = _mock_tool_call(call_id="c1")
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc]),
            _mock_response(content="Working on it..."),
            _mock_response(content="Still working on it..."),
            _mock_response(content="Working on it..."),
        ]
        result = _run(agent)
        assert result["turn_exit_reason"] == "post_tool_placeholder_response"
        # The placeholder text is kept AND the explanation is appended.
        assert "Working on it..." in result["final_response"]
        assert "Incomplete turn" in result["final_response"]
        assert result["api_calls"] == 4  # tools + placeholder + 2 nudged retries

    def test_placeholder_without_tool_round_ends_normally(self, agent):
        """A progress-style sentence in a purely conversational turn (no tool
        calls ran) is a legitimate answer — e.g. replying to "how's the
        report going?" — and must not trigger the nudge."""
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="Still working on the report."),
        ]
        result = _run(agent, prompt="how's the report going?")
        assert result["completed"] is True
        assert result["final_response"] == "Still working on the report."
        assert result["api_calls"] == 1
        assert str(result["turn_exit_reason"]).startswith("text_response")

    def test_real_answer_after_tools_ends_normally(self, agent):
        """Regression guard: a genuine short answer after tool calls must not
        be mistaken for a placeholder — no extra API calls, no footer."""
        tc = _mock_tool_call(call_id="c1")
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc]),
            _mock_response(content="The search returned no matches."),
        ]
        result = _run(agent)
        assert result["final_response"] == "The search returned no matches."
        assert result["api_calls"] == 2
        assert str(result["turn_exit_reason"]).startswith("text_response")
