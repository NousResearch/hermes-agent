"""Truncated reasoning-only promotion must not be delivered as the final answer.

Regression for 2026-10-06: the digest cron emailed the user exactly 36 chars —
"The fix was already applied on 2026-" — because the reasoning-only clean-stop promotion
(agent/turn_final_response.py) had no shape check. The reasoning field was itself cut off
mid-date, so a fragment became the "completed" result of an unattended job.

The pre-existing guard ``looks_like_degenerate_final`` does not cover this case: it is
tuned for wrong-script / leading-punctuation collapses, deliberately does NOT match
all-ASCII alphanumeric text (so "42"/"SQLite" stay answers), and is gated on
``intent_ack_continuation_mode() != "off"`` — "off" on chat_completions, i.e. inert on
this job's transport. These tests pin the ORTHOGONAL truncation predicate and the
promotion gate.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.agent_runtime_helpers import looks_like_degenerate_final
from agent.reasoning_promotion import (
    looks_like_truncated_promoted_reasoning,
    should_promote_reasoning_to_final,
)

DELIVERED_2026_10_06 = "The fix was already applied on 2026-"


class TestTruncationPredicate:
    def test_the_delivered_fragment_is_flagged(self):
        assert looks_like_truncated_promoted_reasoning(DELIVERED_2026_10_06) is True

    def test_the_old_predicate_does_not_cover_it(self):
        """Documents WHY a second predicate exists rather than reusing the first."""
        assert looks_like_degenerate_final(DELIVERED_2026_10_06) is False

    @pytest.mark.parametrize("text", [
        "see issue #",                  # dangling separator
        "committed to path/to/",        # dangling path separator
        "the options are (a",           # unclosed opener
        "fixed in commit abc123,",      # trailing comma
        "and then the",                 # trailing article
        "first read the log and",       # trailing coordinator
        'The error was "connection refus',           # open double quote
        "Here is the fix:\n```python\nprint(1)",     # unclosed ``` fence
    ])
    def test_truncated_shapes_are_flagged(self, text):
        assert looks_like_truncated_promoted_reasoning(text) is True

    @pytest.mark.parametrize("text", [
        "Done.",
        "42",
        "SQLite",
        "[SILENT]",
        "NO_REPLY",
        "Nothing requires action.",
        "The fix was already applied on 2026-10-05 by commit 1003648.",
        "Closed as duplicate of #94041.",
        # Strandable prepositions/copulas legitimately END English sentences; declining any of
        # these would re-ask for an answer that was already whole.
        "Here's what the flag is for",
        "The service is back on",
        "That's the directory it's in",
        "That's the file I was writing to",
        "It is what it is",
        # A quote closing a measurement is not an opening quote (odd raw count, still complete).
        'A 27" monitor is fine',
        # A whole URL that ends in a directory slash is not a path cut mid-token.
        "See the release notes at https://example.com/docs/",
        # A balanced pair of fences is a whole code block.
        "Fixed:\n```python\nprint(1)\n```",
    ])
    def test_complete_answers_are_never_flagged(self, text):
        assert looks_like_truncated_promoted_reasoning(text) is False

    def test_empty_is_not_a_truncation_of_anything(self):
        assert looks_like_truncated_promoted_reasoning("") is False
        assert looks_like_truncated_promoted_reasoning("   ") is False


class TestPromotionGate:
    def test_truncated_fragment_is_not_promoted(self):
        """The regression: this was delivered to the user's email."""
        assert should_promote_reasoning_to_final(DELIVERED_2026_10_06) is False

    def test_two_whole_reasoning_answers_still_promote(self):
        """The positive arm of the predicate: whole synthetic answers are still delivered.

        NOT the 38254- and 9890-char production samples — those are not fixtures in this tree
        (they never were; see the PR body). These two short strings only pin that a complete,
        terminal-punctuation-free sentence is not treated as a truncation.
        """
        assert should_promote_reasoning_to_final("The user asked me to diagnose the cron "
                                                 "failure. I read the job record and found "
                                                 "the script defaults to the wrong path.") is True
        assert should_promote_reasoning_to_final("Failing cron jobs: none. All jobs show a "
                                                 "clean last run.") is True

    def test_empty_reasoning_is_not_promoted(self):
        assert should_promote_reasoning_to_final("") is False


# ── the regression end to end ───────────────────────────────────────────────────────────────
# The unit tests above pin the predicate; this one pins the DELIVERED OUTCOME through the real
# turn loop. Same scenario as the 2026-10-06 cron failure: a reasoning-only clean stop whose
# reasoning was cut mid-date. Red on upstream/main (the fragment IS the final_response, one API
# call); green here, because declining the promotion hands the turn to the empty-response ladder,
# which recovers a real answer.

FRAGMENT = "The fix was already applied on 2026-"
RECOVERED = "The fix was already applied on 2026-10-05 by commit 1003648."


@pytest.fixture()
def loop_agent():
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="http://127.0.0.1:8000/v1",
            model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
            provider="vllm",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False
        return agent


def test_truncated_reasoning_only_stop_is_not_delivered_as_the_answer(loop_agent):
    """This exact 36-char fragment was emailed as the completed result of an unattended job."""
    from tests.agent.test_run_agent import _mock_response

    loop_agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="stop", reasoning_content=FRAGMENT),
        _mock_response(content=RECOVERED, finish_reason="stop"),
    ]
    with (
        patch.object(loop_agent, "_persist_session"),
        patch.object(loop_agent, "_save_trajectory"),
        patch.object(loop_agent, "_cleanup_task_resources"),
    ):
        result = loop_agent.run_conversation("what did the last run report?")

    # The fragment is never the answer; the ladder's second call supplies the real one.
    assert result["final_response"] == RECOVERED
    assert result["api_calls"] == 2


def test_complete_reasoning_only_stop_still_promotes_on_the_first_call(loop_agent):
    """The positive arm: a WHOLE reasoning field is still the answer, with no extra API call.

    Without this arm the gate could satisfy the regression by declining every promotion, which
    would silently delete the feature the promotion exists for (the 38254- and 9890-char
    whole-answer-as-reasoning cases).
    """
    from tests.agent.test_run_agent import _mock_response

    whole = "The last run reported success. All jobs show a clean last run."
    loop_agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="stop", reasoning_content=whole),
    ]
    with (
        patch.object(loop_agent, "_persist_session"),
        patch.object(loop_agent, "_save_trajectory"),
        patch.object(loop_agent, "_cleanup_task_resources"),
    ):
        result = loop_agent.run_conversation("what did the last run report?")

    assert result["final_response"] == whole
    assert result["api_calls"] == 1
