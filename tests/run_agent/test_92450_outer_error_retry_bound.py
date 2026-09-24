"""Tests for issue #92450 — outer-loop error retries must be bounded even
when the turn budget (``max_iterations``) is unlimited.

Before the fix, the outer ``except`` in ``run_conversation`` only left the
loop on a local-processing error (#66267) or when
``api_call_count >= agent.max_iterations - 1``. With the default budget now
unlimited (``sys.maxsize``), a permanent failure that escaped the inner
retry/fallback machinery spun forever (~64 retries/s measured in the issue),
pegged a core, and overwrote days of rotated agent.log history within
minutes.

Injection points reflect the real escape path: exceptions raised INSIDE the
inner retry loop (transport errors from ``create()``, normalization,
fallbacks, ...) never reach the outer handler — that machinery recovers or
terminates on its own. The outer handler sees failures from the final
response assembly, e.g. ``_build_assistant_message`` (a chat-completion-
helpers callable, so such crashes classify as RETRYABLE, never local —
exactly the spin shape reported in the issue).

The fix adds a small per-turn cap on total outer-loop exceptions
(``_MAX_OUTER_LOOP_ERRORS``, scaled down by a tiny explicit
``max_iterations``). These tests pin the behavior contract:

* repeated escaping errors must terminate the turn within the cap;
* a turn that recovers after early escaping failures must complete normally;
* local-processing errors still exit immediately (#66267 unchanged);
* a finite ``max_iterations`` still exhausts via the original near-limit path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client (mirrors test_run_agent's fixture)
    so we can stage responses on ``.chat.completions.create``."""
    from run_agent import AIAgent
    from tests.run_agent.test_run_agent import _mock_response

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
        # Every API call itself SUCCEEDS — #92450's spin happens after the
        # response arrives, when final-response assembly keeps crashing.
        agent.client.chat.completions.create.side_effect = (
            lambda *a, **k: _mock_response(content="ok", finish_reason="stop")
        )
        return agent


class _PermanentError(Exception):
    """An error type no recovery path classifies as retryable-local."""


def _make_local_frame_raiser():
    """Compile a raiser whose frame filename lives in a local-processing
    module, so the production traceback classifier (#66267) sees it as a
    deterministic local bug — without mocking away the classifier itself."""
    namespace = {}
    code = compile(
        "def _raise(exc):\n    raise exc\n",
        filename="/agent/agent_runtime_helpers.py",
        mode="exec",
    )
    exec(code, namespace)
    return namespace["_raise"]


class TestOuterErrorRetryBound:
    def test_repeated_escaping_errors_are_bounded(self, loop_agent):
        """A permanent final-assembly failure that escapes every retry layer
        must end the turn within the per-turn error cap instead of spinning
        forever under the unlimited default budget."""
        boom = _PermanentError("permanent assembly contract violation")

        with (
            patch.object(
                loop_agent, "_build_assistant_message", side_effect=boom
            ),
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("hello")

        # The bound path returns an apology as final_response with
        # failed=True (the turn did not complete successfully), so the
        # meaningful assertions are: bounded call count + dedicated exit
        # reason + apology text + failed flag.
        assert result["api_calls"] <= 8, (
            "The loop must give up within the per-turn error bound instead "
            "of retrying an unlimited-budget failure forever."
        )
        assert result["turn_exit_reason"].startswith("repeated_outer_errors"), (
            f"unexpected exit reason: {result['turn_exit_reason']}"
        )
        assert "repeated errors" in (result["final_response"] or "")
        assert result["failed"] is True
        assert result["completed"] is False

    def test_recovery_after_failures_completes_normally(self, loop_agent):
        """Early escaping failures followed by success must NOT trip the new
        bound — the turn completes and the reply is delivered."""
        good_msg = {
            "role": "assistant",
            "content": "Recovered fine.",
            "finish_reason": "stop",
        }

        with (
            patch.object(
                loop_agent,
                "_build_assistant_message",
                side_effect=[
                    _PermanentError("transient assembly hiccup"),
                    _PermanentError("transient assembly hiccup"),
                    dict(good_msg),
                ],
            ),
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("hello")

        # final_response comes from the raw provider content ("ok"), so the
        # recovery contract is: clean text_response exit exactly on the 3rd
        # call, marked completed — proving the two earlier escapes did not
        # trip the new bound.
        assert result["completed"] is True, (
            f"turn should recover: exit={result.get('turn_exit_reason')}"
        )
        assert result["failed"] is False
        assert result["api_calls"] == 3
        assert result["turn_exit_reason"].startswith("text_response"), (
            f"unexpected exit reason: {result['turn_exit_reason']}"
        )

    def test_local_processing_error_still_exits_immediately(self, loop_agent):
        """#66267 regression guard: a deterministic local bug exits on first
        occurrence — the new counter must not delay or change that exit."""
        raiser = _make_local_frame_raiser()
        boom = TypeError("list content fed into a str regex helper")

        with (
            patch.object(
                loop_agent,
                "_strip_think_blocks",
                side_effect=lambda text: raiser(boom),
            ),
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("hello")

        assert result["turn_exit_reason"].startswith("local_processing_error"), (
            f"unexpected exit reason: {result['turn_exit_reason']}"
        )
        assert result["api_calls"] == 1, (
            "A local processing error must stop immediately on first "
            "occurrence, not consume retries."
        )

    def test_finite_budget_still_governs_near_limit_exit(self, loop_agent):
        """With a small explicit max_iterations the ORIGINAL near-limit path
        fires (unchanged reason string), not the new repeated-error path."""
        loop_agent.max_iterations = 3
        boom = _PermanentError("permanent assembly failure")

        with (
            patch.object(
                loop_agent, "_build_assistant_message", side_effect=boom
            ),
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("hello")

        # Budget of 3 → the legacy guard fires on error #2 (api_call_count 2
        # >= 3 - 1); the new cap would fire at error #3, so the legacy path
        # must win here.
        assert result["turn_exit_reason"].startswith(
            "error_near_max_iterations"
        ), f"legacy near-limit path must govern: {result['turn_exit_reason']}"
