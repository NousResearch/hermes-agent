"""Regression tests for the guarded cleanup chain in finalize_turn.

Issue #8049 — ``run_conversation`` silently drops the final_response when
``_save_trajectory``, ``_cleanup_task_resources``, or ``_persist_session``
raises (disk-full / dead Docker socket / locked SQLite).

Each of the three core cleanup steps must be wrapped in its own try/except so:
  1. One failure cannot skip the remaining steps.
  2. ``final_response`` is always present on the returned result dict.
  3. Failures are surfaced via the ``cleanup_errors`` key.
  4. Clean turns never have a ``cleanup_errors`` key.

Design note: ``finalize_turn`` lazily imports ``logger`` from
``agent.conversation_loop`` to avoid import cycles.  We stub that entire
import with a fake module so tests run without the full project installed.
"""

from __future__ import annotations

import logging
import sys
import types
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stub the heavy dependency chain before importing the module under test.
#
# ``agent/turn_finalizer.py`` imports ``_summarize_user_message_for_log``
# from ``agent.codex_responses_adapter`` at module-level, which in turn
# pulls in ``agent.prompt_builder`` → ``hermes_constants`` (Python ≥3.10
# union syntax) etc.  We short-circuit the whole chain with thin stubs so
# tests run with only stdlib + pytest.
# ---------------------------------------------------------------------------

def _install_stubs():
    """Insert fake modules into sys.modules before any import of the SUT."""
    # agent.codex_responses_adapter — we only need _summarize_user_message_for_log
    _cra = types.ModuleType("agent.codex_responses_adapter")
    _cra._summarize_user_message_for_log = lambda msg: str(msg)
    sys.modules.setdefault("agent.codex_responses_adapter", _cra)

    # agent.conversation_loop — finalize_turn does a lazy import to grab logger
    _cl = types.ModuleType("agent.conversation_loop")
    _cl.logger = logging.getLogger("agent.conversation_loop")
    sys.modules.setdefault("agent.conversation_loop", _cl)

    # hermes_cli.plugins — invoked inside plugin hooks in finalize_turn
    _plugins = types.ModuleType("hermes_cli.plugins")
    _plugins.invoke_hook = lambda *_a, **_kw: iter([])
    sys.modules.setdefault("hermes_cli", types.ModuleType("hermes_cli"))
    sys.modules.setdefault("hermes_cli.plugins", _plugins)


_install_stubs()

# Now we can safely import the module under test.
from agent.turn_finalizer import finalize_turn  # noqa: E402  (after stubs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(
    *,
    save_raises: bool = False,
    cleanup_raises: bool = False,
    persist_raises: bool = False,
):
    """Return a minimal stub agent whose three cleanup methods optionally raise."""
    agent = MagicMock()

    # Attributes read by finalize_turn.
    agent.max_iterations = 10
    agent.quiet_mode = True
    agent.model = "test-model"
    agent.provider = "test-provider"
    agent.base_url = "http://localhost"
    agent.session_id = "sess-0"
    agent.session_input_tokens = 0
    agent.session_output_tokens = 0
    agent.session_cache_read_tokens = 0
    agent.session_cache_write_tokens = 0
    agent.session_reasoning_tokens = 0
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_estimated_cost_usd = 0.0
    agent.session_cost_status = "ok"
    agent.session_cost_source = "pricing"
    agent._tool_guardrail_halt_decision = None
    agent._skill_nudge_interval = 0
    agent._iters_since_skill = 0
    agent._interrupt_message = None
    agent._response_was_previewed = False
    agent.iteration_budget = MagicMock(remaining=5, used=5, max_total=10)
    agent.context_compressor = MagicMock(last_prompt_tokens=0)

    # File-mutation verifier — must return False/None so it doesn't corrupt final_response.
    agent._file_mutation_verifier_enabled.return_value = False
    agent._turn_failed_file_mutations = {}

    # Turn-completion explainer — must return False so it skips its branch.
    agent._turn_completion_explainer_enabled.return_value = False

    # Control which cleanup methods raise.
    if save_raises:
        agent._save_trajectory.side_effect = OSError("disk full")
    if cleanup_raises:
        agent._cleanup_task_resources.side_effect = RuntimeError("Docker dead")
    if persist_raises:
        agent._persist_session.side_effect = IOError("SQLite locked")

    # _drain_pending_steer must return a falsy value so the steer branch is skipped.
    agent._drain_pending_steer.return_value = None

    return agent


def _make_messages():
    return [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def _call_finalize(agent, final_response: str = "The answer is 42") -> dict:
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=3,
        interrupted=False,
        failed=False,
        messages=_make_messages(),
        conversation_history=[],
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="hello",
        original_user_message="hello",
        _should_review_memory=False,
        _turn_exit_reason="text_response",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCleanupGuard:
    def test_all_three_raise_response_preserved(self):
        """All three cleanup methods raise — final_response is still returned."""
        agent = _make_agent(
            save_raises=True, cleanup_raises=True, persist_raises=True
        )
        result = _call_finalize(agent)

        assert result["final_response"] == "The answer is 42"
        assert "cleanup_errors" in result
        assert len(result["cleanup_errors"]) == 3

    def test_save_trajectory_raises_others_still_run(self):
        """_save_trajectory raising does not prevent the other two steps from running."""
        agent = _make_agent(save_raises=True)
        result = _call_finalize(agent)

        assert result["final_response"] == "The answer is 42"
        # Other two cleanup methods were still called.
        agent._cleanup_task_resources.assert_called_once()
        agent._persist_session.assert_called_once()
        # Error surfaced.
        assert "cleanup_errors" in result
        assert any("_save_trajectory" in e for e in result["cleanup_errors"])

    def test_cleanup_task_resources_raises_others_still_run(self):
        """_cleanup_task_resources raising does not skip _persist_session."""
        agent = _make_agent(cleanup_raises=True)
        result = _call_finalize(agent)

        assert result["final_response"] == "The answer is 42"
        agent._save_trajectory.assert_called_once()
        agent._persist_session.assert_called_once()
        assert any("_cleanup_task_resources" in e for e in result["cleanup_errors"])

    def test_persist_session_raises_response_preserved(self):
        """_persist_session raising does not discard the response."""
        agent = _make_agent(persist_raises=True)
        result = _call_finalize(agent)

        assert result["final_response"] == "The answer is 42"
        agent._save_trajectory.assert_called_once()
        agent._cleanup_task_resources.assert_called_once()
        assert any("_persist_session" in e for e in result["cleanup_errors"])

    def test_clean_turn_no_cleanup_errors_key(self):
        """A turn where nothing raises must NOT have a cleanup_errors key."""
        agent = _make_agent()
        result = _call_finalize(agent)

        assert result["final_response"] == "The answer is 42"
        assert "cleanup_errors" not in result
