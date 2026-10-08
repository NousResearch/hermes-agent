"""Targeted tests: run_codex_app_server_turn must forward the configured
run budget to ``CodexAppServerSession.run_turn`` as ``turn_timeout``.

Contract:
- ``agent.run_budget_seconds`` set to a positive value -> ``turn_timeout`` is
  passed as that float (e.g. 900 -> 900.0).
- unset (None), non-numeric, bool, or non-positive -> ``turn_timeout`` is NOT
  passed at all, so ``run_turn`` keeps its 600s default.
- ``agent.turn_liveness`` must not change that deadline.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.codex_runtime import run_codex_app_server_turn


def _make_turn():
    return SimpleNamespace(
        interrupted=False,
        error=None,
        thread_id="thread-1",
        turn_id="turn-1",
        projected_messages=[{"role": "assistant", "content": "CODEX_ASSISTANT"}],
        tool_iterations=0,
        final_text="CODEX_ASSISTANT",
        should_retire=False,
    )


def _make_agent(run_budget):
    agent = MagicMock()
    agent._codex_session = MagicMock()
    agent._codex_session.run_turn.side_effect = lambda **kwargs: (
        _make_turn(), dict(kwargs)
    )[0]
    # Seeded session with no recorded prompt: _ensure_codex_session keeps it.
    agent._codex_session_prompt = None
    agent.compression_checkpoint_required = False
    agent.run_budget_seconds = run_budget
    agent.turn_liveness = SimpleNamespace(timeout_s=15)
    agent._iters_since_skill = 0
    agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent._session_db = None
    agent._session_db_created = True
    agent.session_id = "sess-rb"
    agent.context_compressor = None
    return agent


def _captured_kwargs(run_budget):
    agent = _make_agent(run_budget)
    run_codex_app_server_turn(
        agent,
        user_message="hello",
        original_user_message="hello",
        messages=[{"role": "user", "content": "hello"}],
        effective_task_id="task-1",
    )
    return agent._codex_session.run_turn.call_args.kwargs


def test_positive_run_budget_passed_as_turn_timeout():
    kwargs = _captured_kwargs(900)
    assert kwargs["turn_timeout"] == 900.0
    assert kwargs["user_input"] == "hello"


def test_no_run_budget_keeps_run_turn_default():
    assert "turn_timeout" not in _captured_kwargs(None)


def test_non_numeric_run_budget_keeps_run_turn_default():
    assert "turn_timeout" not in _captured_kwargs("abc")


def test_nonpositive_run_budget_keeps_run_turn_default():
    assert "turn_timeout" not in _captured_kwargs(0)
    assert "turn_timeout" not in _captured_kwargs(-5)


def test_bool_run_budget_keeps_run_turn_default():
    assert "turn_timeout" not in _captured_kwargs(True)
    assert "turn_timeout" not in _captured_kwargs(False)
