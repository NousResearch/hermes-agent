"""/goal re-fold at the compaction boundary (#133643).

The todo list is re-added by every compaction; the goal and its completion contract live in
``state_meta`` and must be re-folded too, or a mid-turn compaction keeps them only inside the
summary's lossy free-text ``## Goal`` section.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.conversation_compression import (
    _is_real_user_message,
    compress_context,
)
from agent.conversation_compression_goal_fold import (
    GOAL_PRESERVED_HEADER,
    GOAL_SNAPSHOT_FLAG,
    fold_goal_snapshot as _fold_goal_snapshot,
)
from hermes_cli.goals import (
    GOAL_PRESERVED_HEADER as GOALS_HEADER,
    GoalContract,
    GoalState,
)


def _agent(session_id: str = "goal-session"):
    return SimpleNamespace(session_id=session_id, _repair_message_sequence=lambda messages: None)


def _active_goal(**overrides) -> GoalState:
    state = GoalState(
        goal="Ship the release train",
        status="active",
        contract=GoalContract(
            outcome="Release shipped",
            verification="pytest -q passes",
            constraints="No public API changes",
            stop_when="A required approval is missing",
        ),
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _goal_rows(messages):
    return [
        message
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "user"
        and GOAL_PRESERVED_HEADER in str(message.get("content") or "")
    ]


def test_header_literal_stays_in_sync_with_goals_module():
    assert GOAL_PRESERVED_HEADER == GOALS_HEADER


def test_fold_goal_snapshot_carries_contract_and_is_synthetic():
    compressed = [{"role": "assistant", "content": "working"}]
    with patch("hermes_cli.goals.load_goal", return_value=_active_goal()):
        _fold_goal_snapshot(_agent(), compressed)

    rows = _goal_rows(compressed)
    assert len(rows) == 1
    row = rows[0]
    assert row[GOAL_SNAPSHOT_FLAG] is True
    text = row["content"]
    assert "Goal: Ship the release train" in text
    assert "Verification: pytest -q passes" in text
    assert "Constraints: No public API changes" in text
    assert "Stop when blocked: A required approval is missing" in text
    # Non-imperative: the preserved block must not tell the model to keep working.
    assert "Continue working" not in text
    # Provenance: scaffolding, never human intent.
    assert _is_real_user_message(row) is False


def test_fold_goal_snapshot_refreshes_rather_than_stacks_across_boundaries():
    compressed = [{"role": "assistant", "content": "working"}]
    with patch("hermes_cli.goals.load_goal", return_value=_active_goal()):
        _fold_goal_snapshot(_agent(), compressed)
        _fold_goal_snapshot(_agent(), compressed)
    assert len(_goal_rows(compressed)) == 1


def test_fold_goal_snapshot_merges_into_trailing_real_user_row():
    compressed = [
        {"role": "assistant", "content": "working"},
        {"role": "user", "content": "Also keep the docs in sync."},
    ]
    with patch("hermes_cli.goals.load_goal", return_value=_active_goal()):
        _fold_goal_snapshot(_agent(), compressed)

    # No new row appended: the block folds into the existing human turn, which stays real.
    assert len(compressed) == 2
    tail = compressed[-1]
    assert GOAL_PRESERVED_HEADER in tail["content"]
    assert tail["content"].startswith("Also keep the docs in sync.")
    assert _is_real_user_message(tail) is True


@pytest.mark.parametrize("status", ["done", "paused", "cleared"])
def test_fold_goal_snapshot_ignores_an_inactive_goal(status):
    compressed = [{"role": "assistant", "content": "working"}]
    with patch("hermes_cli.goals.load_goal", return_value=_active_goal(status=status)):
        _fold_goal_snapshot(_agent(), compressed)
    assert _goal_rows(compressed) == []


def test_fold_goal_snapshot_ignores_a_parked_goal():
    compressed = [{"role": "assistant", "content": "working"}]
    parked = _active_goal(waiting_on_pid=4242, waiting_reason="waiting on CI")
    with patch("hermes_cli.goals.load_goal", return_value=parked):
        _fold_goal_snapshot(_agent(), compressed)
    assert _goal_rows(compressed) == []


def test_fold_goal_snapshot_does_not_duplicate_a_continuation_turn():
    continuation = (
        "[Continuing toward your standing goal]\n"
        "Goal: Ship the release train\n\nCompletion contract:\n- Verification: pytest -q passes\n"
    )
    compressed = [
        {"role": "assistant", "content": "working"},
        {"role": "user", "content": continuation},
    ]
    with patch("hermes_cli.goals.load_goal", return_value=_active_goal()):
        _fold_goal_snapshot(_agent(), compressed)
    assert _goal_rows(compressed) == []


def test_fold_goal_snapshot_preserves_a_snapshot_when_no_goal_is_loadable():
    row = {"role": "user", "content": f"{GOAL_PRESERVED_HEADER}\nGoal: old", GOAL_SNAPSHOT_FLAG: True}
    compressed = [{"role": "assistant", "content": "working"}, row]
    with patch("hermes_cli.goals.load_goal", return_value=None):
        _fold_goal_snapshot(_agent(), compressed)
    assert len(_goal_rows(compressed)) == 1


# ── End-to-end through compress_context ───────────────────────────────


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _valid_zero_user_summary(label: str = "Checked artifacts.") -> str:
    from agent.context_compressor import HISTORICAL_TASK_HEADING, _NO_USER_TASK_SENTINEL

    return f"""{HISTORICAL_TASK_HEADING}
{_NO_USER_TASK_SENTINEL}

## Goal
Historical work only.

## Completed Actions
1. {label}

## Resolved Questions
None.

## Historical Pending User Asks
None.
"""


def _assistant_turns(start: int, count: int) -> list[dict]:
    return [
        {"role": "assistant", "content": f"Scheduled step {idx} completed. " + ("x" * 500)}
        for idx in range(start, start + count)
    ]


def test_compress_context_refolds_an_active_goal(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "goal-refold-lifecycle"
    db.create_session(session_id, source="cron", model="test/model")

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.compression_in_place = True
    agent.context_compressor.protect_first_n = 0
    agent.context_compressor.protect_last_n = 2
    agent.context_compressor.tail_token_budget = 80

    with patch("hermes_cli.goals.load_goal", return_value=_active_goal()), patch(
        "agent.context_compressor.call_llm",
        return_value=_response(_valid_zero_user_summary("did work")),
    ):
        compressed, _ = compress_context(
            agent, _assistant_turns(0, 24), "system", approx_tokens=90_000, force=True,
        )

    rows = _goal_rows(compressed)
    assert len(rows) == 1, "the active goal + contract must survive compaction exactly once"
    assert "Verification: pytest -q passes" in rows[0]["content"]
    db.close()