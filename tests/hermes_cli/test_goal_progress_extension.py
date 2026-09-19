"""Progress-based auto-continuation when a /goal reaches its turn budget (#109804).

Budget exhaustion stops being an unconditional pause and becomes a progress-review checkpoint
(opt-in via ``goals.auto_extend``): concrete recent progress earns another ``max_turns`` window in
the same session (goal, contract and criteria untouched), while a stalled/blocked/unusable review
still pauses with a reason — and ``goals.max_total_turns`` keeps the cumulative spend bounded.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME (same shape as tests/hermes_cli/test_goals.py)."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _enable_auto_extend(home, *, max_total_turns=100, max_turns=20):
    """Write the opt-in config the feature reads (``goals`` block of config.yaml)."""
    (home / "config.yaml").write_text(
        "goals:\n"
        f"  max_turns: {max_turns}\n"
        "  auto_extend: true\n"
        f"  max_total_turns: {max_total_turns}\n",
        encoding="utf-8",
    )


_JUDGE_CONTINUE = ("continue", "still a concrete next step", False, None, False)


def _drive_to_budget(mgr, *, extensions_seen=0, turns_per_window=None):
    """Run windows until the budget boundary fires; returns the last decision."""
    turns = turns_per_window or max(1, mgr.state.max_turns)
    for _ in range(turns):
        with patch("hermes_cli.goals.judge_goal", return_value=_JUDGE_CONTINUE):
            decision = mgr.evaluate_after_turn("did some work")
    return decision


# ──────────────────────────────────────────────────────────────────────
# Config: off by default, opt-in via goals.auto_extend
# ──────────────────────────────────────────────────────────────────────


class TestExtensionConfig:
    def test_defaults_off_with_a_ceiling(self, hermes_home):
        from hermes_cli.goals import _goal_extension_config, DEFAULT_MAX_TOTAL_TURNS

        assert _goal_extension_config() == (False, DEFAULT_MAX_TOTAL_TURNS)

    def test_reads_config_block(self, hermes_home):
        from hermes_cli.goals import _goal_extension_config

        _enable_auto_extend(hermes_home, max_total_turns=40)
        assert _goal_extension_config() == (True, 40)

    def test_string_truthy_and_zero_ceiling(self, hermes_home):
        """YAML-ish string booleans work; 0 disables the ceiling rather than meaning 'pause now'."""
        from hermes_cli.goals import _goal_extension_config

        (hermes_home / "config.yaml").write_text(
            "goals:\n  auto_extend: 'yes'\n  max_total_turns: 0\n", encoding="utf-8"
        )
        assert _goal_extension_config() == (True, 0)

    def test_garbage_falls_back_to_defaults(self, hermes_home):
        """A broken config.yaml must never turn the budget backstop off by accident."""
        from hermes_cli.goals import _goal_extension_config, DEFAULT_MAX_TOTAL_TURNS

        (hermes_home / "config.yaml").write_text(
            "goals:\n  auto_extend: true\n  max_total_turns: not-a-number\n", encoding="utf-8"
        )
        assert _goal_extension_config() == (True, DEFAULT_MAX_TOTAL_TURNS)


# ──────────────────────────────────────────────────────────────────────
# Default behavior unchanged: no auto-extend, no extra review call
# ──────────────────────────────────────────────────────────────────────


class TestDefaultPathUnchanged:
    def test_budget_still_pauses_and_the_review_never_runs(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="ext-off", default_max_turns=2)
        mgr.set("do the thing", max_turns=2)

        review = MagicMock(return_value=("extend", "should never be consulted"))
        with patch("hermes_cli.goals.review_progress", review):
            decision = _drive_to_budget(mgr)

        assert review.call_count == 0, "auto_extend is opt-in; no review call by default"
        assert decision["status"] == "paused"
        assert decision["should_continue"] is False
        assert decision["continuation_prompt"] is None
        assert "turns used" in decision["message"]
        assert "turns total" not in decision["message"]


# ──────────────────────────────────────────────────────────────────────
# The checkpoint: extend / done / stalled / blocked / unusable review
# ──────────────────────────────────────────────────────────────────────


class TestBudgetCheckpoint:
    def _manager(self, sid="ext-on", *, max_turns=2, contract=None, subgoals=None):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id=sid, default_max_turns=max_turns)
        mgr.set("finish the benchmark", max_turns=max_turns, contract=contract)
        for text in subgoals or []:
            mgr.add_subgoal(text)
        return mgr

    def test_extension_grants_another_window_in_place(self, hermes_home):
        """Progress → another max_turns window, goal/contract/criteria preserved, usage visible."""
        from hermes_cli.goals import GoalContract

        _enable_auto_extend(hermes_home)
        mgr = self._manager(
            contract=GoalContract(verification="pytest -q tests/bench passes"),
            subgoals=["keep the CLI flags stable"],
        )

        with patch(
            "hermes_cli.goals.review_progress",
            return_value=("extend", "two of three suites green since the last window"),
        ) as review:
            decision = _drive_to_budget(mgr)

        assert review.call_count == 1
        assert decision["verdict"] == "extended"
        assert decision["status"] == "active"
        assert decision["should_continue"] is True
        # Same session, same goal text, contract and criteria — the loop just keeps going.
        assert "[Continuing toward your standing goal]" in decision["continuation_prompt"]
        assert "finish the benchmark" in decision["continuation_prompt"]
        assert "pytest -q tests/bench passes" in decision["continuation_prompt"]
        assert "keep the CLI flags stable" in decision["continuation_prompt"]
        # Budget: per-window counter restarts, cumulative usage stays visible.
        assert mgr.state.turns_used == 0
        assert mgr.state.max_turns == 2
        assert mgr.state.extensions == 1
        assert mgr.state.total_turns_used == 2
        assert mgr.state.status == "active"
        assert "Goal extended" in decision["message"]
        assert "2/100 turns total" in decision["message"]
        assert "two of three suites green" in decision["message"]

    def test_extension_survives_reload(self, hermes_home):
        """Extensions/usage persist in state_meta, not just in memory."""
        from hermes_cli.goals import GoalManager

        _enable_auto_extend(hermes_home)
        mgr = self._manager(sid="ext-persist")
        with patch("hermes_cli.goals.review_progress", return_value=("extend", "progress")):
            _drive_to_budget(mgr)

        reloaded = GoalManager(session_id="ext-persist")
        assert reloaded.state.extensions == 1
        assert reloaded.state.total_turns_used == 2
        assert reloaded.state.status == "active"

    def test_overall_ceiling_pauses_instead_of_extending(self, hermes_home):
        """The user-approved limit still stops the loop — cumulative, across windows."""
        _enable_auto_extend(hermes_home, max_total_turns=3)
        mgr = self._manager(sid="ext-ceiling", max_turns=2)

        with patch("hermes_cli.goals.review_progress", return_value=("extend", "progress")) as review:
            _drive_to_budget(mgr)                       # window 1 boundary: 2/3 total → extend
            assert mgr.state.extensions == 1
            review.reset_mock()
            decision = _drive_to_budget(mgr)            # window 2 boundary: 4/3 total → pause

        assert review.call_count == 0, "the ceiling is checked before spending a review call"
        assert decision["status"] == "paused"
        assert decision["should_continue"] is False
        assert decision["continuation_prompt"] is None
        assert "overall limit of 3 turns reached" in decision["message"]
        assert mgr.state.total_turns_used == 4

    def test_stalled_review_pauses_with_the_reason(self, hermes_home):
        _enable_auto_extend(hermes_home)
        mgr = self._manager(sid="ext-stalled")

        with patch(
            "hermes_cli.goals.review_progress",
            return_value=("stalled", "same status recap as the previous window, no new evidence"),
        ):
            decision = _drive_to_budget(mgr)

        assert decision["status"] == "paused"
        assert decision["should_continue"] is False
        assert decision["continuation_prompt"] is None
        assert "no new evidence" in decision["message"]
        assert "stalled" in decision["message"]
        assert "/goal resume" in decision["message"]
        assert "2/100 turns total" in decision["message"]
        assert mgr.state.extensions == 0
        assert mgr.state.status == "paused"
        assert "no new evidence" in (mgr.state.paused_reason or "")

    def test_blocked_review_pauses(self, hermes_home):
        _enable_auto_extend(hermes_home)
        mgr = self._manager(sid="ext-blocked")

        with patch(
            "hermes_cli.goals.review_progress",
            return_value=("blocked", "needs the user's API credentials"),
        ):
            decision = _drive_to_budget(mgr)

        assert decision["status"] == "paused"
        assert decision["verdict"] == "blocked"
        assert "needs the user's API credentials" in decision["message"]

    def test_done_review_marks_the_goal_done(self, hermes_home):
        _enable_auto_extend(hermes_home)
        mgr = self._manager(sid="ext-done")

        with patch(
            "hermes_cli.goals.review_progress",
            return_value=("done", "the benchmark ran and every suite is green"),
        ):
            decision = _drive_to_budget(mgr)

        assert decision["status"] == "done"
        assert decision["should_continue"] is False
        assert "Goal achieved" in decision["message"]
        assert mgr.state.status == "done"

    def test_unusable_review_fails_closed(self, hermes_home):
        """No review evidence → pause (not extend) — the opposite of the per-turn judge's fail-open."""
        _enable_auto_extend(hermes_home)
        mgr = self._manager(sid="ext-unusable")

        with patch(
            "hermes_cli.goals.review_progress",
            return_value=("stalled", "progress review failed: TimeoutError"),
        ):
            decision = _drive_to_budget(mgr)

        assert decision["status"] == "paused"
        assert decision["should_continue"] is False
        assert "TimeoutError" in decision["message"]


# ──────────────────────────────────────────────────────────────────────
# review_progress itself
# ──────────────────────────────────────────────────────────────────────


class TestReviewProgress:
    def _fake_llm(self, captured, content: str):
        class _FakeMsg:
            pass

        _FakeMsg.content = content

        class _FakeChoice:
            message = _FakeMsg()

        class _FakeResp:
            choices = [_FakeChoice()]

        def _fake(**kwargs):
            captured.update(kwargs)
            return _FakeResp()

        return _fake

    def test_extend_verdict_parsed(self, hermes_home):
        from hermes_cli import goals

        captured = {}
        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=self._fake_llm(
                captured, '{"verdict": "extend", "reason": "three of four files created"}'
            ),
        ):
            verdict, reason = goals.review_progress("build the four files", "created 3 of 4")

        assert verdict == "extend"
        assert reason == "three of four files created"

    def test_prompt_carries_goal_and_contract(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalContract

        captured = {}
        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=self._fake_llm(captured, '{"verdict": "extend", "reason": "progress"}'),
        ):
            goals.review_progress(
                "migrate auth to JWT",
                "auth service ported, tests still failing",
                contract=GoalContract(verification="pytest tests/auth passes"),
            )

        user_msg = next(
            (m["content"] for m in (captured.get("messages") or []) if m["role"] == "user"), ""
        )
        assert "migrate auth to JWT" in user_msg
        assert "pytest tests/auth passes" in user_msg
        assert "auth service ported" in user_msg

    def test_api_error_fails_closed_to_stalled(self, hermes_home):
        from hermes_cli import goals

        with patch("agent.auxiliary_client.call_llm", side_effect=RuntimeError("boom")):
            verdict, reason = goals.review_progress("g", "some response")

        assert verdict == "stalled"
        assert "RuntimeError" in reason

    def test_non_json_reply_fails_closed(self, hermes_home):
        from hermes_cli import goals

        with patch("agent.auxiliary_client.call_llm", return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="I think it's going well!"))]
        )):
            verdict, reason = goals.review_progress("g", "some response")

        assert verdict == "stalled"
        assert "not JSON" in reason

    def test_unknown_verdict_fails_closed(self, hermes_home):
        from hermes_cli import goals

        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=self._fake_llm({}, '{"verdict": "maybe", "reason": "unclear"}'),
        ):
            verdict, reason = goals.review_progress("g", "some response")

        assert verdict == "stalled"
        assert "no usable verdict" in reason

    def test_empty_response_is_stalled_without_calling_the_model(self, hermes_home):
        from hermes_cli import goals

        with patch("agent.auxiliary_client.call_llm") as call:
            verdict, reason = goals.review_progress("g", "   ")

        assert verdict == "stalled"
        assert "empty response" in reason
        call.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# Cumulative usage stays visible; /goal resume doesn't wipe it
# ──────────────────────────────────────────────────────────────────────


class TestCumulativeUsageVisible:
    def test_resume_resets_the_window_but_not_the_cumulative_count(self, hermes_home):
        from hermes_cli.goals import GoalManager

        _enable_auto_extend(hermes_home)
        mgr = GoalManager(session_id="ext-resume", default_max_turns=2)
        mgr.set("do the thing", max_turns=2)
        with patch("hermes_cli.goals.review_progress", return_value=("extend", "progress")):
            _drive_to_budget(mgr)
        assert (mgr.state.turns_used, mgr.state.total_turns_used, mgr.state.extensions) == (0, 2, 1)

        mgr.resume()
        assert mgr.state.turns_used == 0
        assert mgr.state.total_turns_used == 2
        assert mgr.state.extensions == 1

    def test_status_line_shows_cumulative_turns_and_extensions(self, hermes_home):
        from hermes_cli.goals import GoalManager

        _enable_auto_extend(hermes_home)
        mgr = GoalManager(session_id="ext-status", default_max_turns=2)
        mgr.set("do the thing", max_turns=2)
        with patch("hermes_cli.goals.review_progress", return_value=("extend", "progress")):
            _drive_to_budget(mgr)

        line = mgr.status_line()
        assert "0/2 turns" in line
        assert "2 total" in line
        assert "1 budget extension" in line

    def test_status_line_stays_quiet_before_any_extension(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="ext-status-quiet", default_max_turns=2)
        mgr.set("do the thing", max_turns=2)
        assert "total" not in mgr.status_line()


# ──────────────────────────────────────────────────────────────────────
# Back-compat
# ──────────────────────────────────────────────────────────────────────


class TestBackCompat:
    def test_legacy_state_row_loads_with_zeroed_counters(self, hermes_home):
        from hermes_cli.goals import GoalState

        state = GoalState.from_json('{"goal": "old goal", "status": "active", "turns_used": 2}')
        assert state.extensions == 0
        assert state.total_turns_used == 0
        # A legacy goal mid-window still pauses at the budget (no extension configured for it).
        assert state.max_turns == 20

    def test_state_roundtrips_the_new_counters(self, hermes_home):
        from hermes_cli.goals import GoalState

        state = GoalState(goal="g", turns_used=1, extensions=2, total_turns_used=41)
        restored = GoalState.from_json(state.to_json())
        assert (restored.extensions, restored.total_turns_used) == (2, 41)
