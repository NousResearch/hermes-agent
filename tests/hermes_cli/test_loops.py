"""Tests for hermes_cli/loops.py — /loop recurring in-session wakeups."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes don't clobber the real one."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


# ──────────────────────────────────────────────────────────────────────
# Interval / argument parsing
# ──────────────────────────────────────────────────────────────────────


class TestParseIntervalToken:
    def test_minutes(self):
        from hermes_cli.loops import parse_interval_token

        assert parse_interval_token("5m") == 300

    def test_seconds(self):
        from hermes_cli.loops import parse_interval_token

        assert parse_interval_token("30s") == 30

    def test_hours(self):
        from hermes_cli.loops import parse_interval_token

        assert parse_interval_token("2h") == 7200

    def test_compound(self):
        from hermes_cli.loops import parse_interval_token

        assert parse_interval_token("1h30m") == 5400

    def test_case_insensitive(self):
        from hermes_cli.loops import parse_interval_token

        assert parse_interval_token("5M") == 300

    def test_bare_number_is_not_interval(self):
        from hermes_cli.loops import parse_interval_token

        assert parse_interval_token("3") is None

    def test_prose_is_not_interval(self):
        from hermes_cli.loops import parse_interval_token

        assert parse_interval_token("check") is None
        assert parse_interval_token("") is None
        assert parse_interval_token("5x") is None

    def test_zero_rejected(self):
        from hermes_cli.loops import parse_interval_token

        assert parse_interval_token("0m") is None
        assert parse_interval_token("0s") is None


class TestParseLoopArgs:
    def test_fixed_interval(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("5m check the deploy status")
        assert p["interval_seconds"] == 300
        assert p["prompt"] == "check the deploy status"
        assert p["error"] is None

    def test_every_sugar(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("every 10m /recap")
        assert p["interval_seconds"] == 600
        assert p["prompt"] == "/recap"

    def test_self_paced(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("keep refining the failing test until the suite passes")
        assert p["interval_seconds"] is None
        assert p["prompt"].startswith("keep refining")

    def test_times_flag(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("2m poll CI --times 30")
        assert p["interval_seconds"] == 120
        assert p["prompt"] == "poll CI"
        assert p["times"] == 30

    def test_until_flag(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("5m watch the queue --until queue depth reaches zero")
        assert p["interval_seconds"] == 300
        assert p["prompt"] == "watch the queue"
        assert p["until"] == "queue depth reaches zero"

    def test_until_and_times_together(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("2m poll --times 5 --until it is green")
        assert p["times"] == 5
        assert p["until"] == "it is green"
        assert p["prompt"] == "poll"

    def test_bad_times(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("2m poll --times zero")
        assert p["error"] is not None

    def test_no_start_now_flag_anymore(self):
        from hermes_cli.loops import parse_loop_args

        # Immediate first wakeup is the default now; --start-now was never
        # released, so the token is NOT parsed as a flag.
        p = parse_loop_args("1h check the deploy status")
        assert "start_now" not in p
        assert p["interval_seconds"] == 3600
        assert p["prompt"] == "check the deploy status"
        assert p["error"] is None

    def test_start_now_word_in_prompt_kept_verbatim(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("1h read the file called start-now.md")
        assert p["prompt"] == "read the file called start-now.md"

    def test_interval_only_is_error(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("5m")
        assert p["error"] is not None

    def test_empty(self):
        from hermes_cli.loops import parse_loop_args

        assert parse_loop_args("")["error"] == "empty"

    def test_prompt_with_leading_number_not_eaten(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("3 things to verify in the repo")
        assert p["interval_seconds"] is None
        assert p["prompt"].startswith("3 things")


class TestFormatInterval:
    def test_render(self):
        from hermes_cli.loops import format_interval

        assert format_interval(30) == "30s"
        assert format_interval(300) == "5m"
        assert format_interval(5400) == "1h30m"
        assert format_interval(90) == "1m30s"
        assert format_interval(0) == "0s"


# ──────────────────────────────────────────────────────────────────────
# LOOP_COMPLETE marker
# ──────────────────────────────────────────────────────────────────────


class TestResponseSignalsComplete:
    def test_marker_on_own_line(self):
        from hermes_cli.loops import response_signals_complete

        assert response_signals_complete("Deploy is live.\nLOOP_COMPLETE") is True

    def test_marker_with_trailing_period(self):
        from hermes_cli.loops import response_signals_complete

        assert response_signals_complete("done\nLOOP_COMPLETE.") is True

    def test_marker_mid_sentence_does_not_count(self):
        from hermes_cli.loops import response_signals_complete

        assert response_signals_complete("I will emit LOOP_COMPLETE when finished") is False

    def test_no_marker(self):
        from hermes_cli.loops import response_signals_complete

        assert response_signals_complete("still building") is False
        assert response_signals_complete("") is False


# ──────────────────────────────────────────────────────────────────────
# LoopState round-trip
# ──────────────────────────────────────────────────────────────────────


class TestLoopStateSerde:
    def test_round_trip(self):
        from hermes_cli.loops import LoopState

        s = LoopState(
            prompt="check CI",
            mode="interval",
            interval_seconds=300.0,
            current_delay=300.0,
            times=5,
            until="ci is green",
            ticks_fired=2,
            route={"platform": "telegram", "chat_id": "123"},
        )
        s2 = LoopState.from_json(s.to_json())
        assert s2.prompt == "check CI"
        assert s2.interval_seconds == 300.0
        assert s2.times == 5
        assert s2.until == "ci is green"
        assert s2.ticks_fired == 2
        assert s2.route == {"platform": "telegram", "chat_id": "123"}

    def test_old_row_missing_fields(self):
        from hermes_cli.loops import LoopState

        s = LoopState.from_json('{"prompt": "p"}')
        assert s.prompt == "p"
        assert s.status == "active"
        assert s.route == {}


# ──────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────


class TestPersistence:
    def test_save_load_clear(self, hermes_home):
        from hermes_cli.loops import LoopManager, load_loop

        mgr = LoopManager(session_id="sess-1")
        mgr.set("check the deploy", interval_seconds=300)
        loaded = load_loop("sess-1")
        assert loaded is not None
        assert loaded.prompt == "check the deploy"
        assert loaded.status == "active"

        assert mgr.clear() is True
        cleared = load_loop("sess-1")
        assert cleared is not None and cleared.status == "cleared"

    def test_list_active_loops(self, hermes_home):
        from hermes_cli.loops import LoopManager, list_active_loops

        LoopManager(session_id="a").set("task a", interval_seconds=60)
        mgr_b = LoopManager(session_id="b")
        mgr_b.set("task b", interval_seconds=60)
        mgr_b.pause()

        active = dict(list_active_loops())
        assert "a" in active
        assert "b" not in active

    def test_migrate_to_session(self, hermes_home):
        from hermes_cli.loops import LoopManager, load_loop, migrate_loop_to_session

        LoopManager(session_id="parent").set("watch it", interval_seconds=60)
        assert migrate_loop_to_session("parent", "child", reason="compression") is True
        child = load_loop("child")
        assert child is not None and child.prompt == "watch it"
        parent = load_loop("parent")
        assert parent is not None and parent.status == "cleared"

    def test_migrate_no_source(self, hermes_home):
        from hermes_cli.loops import migrate_loop_to_session

        assert migrate_loop_to_session("nope", "child2") is False
        assert migrate_loop_to_session("same", "same") is False

    def test_migrate_does_not_clobber_child(self, hermes_home):
        from hermes_cli.loops import LoopManager, load_loop, migrate_loop_to_session

        LoopManager(session_id="p2").set("parent loop", interval_seconds=60)
        LoopManager(session_id="c2").set("child loop", interval_seconds=60)
        assert migrate_loop_to_session("p2", "c2") is False
        assert load_loop("c2").prompt == "child loop"


# ──────────────────────────────────────────────────────────────────────
# Tick lifecycle
# ──────────────────────────────────────────────────────────────────────


class TestTickLifecycle:
    def test_due_immediately_on_create(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t1")
        mgr.set("poll", interval_seconds=300)
        assert mgr.is_due() is True

    def test_due_after_interval(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t2")
        state = mgr.set("poll", interval_seconds=300)
        state.next_due_at = time.time() - 1
        assert mgr.is_due() is True

    def test_not_due_once_rescheduled(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t2a")
        state = mgr.set("poll", interval_seconds=300)
        state.next_due_at = time.time() + 300
        assert mgr.is_due() is False

    def test_self_paced_due_immediately_on_create(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t2c")
        state = mgr.set("keep refining")
        assert state.mode == "self_paced"
        assert mgr.is_due() is True

    def test_fire_marks_awaiting_and_blocks_refire(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t3")
        state = mgr.set("poll the build", interval_seconds=300)
        state.next_due_at = time.time() - 1
        wakeup = mgr.fire_tick()
        assert wakeup is not None
        assert "[/loop wakeup #1" in wakeup
        assert "poll the build" in wakeup
        assert "LOOP_COMPLETE" in wakeup
        assert mgr.state.awaiting_response is True
        assert mgr.is_due() is False  # can't double-fire mid-turn
        assert mgr.fire_tick() is None

    def test_slash_prompt_returned_raw(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t4")
        state = mgr.set("/recap", interval_seconds=300)
        state.next_due_at = time.time() - 1
        assert mgr.fire_tick() == "/recap"

    def test_abandon_tick_rolls_back(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t5")
        state = mgr.set("poll", interval_seconds=300)
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        mgr.abandon_tick()
        assert mgr.state.awaiting_response is False
        assert mgr.state.ticks_fired == 0

    def test_complete_tick_marker_stops(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t6")
        state = mgr.set("poll", interval_seconds=300)
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        decision = mgr.complete_tick("The deploy is live.\nLOOP_COMPLETE")
        assert decision["stopped"] is True
        assert decision["status"] == "done"

    def test_complete_tick_times_cap(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t7")
        state = mgr.set("poll", interval_seconds=300, times=1)
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        decision = mgr.complete_tick("still building")
        assert decision["stopped"] is True
        assert decision["status"] == "done"
        assert "1/1" in decision["message"]

    def test_complete_tick_continues_and_reschedules(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t8")
        state = mgr.set("poll", interval_seconds=300)
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        decision = mgr.complete_tick("still building")
        assert decision["stopped"] is False
        assert decision["status"] == "active"
        assert mgr.state.awaiting_response is False
        assert mgr.state.next_due_at > time.time() + 250

    def test_complete_tick_max_ticks_pauses(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t9")
        state = mgr.set("poll", interval_seconds=300)
        state.max_ticks = 1
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        decision = mgr.complete_tick("still building")
        assert decision["stopped"] is True
        assert decision["status"] == "paused"

    def test_until_judge_done_stops(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t10")
        state = mgr.set("poll", interval_seconds=300, until="the suite is green")
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        with patch("hermes_cli.goals.judge_goal", return_value=("done", "suite green", False, None, False)):
            decision = mgr.complete_tick("All 500 tests passed.")
        assert decision["stopped"] is True
        assert decision["status"] == "done"

    def test_until_judge_continue_loops(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t11")
        state = mgr.set("poll", interval_seconds=300, until="the suite is green")
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        with patch("hermes_cli.goals.judge_goal", return_value=("continue", "3 failures", False, None, False)):
            decision = mgr.complete_tick("3 tests still failing")
        assert decision["stopped"] is False

    def test_until_judge_blocked_pauses(self, hermes_home):
        """An unachievable stop condition pauses the loop instead of spinning to the tick budget."""
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t11b")
        state = mgr.set("poll", interval_seconds=300, until="the deleted repo's CI is green")
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        with patch("hermes_cli.goals.judge_goal", return_value=("blocked", "repo no longer exists", False, None, False)):
            decision = mgr.complete_tick("The repository was deleted; there is no CI to watch.")
        assert decision["stopped"] is True
        assert decision["status"] == "paused"
        assert "unachievable" in decision["message"]

    def test_until_judge_error_fails_open(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t12")
        state = mgr.set("poll", interval_seconds=300, until="green")
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        with patch("hermes_cli.goals.judge_goal", side_effect=RuntimeError("api down")):
            decision = mgr.complete_tick("some output")
        assert decision["stopped"] is False  # fail-open: keep looping


class TestSelfPacedBackoff:
    def test_backoff_doubles_on_unchanged_and_resets_on_change(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="sp1")
        state = mgr.set("watch the queue")  # self-paced
        floor = state.current_delay
        assert state.mode == "self_paced"

        # Tick 1: response A → delay stays at floor (change from empty digest).
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        mgr.complete_tick("queue depth is 5")
        assert mgr.state.current_delay == floor

        # Tick 2: same response → backoff doubles.
        mgr.state.next_due_at = time.time() - 1
        mgr.fire_tick()
        mgr.complete_tick("queue depth is 5")
        assert mgr.state.current_delay == floor * 2

        # Tick 3: same again → doubles again.
        mgr.state.next_due_at = time.time() - 1
        mgr.fire_tick()
        mgr.complete_tick("queue depth is 5")
        assert mgr.state.current_delay == floor * 4

        # Tick 4: changed response → snaps back to floor.
        mgr.state.next_due_at = time.time() - 1
        mgr.fire_tick()
        mgr.complete_tick("queue depth is 2 — draining")
        assert mgr.state.current_delay == floor

    def test_timestamp_only_changes_do_not_reset(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="sp2")
        state = mgr.set("watch")
        floor = state.current_delay
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        mgr.complete_tick("Still building. Checked at 14:02:33")
        mgr.state.next_due_at = time.time() - 1
        mgr.fire_tick()
        mgr.complete_tick("Still building. Checked at 14:07:33")
        assert mgr.state.current_delay == floor * 2  # digest ignored the clock


# ──────────────────────────────────────────────────────────────────────
# Controls (pause/resume/clear) + min interval + status
# ──────────────────────────────────────────────────────────────────────


class TestControls:
    def test_min_interval_enforced(self, hermes_home):
        from hermes_cli.loops import LoopManager, min_interval_seconds

        mgr = LoopManager(session_id="c1")
        state = mgr.set("poll", interval_seconds=1)
        assert state.interval_seconds >= min_interval_seconds()

    def test_pause_resume(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="c2")
        state = mgr.set("poll", interval_seconds=300)
        state.next_due_at = time.time() - 1
        mgr.pause()
        assert mgr.is_active() is False
        assert mgr.is_due() is False
        mgr.resume()
        assert mgr.is_active() is True

    def test_paused_mid_tick_clears_awaiting(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="c3")
        state = mgr.set("poll", interval_seconds=300)
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        mgr.pause(reason="user-interrupted")
        assert mgr.state.awaiting_response is False

    def test_status_line_shapes(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="c4")
        assert "No loop set" in mgr.status_line()
        mgr.set("poll the build", interval_seconds=300)
        assert "active" in mgr.status_line()
        assert "poll the build" in mgr.status_line()
        mgr.pause()
        assert "paused" in mgr.status_line()
        mgr.clear()
        assert "No loop set" in mgr.status_line()


# ──────────────────────────────────────────────────────────────────────
# /goal mixing
# ──────────────────────────────────────────────────────────────────────


class TestGoalMixing:
    def test_active_goal_blocks_tick(self, hermes_home):
        from hermes_cli.goals import GoalManager
        from hermes_cli.loops import goal_blocks_loop_tick

        GoalManager(session_id="g1").set("finish the migration")
        assert goal_blocks_loop_tick("g1") is True

    def test_no_goal_does_not_block(self, hermes_home):
        from hermes_cli.loops import goal_blocks_loop_tick

        assert goal_blocks_loop_tick("g2") is False

    def test_paused_goal_does_not_block(self, hermes_home):
        from hermes_cli.goals import GoalManager
        from hermes_cli.loops import goal_blocks_loop_tick

        gm = GoalManager(session_id="g3")
        gm.set("finish it")
        gm.pause()
        assert goal_blocks_loop_tick("g3") is False

    def test_parked_goal_does_not_block(self, hermes_home):
        from hermes_cli.goals import GoalManager
        from hermes_cli.loops import goal_blocks_loop_tick

        gm = GoalManager(session_id="g4")
        gm.set("finish it")
        gm.wait_for_seconds(3600, reason="waiting on CI")
        assert goal_blocks_loop_tick("g4") is False


# ──────────────────────────────────────────────────────────────────────
# dispatch_loop_command (shared slash handler)
# ──────────────────────────────────────────────────────────────────────


class TestDispatchLoopCommand:
    def test_create_fixed(self, hermes_home):
        from hermes_cli.loops import LoopManager, dispatch_loop_command

        mgr = LoopManager(session_id="d1")
        result = dispatch_loop_command(mgr, "5m check the deploy")
        assert result["created"] is True
        assert "Loop set" in result["output"]
        assert "every 5m" in result["output"]

    def test_create_self_paced(self, hermes_home):
        from hermes_cli.loops import LoopManager, dispatch_loop_command

        mgr = LoopManager(session_id="d2")
        result = dispatch_loop_command(mgr, "keep fixing the tests")
        assert result["created"] is True
        assert "Self-paced" in result["output"]

    def test_create_fires_immediately(self, hermes_home):
        from hermes_cli.loops import LoopManager, dispatch_loop_command

        mgr = LoopManager(session_id="d2a")
        result = dispatch_loop_command(mgr, "1h check the deploy")
        assert result["created"] is True
        assert "Loop set" in result["output"]
        assert "fires now" in result["output"]
        assert mgr.is_due() is True

    def test_status_empty(self, hermes_home):
        from hermes_cli.loops import LoopManager, dispatch_loop_command

        mgr = LoopManager(session_id="d3")
        result = dispatch_loop_command(mgr, "")
        assert result["created"] is False
        assert "No loop set" in result["output"]

    def test_pause_resume_stop(self, hermes_home):
        from hermes_cli.loops import LoopManager, dispatch_loop_command

        mgr = LoopManager(session_id="d4")
        dispatch_loop_command(mgr, "5m poll")
        assert "paused" in dispatch_loop_command(mgr, "pause")["output"].lower()
        assert "resumed" in dispatch_loop_command(mgr, "resume")["output"].lower()
        assert "stopped" in dispatch_loop_command(mgr, "stop")["output"].lower()
        assert "No active loop" in dispatch_loop_command(mgr, "stop")["output"]

    def test_route_stored(self, hermes_home):
        from hermes_cli.loops import LoopManager, dispatch_loop_command, load_loop

        mgr = LoopManager(session_id="d5")
        route = {"platform": "telegram", "chat_id": "42", "chat_type": "private"}
        dispatch_loop_command(mgr, "5m ping", route=route)
        assert load_loop("d5").route == route

    def test_help(self, hermes_home):
        from hermes_cli.loops import LoopManager, dispatch_loop_command

        mgr = LoopManager(session_id="d6")
        out = dispatch_loop_command(mgr, "help")["output"]
        assert "Usage" in out
        assert "--times" in out

    def test_bad_times_error(self, hermes_home):
        from hermes_cli.loops import LoopManager, dispatch_loop_command

        mgr = LoopManager(session_id="d7")
        result = dispatch_loop_command(mgr, "5m poll --times banana")
        assert result["created"] is False
        assert "--times" in result["output"]


# ──────────────────────────────────────────────────────────────────────
# Command registry
# ──────────────────────────────────────────────────────────────────────


class TestCommandRegistry:
    def test_loop_registered_with_proactive_alias(self):
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("loop")
        assert cmd is not None
        assert cmd.name == "loop"
        alias = resolve_command("proactive")
        assert alias is not None
        assert alias.name == "loop"


# ──────────────────────────────────────────────────────────────────────
# SessionDB.list_meta_prefix
# ──────────────────────────────────────────────────────────────────────


class TestListMetaPrefix:
    def test_prefix_scan(self, hermes_home):
        from hermes_state import SessionDB

        db = SessionDB()
        db.set_meta("lmp-test:aaa", "1")
        db.set_meta("lmp-test:bbb", "2")
        db.set_meta("other:aaa", "3")
        rows = dict(db.list_meta_prefix("lmp-test:"))
        assert rows == {"lmp-test:aaa": "1", "lmp-test:bbb": "2"}

    def test_wildcards_escaped(self, hermes_home):
        from hermes_state import SessionDB

        db = SessionDB()
        db.set_meta("pre%fix:x", "1")
        db.set_meta("prefix:y", "2")
        assert db.list_meta_prefix("pre%") == [("pre%fix:x", "1")]

    def test_empty_prefix(self, hermes_home):
        from hermes_state import SessionDB

        db = SessionDB()
        assert db.list_meta_prefix("") == []
