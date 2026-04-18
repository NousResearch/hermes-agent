"""Tests for the FanoutReviewFSM — state machine for /fanout command review."""

import pytest

from hermes_cli.fanout_fsm import (
    FanoutPlan,
    FanoutReviewFSM,
    FanoutState,
    FanoutStory,
    FanoutTransitionError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _story(sid: str = "001", name: str = "test", deps: list[str] | None = None,
           acceptance: list[str] | None = None) -> FanoutStory:
    return FanoutStory(
        id=sid,
        name=name,
        description=f"Description for {name}",
        dependencies=deps or [],
        acceptance=acceptance or ["passes tests"],
    )


# ---------------------------------------------------------------------------
# FanoutStory
# ---------------------------------------------------------------------------

class TestFanoutStory:
    def test_to_dict_roundtrip(self):
        s = _story("001", "auth", deps=["000"], acceptance=["a", "b"])
        d = s.to_dict()
        restored = FanoutStory.from_dict(d)
        assert restored.id == "001"
        assert restored.name == "auth"
        assert restored.dependencies == ["000"]
        assert restored.acceptance == ["a", "b"]

    def test_defaults(self):
        s = FanoutStory(id="1", name="x", description="d")
        assert s.dependencies == []
        assert s.acceptance == []

    def test_from_dict_missing_fields(self):
        s = FanoutStory.from_dict({"id": "1"})
        assert s.name == "unnamed"
        assert s.description == ""


# ---------------------------------------------------------------------------
# FanoutPlan
# ---------------------------------------------------------------------------

class TestFanoutPlan:
    def test_is_complete_false_when_none_done(self):
        plan = FanoutPlan(task="t", stories=[_story("001")])
        assert not plan.is_complete

    def test_is_complete_true_when_all_done(self):
        plan = FanoutPlan(task="t", stories=[_story("001"), _story("002")])
        plan.completed = {"001", "002"}
        assert plan.is_complete

    def test_is_complete_false_when_partial(self):
        plan = FanoutPlan(task="t", stories=[_story("001"), _story("002")])
        plan.completed = {"001"}
        assert not plan.is_complete

    def test_is_complete_false_with_no_stories(self):
        plan = FanoutPlan(task="t", stories=[])
        assert not plan.is_complete

    def test_remaining_stories(self):
        plan = FanoutPlan(task="t", stories=[_story("001"), _story("002"), _story("003")])
        plan.completed = {"002"}
        remaining = plan.remaining_stories
        assert len(remaining) == 2
        assert {s.id for s in remaining} == {"001", "003"}

    def test_ready_stories_no_deps(self):
        plan = FanoutPlan(task="t", stories=[_story("001"), _story("002")])
        ready = plan.ready_stories
        assert {s.id for s in ready} == {"001", "002"}

    def test_ready_stories_deps_satisfied(self):
        plan = FanoutPlan(task="t", stories=[
            _story("001", "A", deps=[]),
            _story("002", "B", deps=["001"]),
        ])
        plan.completed = {"001"}
        ready = plan.ready_stories
        assert [s.id for s in ready] == ["002"]

    def test_ready_stories_deps_not_satisfied(self):
        plan = FanoutPlan(task="t", stories=[
            _story("001", "A", deps=[]),
            _story("002", "B", deps=["001"]),
        ])
        ready = plan.ready_stories
        assert [s.id for s in ready] == ["001"]

    def test_blocked_stories(self):
        plan = FanoutPlan(task="t", stories=[
            _story("001", "A", deps=[]),
            _story("002", "B", deps=["001"]),
        ])
        blocked = plan.blocked_stories
        assert [s.id for s in blocked] == ["002"]

    def test_to_dict_roundtrip(self):
        plan = FanoutPlan(task="build app", stories=[_story("001", "auth")])
        plan.completed = {"001"}
        plan.critique_history = ["needs more validation"]
        d = plan.to_dict()
        restored = FanoutPlan.from_dict(d)
        assert restored.task == "build app"
        assert len(restored.stories) == 1
        assert restored.stories[0].id == "001"
        assert restored.completed == {"001"}
        assert restored.critique_history == ["needs more validation"]


# ---------------------------------------------------------------------------
# FanoutReviewFSM — basic lifecycle
# ---------------------------------------------------------------------------

class TestFanoutReviewFSMLifecycle:
    def test_initial_state_is_idle(self):
        fsm = FanoutReviewFSM()
        assert fsm.state == FanoutState.IDLE
        assert fsm.plan is None

    def test_full_happy_path(self):
        fsm = FanoutReviewFSM()

        fsm.start("Build a web scraper")
        assert fsm.state == FanoutState.DECOMPOSING
        assert fsm.plan is not None
        assert fsm.plan.task == "Build a web scraper"

        stories = [_story("001", "HTTP client"), _story("002", "Parser")]
        fsm.decomposition_done(stories)
        assert fsm.state == FanoutState.PLAN_READY
        assert len(fsm.plan.stories) == 2

        fsm.accept()
        assert fsm.state == FanoutState.CONFIRMED

        fsm.execute()
        assert fsm.state == FanoutState.EXECUTING

        fsm.story_done("001")
        assert fsm.state == FanoutState.EXECUTING
        assert fsm.plan.completed == {"001"}

        fsm.story_done("002")
        fsm.all_stories_done()
        assert fsm.state == FanoutState.COMPLETE

    def test_history_records_transitions(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        fsm.decomposition_done([_story("001")])
        fsm.accept()
        fsm.execute()
        fsm.story_done("001")
        fsm.all_stories_done()

        history = fsm.history
        assert len(history) == 6
        assert history[0] == ("start", FanoutState.IDLE, FanoutState.DECOMPOSING)
        assert history[1] == ("decomposition_done", FanoutState.DECOMPOSING, FanoutState.PLAN_READY)
        assert history[2] == ("accept", FanoutState.PLAN_READY, FanoutState.CONFIRMED)
        assert history[3] == ("execute", FanoutState.CONFIRMED, FanoutState.EXECUTING)
        assert history[4] == ("story_done", FanoutState.EXECUTING, FanoutState.EXECUTING)
        assert history[5] == ("all_stories_done", FanoutState.EXECUTING, FanoutState.COMPLETE)


# ---------------------------------------------------------------------------
# FanoutReviewFSM — review gate: critique loop
# ---------------------------------------------------------------------------

class TestFanoutReviewFSMCritiqueLoop:
    def test_critique_then_re_decompose(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        fsm.decomposition_done([_story("001")])

        fsm.critique("stories are too broad")
        assert fsm.state == FanoutState.CRITIQUING
        assert fsm.accumulated_critique == "stories are too broad"

        fsm.re_decompose()
        assert fsm.state == FanoutState.DECOMPOSING
        assert len(fsm.plan.critique_history) == 1

        # Second decomposition
        fsm.decomposition_done([_story("001a"), _story("002a")])
        fsm.critique("still not granular enough")
        fsm.re_decompose()

        assert fsm.accumulated_critique == "stories are too broad\n\nstill not granular enough"

    def test_multiple_critiques_accumulate(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        fsm.decomposition_done([_story("001")])

        fsm.critique("first issue")
        fsm.re_decompose()
        fsm.decomposition_done([_story("001")])
        fsm.critique("second issue")
        fsm.re_decompose()

        assert len(fsm.plan.critique_history) == 2
        assert "first issue" in fsm.accumulated_critique
        assert "second issue" in fsm.accumulated_critique


# ---------------------------------------------------------------------------
# FanoutReviewFSM — edit flow
# ---------------------------------------------------------------------------

class TestFanoutReviewFSMEditFlow:
    def test_edit_and_resume(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        fsm.decomposition_done([_story("001")])

        fsm.edit()
        assert fsm.state == FanoutState.EDITING

        fsm.resume_review()
        assert fsm.state == FanoutState.PLAN_READY

    def test_edit_then_accept(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        fsm.decomposition_done([_story("001")])

        fsm.edit()
        fsm.resume_review()
        fsm.accept()
        assert fsm.state == FanoutState.CONFIRMED


# ---------------------------------------------------------------------------
# FanoutReviewFSM — abort and retry
# ---------------------------------------------------------------------------

class TestFanoutReviewFSMAbortRetry:
    def test_abort_from_idle(self):
        fsm = FanoutReviewFSM()
        fsm.abort()
        assert fsm.state == FanoutState.ABORTED

    def test_abort_from_decomposing(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        fsm.abort()
        assert fsm.state == FanoutState.ABORTED

    def test_abort_from_plan_ready(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        fsm.decomposition_done([_story("001")])
        fsm.abort()
        assert fsm.state == FanoutState.ABORTED

    def test_abort_from_executing(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        fsm.decomposition_done([_story("001")])
        fsm.accept()
        fsm.execute()
        fsm.abort()
        assert fsm.state == FanoutState.ABORTED

    def test_abort_from_confirmed(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        fsm.decomposition_done([_story("001")])
        fsm.accept()
        fsm.abort()
        assert fsm.state == FanoutState.ABORTED

    def test_retry_from_aborted(self):
        fsm = FanoutReviewFSM()
        fsm.abort()
        fsm.retry()
        assert fsm.state == FanoutState.IDLE
        assert fsm.plan is None

    def test_retry_from_complete(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        fsm.decomposition_done([_story("001")])
        fsm.accept()
        fsm.execute()
        fsm.story_done("001")
        fsm.all_stories_done()
        fsm.retry()
        assert fsm.state == FanoutState.IDLE
        assert fsm.plan is None

    def test_abort_records_in_history(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        fsm.abort()
        assert fsm.history[-1] == ("abort", FanoutState.DECOMPOSING, FanoutState.ABORTED)


# ---------------------------------------------------------------------------
# FanoutReviewFSM — invalid transitions
# ---------------------------------------------------------------------------

class TestFanoutReviewFSMInvalidTransitions:
    def test_decomposition_done_from_idle_raises(self):
        fsm = FanoutReviewFSM()
        with pytest.raises(FanoutTransitionError) as exc:
            fsm.decomposition_done([_story("001")])
        assert exc.value.current == FanoutState.IDLE
        assert exc.value.action == "decomposition_done"

    def test_accept_from_idle_raises(self):
        fsm = FanoutReviewFSM()
        with pytest.raises(FanoutTransitionError):
            fsm.accept()

    def test_execute_from_idle_raises(self):
        fsm = FanoutReviewFSM()
        with pytest.raises(FanoutTransitionError):
            fsm.execute()

    def test_story_done_from_idle_raises(self):
        fsm = FanoutReviewFSM()
        with pytest.raises(FanoutTransitionError):
            fsm.story_done("001")

    def test_critique_from_idle_raises(self):
        fsm = FanoutReviewFSM()
        with pytest.raises(FanoutTransitionError):
            fsm.critique("bad")

    def test_edit_from_idle_raises(self):
        fsm = FanoutReviewFSM()
        with pytest.raises(FanoutTransitionError):
            fsm.edit()

    def test_re_decompose_from_idle_raises(self):
        fsm = FanoutReviewFSM()
        with pytest.raises(FanoutTransitionError):
            fsm.re_decompose()

    def test_resume_review_from_idle_raises(self):
        fsm = FanoutReviewFSM()
        with pytest.raises(FanoutTransitionError):
            fsm.resume_review()

    def test_transition_error_message_is_helpful(self):
        fsm = FanoutReviewFSM()
        try:
            fsm.accept()
        except FanoutTransitionError as e:
            assert "Cannot 'accept'" in str(e)
            assert "idle" in str(e)
            assert "start" in str(e)
            assert "abort" in str(e)


# ---------------------------------------------------------------------------
# FanoutReviewFSM — decomposition failure
# ---------------------------------------------------------------------------

class TestFanoutReviewFSMDecompositionFailure:
    def test_decomposition_fail_returns_to_idle(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        fsm.decomposition_fail()
        assert fsm.state == FanoutState.IDLE

    def test_can_restart_after_failure(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        fsm.decomposition_fail()
        fsm.start("new task")
        assert fsm.state == FanoutState.DECOMPOSING
        assert fsm.plan.task == "new task"


# ---------------------------------------------------------------------------
# FanoutReviewFSM — queries
# ---------------------------------------------------------------------------

class TestFanoutReviewFSMQueries:
    def test_is_terminal_complete(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        fsm.decomposition_done([_story("001")])
        fsm.accept()
        fsm.execute()
        fsm.story_done("001")
        fsm.all_stories_done()
        assert fsm.is_terminal

    def test_is_terminal_aborted(self):
        fsm = FanoutReviewFSM()
        fsm.abort()
        assert fsm.is_terminal

    def test_is_terminal_false_when_active(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        assert not fsm.is_terminal

    def test_pending_story_ids(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        fsm.decomposition_done([_story("001"), _story("002")])
        assert fsm.pending_story_ids == ["001", "002"]

        fsm.accept()
        fsm.execute()
        fsm.story_done("001")
        assert fsm.pending_story_ids == ["002"]

    def test_ready_story_ids_with_deps(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        fsm.decomposition_done([
            _story("001", "A", deps=[]),
            _story("002", "B", deps=["001"]),
            _story("003", "C", deps=["002"]),
        ])
        assert fsm.ready_story_ids == ["001"]

        fsm.accept()
        fsm.execute()
        fsm.story_done("001")
        assert fsm.ready_story_ids == ["002"]

        fsm.story_done("002")
        assert fsm.ready_story_ids == ["003"]

    def test_blocked_story_ids(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        fsm.decomposition_done([
            _story("001", "A", deps=[]),
            _story("002", "B", deps=["001"]),
        ])
        assert fsm.blocked_story_ids == ["002"]

        fsm.accept()
        fsm.execute()
        fsm.story_done("001")
        assert fsm.blocked_story_ids == []

    def test_story_lookup(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        s = _story("001", "auth")
        fsm.decomposition_done([s])
        assert fsm.story("001") == s
        assert fsm.story("999") is None

    def test_story_lookup_returns_none_before_decomposition(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        assert fsm.story("001") is None

    def test_queries_empty_before_plan(self):
        fsm = FanoutReviewFSM()
        assert fsm.pending_story_ids == []
        assert fsm.ready_story_ids == []
        assert fsm.blocked_story_ids == []
        assert fsm.accumulated_critique == ""


# ---------------------------------------------------------------------------
# FanoutReviewFSM — reset
# ---------------------------------------------------------------------------

class TestFanoutReviewFSMReset:
    def test_reset_clears_everything(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        fsm.decomposition_done([_story("001")])
        fsm.critique("needs work")
        fsm.reset()
        assert fsm.state == FanoutState.IDLE
        assert fsm.plan is None
        assert fsm.history == []


# ---------------------------------------------------------------------------
# FanoutReviewFSM — plan dict roundtrip via from_dict
# ---------------------------------------------------------------------------

class TestFanoutReviewFSMPlanPersistence:
    def test_plan_can_be_restored_from_dict(self):
        fsm = FanoutReviewFSM()
        fsm.start("task")
        fsm.decomposition_done([_story("001", "auth", deps=[], acceptance=["a", "b"])])

        d = fsm.plan.to_dict()
        fsm.reset()

        fsm._plan = FanoutPlan.from_dict(d)
        assert fsm.plan.task == "task"
        assert len(fsm.plan.stories) == 1
        assert fsm.plan.stories[0].id == "001"
        assert fsm.plan.stories[0].acceptance == ["a", "b"]
