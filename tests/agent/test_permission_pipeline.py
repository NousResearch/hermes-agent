# tests/agent/test_permission_pipeline.py
import pytest
import asyncio

from agent.permission_pipeline import (
    PermissionPipeline,
    PermissionResult,
    PermissionLevel,
    is_terminal,
)


class TestPermissionPipeline:
    """Tests for the PermissionPipeline staged approval system."""

    def test_add_stage(self):
        """add_stage() adds a stage to the pipeline."""
        pipeline = PermissionPipeline()

        def checker(cmd, ctx):
            return PermissionResult(PermissionLevel.APPROVE, True, "ok")

        pipeline.add_stage("test_stage", checker, priority=50)

        assert ("test_stage",) in [(name,) for prio, name in pipeline.stages]

    def test_add_stage_priority_ordering(self):
        """Stages are sorted by priority (lower number = runs first)."""
        pipeline = PermissionPipeline()

        def checker(cmd, ctx):
            return PermissionResult(PermissionLevel.APPROVE, True, "ok")

        pipeline.add_stage("stage_100", checker, priority=100)
        pipeline.add_stage("stage_10", checker, priority=10)
        pipeline.add_stage("stage_50", checker, priority=50)

        stages = pipeline.stages
        priorities = [prio for prio, _ in stages]

        # Should be sorted ascending
        assert priorities == [10, 50, 100]
        names = [name for _, name in stages]
        assert names == ["stage_10", "stage_50", "stage_100"]

    def test_block_is_terminal(self):
        """BLOCK result returns immediately without checking subsequent stages."""
        pipeline = PermissionPipeline()
        call_order = []

        def blocker(cmd, ctx):
            call_order.append("blocker")
            return PermissionResult(PermissionLevel.BLOCK, False, "blocked")

        def never_called(cmd, ctx):
            call_order.append("never_called")
            return PermissionResult(PermissionLevel.APPROVE, True, "ok")

        pipeline.add_stage("blocker", blocker, priority=10)
        pipeline.add_stage("never_called", never_called, priority=50)

        result = asyncio.run(pipeline.check("any command"))

        assert result.level == PermissionLevel.BLOCK
        assert result.approved is False
        assert call_order == ["blocker"]  # Second stage never ran

    def test_skip_is_terminal(self):
        """SKIP result returns immediately without checking subsequent stages."""
        pipeline = PermissionPipeline()
        call_order = []

        def skipper(cmd, ctx):
            call_order.append("skipper")
            return PermissionResult(PermissionLevel.SKIP, True, "skipped")

        def never_called(cmd, ctx):
            call_order.append("never_called")
            return PermissionResult(PermissionLevel.APPROVE, True, "ok")

        pipeline.add_stage("skipper", skipper, priority=10)
        pipeline.add_stage("never_called", never_called, priority=50)

        result = asyncio.run(pipeline.check("any command"))

        assert result.level == PermissionLevel.SKIP
        assert result.approved is True
        assert call_order == ["skipper"]  # Second stage never ran

    def test_review_is_terminal(self):
        """REVIEW result returns immediately without checking subsequent stages."""
        pipeline = PermissionPipeline()
        call_order = []

        def reviewer(cmd, ctx):
            call_order.append("reviewer")
            return PermissionResult(PermissionLevel.REVIEW, False, "needs review")

        def never_called(cmd, ctx):
            call_order.append("never_called")
            return PermissionResult(PermissionLevel.APPROVE, True, "ok")

        pipeline.add_stage("reviewer", reviewer, priority=10)
        pipeline.add_stage("never_called", never_called, priority=50)

        result = asyncio.run(pipeline.check("any command"))

        assert result.level == PermissionLevel.REVIEW
        assert result.approved is False
        assert call_order == ["reviewer"]  # Second stage never ran

    def test_approve_continues_to_next_stage(self):
        """APPROVE result continues to the next stage."""
        pipeline = PermissionPipeline()
        call_order = []

        def approver1(cmd, ctx):
            call_order.append("approver1")
            return PermissionResult(PermissionLevel.APPROVE, True, "approved by 1")

        def approver2(cmd, ctx):
            call_order.append("approver2")
            return PermissionResult(PermissionLevel.APPROVE, True, "approved by 2")

        pipeline.add_stage("approver1", approver1, priority=10)
        pipeline.add_stage("approver2", approver2, priority=50)

        result = asyncio.run(pipeline.check("any command"))

        assert result.level == PermissionLevel.APPROVE
        assert call_order == ["approver1", "approver2"]

    def test_empty_pipeline_returns_explicit_approve(self):
        """Empty pipeline returns explicit APPROVE with default reason."""
        pipeline = PermissionPipeline()

        result = asyncio.run(pipeline.check("any command"))

        assert result.level == PermissionLevel.APPROVE
        assert result.approved is True
        assert "Default approval" in result.reason

    def test_all_approve_returns_last_approve_result(self):
        """When all stages return APPROVE, returns the last APPROVE result."""
        pipeline = PermissionPipeline()

        def checker1(cmd, ctx):
            return PermissionResult(PermissionLevel.APPROVE, True, "first approval")

        def checker2(cmd, ctx):
            return PermissionResult(PermissionLevel.APPROVE, True, "second approval")

        pipeline.add_stage("checker1", checker1, priority=10)
        pipeline.add_stage("checker2", checker2, priority=50)

        result = asyncio.run(pipeline.check("any command"))

        assert result.level == PermissionLevel.APPROVE
        assert result.reason == "second approval"

    def test_last_result_reason_preserved(self):
        """Last APPROVE result's reason is preserved in the final result."""
        pipeline = PermissionPipeline()
        reasons = []

        def tracker1(cmd, ctx):
            reasons.append("tracker1")
            return PermissionResult(PermissionLevel.APPROVE, True, "reason1")

        def tracker2(cmd, ctx):
            reasons.append("tracker2")
            return PermissionResult(PermissionLevel.APPROVE, True, "reason2")

        pipeline.add_stage("tracker1", tracker1, priority=10)
        pipeline.add_stage("tracker2", tracker2, priority=50)

        result = asyncio.run(pipeline.check("any command"))

        assert result.reason == "reason2"
        assert reasons == ["tracker1", "tracker2"]

    def test_priority_lower_number_runs_first(self):
        """Lower priority number means the stage runs first."""
        pipeline = PermissionPipeline()
        call_order = []

        def stage(prio):
            def checker(cmd, ctx):
                call_order.append(prio)
                return PermissionResult(PermissionLevel.APPROVE, True, f"prio {prio}")
            return checker

        # Add in random order
        pipeline.add_stage("p100", stage(100), priority=100)
        pipeline.add_stage("p10", stage(10), priority=10)
        pipeline.add_stage("p50", stage(50), priority=50)

        asyncio.run(pipeline.check("cmd"))

        # Should execute in priority order
        assert call_order == [10, 50, 100]

    def test_is_terminal_function(self):
        """is_terminal() returns True for BLOCK, SKIP, REVIEW."""
        block = PermissionResult(PermissionLevel.BLOCK, False, "blocked")
        skip = PermissionResult(PermissionLevel.SKIP, True, "skipped")
        review = PermissionResult(PermissionLevel.REVIEW, False, "review")
        approve = PermissionResult(PermissionLevel.APPROVE, True, "ok")

        assert is_terminal(block) is True
        assert is_terminal(skip) is True
        assert is_terminal(review) is True
        assert is_terminal(approve) is False

    def test_checker_exception_treated_as_approve(self):
        """If a checker raises an exception, it's treated as APPROVE (fail-open)."""
        pipeline = PermissionPipeline()
        call_order = []

        def bad_checker(cmd, ctx):
            call_order.append("bad")
            raise RuntimeError("checker error")

        def next_checker(cmd, ctx):
            call_order.append("next")
            return PermissionResult(PermissionLevel.APPROVE, True, "ok")

        pipeline.add_stage("bad", bad_checker, priority=10)
        pipeline.add_stage("next", next_checker, priority=50)

        result = asyncio.run(pipeline.check("any command"))

        # Fail-open: exception treated as APPROVE, pipeline continues to next checker
        assert result.level == PermissionLevel.APPROVE
        # Pipeline continued to next checker, which returned "ok"
        assert result.reason == "ok"
        assert call_order == ["bad", "next"]

    def test_context_passed_to_checker(self):
        """Context dict is passed to each checker."""
        pipeline = PermissionPipeline()
        received_context = {}

        def checker(cmd, ctx):
            nonlocal received_context
            received_context = ctx
            return PermissionResult(PermissionLevel.APPROVE, True, "ok")

        pipeline.add_stage("test", checker, priority=10)

        test_context = {"session_key": "abc123", "env_type": "test"}
        asyncio.run(pipeline.check("any command", context=test_context))

        assert received_context == test_context

    def test_command_passed_to_checker(self):
        """Command string is passed to each checker."""
        pipeline = PermissionPipeline()
        received_command = None

        def checker(cmd, ctx):
            nonlocal received_command
            received_command = cmd
            return PermissionResult(PermissionLevel.APPROVE, True, "ok")

        pipeline.add_stage("test", checker, priority=10)

        asyncio.run(pipeline.check("rm -rf /tmp/test"))

        assert received_command == "rm -rf /tmp/test"

    def test_remove_stage(self):
        """remove_stage() removes a stage by name."""
        pipeline = PermissionPipeline()

        def checker(cmd, ctx):
            return PermissionResult(PermissionLevel.APPROVE, True, "ok")

        pipeline.add_stage("to_remove", checker, priority=10)
        pipeline.add_stage("to_keep", checker, priority=50)

        removed = pipeline.remove_stage("to_remove")
        assert removed is True

        stages = pipeline.stages
        names = [name for _, name in stages]
        assert "to_remove" not in names
        assert "to_keep" in names

    def test_remove_stage_returns_false_if_not_found(self):
        """remove_stage() returns False if stage name not found."""
        pipeline = PermissionPipeline()
        result = pipeline.remove_stage("nonexistent")
        assert result is False

    def test_clear_removes_all_stages(self):
        """clear() removes all stages from the pipeline."""
        pipeline = PermissionPipeline()

        def checker(cmd, ctx):
            return PermissionResult(PermissionLevel.APPROVE, True, "ok")

        pipeline.add_stage("s1", checker, priority=10)
        pipeline.add_stage("s2", checker, priority=20)

        pipeline.clear()

        assert pipeline.stages == []

    def test_async_checker_supported(self):
        """Async checker functions are properly awaited."""
        pipeline = PermissionPipeline()
        call_order = []

        async def async_checker(cmd, ctx):
            call_order.append("async")
            await asyncio.sleep(0)  # yield
            return PermissionResult(PermissionLevel.APPROVE, True, "async ok")

        def sync_checker(cmd, ctx):
            call_order.append("sync")
            return PermissionResult(PermissionLevel.APPROVE, True, "sync ok")

        pipeline.add_stage("async", async_checker, priority=10)
        pipeline.add_stage("sync", sync_checker, priority=50)

        result = asyncio.run(pipeline.check("cmd"))

        assert result.level == PermissionLevel.APPROVE
        assert call_order == ["async", "sync"]
