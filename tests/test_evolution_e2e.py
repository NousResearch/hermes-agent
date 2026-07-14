"""End-to-end tests for the Hermes Agent Evolution Engine (HAEE).

Tests the full evolution pipeline: task definition → trajectory capture →
evaluation → failure analysis → improvement proposal → regression gate.
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.evolution import (
    EvalCheck,
    EvalResult,
    EvolutionConfig,
    EvolutionManager,
    EvolutionStore,
    FailureAnalysis,
    FailureAnalyzer,
    FailureCategory,
    FailureFinding,
    GateResult,
    GateVerdict,
    HarnessVariant,
    ImprovementActionType,
    ImprovementProposal,
    ImprovementProposer,
    RegressionGate,
    SuccessCriterion,
    SuccessCriterionType,
    TaskDefinition,
    TaskEvaluator,
    TaskStatus,
    Trajectory,
    TrajectoryCollector,
    VariantManager,
    delete_task,
    get_evolution_store,
    list_tasks,
    load_task,
    quick_evaluate,
    save_task,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_hermes_home():
    """Create a temporary HERMES_HOME for isolated testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        home = Path(tmpdir) / ".hermes"
        home.mkdir(parents=True)
        with patch("hermes_constants.get_hermes_home", return_value=home):
            yield home


@pytest.fixture
def sample_task():
    """A sample task definition for testing."""
    return TaskDefinition(
        name="hello-world",
        description='Write "hello world" to a file and verify it exists',
        success_criteria=[
            SuccessCriterion(
                type=SuccessCriterionType.FILE_EXISTS,
                path="/tmp/hello.txt",
            ),
            SuccessCriterion(
                type=SuccessCriterionType.CONTENT_MATCH,
                path="/tmp/hello.txt",
                pattern=r"hello.*world",
            ),
        ],
        domain="test",
        complexity=1,
        timeout_seconds=30,
        max_turns=5,
    )


@pytest.fixture
def sample_trajectory():
    """A sample trajectory for testing."""
    traj = Trajectory(
        task_name="hello-world",
        run_id="evo_test123",
        status="completed",
        total_turns=3,
        total_tool_calls=2,
        total_tokens=500,
    )
    from agent.evolution.trajectory_collector import TraceStep

    traj.steps = [
        TraceStep(
            step=1,
            type="model_call",
            summary="I will write hello world to a file",
            extra={"model": "test-model", "tool_calls": ["write_file"]},
        ),
        TraceStep(
            step=2,
            type="tool_execution",
            status="success",
            summary="Wrote 'hello world' to /tmp/hello.txt",
            extra={"tool": "write_file"},
        ),
        TraceStep(
            step=3,
            type="model_call",
            summary="Task complete — file has been written",
            extra={"model": "test-model", "tool_calls": []},
        ),
    ]
    return traj


@pytest.fixture
def evolution_config():
    """Evolution config with all features enabled."""
    return EvolutionConfig(
        enabled=True,
        mode="on_failure",
        max_iterations=3,
        regression_enabled=True,
        max_regression_tasks=10,
    )


# ---------------------------------------------------------------------------
# Task Definition Tests
# ---------------------------------------------------------------------------


class TestTaskDefinition:
    def test_create_and_validate(self):
        task = TaskDefinition(
            name="test-task",
            description="A test",
            success_criteria=[
                SuccessCriterion(type=SuccessCriterionType.TEST_PASS, command="true"),
            ],
        )
        errors = task.validate()
        assert not errors, f"Unexpected errors: {errors}"

    def test_validate_missing_name(self):
        task = TaskDefinition(name="", description="test", success_criteria=[])
        errors = task.validate()
        assert any("name" in e.lower() for e in errors)

    def test_validate_missing_criteria(self):
        task = TaskDefinition(name="test", description="test", success_criteria=[])
        errors = task.validate()
        assert any("criterion" in e.lower() or "criteria" in e.lower() for e in errors)

    def test_validate_bad_complexity(self):
        task = TaskDefinition(
            name="test",
            description="test",
            complexity=99,
            success_criteria=[SuccessCriterion(type=SuccessCriterionType.TEST_PASS, command="true")],
        )
        errors = task.validate()
        assert any("complexity" in e.lower() for e in errors)

    def test_yaml_roundtrip(self, temp_hermes_home, sample_task):
        path = save_task(sample_task)
        assert path.exists()
        loaded = load_task("hello-world")
        assert loaded is not None
        assert loaded.name == sample_task.name
        assert len(loaded.success_criteria) == len(sample_task.success_criteria)
        delete_task("hello-world")

    def test_list_tasks(self, temp_hermes_home, sample_task):
        save_task(sample_task)
        tasks = list_tasks()
        assert any(t.name == "hello-world" for t in tasks)
        delete_task("hello-world")

    def test_to_dict_from_dict(self, sample_task):
        d = sample_task.to_dict()
        task2 = TaskDefinition.from_dict(d)
        assert task2.name == sample_task.name
        assert len(task2.success_criteria) == len(sample_task.success_criteria)


# ---------------------------------------------------------------------------
# Evaluator Tests
# ---------------------------------------------------------------------------


class TestEvaluator:
    def test_file_exists_pass(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test")
            path = f.name

        try:
            task = TaskDefinition(
                name="test",
                description="test",
                success_criteria=[
                    SuccessCriterion(type=SuccessCriterionType.FILE_EXISTS, path=path),
                ],
            )
            evaluator = TaskEvaluator()
            result = evaluator.evaluate(task, Trajectory())
            assert result.passed
            assert result.score == 1.0
        finally:
            os.unlink(path)

    def test_file_exists_fail(self):
        task = TaskDefinition(
            name="test",
            description="test",
            success_criteria=[
                SuccessCriterion(
                    type=SuccessCriterionType.FILE_EXISTS,
                    path="/nonexistent/path/12345",
                ),
            ],
        )
        evaluator = TaskEvaluator()
        result = evaluator.evaluate(task, Trajectory())
        assert not result.passed

    def test_test_pass_success(self):
        task = TaskDefinition(
            name="test",
            description="test",
            success_criteria=[
                SuccessCriterion(type=SuccessCriterionType.TEST_PASS, command="echo ok"),
            ],
        )
        evaluator = TaskEvaluator()
        result = evaluator.evaluate(task, Trajectory())
        assert result.passed

    def test_test_pass_fail(self):
        task = TaskDefinition(
            name="test",
            description="test",
            success_criteria=[
                SuccessCriterion(type=SuccessCriterionType.TEST_PASS, command="exit 1"),
            ],
        )
        evaluator = TaskEvaluator()
        result = evaluator.evaluate(task, Trajectory())
        assert not result.passed

    def test_content_match(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world\nfoo bar\n")
            path = f.name

        try:
            task = TaskDefinition(
                name="test",
                description="test",
                success_criteria=[
                    SuccessCriterion(
                        type=SuccessCriterionType.CONTENT_MATCH,
                        path=path,
                        pattern=r"hello.*world",
                    ),
                ],
            )
            evaluator = TaskEvaluator()
            result = evaluator.evaluate(task, Trajectory())
            assert result.passed
        finally:
            os.unlink(path)

    def test_composite_scoring(self):
        task = TaskDefinition(
            name="test",
            description="test",
            success_criteria=[
                SuccessCriterion(type=SuccessCriterionType.TEST_PASS, command="true", weight=0.5),
                SuccessCriterion(type=SuccessCriterionType.TEST_PASS, command="false", weight=0.5),
            ],
        )
        evaluator = TaskEvaluator()
        result = evaluator.evaluate(task, Trajectory())
        assert not result.passed  # One criterion fails
        assert result.score == 0.5  # 50% weighted score

    def test_quick_evaluate(self):
        task = TaskDefinition(
            name="test",
            description="test",
            success_criteria=[
                SuccessCriterion(type=SuccessCriterionType.TEST_PASS, command="true"),
                SuccessCriterion(type=SuccessCriterionType.LLM_JUDGE, rubric="Is it good?"),
            ],
        )
        result = quick_evaluate(task)
        assert result.passed  # LLM judge is skipped, only test_pass runs


# ---------------------------------------------------------------------------
# Evolution Store Tests
# ---------------------------------------------------------------------------


class TestEvolutionStore:
    def test_create_and_get_run(self, temp_hermes_home):
        store = EvolutionStore()
        try:
            run_id = store.create_run(task_name="test", task_domain="coding")
            assert run_id.startswith("evo_")

            run = store.get_run(run_id)
            assert run is not None
            assert run["task_name"] == "test"
            assert run["status"] == "pending"
        finally:
            store.close()

    def test_update_run_status(self, temp_hermes_home):
        store = EvolutionStore()
        try:
            run_id = store.create_run(task_name="test")
            store.update_run_status(run_id, "succeeded", final_score=0.95)
            run = store.get_run(run_id)
            assert run["status"] == "succeeded"
            assert run["final_score"] == 0.95
        finally:
            store.close()

    def test_list_runs(self, temp_hermes_home):
        store = EvolutionStore()
        try:
            store.create_run(task_name="task-a")
            store.create_run(task_name="task-b")
            runs = store.list_runs()
            assert len(runs) >= 2
        finally:
            store.close()

    def test_iterations(self, temp_hermes_home):
        store = EvolutionStore()
        try:
            run_id = store.create_run(task_name="test")
            store.add_iteration(run_id, 1, "attempting", score=0.5)
            store.add_iteration(run_id, 2, "evaluating", score=0.8)
            iters = store.get_iterations(run_id)
            assert len(iters) == 2
            assert iters[0]["iteration_num"] == 1
            assert iters[1]["iteration_num"] == 2
        finally:
            store.close()

    def test_regression_baseline(self, temp_hermes_home):
        store = EvolutionStore()
        try:
            store.set_baseline("task-x", 0.95, task_domain="coding")
            baseline = store.get_baseline("task-x")
            assert baseline is not None
            assert baseline["baseline_score"] == 0.95
            store.delete_baseline("task-x")
            assert store.get_baseline("task-x") is None
        finally:
            store.close()

    def test_singleton(self, temp_hermes_home):
        store1 = get_evolution_store()
        store2 = get_evolution_store()
        assert store1 is store2
        store1.close()


# ---------------------------------------------------------------------------
# Trajectory Collector Tests
# ---------------------------------------------------------------------------


class TestTrajectoryCollector:
    def test_basic_collection(self):
        collector = TrajectoryCollector(task_name="test", run_id="evo_123")
        collector.start()
        assert collector.is_active

        collector.record_model_call(
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            tool_calls=["read_file"],
            summary="Testing",
        )
        collector.record_tool_call(
            tool_name="read_file",
            tool_args={"path": "/tmp/x"},
            status="success",
            result_summary="Read 10 lines",
        )

        traj = collector.stop("completed")
        assert traj.total_turns == 1
        assert traj.total_tool_calls == 1
        assert len(traj.steps) == 2

    def test_save_and_load(self, temp_hermes_home):
        collector = TrajectoryCollector(task_name="test-save", run_id="evo_save")
        collector.start()
        collector.record_model_call(summary="Test", model="m")
        collector.stop()

        path = collector.save()
        assert path.exists()
        assert path.suffix == ".yaml"

    def test_error_recording(self):
        collector = TrajectoryCollector(task_name="test", run_id="evo_err")
        collector.start()
        collector.record_tool_call(
            tool_name="terminal",
            status="error",
            error_message="Command not found",
        )
        collector.record_error("Something went wrong")
        traj = collector.stop()
        assert len(traj.errors) >= 1

    def test_max_steps_truncation(self):
        collector = TrajectoryCollector(task_name="test", run_id="evo_max", max_steps=3)
        collector.start()
        for i in range(10):
            collector.record_model_call(summary=f"Step {i}")
        traj = collector.stop()
        assert len(traj.steps) <= 3


# ---------------------------------------------------------------------------
# Failure Analyzer Tests
# ---------------------------------------------------------------------------


class TestFailureAnalyzer:
    def test_timeout_detection(self):
        analyzer = FailureAnalyzer()
        traj = Trajectory(status="timeout", total_turns=30, total_tool_calls=2)
        findings = analyzer._rule_based_analysis(traj, EvalResult(passed=False, score=0.0))
        assert any(f.category == FailureCategory.TIMEOUT for f in findings)

    def test_premature_completion_detection(self):
        analyzer = FailureAnalyzer()
        traj = Trajectory(status="completed", total_turns=5, total_tool_calls=3)
        findings = analyzer._rule_based_analysis(traj, EvalResult(passed=False, score=0.33))
        assert any(f.category == FailureCategory.PREMATURE_COMPLETION for f in findings)

    def test_execution_error_detection(self):
        analyzer = FailureAnalyzer()
        traj = Trajectory(status="failed", total_turns=5, total_tool_calls=3)
        traj.errors = [{"step": 3, "tool": "terminal", "message": "permission denied"}]
        findings = analyzer._rule_based_analysis(traj, EvalResult(passed=False, score=0.0))
        assert any(f.category == FailureCategory.EXECUTION_ERROR for f in findings)

    def test_analysis_serialization(self):
        analysis = FailureAnalysis(
            run_id="evo_1",
            task_name="test",
            overall_score=0.5,
            findings=[
                FailureFinding(
                    category=FailureCategory.TIMEOUT,
                    confidence=0.95,
                    description="Task timed out",
                    evidence="Status: timeout",
                ),
            ],
        )
        d = analysis.to_dict()
        assert d["findings"][0]["category"] == "timeout"
        json_str = analysis.to_json()
        assert "timeout" in json_str

        # Roundtrip
        analysis2 = FailureAnalysis.from_dict(d)
        assert analysis2.findings[0].category == FailureCategory.TIMEOUT


# ---------------------------------------------------------------------------
# Improvement Proposer Tests
# ---------------------------------------------------------------------------


class TestImprovementProposer:
    def test_rule_based_proposals(self):
        proposer = ImprovementProposer()
        analysis = FailureAnalysis(
            run_id="evo_1",
            task_name="test",
            findings=[
                FailureFinding(
                    category=FailureCategory.MISSING_TOOL,
                    confidence=0.8,
                    description="Agent needed a Kubernetes tool",
                    evidence="Trace shows kubectl not available",
                    suggested_fix_category="tool",
                ),
                FailureFinding(
                    category=FailureCategory.PREMATURE_COMPLETION,
                    confidence=0.85,
                    description="Agent didn't verify",
                    evidence="No verification step in trajectory",
                    suggested_fix_category="prompt",
                ),
            ],
        )
        proposals = proposer._rule_based_proposals(analysis)
        assert len(proposals) >= 2
        # Both findings generate SKILL_CREATE proposals with real content
        types = {p.action_type for p in proposals}
        assert ImprovementActionType.SKILL_CREATE in types
        # Verify proposals have real content, not empty placeholders
        for p in proposals:
            assert len(p.content) > 200, f"Proposal {p.target} has no real content ({len(p.content)} bytes)"
            assert "---" in p.content, f"Proposal {p.target} missing YAML frontmatter"

    def test_deduplication(self):
        proposer = ImprovementProposer()
        analysis = FailureAnalysis(
            run_id="test", task_name="test-task",
            findings=[
                FailureFinding(category=FailureCategory.MISSING_TOOL, confidence=0.8,
                              description="Need tool", evidence="No kubectl found",
                              suggested_fix_category="tool"),
            ],
        )
        proposals = proposer._rule_based_proposals(analysis)
        # Proposer should not create duplicate proposals for the same (type, target)
        targets = {(p.action_type, p.target) for p in proposals}
        assert len(targets) == len(proposals)  # All unique
        # Proposals must have real content
        for p in proposals:
            assert len(p.content) > 100, f"Proposal {p.target} has empty content"

    def test_proposal_serialization(self):
        proposal = ImprovementProposal(
            action_type=ImprovementActionType.SKILL_CREATE,
            target="test-skill",
            description="Test",
            rationale="Because",
            content="---\nname: test\n---\n\n# Test",
            confidence=0.7,
        )
        d = proposal.to_dict()
        p2 = ImprovementProposal.from_dict(d)
        assert p2.target == proposal.target
        assert p2.action_type == proposal.action_type


# ---------------------------------------------------------------------------
# Regression Gate Tests
# ---------------------------------------------------------------------------


class TestRegressionGate:
    def test_accept_valid_skill(self):
        gate = RegressionGate()
        proposal = ImprovementProposal(
            action_type=ImprovementActionType.SKILL_CREATE,
            target="valid-skill",
            description="A valid skill proposal",
            rationale="Testing",
            content="---\nname: valid-skill\ndescription: Valid skill.\n---\n\n# Valid Skill\n\nContent.",
        )
        result = gate.evaluate(proposal)
        assert result.verdict == GateVerdict.ACCEPT
        assert len(result.failures) == 0

    def test_reject_missing_target(self):
        gate = RegressionGate()
        proposal = ImprovementProposal(
            action_type=ImprovementActionType.SKILL_CREATE,
            target="",
            description="",
            rationale="",
        )
        result = gate.evaluate(proposal)
        assert result.verdict == GateVerdict.REJECT

    def test_reject_bad_python_syntax(self):
        gate = RegressionGate()
        proposal = ImprovementProposal(
            action_type=ImprovementActionType.TOOL_CREATE,
            target="bad-tool",
            description="Bad Python",
            rationale="Testing",
            content="def broken(:\n    return",
        )
        result = gate.evaluate(proposal)
        assert result.verdict == GateVerdict.REJECT
        assert any("syntax" in f.lower() for f in result.failures)

    def test_reject_skill_without_frontmatter(self):
        gate = RegressionGate()
        proposal = ImprovementProposal(
            action_type=ImprovementActionType.SKILL_CREATE,
            target="no-frontmatter",
            description="Missing frontmatter",
            rationale="Testing",
            content="# Just a heading\n\nNo frontmatter here.",
        )
        result = gate.evaluate(proposal)
        assert result.verdict == GateVerdict.REJECT

    def test_reject_oversized_skill(self):
        gate = RegressionGate()
        proposal = ImprovementProposal(
            action_type=ImprovementActionType.SKILL_CREATE,
            target="huge-skill",
            description="Too big",
            rationale="Testing",
            content="---\nname: huge\ndescription: Big.\n---\n\n" + ("x" * 20000),
        )
        result = gate.evaluate(proposal)
        # Size warnings go to NEEDS_REVIEW (not hard REJECT) — 15KB limit exceeded
        assert result.verdict == GateVerdict.NEEDS_REVIEW
        assert len(result.warnings) > 0

    def test_accept_valid_tool_code(self):
        gate = RegressionGate()
        proposal = ImprovementProposal(
            action_type=ImprovementActionType.TOOL_CREATE,
            target="valid-tool",
            description="Valid tool",
            rationale="Testing",
            content="def my_tool(path: str) -> str:\n    \"\"\"A valid tool.\"\"\"\n    return open(path).read()",
        )
        result = gate.evaluate(proposal)
        # Valid Python but import might fail in gate env — should at least pass manifest + content
        assert result.verdict in (GateVerdict.ACCEPT, GateVerdict.NEEDS_REVIEW)

    def test_reject_empty_patch(self):
        gate = RegressionGate()
        proposal = ImprovementProposal(
            action_type=ImprovementActionType.SKILL_PATCH,
            target="some-skill",
            description="Empty patch",
            rationale="Testing",
            old_string="",
            new_string="",
        )
        result = gate.evaluate(proposal)
        assert result.verdict == GateVerdict.REJECT


# ---------------------------------------------------------------------------
# Harness Variants Tests
# ---------------------------------------------------------------------------


class TestHarnessVariants:
    def test_default_variant_exists(self):
        vm = VariantManager()
        assert len(vm.active_variants) == 1
        assert vm.active_variant.name == "default"

    def test_fork_variant(self):
        vm = VariantManager()
        parent = vm.active_variant
        child = vm.fork_variant(parent, "test-change", name="experimental")
        assert len(vm.active_variants) == 2
        assert child.parent_variant == parent.variant_id
        assert child.name == "experimental"

    def test_max_variants(self):
        vm = VariantManager(max_variants=2)
        parent = vm.active_variant
        vm.fork_variant(parent, "change-1", name="v1")
        assert len(vm.active_variants) == 2
        vm.fork_variant(parent, "change-2", name="v2")
        # Should have retired the lowest performer
        assert len(vm.active_variants) == 2

    def test_retire_variant(self):
        vm = VariantManager()
        child = vm.fork_variant(vm.active_variant, "change", name="to-retire")
        assert vm.retire_variant(child.variant_id)
        assert not child.is_active
        assert len(vm.active_variants) == 1

    def test_cannot_retire_default(self):
        vm = VariantManager()
        default = vm.active_variant
        assert not vm.retire_variant(default.variant_id)

    def test_task_routing(self):
        vm = VariantManager()
        parent = vm.active_variant
        parent.record_result("task-a", 0.8, True)
        child = vm.fork_variant(parent, "change", name="forked")
        child.record_result("task-a", 0.95, True)

        routed = vm.route_task("task-a")
        assert routed.variant_id == child.variant_id  # Forked has better score

        routed_b = vm.route_task("task-b")
        assert routed_b.variant_id == parent.variant_id  # Default for unseen task

    def test_persistence(self, temp_hermes_home):
        vm = VariantManager()
        vm.fork_variant(vm.active_variant, "change", name="saved")
        path = vm.save()
        assert path.exists()

        vm2 = VariantManager.load(path)
        assert len(vm2.active_variants) == 2


# ---------------------------------------------------------------------------
# Evolution Manager Integration Tests
# ---------------------------------------------------------------------------


class TestEvolutionManager:
    def test_initialize_and_shutdown(self, temp_hermes_home):
        config = EvolutionConfig(enabled=True, mode="on_failure", max_iterations=3)
        mgr = EvolutionManager()
        mgr.initialize(session_id="test-session", config=config)
        assert mgr.is_enabled
        mgr.shutdown()

    def test_disabled_when_config_says_no(self, temp_hermes_home):
        config = EvolutionConfig(enabled=False)
        mgr = EvolutionManager()
        mgr.initialize(session_id="test", config=config)
        assert not mgr.is_enabled

    def test_full_task_lifecycle(self, temp_hermes_home):
        config = EvolutionConfig(enabled=True, max_iterations=3)
        mgr = EvolutionManager()
        mgr.initialize(session_id="test-lifecycle", config=config)

        task = TaskDefinition(
            name="lifecycle-test",
            description="Test full lifecycle",
            success_criteria=[
                SuccessCriterion(type=SuccessCriterionType.TEST_PASS, command="true"),
            ],
        )

        # Start
        run = mgr.start_task(task)
        assert run.status == TaskStatus.PENDING  # Pending until collector starts
        assert run.collector.is_active

        # Simulate agent execution
        run.collector.record_model_call(
            model="test", input_tokens=100, output_tokens=50, summary="Working"
        )
        run.collector.record_tool_call(
            tool_name="terminal", status="success", result_summary="ok"
        )
        run.trajectory = run.collector.stop()

        # Evaluate
        result = mgr.evaluate(run)
        assert result.passed
        assert run.status == TaskStatus.SUCCEEDED

        # End task to persist
        mgr.end_task(run)

        # Verify store
        stored = mgr.get_run_status(run.run_id)
        assert stored is not None
        assert stored["status"] == "succeeded"

        mgr.shutdown()

    def test_failed_task_triggers_analysis(self, temp_hermes_home):
        config = EvolutionConfig(enabled=True, max_iterations=3)
        mgr = EvolutionManager()
        mgr.initialize(session_id="test-fail", config=config)

        task = TaskDefinition(
            name="fail-test",
            description="This will fail",
            success_criteria=[
                SuccessCriterion(
                    type=SuccessCriterionType.FILE_EXISTS,
                    path="/definitely/does/not/exist.txt",
                ),
            ],
        )

        run = mgr.start_task(task)
        run.collector.record_model_call(summary="I'll create the file")
        run.collector.record_tool_call(tool_name="write_file", status="error",
                                        error_message="Permission denied")
        run.trajectory = run.collector.stop()

        result = mgr.evaluate(run)
        assert not result.passed
        # After evaluation of a failed task, status transitions to FAILED
        # (evaluate sets FAILED if not passed, SUCCEEDED if passed)
        assert run.status in (TaskStatus.FAILED, TaskStatus.EVALUATING)

        # Analysis should have findings
        iterations = mgr.get_run_iterations(run.run_id)
        assert len(iterations) >= 1

        mgr.shutdown()

    def test_list_runs(self, temp_hermes_home):
        config = EvolutionConfig(enabled=True)
        mgr = EvolutionManager()
        mgr.initialize(session_id="test-list", config=config)

        task = TaskDefinition(
            name="list-test",
            description="Test",
            success_criteria=[SuccessCriterion(type=SuccessCriterionType.TEST_PASS, command="true")],
        )
        run = mgr.start_task(task)
        run.collector.record_model_call(summary="Done")
        run.trajectory = run.collector.stop()
        mgr.evaluate(run)

        runs = mgr.list_runs(limit=5)
        assert len(runs) >= 1

        mgr.shutdown()

    def test_variant_stats(self, temp_hermes_home):
        config = EvolutionConfig(enabled=True)
        mgr = EvolutionManager()
        mgr.initialize(session_id="test-variants", config=config)

        stats = mgr.get_variant_stats()
        assert len(stats) >= 1
        assert stats[0]["name"] == "default"

        mgr.shutdown()


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


class TestEvolutionConfig:
    def test_defaults(self):
        config = EvolutionConfig()
        assert not config.enabled
        assert config.mode == "on_failure"
        assert config.max_iterations == 5

    def test_validate_valid(self):
        config = EvolutionConfig()
        errors = config.validate()
        assert not errors

    def test_validate_invalid_mode(self):
        config = EvolutionConfig(mode="invalid_mode")
        errors = config.validate()
        assert any("mode" in e.lower() for e in errors)

    def test_needs_approval(self):
        config = EvolutionConfig()
        assert config.needs_approval("tool_create")
        assert not config.needs_approval("skill_create")
        assert not config.needs_approval("skill_patch")
        # Unknown actions default to requiring approval
        assert config.needs_approval("unknown_action")

    def test_from_config_dict(self):
        config = EvolutionConfig.from_config({
            "evolution": {
                "enabled": True,
                "mode": "continuous",
                "max_iterations": 10,
            }
        })
        assert config.enabled
        assert config.mode == "continuous"
        assert config.max_iterations == 10


# ---------------------------------------------------------------------------
# Serialization / Roundtrip Tests
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_trajectory_json_roundtrip(self, sample_trajectory):
        json_str = sample_trajectory.to_json()
        data = json.loads(json_str)
        assert data["task_name"] == "hello-world"
        assert len(data["steps"]) == 3

    def test_task_definition_serialization(self, sample_task):
        d = sample_task.to_dict()
        assert d["name"] == "hello-world"
        task2 = TaskDefinition.from_dict(d)
        assert task2.name == sample_task.name
