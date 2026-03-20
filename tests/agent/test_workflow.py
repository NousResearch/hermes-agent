"""Tests for the workflow recorder, runner, and storage."""

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.workflow import (
    Workflow,
    WorkflowRecorder,
    WorkflowRunner,
    WorkflowStep,
    delete_workflow,
    format_workflow_detail,
    format_workflow_list,
    list_workflows,
    load_workflow,
    save_workflow,
    _sanitize_name,
)


@pytest.fixture
def workflows_dir(tmp_path):
    """Use a temp directory for workflow storage."""
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir()
    with patch("agent.workflow._get_workflows_dir", return_value=wf_dir):
        yield wf_dir


# ---------------------------------------------------------------------------
# Name sanitization
# ---------------------------------------------------------------------------

class TestSanitizeName:
    def test_basic(self):
        assert _sanitize_name("deploy-check") == "deploy-check"

    def test_spaces_to_dashes(self):
        assert _sanitize_name("my workflow") == "my-workflow"

    def test_special_chars(self):
        assert _sanitize_name("test@#$%!") == "test"

    def test_empty(self):
        assert _sanitize_name("") == "unnamed"

    def test_only_special(self):
        assert _sanitize_name("@#$%") == "unnamed"

    def test_long_name_truncated(self):
        result = _sanitize_name("a" * 100)
        assert len(result) <= 60

    def test_unicode(self):
        result = _sanitize_name("workflow-test")
        assert result == "workflow-test"

    def test_consecutive_dashes_collapsed(self):
        assert _sanitize_name("a---b---c") == "a-b-c"


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class TestWorkflowRecorder:
    def test_record_steps(self):
        rec = WorkflowRecorder("test-wf")
        rec.record_step("Run pytest")
        rec.record_step("Check git status")
        assert rec.step_count == 2
        assert rec.is_recording

    def test_skips_slash_commands(self):
        rec = WorkflowRecorder("test")
        rec.record_step("Hello")
        rec.record_step("/workflow stop")
        rec.record_step("/help")
        assert rec.step_count == 1

    def test_stop_returns_workflow(self):
        rec = WorkflowRecorder("test", description="desc")
        rec.record_step("step 1")
        wf = rec.stop()
        assert wf.name == "test"
        assert wf.description == "desc"
        assert len(wf.steps) == 1
        assert not rec.is_recording

    def test_record_after_stop_ignored(self):
        rec = WorkflowRecorder("test")
        rec.record_step("step 1")
        rec.stop()
        rec.record_step("step 2")
        assert rec.step_count == 1

    def test_record_with_tool_names(self):
        rec = WorkflowRecorder("test")
        rec.record_step("Read the file", tool_names=["read_file", "terminal"])
        wf = rec.stop()
        assert wf.steps[0].expect_tool == "read_file, terminal"

    def test_save_to_disk(self, workflows_dir):
        rec = WorkflowRecorder("save-test")
        rec.record_step("step 1")
        rec.record_step("step 2")
        path = rec.save()
        assert path.exists()
        assert path.suffix == ".yaml"


# ---------------------------------------------------------------------------
# Storage (save / load / list / delete)
# ---------------------------------------------------------------------------

class TestWorkflowStorage:
    def test_save_and_load(self, workflows_dir):
        wf = Workflow(
            name="test-save",
            description="A test workflow",
            steps=[
                WorkflowStep(prompt="Run pytest"),
                WorkflowStep(prompt="Check git", expect_tool="terminal"),
            ],
            created_at=time.time(),
        )
        save_workflow(wf)
        loaded = load_workflow("test-save")
        assert loaded is not None
        assert loaded.name == "test-save"
        assert loaded.description == "A test workflow"
        assert len(loaded.steps) == 2
        assert loaded.steps[0].prompt == "Run pytest"
        assert loaded.steps[1].expect_tool == "terminal"

    def test_load_nonexistent(self, workflows_dir):
        assert load_workflow("nonexistent") is None

    def test_list_workflows(self, workflows_dir):
        save_workflow(Workflow(name="wf-a", steps=[WorkflowStep(prompt="a")]))
        save_workflow(Workflow(name="wf-b", steps=[WorkflowStep(prompt="b")]))
        wfs = list_workflows()
        assert len(wfs) == 2
        names = {wf.name for wf in wfs}
        assert "wf-a" in names
        assert "wf-b" in names

    def test_delete_workflow(self, workflows_dir):
        save_workflow(Workflow(name="to-delete", steps=[WorkflowStep(prompt="x")]))
        assert delete_workflow("to-delete") is True
        assert load_workflow("to-delete") is None

    def test_delete_nonexistent(self, workflows_dir):
        assert delete_workflow("nope") is False

    def test_run_count_persisted(self, workflows_dir):
        wf = Workflow(name="counted", steps=[WorkflowStep(prompt="x")])
        save_workflow(wf)
        loaded = load_workflow("counted")
        assert loaded.run_count == 0
        loaded.run_count = 3
        save_workflow(loaded)
        reloaded = load_workflow("counted")
        assert reloaded.run_count == 3


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class TestWorkflowRunner:
    def test_run_all_steps(self, workflows_dir):
        wf = Workflow(
            name="run-test",
            steps=[
                WorkflowStep(prompt="step 1"),
                WorkflowStep(prompt="step 2"),
                WorkflowStep(prompt="step 3"),
            ],
        )
        save_workflow(wf)

        responses = []
        def mock_send(prompt):
            responses.append(prompt)
            return f"Response to: {prompt}"

        runner = WorkflowRunner(wf)
        results = runner.run(mock_send)

        assert len(results) == 3
        assert all(r["status"] == "ok" for r in results)
        assert len(responses) == 3
        assert responses[0] == "step 1"

    def test_run_updates_stats(self, workflows_dir):
        wf = Workflow(name="stats-test", steps=[WorkflowStep(prompt="x")])
        save_workflow(wf)

        runner = WorkflowRunner(wf)
        runner.run(lambda p: "ok")

        reloaded = load_workflow("stats-test")
        assert reloaded.run_count == 1
        assert reloaded.last_run_at > 0

    def test_run_handles_exception(self, workflows_dir):
        wf = Workflow(
            name="error-test",
            steps=[
                WorkflowStep(prompt="good"),
                WorkflowStep(prompt="bad"),
                WorkflowStep(prompt="after"),
            ],
        )
        save_workflow(wf)

        call_count = [0]
        def failing_send(prompt):
            call_count[0] += 1
            if prompt == "bad":
                raise RuntimeError("API down")
            return "ok"

        runner = WorkflowRunner(wf)
        results = runner.run(failing_send)

        assert len(results) == 3
        assert results[0]["status"] == "ok"
        assert results[1]["status"] == "error"
        assert "API down" in results[1]["response"]
        assert results[2]["status"] == "ok"

    def test_run_cancel(self, workflows_dir):
        wf = Workflow(
            name="cancel-test",
            steps=[
                WorkflowStep(prompt="step 1"),
                WorkflowStep(prompt="step 2"),
                WorkflowStep(prompt="step 3"),
            ],
        )
        save_workflow(wf)

        runner = WorkflowRunner(wf)

        call_count = [0]
        def cancelling_send(prompt):
            call_count[0] += 1
            if call_count[0] >= 2:
                runner.cancel()
            return "ok"

        results = runner.run(cancelling_send)

        # Step 2 completes, step 3 is cancelled
        assert any(r["status"] == "cancelled" for r in results)
        assert call_count[0] == 2

    def test_progress_tracking(self, workflows_dir):
        wf = Workflow(
            name="progress-test",
            steps=[WorkflowStep(prompt="a"), WorkflowStep(prompt="b")],
        )
        save_workflow(wf)

        runner = WorkflowRunner(wf)
        assert runner.progress == "0/2"
        assert not runner.is_running

        starts = []
        dones = []

        def on_start(i, step):
            starts.append(i)

        def on_done(i, step, resp):
            dones.append(i)

        runner.run(lambda p: "ok", on_step_start=on_start, on_step_done=on_done)

        assert starts == [0, 1]
        assert dones == [0, 1]
        assert not runner.is_running

    def test_empty_workflow(self, workflows_dir):
        wf = Workflow(name="empty", steps=[])
        save_workflow(wf)
        runner = WorkflowRunner(wf)
        results = runner.run(lambda p: "ok")
        assert results == []


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestWorkflowFormatting:
    def test_format_empty_list(self):
        result = format_workflow_list([])
        assert "No saved workflows" in result

    def test_format_list(self):
        wfs = [
            Workflow(name="wf-1", steps=[WorkflowStep(prompt="a")], run_count=3),
            Workflow(name="wf-2", description="desc", steps=[WorkflowStep(prompt="b"), WorkflowStep(prompt="c")]),
        ]
        result = format_workflow_list(wfs)
        assert "wf-1" in result
        assert "1 steps" in result
        assert "3 runs" in result
        assert "wf-2" in result
        assert "desc" in result

    def test_format_detail(self):
        wf = Workflow(
            name="detail-test",
            description="A detailed workflow",
            steps=[
                WorkflowStep(prompt="Run pytest"),
                WorkflowStep(prompt="Check git status", expect_tool="terminal"),
            ],
            run_count=5,
        )
        result = format_workflow_detail(wf)
        assert "detail-test" in result
        assert "A detailed workflow" in result
        assert "Run pytest" in result
        assert "Check git status" in result
        assert "[terminal]" in result
        assert "5" in result
