"""Tests for hermes_cli.run_registry — persistent chat run tracking."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli.run_registry import (
    RunRegistry,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_TIMEOUT,
)


@pytest.fixture
def registry(tmp_path: Path) -> RunRegistry:
    """Fresh in-memory (tmp) registry per test."""
    return RunRegistry(db_path=tmp_path / "test_runs.db")


class TestCreate:
    def test_creates_run_with_queued_status(self, registry: RunRegistry):
        registry.create_run("run_001", "session_abc", "hello")
        run = registry.get_run("run_001")
        assert run is not None
        assert run["run_id"] == "run_001"
        assert run["session_id"] == "session_abc"
        assert run["status"] == STATUS_QUEUED
        assert run["messages"] == []
        assert run["error"] is None
        assert run["started_at"] is not None
        assert run["completed_at"] is None

    def test_create_is_idempotent(self, registry: RunRegistry):
        """INSERT OR IGNORE means double-create is safe."""
        registry.create_run("run_002", "s1", "first")
        registry.create_run("run_002", "s2", "second")  # ignored
        run = registry.get_run("run_002")
        assert run["session_id"] == "s1"

    def test_get_nonexistent_returns_none(self, registry: RunRegistry):
        assert registry.get_run("nonexistent") is None


class TestLifecycle:
    def test_queued_to_running(self, registry: RunRegistry):
        registry.create_run("run_003", "s", "msg")
        registry.start_run("run_003")
        run = registry.get_run("run_003")
        assert run["status"] == STATUS_RUNNING

    def test_running_to_completed(self, registry: RunRegistry):
        registry.create_run("run_004", "s", "msg")
        registry.start_run("run_004")
        registry.complete_run(
            "run_004",
            messages=[{"id": 1, "role": "assistant", "content": "hi"}],
            session_id="s",
        )
        run = registry.get_run("run_004")
        assert run["status"] == STATUS_COMPLETED
        assert len(run["messages"]) == 1
        assert run["completed_at"] is not None

    def test_running_to_failed(self, registry: RunRegistry):
        registry.create_run("run_005", "s", "msg")
        registry.start_run("run_005")
        registry.fail_run("run_005", "model error")
        run = registry.get_run("run_005")
        assert run["status"] == STATUS_FAILED
        assert run["error"] == "model error"
        assert run["completed_at"] is not None

    def test_running_to_timeout(self, registry: RunRegistry):
        registry.create_run("run_006", "s", "msg")
        registry.start_run("run_006")
        registry.timeout_run("run_006")
        run = registry.get_run("run_006")
        assert run["status"] == STATUS_TIMEOUT
        assert run["completed_at"] is not None

    def test_complete_is_idempotent_when_already_terminal(self, registry: RunRegistry):
        """Once completed, further complete/fail/timeout calls are ignored."""
        registry.create_run("run_007", "s", "msg")
        registry.start_run("run_007")
        registry.complete_run("run_007", messages=[{"role": "assistant", "content": "done"}])
        # Second complete call should not overwrite
        registry.complete_run("run_007", messages=[{"role": "assistant", "content": "overwrite"}])
        run = registry.get_run("run_007")
        assert run["status"] == STATUS_COMPLETED
        assert run["messages"][0]["content"] == "done"

    def test_fail_is_idempotent_when_already_terminal(self, registry: RunRegistry):
        registry.create_run("run_008", "s", "msg")
        registry.fail_run("run_008", "first error")
        registry.fail_run("run_008", "second error")  # ignored
        run = registry.get_run("run_008")
        assert run["error"] == "first error"

    def test_timeout_is_idempotent_when_already_completed(self, registry: RunRegistry):
        registry.create_run("run_009", "s", "msg")
        registry.complete_run("run_009", messages=[])
        registry.timeout_run("run_009")  # ignored
        run = registry.get_run("run_009")
        assert run["status"] == STATUS_COMPLETED


class TestOrphanSweep:
    def test_get_stale_running_returns_old_runs(self, registry: RunRegistry):
        registry.create_run("run_010", "s", "msg")
        registry.start_run("run_010")

        # Backdate the started_at to simulate an old run
        with registry._lock:
            registry._get_conn().execute(
                "UPDATE chat_runs SET started_at=? WHERE run_id=?",
                (time.time() - 700, "run_010"),
            )

        stale = registry.get_stale_running_runs(max_age_seconds=600)
        assert "run_010" in stale

    def test_get_stale_excludes_recent_runs(self, registry: RunRegistry):
        registry.create_run("run_011", "s", "msg")
        registry.start_run("run_011")
        stale = registry.get_stale_running_runs(max_age_seconds=600)
        assert "run_011" not in stale

    def test_get_stale_excludes_terminal_runs(self, registry: RunRegistry):
        registry.create_run("run_012", "s", "msg")
        registry.complete_run("run_012", messages=[])
        with registry._lock:
            registry._get_conn().execute(
                "UPDATE chat_runs SET started_at=? WHERE run_id=?",
                (time.time() - 700, "run_012"),
            )
        stale = registry.get_stale_running_runs(max_age_seconds=600)
        assert "run_012" not in stale


class TestCleanup:
    def test_delete_old_terminal_runs(self, registry: RunRegistry):
        registry.create_run("run_020", "s", "msg")
        registry.complete_run("run_020", messages=[])
        # Backdate completed_at to 8 days ago
        with registry._lock:
            registry._get_conn().execute(
                "UPDATE chat_runs SET completed_at=? WHERE run_id=?",
                (time.time() - 8 * 86400, "run_020"),
            )
        deleted = registry.delete_old_runs(max_age_seconds=7 * 86400)
        assert deleted == 1
        assert registry.get_run("run_020") is None

    def test_delete_preserves_recent_runs(self, registry: RunRegistry):
        registry.create_run("run_021", "s", "msg")
        registry.complete_run("run_021", messages=[])
        deleted = registry.delete_old_runs(max_age_seconds=7 * 86400)
        assert deleted == 0
        assert registry.get_run("run_021") is not None

    def test_delete_preserves_running_runs(self, registry: RunRegistry):
        registry.create_run("run_022", "s", "msg")
        registry.start_run("run_022")
        # Even if very old, running runs are NOT deleted by delete_old_runs
        with registry._lock:
            registry._get_conn().execute(
                "UPDATE chat_runs SET started_at=? WHERE run_id=?",
                (time.time() - 30 * 86400, "run_022"),
            )
        deleted = registry.delete_old_runs(max_age_seconds=7 * 86400)
        assert deleted == 0  # running is not terminal
        assert registry.get_run("run_022") is not None


class TestConcurrency:
    def test_multiple_runs_independent(self, registry: RunRegistry):
        for i in range(10):
            registry.create_run(f"run_c{i}", "s", f"msg{i}")
        for i in range(10):
            registry.start_run(f"run_c{i}")
        for i in range(0, 10, 2):
            registry.complete_run(f"run_c{i}", messages=[])
        for i in range(1, 10, 2):
            registry.fail_run(f"run_c{i}", "err")

        for i in range(0, 10, 2):
            assert registry.get_run(f"run_c{i}")["status"] == STATUS_COMPLETED
        for i in range(1, 10, 2):
            assert registry.get_run(f"run_c{i}")["status"] == STATUS_FAILED


class TestSerialization:
    def test_messages_stored_and_retrieved_as_list(self, registry: RunRegistry):
        msgs = [
            {"id": 1, "role": "user", "content": "hello", "timestamp": 1000},
            {"id": 2, "role": "assistant", "content": "world", "timestamp": 1001},
        ]
        registry.create_run("run_s1", "s", "hello")
        registry.complete_run("run_s1", messages=msgs)
        run = registry.get_run("run_s1")
        assert run["messages"] == msgs

    def test_empty_messages_returns_empty_list(self, registry: RunRegistry):
        registry.create_run("run_s2", "s", "x")
        registry.complete_run("run_s2", messages=[])
        run = registry.get_run("run_s2")
        assert run["messages"] == []

    def test_timestamps_are_iso_strings(self, registry: RunRegistry):
        registry.create_run("run_s3", "s", "x")
        run = registry.get_run("run_s3")
        assert "T" in run["started_at"]
        assert "T" in run["updated_at"]
        assert run["completed_at"] is None


class TestRunSteps:
    def test_create_step_basic(self, registry: RunRegistry):
        registry.create_step(
            run_id="run_r1",
            step_id="step_001",
            type_="log",
            title="Iniciando",
        )
        steps = registry.get_steps("run_r1")
        assert len(steps) == 1
        s = steps[0]
        assert s["id"] == "step_001"
        assert s["run_id"] == "run_r1"
        assert s["type"] == "log"
        assert s["title"] == "Iniciando"
        assert s["status"] == "running"
        assert s["content"] is None
        assert "T" in s["created_at"]
        assert "T" in s["updated_at"]

    def test_create_step_with_content(self, registry: RunRegistry):
        registry.create_step(
            run_id="run_r2",
            step_id="step_002",
            type_="tool_call",
            title="Chamando bash",
            content='{"cmd": "ls"}',
            status="running",
        )
        steps = registry.get_steps("run_r2")
        assert steps[0]["content"] == '{"cmd": "ls"}'
        assert steps[0]["type"] == "tool_call"

    def test_create_step_idempotent(self, registry: RunRegistry):
        registry.create_step(run_id="run_r3", step_id="step_003", type_="log", title="A")
        registry.create_step(run_id="run_r3", step_id="step_003", type_="log", title="B")
        steps = registry.get_steps("run_r3")
        assert len(steps) == 1
        assert steps[0]["title"] == "A"

    def test_update_step_status(self, registry: RunRegistry):
        registry.create_step(run_id="run_r4", step_id="step_004", type_="log", title="Running")
        registry.update_step(step_id="step_004", status="completed")
        steps = registry.get_steps("run_r4")
        assert steps[0]["status"] == "completed"

    def test_update_step_with_content(self, registry: RunRegistry):
        registry.create_step(run_id="run_r5", step_id="step_005", type_="tool_call", title="T")
        registry.update_step(step_id="step_005", status="completed", content="result text")
        steps = registry.get_steps("run_r5")
        assert steps[0]["content"] == "result text"
        assert steps[0]["status"] == "completed"

    def test_update_step_with_title(self, registry: RunRegistry):
        registry.create_step(run_id="run_r6", step_id="step_006", type_="log", title="Old")
        registry.update_step(step_id="step_006", status="completed", title="New")
        steps = registry.get_steps("run_r6")
        assert steps[0]["title"] == "New"
        assert steps[0]["status"] == "completed"

    def test_complete_step_helper(self, registry: RunRegistry):
        registry.create_step(run_id="run_r7", step_id="step_007", type_="log", title="T")
        registry.complete_step(step_id="step_007", content="done")
        steps = registry.get_steps("run_r7")
        assert steps[0]["status"] == "completed"
        assert steps[0]["content"] == "done"

    def test_fail_step_helper(self, registry: RunRegistry):
        registry.create_step(run_id="run_r8", step_id="step_008", type_="log", title="T")
        registry.fail_step(step_id="step_008", error="something broke")
        steps = registry.get_steps("run_r8")
        assert steps[0]["status"] == "failed"
        assert steps[0]["content"] == "something broke"

    def test_get_steps_returns_empty_for_unknown_run(self, registry: RunRegistry):
        steps = registry.get_steps("nonexistent_run")
        assert steps == []

    def test_get_steps_ordered_by_created_at(self, registry: RunRegistry):
        registry.create_step(run_id="run_r9", step_id="step_r9_a", type_="log", title="First")
        registry.create_step(run_id="run_r9", step_id="step_r9_b", type_="tool_call", title="Second")
        registry.create_step(run_id="run_r9", step_id="step_r9_c", type_="message", title="Third")
        steps = registry.get_steps("run_r9")
        assert len(steps) == 3
        assert steps[0]["id"] == "step_r9_a"
        assert steps[1]["id"] == "step_r9_b"
        assert steps[2]["id"] == "step_r9_c"

    def test_steps_isolated_between_runs(self, registry: RunRegistry):
        registry.create_step(run_id="run_ra", step_id="step_ra1", type_="log", title="A")
        registry.create_step(run_id="run_rb", step_id="step_rb1", type_="log", title="B")
        steps_a = registry.get_steps("run_ra")
        steps_b = registry.get_steps("run_rb")
        assert len(steps_a) == 1
        assert steps_a[0]["title"] == "A"
        assert len(steps_b) == 1
        assert steps_b[0]["title"] == "B"
