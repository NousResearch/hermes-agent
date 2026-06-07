"""Tests for hermes_orchestrator (OMO 11.2 borrowing: boulder.json + ultragoal)."""
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_orchestrator import (
    OrchestratorContext,
    append_step,
    boulder_path,
    complete_step,
    create_boulder,
    finish_boulder,
    list_boulders,
    load_boulder,
    load_goal,
    make_task_id,
    orchestrator_root,
    record_evidence,
    set_goal,
    ultragoal_dir,
    write_audit,
)


@pytest.fixture(autouse=True)
def clean_orchestrator_dir(tmp_path, monkeypatch):
    """把 ~/.hermes/orchestrator 重定向到 tmp，避免污染真实数据。"""
    fake_home = tmp_path / "hermes"
    fake_home.mkdir()
    (fake_home / "orchestrator").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    yield fake_home


# ---------------------------------------------------------------------------
# Task ID 生成
# ---------------------------------------------------------------------------

class TestMakeTaskId:
    def test_starts_with_timestamp(self):
        tid = make_task_id("扫目录找 contract")
        # 14 位时间戳前缀
        assert tid[:14].isdigit()

    def test_includes_slug(self):
        tid = make_task_id("Search for contract reviews")
        assert "search" in tid or "contract" in tid

    def test_unique_per_call(self):
        """不同调用产生不同 task_id（hash 防冲突）"""
        ids = {make_task_id("test") for _ in range(100)}
        assert len(ids) == 100  # 全部唯一


# ---------------------------------------------------------------------------
# Boulder CRUD
# ---------------------------------------------------------------------------

class TestBoulderCRUD:
    def test_create_boulder(self):
        b = create_boulder("测试任务", model="claude-opus-4", toolsets=["file"])
        assert b.goal == "测试任务"
        assert b.model == "claude-opus-4"
        assert b.status == "created"
        assert b.toolsets == ["file"]

    def test_load_boulder(self):
        b = create_boulder("load test")
        loaded = load_boulder(b.task_id)
        assert loaded is not None
        assert loaded.goal == "load test"
        assert loaded.task_id == b.task_id

    def test_load_nonexistent_returns_none(self):
        assert load_boulder("nonexistent-task-id") is None

    def test_save_boulder_updates_timestamp(self):
        b = create_boulder("ts test")
        original_ts = b.updated_at
        import time
        time.sleep(0.01)
        b.status = "running"
        # 走 append_step 触发 save
        append_step(b.task_id, tool="terminal", args_summary="x")
        loaded = load_boulder(b.task_id)
        assert loaded.updated_at > original_ts


# ---------------------------------------------------------------------------
# Step 记录
# ---------------------------------------------------------------------------

class TestSteps:
    def test_append_step_creates_running_status(self):
        """第一次 append_step 应把 status 从 created → running"""
        b = create_boulder("step test")
        assert b.status == "created"
        idx = append_step(b.task_id, tool="terminal", args_summary="ls")
        loaded = load_boulder(b.task_id)
        assert loaded.status == "running"
        assert len(loaded.steps) == 1
        assert loaded.steps[0].tool == "terminal"
        assert loaded.steps[0].status == "running"

    def test_complete_step_records_result(self):
        b = create_boulder("complete test")
        idx = append_step(b.task_id, tool="read_file", args_summary="README")
        complete_step(b.task_id, idx, result_summary="100 行", status="success")
        loaded = load_boulder(b.task_id)
        assert loaded.steps[0].status == "success"
        assert loaded.steps[0].result_summary == "100 行"
        assert loaded.steps[0].duration_ms is not None
        assert loaded.steps[0].duration_ms >= 0

    def test_complete_step_with_error(self):
        b = create_boulder("error test")
        idx = append_step(b.task_id, tool="terminal", args_summary="bad cmd")
        complete_step(b.task_id, idx, status="failed", error="command not found")
        loaded = load_boulder(b.task_id)
        assert loaded.steps[0].status == "failed"
        assert loaded.steps[0].error == "command not found"

    def test_multiple_steps(self):
        b = create_boulder("multi step")
        for i in range(3):
            idx = append_step(b.task_id, tool="terminal", args_summary=f"cmd {i}")
            complete_step(b.task_id, idx, result_summary=f"result {i}")
        loaded = load_boulder(b.task_id)
        assert len(loaded.steps) == 3
        assert all(s.status == "success" for s in loaded.steps)


# ---------------------------------------------------------------------------
# Ultragoal
# ---------------------------------------------------------------------------

class TestUltragoal:
    def test_set_and_load_goal(self):
        b = create_boulder("goal test")
        set_goal(
            b.task_id,
            "goal test",
            success_criteria=["条件 1", "条件 2"],
            anti_criteria=["失败 1"],
        )
        g = load_goal(b.task_id)
        assert g is not None
        assert g.goal == "goal test"
        assert g.success_criteria == ["条件 1", "条件 2"]
        assert g.anti_criteria == ["失败 1"]

    def test_record_evidence(self):
        b = create_boulder("evidence test")
        record_evidence(
            b.task_id, 0,
            source="工具输出",
            content="ls 输出 3 个文件",
            citation="terminal: ls",
            confidence="high",
        )
        evidence_dir = ultragoal_dir(b.task_id) / "evidence"
        files = list(evidence_dir.glob("step_*.json"))
        assert len(files) == 1
        rec = json.loads(files[0].read_text(encoding="utf-8"))
        assert rec["source"] == "工具输出"
        assert rec["content"] == "ls 输出 3 个文件"
        assert rec["citation"] == "terminal: ls"
        assert rec["confidence"] == "high"

    def test_record_evidence_firsthand_vs_inference(self):
        """不同 source 类型都应该被记录——这是反目标漂移的关键"""
        b = create_boulder("source test")
        record_evidence(b.task_id, 0, source="一手", content="从源码读到 X")
        record_evidence(b.task_id, 1, source="二手", content="从转述听到 X")
        record_evidence(b.task_id, 2, source="推断", content="我猜 X")
        files = sorted((ultragoal_dir(b.task_id) / "evidence").glob("step_*.json"))
        assert len(files) == 3
        sources = [json.loads(f.read_text())["source"] for f in files]
        assert sources == ["一手", "二手", "推断"]


# ---------------------------------------------------------------------------
# Audit Report
# ---------------------------------------------------------------------------

class TestAuditReport:
    def test_audit_contains_all_sections_with_goal(self):
        """传了 success_criteria 时 audit.md 必须包含目标/成功标准/进度/证据链"""
        with OrchestratorContext(
            "audit test",
            model="claude-opus-4",
            success_criteria=["条件 1"],
            anti_criteria=["失败 1"],
        ) as ctx:
            ctx.append_step(tool="terminal", args_summary="ls")
            ctx.complete_step(result_summary="3 files")
            ctx.record_evidence(source="工具输出", content="ls 输出")
        audit = ultragoal_dir(ctx.task_id) / "audit.md"
        content = audit.read_text(encoding="utf-8")
        assert "🎯 Goal" in content
        assert "✅ Success Criteria" in content
        assert "❌ Anti-Criteria" in content
        assert "📊 Progress" in content
        assert "Evidence Chain" in content
        assert "**Step 0**" in content

    def test_audit_minimal_without_goal(self):
        """不传 success_criteria 时只有 progress + steps 段"""
        with OrchestratorContext("md test") as ctx:
            ctx.append_step(tool="x", args_summary="y")
            ctx.complete_step(result_summary="ok")
        audit = ultragoal_dir(ctx.task_id) / "audit.md"
        content = audit.read_text(encoding="utf-8")
        # 至少 1 个 ## 段（progress）
        assert content.count("\n## ") >= 1
        # 没有 goal 段
        assert "🎯 Goal" not in content
        # 但有 step 记录
        assert "Step 0" in content


# ---------------------------------------------------------------------------
# OrchestratorContext（高层 API）
# ---------------------------------------------------------------------------

class TestOrchestratorContext:
    def test_context_manager_auto_finish_on_success(self):
        """正常退出 → 自动 finish(status=completed)"""
        with OrchestratorContext("ctx test") as ctx:
            ctx.append_step(tool="x", args_summary="y")
            ctx.complete_step(result_summary="ok")
        b = load_boulder(ctx.task_id)
        assert b.status == "completed"

    def test_context_manager_auto_finish_on_exception(self):
        """异常退出 → 自动 finish(status=failed)"""
        with pytest.raises(ValueError):
            with OrchestratorContext("ctx fail test") as ctx:
                ctx.append_step(tool="x", args_summary="y")
                raise ValueError("boom")
        b = load_boulder(ctx.task_id)
        assert b.status == "failed"
        assert "boom" in (b.error or "")

    def test_record_evidence_with_citation(self):
        """evidence 必须可引用——这是 audit 价值的关键"""
        with OrchestratorContext("citation test") as ctx:
            ctx.append_step(tool="terminal", args_summary="ls")
            ctx.record_evidence(
                source="工具输出",
                content="找到 3 个文件",
                citation="https://example.com/docs/file.md",
            )
        evidence = list((ultragoal_dir(ctx.task_id) / "evidence").glob("step_*.json"))
        rec = json.loads(evidence[0].read_text())
        assert rec["citation"].startswith("https://")


# ---------------------------------------------------------------------------
# list_boulders
# ---------------------------------------------------------------------------

class TestListBoulders:
    def test_list_returns_recent(self):
        create_boulder("a")
        create_boulder("b")
        create_boulder("c")
        all_b = list_boulders(limit=10)
        assert len(all_b) == 3

    def test_list_filter_by_status(self):
        b1 = create_boulder("running task")
        # 触发 running
        append_step(b1.task_id, tool="x", args_summary="y")
        b2 = create_boulder("still created")
        # b2 还是 created
        running = list_boulders(status="running")
        created = list_boulders(status="created")
        assert any(b.task_id == b1.task_id for b in running)
        assert any(b.task_id == b2.task_id for b in created)
        assert not any(b.task_id == b1.task_id for b in created)

    def test_list_empty(self, clean_orchestrator_dir):
        """空目录应该返回空列表（不报错）"""
        # 重命名 orchestrator 让 list 找不到
        (clean_orchestrator_dir / "orchestrator").rmdir()
        result = list_boulders()
        assert result == []
