"""Swarm module test suite — covers scheduler, router, planner, messaging,
verifier, artifacts, approval, monitor, orchestrator, and worker."""

import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from swarm.types import (
    ArtifactRef, MessageType, SwarmConfig, SwarmMessage, SwarmResult,
    SwarmTask, SwarmWorker, TaskState, WorkerState,
)
from swarm.exceptions import (
    BudgetExceededError, DependencyFailedError, SwarmError,
    TaskFailedError, WorkerUnavailableError,
)


# ── Scheduler ────────────────────────────────────────────────────────────


class TestScheduler:
    def _make_scheduler(self):
        from swarm.scheduler import SwarmScheduler
        return SwarmScheduler()

    def test_add_task(self):
        s = self._make_scheduler()
        t = SwarmTask(name="a", prompt="do a")
        s.add_task(t)
        assert s.get_status() == {"pending": 1}

    def test_duplicate_task_raises(self):
        s = self._make_scheduler()
        t = SwarmTask(name="a", prompt="do a", id="dup")
        s.add_task(t)
        with pytest.raises(ValueError, match="already exists"):
            s.add_task(SwarmTask(name="b", prompt="do b", id="dup"))

    def test_get_ready_no_deps(self):
        s = self._make_scheduler()
        t1 = SwarmTask(name="a", prompt="a")
        t2 = SwarmTask(name="b", prompt="b")
        s.add_task(t1)
        s.add_task(t2)
        ready = s.get_ready_tasks()
        assert len(ready) == 2

    def test_get_ready_with_deps(self):
        s = self._make_scheduler()
        t1 = SwarmTask(name="a", prompt="a", id="t1")
        t2 = SwarmTask(name="b", prompt="b", id="t2", deps=["t1"])
        s.add_task(t1)
        s.add_task(t2)
        ready = s.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t1"

    def test_dep_unblocks_after_completion(self):
        s = self._make_scheduler()
        t1 = SwarmTask(name="a", prompt="a", id="t1")
        t2 = SwarmTask(name="b", prompt="b", id="t2", deps=["t1"])
        s.add_task(t1)
        s.add_task(t2)
        s.mark_running("t1", "w1")
        s.mark_completed("t1", SwarmResult(task_id="t1", success=True))
        ready = s.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t2"

    def test_mark_running_wrong_state_raises(self):
        s = self._make_scheduler()
        t = SwarmTask(name="a", prompt="a", id="t1")
        s.add_task(t)
        s.mark_running("t1", "w1")
        with pytest.raises(ValueError, match="expected pending"):
            s.mark_running("t1", "w2")

    def test_mark_failed_retries(self):
        s = self._make_scheduler()
        t = SwarmTask(name="a", prompt="a", id="t1", max_retries=3)
        s.add_task(t)
        s.mark_running("t1", "w1")
        s.mark_failed("t1", "oops")
        # Should go back to pending (retry 1 of 3)
        assert s._tasks["t1"].state == TaskState.pending
        assert s._tasks["t1"].retries == 1

    def test_mark_failed_exhausted(self):
        s = self._make_scheduler()
        t = SwarmTask(name="a", prompt="a", id="t1", max_retries=1)
        s.add_task(t)
        s.mark_running("t1", "w1")
        s.mark_failed("t1", "oops")
        assert s._tasks["t1"].state == TaskState.failed

    def test_cancel_downstream(self):
        s = self._make_scheduler()
        t1 = SwarmTask(name="a", prompt="a", id="t1")
        t2 = SwarmTask(name="b", prompt="b", id="t2", deps=["t1"])
        t3 = SwarmTask(name="c", prompt="c", id="t3", deps=["t2"])
        s.add_task(t1)
        s.add_task(t2)
        s.add_task(t3)
        s.cancel_downstream("t1")
        assert s._tasks["t2"].state == TaskState.cancelled
        assert s._tasks["t3"].state == TaskState.cancelled

    def test_is_complete(self):
        s = self._make_scheduler()
        t = SwarmTask(name="a", prompt="a", id="t1")
        s.add_task(t)
        assert not s.is_complete()
        s.mark_running("t1", "w1")
        assert not s.is_complete()
        s.mark_completed("t1", SwarmResult(task_id="t1", success=True))
        assert s.is_complete()

    def test_serialization_roundtrip(self):
        s = self._make_scheduler()
        t = SwarmTask(name="a", prompt="do a", id="t1", role="executor")
        s.add_task(t)
        s.mark_running("t1", "w1")
        s.mark_completed("t1", SwarmResult(task_id="t1", success=True, output="done"))
        data = s.to_dict()
        from swarm.scheduler import SwarmScheduler
        s2 = SwarmScheduler.from_dict(data)
        assert s2._tasks["t1"].state == TaskState.completed

    def test_add_dependency(self):
        s = self._make_scheduler()
        t1 = SwarmTask(name="a", prompt="a", id="t1")
        t2 = SwarmTask(name="b", prompt="b", id="t2")
        s.add_task(t1)
        s.add_task(t2)
        s.add_dependency("t2", "t1")
        ready = s.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t1"


# ── Router ───────────────────────────────────────────────────────────────


class TestRouter:
    def _make_router(self, budget=0.0):
        from swarm.router import ModelRouter, ProviderConfig
        config = SwarmConfig(budget_limit_usd=budget)
        providers = {
            "local": ProviderConfig(
                name="local", api_key="local",
                base_url="http://localhost:8800/v1",
                allowed_models=["local/qwen3.5-9b"],
            ),
        }
        return ModelRouter(config, providers=providers)

    def test_select_model_local(self):
        r = self._make_router()
        model = r.select_model("local")
        assert model == "local/qwen3.5-9b"

    def test_available_models(self):
        r = self._make_router()
        models = r.available_models()
        assert "local/qwen3.5-9b" in models

    def test_estimate_cost_local_is_zero(self):
        r = self._make_router()
        cost = r.estimate_cost("local/qwen3.5-9b", 10000, 5000)
        assert cost == 0.0

    def test_estimate_cost_paid_model(self):
        r = self._make_router()
        cost = r.estimate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost > 0

    def test_log_usage_tracks_spend(self):
        r = self._make_router()
        r.log_usage("local/qwen3.5-9b", 1000, 500, 0.0, role="executor")
        summary = r.get_spend_summary()
        assert summary["num_requests"] == 1
        assert summary["total_input_tokens"] == 1000

    def test_budget_check(self):
        r = self._make_router(budget=1.0)
        assert r.is_within_budget(0.5)
        assert not r.is_within_budget(1.5)

    def test_spend_summary_shape(self):
        r = self._make_router()
        s = r.get_spend_summary()
        assert "total_cost_usd" in s
        assert "budget_limit_usd" in s
        assert "available_models" in s

    def test_get_client_kwargs(self):
        r = self._make_router()
        kwargs = r.get_client_kwargs("local/qwen3.5-9b")
        assert "api_key" in kwargs
        assert "base_url" in kwargs

    def test_add_provider(self):
        from swarm.router import ProviderConfig
        r = self._make_router()
        r.add_provider(ProviderConfig(
            name="test", api_key="key", allowed_models=["test-model"],
        ))
        assert "test" in r._providers


# ── Planner ──────────────────────────────────────────────────────────────


class TestPlanner:
    def _make_planner(self):
        from swarm.planner import TaskPlanner
        return TaskPlanner()

    def test_decompose_single_goal(self):
        p = self._make_planner()
        tasks = p.decompose("build a web scraper")
        assert len(tasks) >= 1
        assert all(isinstance(t, SwarmTask) for t in tasks)

    def test_decompose_numbered_steps(self):
        p = self._make_planner()
        tasks = p.decompose(
            "1. Research the API\n2. Implement the client\n3. Write tests"
        )
        assert len(tasks) == 3

    def test_decompose_bullet_steps(self):
        p = self._make_planner()
        tasks = p.decompose(
            "- Search for docs\n- Build the parser\n- Verify output"
        )
        assert len(tasks) == 3

    def test_role_inference_executor(self):
        from swarm.planner import _infer_role
        assert _infer_role("implement the login system") == "executor"

    def test_role_inference_researcher(self):
        from swarm.planner import _infer_role
        assert _infer_role("research the best approaches and explore options") == "researcher"

    def test_role_inference_verifier(self):
        from swarm.planner import _infer_role
        assert _infer_role("test and verify the output") == "verifier"

    def test_dep_signal_detection(self):
        from swarm.planner import _has_dep_signal
        assert _has_dep_signal("then merge the results")
        assert _has_dep_signal("after that, test it")
        assert not _has_dep_signal("search for documentation")

    def test_explicit_dep_extraction(self):
        from swarm.planner import _extract_explicit_dep
        refs = _extract_explicit_dep("after step 2, verify")
        assert refs == [1]  # 0-indexed

    def test_plan_from_steps(self):
        p = self._make_planner()
        tasks = p.plan_from_steps(
            ["do A", "do B", "do C"],
            deps={2: [0, 1]},
        )
        assert len(tasks) == 3
        assert len(tasks[2].deps) == 2

    def test_estimate_parallelism(self):
        p = self._make_planner()
        tasks = p.plan_from_steps(["a", "b", "c"])
        assert p.estimate_parallelism(tasks) == 3  # all parallel

    def test_estimate_parallelism_chain(self):
        p = self._make_planner()
        tasks = p.plan_from_steps(["a", "b", "c"], deps={1: [0], 2: [1]})
        assert p.estimate_parallelism(tasks) == 1  # linear chain

    def test_validate_plan_clean(self):
        p = self._make_planner()
        tasks = p.plan_from_steps(["a", "b"])
        errors = p.validate_plan(tasks)
        assert errors == []

    def test_validate_plan_missing_dep(self):
        p = self._make_planner()
        tasks = [SwarmTask(name="a", prompt="a", deps=["nonexistent"])]
        errors = p.validate_plan(tasks)
        assert any("unknown" in e for e in errors)

    def test_format_plan(self):
        p = self._make_planner()
        tasks = p.plan_from_steps(["do A", "do B"])
        output = p.format_plan(tasks)
        assert "Plan" in output
        assert "step_0" in output


# ── Messaging ────────────────────────────────────────────────────────────


class TestMessageBus:
    def _make_bus(self):
        from swarm.messaging import MessageBus
        return MessageBus()

    def test_send_direct(self):
        bus = self._make_bus()
        bus._ensure_agent("alice")
        bus._ensure_agent("bob")
        msg = SwarmMessage(
            from_agent="alice", to_agent="bob",
            msg_type=MessageType.data, payload={"x": 1},
        )
        bus.send(msg)
        received = bus.receive("bob", block=False)
        assert received is not None
        assert received.payload == {"x": 1}

    def test_broadcast(self):
        bus = self._make_bus()
        bus._ensure_agent("alice")
        bus._ensure_agent("bob")
        bus._ensure_agent("carol")
        bus.broadcast("alice", "status", {"info": "hello"})
        assert bus.receive("bob", block=False) is not None
        assert bus.receive("carol", block=False) is not None
        # Sender shouldn't receive own broadcast
        assert bus.receive("alice", block=False) is None

    def test_receive_timeout(self):
        bus = self._make_bus()
        bus._ensure_agent("a")
        result = bus.receive("a", block=True, timeout=0.01)
        assert result is None

    def test_peek(self):
        bus = self._make_bus()
        bus._ensure_agent("a")
        bus.broadcast("b", "data", {"v": 42})
        peeked = bus.peek("a")
        assert len(peeked) == 1

    def test_subscribe_callback(self):
        bus = self._make_bus()
        bus._ensure_agent("a")
        received = []
        bus.subscribe("a", "status", lambda msg: received.append(msg))
        bus.broadcast("b", "status", {"ping": True})
        assert len(received) == 1

    def test_clear_inbox(self):
        bus = self._make_bus()
        bus._ensure_agent("a")
        bus.broadcast("b", "data", {"v": 1})
        bus.broadcast("b", "data", {"v": 2})
        bus.clear("a")
        assert bus.receive("a", block=False) is None

    def test_history(self):
        bus = self._make_bus()
        bus._ensure_agent("a")
        bus.broadcast("a", "status", {"x": 1})
        history = bus.get_history()
        assert len(history) == 1
        agent_history = bus.get_history("a")
        assert len(agent_history) == 1


# ── Verifier ─────────────────────────────────────────────────────────────


class TestVerifier:
    def _make_verifier(self):
        from swarm.verifier import Verifier
        return Verifier()

    def test_verify_good_result(self):
        v = self._make_verifier()
        task = SwarmTask(name="t", prompt="do it", id="t1")
        result = SwarmResult(task_id="t1", success=True, output="All good!")
        vr = v.verify(task, result)
        assert vr.passed
        assert vr.score > 0.5

    def test_verify_empty_output(self):
        v = self._make_verifier()
        task = SwarmTask(name="t", prompt="do it", id="t1")
        result = SwarmResult(task_id="t1", success=True, output="")
        vr = v.verify(task, result)
        # non_empty_output should fail
        assert any(c.name == "non_empty_output" and not c.passed for c in vr.checks)

    def test_verify_error_in_output(self):
        v = self._make_verifier()
        task = SwarmTask(name="t", prompt="do it", id="t1")
        result = SwarmResult(task_id="t1", success=True, output="Traceback most recent call\nError: boom")
        vr = v.verify(task, result)
        assert any(c.name == "no_error" and not c.passed for c in vr.checks)

    def test_verify_python_syntax(self):
        v = self._make_verifier()
        task = SwarmTask(name="t", prompt="do it", id="t1")
        result = SwarmResult(task_id="t1", success=True, output="def foo(:\n  pass")
        vr = v.verify(task, result)
        assert any(c.name == "syntax" and not c.passed for c in vr.checks)

    def test_verify_valid_python(self):
        v = self._make_verifier()
        task = SwarmTask(name="t", prompt="do it", id="t1")
        result = SwarmResult(task_id="t1", success=True, output="def foo():\n    return 42")
        vr = v.verify(task, result)
        assert any(c.name == "syntax" and c.passed for c in vr.checks)

    def test_diff_size_check(self):
        from swarm.verifier import DiffSizeCheck
        check = DiffSizeCheck(max_lines=5)
        task = SwarmTask(name="t", prompt="do it", id="t1")
        result = SwarmResult(task_id="t1", success=True, output="a\n" * 100)
        cr = check.check(task, result)
        assert not cr.passed

    def test_format_report(self):
        from swarm.verifier import format_report
        v = self._make_verifier()
        task = SwarmTask(name="t", prompt="do it", id="t1")
        result = SwarmResult(task_id="t1", success=True, output="hello")
        vr = v.verify(task, result)
        report = format_report(vr)
        assert "Verification Report" in report
        assert "PASSED" in report or "FAILED" in report

    def test_custom_threshold(self):
        from swarm.verifier import Verifier, NonEmptyOutputCheck
        v = Verifier(checks=[NonEmptyOutputCheck()], threshold=1.0)
        task = SwarmTask(name="t", prompt="do it", id="t1")
        result = SwarmResult(task_id="t1", success=True, output="ok")
        vr = v.verify(task, result)
        assert vr.passed
        assert vr.score == 1.0


# ── Artifacts ────────────────────────────────────────────────────────────


class TestArtifacts:
    def test_store_and_retrieve(self):
        from swarm.artifacts import ArtifactStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            ref = store.store_bytes("task1", b"hello world", "test.txt")
            assert ref.size_bytes == 11
            assert ref.checksum
            path = store.get(ref.id)
            assert path is not None
            assert path.read_bytes() == b"hello world"

    def test_store_file(self):
        from swarm.artifacts import ArtifactStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            src = Path(tmpdir) / "source.txt"
            src.write_text("file content")
            ref = store.store("task1", src)
            assert ref.mime_type == "text/plain"
            retrieved = store.get(ref.id)
            assert retrieved.read_text() == "file content"

    def test_get_by_task(self):
        from swarm.artifacts import ArtifactStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            store.store_bytes("task1", b"a", "a.txt")
            store.store_bytes("task1", b"b", "b.txt")
            store.store_bytes("task2", b"c", "c.txt")
            task1_refs = store.get_by_task("task1")
            assert len(task1_refs) == 2

    def test_delete(self):
        from swarm.artifacts import ArtifactStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            ref = store.store_bytes("task1", b"x", "x.txt")
            store.delete(ref.id)
            assert store.get(ref.id) is None

    def test_cleanup_task(self):
        from swarm.artifacts import ArtifactStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            store.store_bytes("task1", b"a", "a.txt")
            store.store_bytes("task1", b"b", "b.txt")
            store.cleanup_task("task1")
            assert store.get_by_task("task1") == []

    def test_export_manifest(self):
        from swarm.artifacts import ArtifactStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            store.store_bytes("t1", b"data", "f.bin")
            manifest = store.export_manifest()
            assert len(manifest["artifacts"]) == 1
            assert manifest["artifacts"][0]["mime_type"] == "application/octet-stream"


# ── Approval ─────────────────────────────────────────────────────────────


class TestApproval:
    def _make_gate(self):
        from swarm.approval import ApprovalGate
        return ApprovalGate()

    def test_low_risk_auto_approved(self):
        gate = self._make_gate()
        task = SwarmTask(name="t", prompt="read file", id="t1")
        req = gate.check(task, "read")
        assert req.status == "approved"
        assert req.decided_by == "auto"

    def test_high_risk_needs_approval(self):
        gate = self._make_gate()
        task = SwarmTask(name="t", prompt="run bash", id="t1")
        req = gate.check(task, "terminal")
        assert req.status == "pending"

    def test_approve_flow(self):
        gate = self._make_gate()
        task = SwarmTask(name="t", prompt="deploy", id="t1")
        req = gate.check(task, "deploy")
        gate.request_approval(req)
        assert len(gate.get_pending()) == 1
        gate.approve(req.id, by="admin")
        assert req.status == "approved"
        assert len(gate.get_pending()) == 0

    def test_deny_flow(self):
        gate = self._make_gate()
        task = SwarmTask(name="t", prompt="push", id="t1")
        req = gate.check(task, "git_push")
        gate.request_approval(req)
        gate.deny(req.id, reason="not ready")
        assert req.status == "denied"

    def test_spend_threshold(self):
        gate = self._make_gate()
        task = SwarmTask(name="t", prompt="costly op", id="t1")
        req = gate.check(task, "read", context={"spend_usd": 5.0})
        assert req.status == "pending"
        assert "spend" in req.reason

    def test_role_requires_approval(self):
        gate = self._make_gate()
        task = SwarmTask(name="t", prompt="merge", id="t1", role="merger")
        req = gate.check(task, "read")
        assert req.status == "pending"
        assert "merger" in req.reason

    def test_wait_for_approval(self):
        gate = self._make_gate()
        task = SwarmTask(name="t", prompt="bash cmd", id="t1")
        req = gate.check(task, "bash")
        gate.request_approval(req)
        # Approve in a thread
        def _approve():
            time.sleep(0.05)
            gate.approve(req.id)
        t = threading.Thread(target=_approve, daemon=True)
        t.start()
        result = gate.wait_for_approval(req.id, timeout=2.0)
        assert result is True


# ── Monitor ──────────────────────────────────────────────────────────────


class TestMonitor:
    def _make_monitor(self):
        from swarm.monitor import SwarmMonitor
        return SwarmMonitor(log_dir=Path(tempfile.mkdtemp()))

    def test_record_and_timeline(self):
        m = self._make_monitor()
        m.record_event("test_event", task_id="t1")
        timeline = m.get_timeline()
        assert len(timeline) == 1
        assert timeline[0]["type"] == "test_event"

    def test_record_api_call(self):
        m = self._make_monitor()
        m.record_api_call("gpt-4o", 1000, 500, 0.5, 0.01, task_id="t1")
        summary = m.get_summary()
        assert summary["total_input_tokens"] == 1000
        assert summary["total_output_tokens"] == 500

    def test_record_tool_call(self):
        m = self._make_monitor()
        m.record_tool_call("bash", 0.1, True, task_id="t1")
        m.record_tool_call("bash", 0.2, False, task_id="t1")
        summary = m.get_summary()
        assert summary["by_tool"]["bash"]["calls"] == 2
        assert summary["by_tool"]["bash"]["successes"] == 1
        assert summary["by_tool"]["bash"]["failures"] == 1

    def test_record_task_transition(self):
        m = self._make_monitor()
        m.record_task_transition("t1", "pending", "running")
        m.record_task_transition("t1", "running", "completed")
        summary = m.get_summary()
        assert summary["completed"] == 1

    def test_cost_breakdown(self):
        m = self._make_monitor()
        m.record_api_call("gpt-4o", 1000, 500, 0.5, 0.01, task_id="t1")
        m.record_api_call("gpt-4o", 2000, 1000, 0.8, 0.02, task_id="t2")
        cb = m.get_cost_breakdown()
        assert cb["total_cost_usd"] == pytest.approx(0.03)
        assert "gpt-4o" in cb["by_model"]

    def test_export_json(self):
        m = self._make_monitor()
        m.record_event("test", task_id="t1")
        path = m.export_json()
        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert "summary" in data
        assert "events" in data

    def test_summary_empty(self):
        m = self._make_monitor()
        summary = m.get_summary()
        assert summary["total_events"] == 0
        assert summary["total_cost_usd"] == 0.0


# ── Roles ────────────────────────────────────────────────────────────────


class TestRoles:
    def test_get_role(self):
        from swarm.roles import get_role
        role = get_role("executor")
        assert role.name == "executor"
        assert role.default_model

    def test_get_role_unknown(self):
        from swarm.roles import get_role
        with pytest.raises(KeyError, match="Unknown role"):
            get_role("nonexistent")

    def test_list_roles(self):
        from swarm.roles import list_roles
        roles = list_roles()
        names = {r.name for r in roles}
        assert "planner" in names
        assert "executor" in names
        assert "critic" in names
        assert "verifier" in names
        assert "merger" in names
        assert "researcher" in names


# ── Exceptions ───────────────────────────────────────────────────────────


class TestExceptions:
    def test_swarm_error(self):
        with pytest.raises(SwarmError):
            raise SwarmError("test")

    def test_task_failed(self):
        e = TaskFailedError("t1", "oops")
        assert e.task_id == "t1"
        assert "oops" in str(e)

    def test_worker_unavailable(self):
        e = WorkerUnavailableError("w1", "busy")
        assert "w1" in str(e)

    def test_budget_exceeded(self):
        e = BudgetExceededError(spent=5.0, limit=3.0)
        assert e.spent == 5.0
        assert e.limit == 3.0

    def test_dependency_failed(self):
        e = DependencyFailedError("t2", "t1")
        assert e.task_id == "t2"
        assert e.dep_id == "t1"


# ── Worker Pool ──────────────────────────────────────────────────────────


class TestWorkerPool:
    def test_scale_up_and_status(self):
        from swarm.worker import WorkerPool
        config = SwarmConfig(max_workers=4)
        pool = WorkerPool(config)
        pool.scale_up(3)
        status = pool.get_status()
        assert status["total"] == 3
        assert status["states"].get("idle") == 3

    def test_get_available_worker(self):
        from swarm.worker import WorkerPool
        config = SwarmConfig(max_workers=4)
        pool = WorkerPool(config)
        pool.scale_up(2)
        w = pool.get_available_worker()
        assert w is not None
        assert w.state == WorkerState.idle

    def test_shutdown(self):
        from swarm.worker import WorkerPool
        config = SwarmConfig(max_workers=4)
        pool = WorkerPool(config)
        pool.scale_up(2)
        pool.shutdown()
        assert pool.get_status()["total"] == 0


# ── Orchestrator Integration ─────────────────────────────────────────────


class TestOrchestrator:
    def test_add_task_and_run(self):
        from swarm.orchestrator import SwarmOrchestrator
        orch = SwarmOrchestrator(SwarmConfig(max_workers=2))
        tid = orch.add_task("hello", "say hello")
        assert tid
        summary = orch.run(max_turns=50)
        assert "tasks" in summary

    def test_add_plan_and_run(self):
        from swarm.orchestrator import SwarmOrchestrator
        orch = SwarmOrchestrator(SwarmConfig(max_workers=2))
        plan = [
            {"name": "step1", "prompt": "do A"},
            {"name": "step2", "prompt": "do B", "deps": ["step1"]},
        ]
        ids = orch.add_plan(plan)
        assert len(ids) == 2
        summary = orch.run(max_turns=100)
        total = summary.get("total_tasks", 0)
        assert total == 2

    def test_cancel(self):
        from swarm.orchestrator import SwarmOrchestrator
        orch = SwarmOrchestrator(SwarmConfig(max_workers=1))
        orch.add_task("t", "do something")
        orch.cancel()
        assert orch.get_status()["cancelled"]

    def test_get_status(self):
        from swarm.orchestrator import SwarmOrchestrator
        orch = SwarmOrchestrator(SwarmConfig())
        orch.add_task("t", "task")
        status = orch.get_status()
        assert "scheduler" in status
        assert "workers" in status
        assert "router" in status

    def test_run_parallel_tasks(self):
        from swarm.orchestrator import SwarmOrchestrator
        orch = SwarmOrchestrator(SwarmConfig(max_workers=4))
        for i in range(4):
            orch.add_task(f"task_{i}", f"do {i}")
        summary = orch.run(max_turns=200)
        assert summary["total_tasks"] == 4

    def test_run_with_display(self):
        """Live display mode runs without crashing."""
        from swarm.orchestrator import SwarmOrchestrator
        from rich.console import Console
        import io
        orch = SwarmOrchestrator(SwarmConfig(max_workers=2))
        orch.add_task("hello", "say hello")
        console = Console(file=io.StringIO(), force_terminal=True)
        summary = orch.run_with_display(console=console, max_turns=50)
        assert "tasks" in summary

    def test_run_with_display_parallel(self):
        """Live display with multiple parallel tasks."""
        from swarm.orchestrator import SwarmOrchestrator
        from rich.console import Console
        import io
        orch = SwarmOrchestrator(SwarmConfig(max_workers=4))
        for i in range(3):
            orch.add_task(f"task_{i}", f"do {i}")
        console = Console(file=io.StringIO(), force_terminal=True)
        summary = orch.run_with_display(console=console, max_turns=200)
        assert summary["total_tasks"] == 3

    def test_build_live_table(self):
        """Live table builds without error."""
        from swarm.orchestrator import SwarmOrchestrator
        orch = SwarmOrchestrator(SwarmConfig(max_workers=2))
        orch.add_task("test", "do test")
        orch._ensure_workers()
        table = orch._build_live_table()
        assert table is not None
        assert table.title == "🐝 Swarm"
