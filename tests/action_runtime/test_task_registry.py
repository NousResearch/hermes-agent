"""Tests for action_runtime.task_registry — Phase 5 Step 1.

Coverage per the migration plan: register/complete/interrupt round-trip,
pause flag, TTL eviction for idempotency keys — plus the Q3 ruling pins
(TaskStatus is separate; contract.Status stays terminal-only).
"""

from __future__ import annotations

import weakref

from action_runtime.contract import ErrorType, ExecError, ExecutionResult, Status
from action_runtime.task_registry import (
    TERMINAL_STATUSES,
    AgentTaskRecord,
    AgentTaskRegistry,
    TaskStatus,
    get_registry,
)


def _result(status: Status = Status.SUCCEEDED) -> ExecutionResult:
    error = None
    if status is not Status.SUCCEEDED:
        error = ExecError(type=ErrorType.INTERNAL, message="boom", retryable=False)
    return ExecutionResult(task_id="t-1", status=status, outputs={"output": "ok"}, error=error)


class _FakeAgent:
    def __init__(self):
        self.interrupted = False

    def interrupt(self):
        self.interrupted = True


# ---------------------------------------------------------------------------
# Q3 ruling pins
# ---------------------------------------------------------------------------

def test_contract_status_stays_terminal_only():
    """Option B: RUNNING must never appear in contract.Status."""
    assert "running" not in {s.value for s in Status}


def test_task_status_terminal_values_mirror_contract_status():
    """A completed record's status string equals its result's status string."""
    contract_values = {s.value for s in Status}
    terminal_values = {s.value for s in TERMINAL_STATUSES}
    assert terminal_values == contract_values
    assert TaskStatus.RUNNING not in TERMINAL_STATUSES


# ---------------------------------------------------------------------------
# register / get / list_active / complete
# ---------------------------------------------------------------------------

def test_register_get_round_trip_and_started_at_stamp():
    reg = AgentTaskRegistry()
    rec = AgentTaskRecord(task_id="sa-0-aaaa", goal="do x", intent="delegate")
    reg.register(rec)

    got = reg.get("sa-0-aaaa")
    assert got is rec
    assert got.status is TaskStatus.RUNNING
    assert got.started_at > 0  # stamped by register()
    assert reg.get("missing") is None


def test_list_active_returns_only_running():
    reg = AgentTaskRegistry()
    reg.register(AgentTaskRecord(task_id="a"))
    reg.register(AgentTaskRecord(task_id="b"))
    assert {r.task_id for r in reg.list_active()} == {"a", "b"}

    assert reg.complete("a", _result()) is True
    assert {r.task_id for r in reg.list_active()} == {"b"}


def test_complete_mirrors_result_status_and_drops_agent_ref():
    reg = AgentTaskRegistry()
    agent = _FakeAgent()
    reg.register(AgentTaskRecord(task_id="t", agent_ref=agent))

    assert reg.complete("t", _result(Status.PARTIAL)) is True
    rec = reg.get("t")
    assert rec.status is TaskStatus.PARTIAL
    assert rec.agent_ref is None
    assert rec.finished_at is not None
    assert rec.result.status is Status.PARTIAL


def test_complete_with_none_result_maps_to_failed():
    """The timeout/interrupt path: no result, terminal FAILED."""
    reg = AgentTaskRegistry()
    reg.register(AgentTaskRecord(task_id="t"))
    assert reg.complete("t", None) is True
    rec = reg.get("t")
    assert rec.status is TaskStatus.FAILED
    assert rec.result is None


def test_complete_unknown_task_returns_false():
    assert AgentTaskRegistry().complete("nope", _result()) is False


# ---------------------------------------------------------------------------
# interrupt
# ---------------------------------------------------------------------------

def test_interrupt_calls_agent_interrupt():
    reg = AgentTaskRegistry()
    agent = _FakeAgent()
    reg.register(AgentTaskRecord(task_id="t", agent_ref=agent))

    assert reg.interrupt("t") is True
    assert agent.interrupted is True


def test_interrupt_resolves_weakref_agent():
    reg = AgentTaskRegistry()
    agent = _FakeAgent()
    reg.register(AgentTaskRecord(task_id="t", agent_ref=weakref.ref(agent)))

    assert reg.interrupt("t") is True
    assert agent.interrupted is True


def test_interrupt_unknown_terminal_or_dead_agent_returns_false():
    reg = AgentTaskRegistry()
    assert reg.interrupt("missing") is False

    reg.register(AgentTaskRecord(task_id="done", agent_ref=_FakeAgent()))
    reg.complete("done", _result())
    assert reg.interrupt("done") is False  # terminal

    reg.register(AgentTaskRecord(task_id="gone", agent_ref=None))
    assert reg.interrupt("gone") is False  # no live agent


# ---------------------------------------------------------------------------
# spawn pause flag
# ---------------------------------------------------------------------------

def test_pause_spawns_round_trip():
    reg = AgentTaskRegistry()
    assert reg.spawns_paused() is False
    reg.pause_spawns(True)
    assert reg.spawns_paused() is True
    reg.pause_spawns(False)
    assert reg.spawns_paused() is False


# ---------------------------------------------------------------------------
# idempotency replay store (ephemeral, TTL + cap — Q2 ruling)
# ---------------------------------------------------------------------------

def test_replay_remember_recall_round_trip():
    reg = AgentTaskRegistry()
    reg.remember("key-1", {"status": "succeeded"}, now=1000.0)
    assert reg.recall("key-1", now=1000.0) == {"status": "succeeded"}
    assert reg.recall("missing", now=1000.0) is None


def test_replay_ttl_eviction():
    reg = AgentTaskRegistry()
    reg.remember("key-1", {"status": "succeeded"}, now=1000.0)
    # one second past the TTL → evicted
    assert reg.recall("key-1", now=1000.0 + AgentTaskRegistry.REPLAY_TTL_S + 1) is None


def test_replay_cap_evicts_oldest():
    reg = AgentTaskRegistry()
    for i in range(AgentTaskRegistry.REPLAY_CAP):
        reg.remember(f"k{i}", {"i": i}, now=1000.0 + i)
    reg.remember("overflow", {"i": -1}, now=2500.0)
    assert reg.recall("k0", now=2500.0) is None  # oldest evicted
    assert reg.recall("overflow", now=2500.0) == {"i": -1}


def test_complete_with_idempotency_key_populates_replay():
    reg = AgentTaskRegistry()
    reg.register(AgentTaskRecord(task_id="t", idempotency_key="idem-1"))
    reg.complete("t", _result())
    replayed = reg.recall("idem-1")
    assert replayed is not None
    assert replayed["status"] == "succeeded"
    assert replayed["task_id"] == "t"


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------

def test_snapshot_excludes_agent_ref_and_embeds_rich_result():
    rec = AgentTaskRecord(
        task_id="t", goal="g", intent="delegate", agent_ref=_FakeAgent(),
        started_at=1.0,
    )
    snap = rec.snapshot()
    assert "agent_ref" not in snap
    assert snap["status"] == "running"
    assert snap["result"] is None

    rec.result = _result()
    rec.status = TaskStatus.SUCCEEDED
    snap2 = rec.snapshot()
    assert snap2["result"]["status"] == "succeeded"


# ---------------------------------------------------------------------------
# registry-backed persistence (Step 6a)
# ---------------------------------------------------------------------------

def _patch_home(monkeypatch, tmp_path):
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)


def _tasks_file(tmp_path, session_id):
    return tmp_path / "spawn-trees" / session_id / "_tasks.jsonl"


def test_complete_with_session_id_appends_one_jsonl_line(tmp_path, monkeypatch):
    import json

    _patch_home(monkeypatch, tmp_path)
    reg = AgentTaskRegistry()
    reg.register(AgentTaskRecord(task_id="sa-0-aaaa", session_id="sess-1", goal="g"))
    assert reg.complete("sa-0-aaaa", _result()) is True

    lines = _tasks_file(tmp_path, "sess-1").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["task_id"] == "sa-0-aaaa"
    assert entry["session_id"] == "sess-1"
    assert entry["status"] == "succeeded"
    assert entry["result"]["status"] == "succeeded"


def test_complete_without_session_id_writes_nothing(tmp_path, monkeypatch):
    _patch_home(monkeypatch, tmp_path)
    reg = AgentTaskRegistry()
    reg.register(AgentTaskRecord(task_id="t"))
    assert reg.complete("t", _result()) is True
    assert not (tmp_path / "spawn-trees").exists()


def test_persist_failure_never_breaks_complete(tmp_path, monkeypatch):
    from pathlib import Path

    _patch_home(monkeypatch, tmp_path)

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", _boom)

    reg = AgentTaskRegistry()
    reg.register(AgentTaskRecord(task_id="t", session_id="sess-1", agent_ref=_FakeAgent()))
    assert reg.complete("t", _result()) is True  # persist failure swallowed

    rec = reg.get("t")
    assert rec.status is TaskStatus.SUCCEEDED  # record still terminalized
    assert rec.agent_ref is None
    assert rec.finished_at is not None


def test_snapshot_includes_session_id_only_when_set():
    """Wire-compat pin: task.status/task.list spread snapshot() byte-identical,
    so the key must be absent on records without a session (additive-first)."""
    assert "session_id" not in AgentTaskRecord(task_id="t").snapshot()
    assert AgentTaskRecord(task_id="t", session_id="s").snapshot()["session_id"] == "s"


def test_two_completes_append_two_lines(tmp_path, monkeypatch):
    import json

    _patch_home(monkeypatch, tmp_path)
    reg = AgentTaskRegistry()
    reg.register(AgentTaskRecord(task_id="sa-0-aaaa", session_id="sess-1"))
    reg.register(AgentTaskRecord(task_id="sa-1-bbbb", session_id="sess-1"))
    assert reg.complete("sa-0-aaaa", _result()) is True
    assert reg.complete("sa-1-bbbb", _result(Status.FAILED)) is True

    lines = _tasks_file(tmp_path, "sess-1").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    entries = [json.loads(line) for line in lines]
    assert [e["task_id"] for e in entries] == ["sa-0-aaaa", "sa-1-bbbb"]
    assert [e["status"] for e in entries] == ["succeeded", "failed"]


# ---------------------------------------------------------------------------
# singleton
# ---------------------------------------------------------------------------

def test_get_registry_singleton():
    assert get_registry() is get_registry()
