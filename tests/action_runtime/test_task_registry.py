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
    TOOLS_TAIL_CAP,
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
    # Q5 parity: no message exists on the None-result path — never invented.
    assert rec.error is None


def test_complete_lifts_error_summary_from_result():
    """Q5 parity: complete() copies result.error.message onto the record's
    one-line "error" field; SUCCEEDED results leave it unset."""
    reg = AgentTaskRegistry()
    reg.register(AgentTaskRecord(task_id="fail"))
    reg.register(AgentTaskRecord(task_id="ok"))

    assert reg.complete("fail", _result(Status.FAILED)) is True
    assert reg.get("fail").error == "boom"  # _result's ExecError message
    assert reg.get_snapshot("fail")["error"] == "boom"

    assert reg.complete("ok", _result()) is True
    assert reg.get("ok").error is None
    assert "error" not in reg.get_snapshot("ok")


def test_complete_unknown_task_returns_false():
    assert AgentTaskRegistry().complete("nope", _result()) is False


def test_second_complete_is_rejected_and_record_unchanged(tmp_path, monkeypatch):
    """A late second complete() must not falsify the first terminal result
    nor append a second _tasks.jsonl line."""
    import json

    _patch_home(monkeypatch, tmp_path)
    reg = AgentTaskRegistry()
    reg.register(AgentTaskRecord(task_id="t", session_id="sess-1"))
    assert reg.complete("t", _result()) is True
    first_finished = reg.get("t").finished_at

    assert reg.complete("t", _result(Status.FAILED)) is False
    rec = reg.get("t")
    assert rec.status is TaskStatus.SUCCEEDED  # not overwritten
    assert rec.result.status is Status.SUCCEEDED
    assert rec.finished_at == first_finished

    lines = _tasks_file(tmp_path, "sess-1").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "succeeded"


def test_terminal_cap_evicts_oldest_terminal_never_running():
    reg = AgentTaskRegistry()
    reg.RECORDS_TERMINAL_CAP = 2
    reg.register(AgentTaskRecord(task_id="run"))  # stays RUNNING throughout
    for i in range(3):
        reg.register(AgentTaskRecord(task_id=f"t{i}"))
        assert reg.complete(f"t{i}", _result()) is True

    assert reg.get("run") is not None  # RUNNING is never evicted
    assert reg.get("t0") is None  # oldest terminal by finished_at evicted
    assert reg.get("t1") is not None
    assert reg.get("t2") is not None
    assert {r.task_id for r in reg.list_active()} == {"run"}


# ---------------------------------------------------------------------------
# update_progress
# ---------------------------------------------------------------------------

def test_update_progress_updates_running_record_fields():
    reg = AgentTaskRegistry()
    reg.register(AgentTaskRecord(task_id="t"))

    assert reg.update_progress("t", tool_count=3, last_tool="terminal") is True
    rec = reg.get("t")
    assert rec.tool_count == 3
    assert rec.last_tool == "terminal"

    # Partial update: an omitted field keeps its previous value.
    assert reg.update_progress("t", tool_count=4) is True
    rec = reg.get("t")
    assert rec.tool_count == 4
    assert rec.last_tool == "terminal"


def test_update_progress_accumulates_tools_tail():
    """Q5 parity: last_tool updates accumulate into the tools tail —
    distinct-consecutive names, last TOOLS_TAIL_CAP kept (the TUI's
    pushUnique(8) semantics), empty names never recorded."""
    reg = AgentTaskRegistry()
    reg.register(AgentTaskRecord(task_id="t"))

    reg.update_progress("t", last_tool="bash")
    reg.update_progress("t", last_tool="bash")  # consecutive repeat collapses
    reg.update_progress("t", last_tool="web_search")
    reg.update_progress("t", last_tool="")  # empty: last_tool set, not appended
    reg.update_progress("t", tool_count=9)  # no last_tool: tail untouched
    rec = reg.get("t")
    assert rec.tools == ["bash", "web_search"]
    assert rec.last_tool == ""
    assert rec.tool_count == 9

    for i in range(TOOLS_TAIL_CAP + 3):
        reg.update_progress("t", last_tool=f"tool-{i}")
    tail = reg.get("t").tools
    assert len(tail) == TOOLS_TAIL_CAP
    assert tail[-1] == f"tool-{TOOLS_TAIL_CAP + 2}"  # newest kept, oldest dropped


def test_update_progress_noop_on_missing_or_terminal():
    reg = AgentTaskRegistry()
    assert reg.update_progress("missing", tool_count=1) is False

    reg.register(AgentTaskRecord(task_id="t"))
    reg.update_progress("t", tool_count=2, last_tool="web_search")
    assert reg.complete("t", _result()) is True
    # A late callback after complete() must never mutate a terminal record.
    assert reg.update_progress("t", tool_count=99, last_tool="late") is False
    rec = reg.get("t")
    assert rec.tool_count == 2
    assert rec.last_tool == "web_search"


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


def test_interrupt_swallows_agent_exception():
    """Parity with interrupt_subagent: a raising agent yields False, never
    an exception escaping into the RPC dispatcher."""

    class _RaisingAgent:
        def interrupt(self, message=None):
            raise RuntimeError("boom")

    reg = AgentTaskRegistry()
    reg.register(AgentTaskRecord(task_id="t", agent_ref=_RaisingAgent()))
    assert reg.interrupt("t") is False


def test_interrupt_forwards_reason_when_given():
    class _ReasonAgent:
        def __init__(self):
            self.message = None

        def interrupt(self, message=None):
            self.message = message

    reg = AgentTaskRegistry()
    agent = _ReasonAgent()
    reg.register(AgentTaskRecord(task_id="t", agent_ref=agent))
    assert reg.interrupt("t", reason="user cancelled") is True
    assert agent.message == "user cancelled"


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
    # task_id matches _result()'s: since the R5 fix the replay dict is the
    # rich result wire, so its task_id comes from the ExecutionResult (which
    # in production always equals the record's id).
    reg = AgentTaskRegistry()
    reg.register(AgentTaskRecord(task_id="t-1", idempotency_key="idem-1"))
    reg.complete("t-1", _result())
    replayed = reg.recall("idem-1")
    assert replayed is not None
    assert replayed["status"] == "succeeded"
    assert replayed["task_id"] == "t-1"


def test_replay_stores_rich_wire_shape():
    """task.submit replays the stored dict directly as the RPC result, so it
    must be the result_to_wire_rich shape — not a record snapshot."""
    reg = AgentTaskRegistry()
    reg.register(AgentTaskRecord(task_id="t", idempotency_key="idem-1"))
    reg.complete("t", _result())
    replayed = reg.recall("idem-1")
    assert set(replayed) == {"task_id", "status", "outputs", "error", "side_effects"}
    assert replayed["outputs"] == {"output": "ok"}


def test_find_running_by_key_matches_only_live_records():
    """The in-flight half of idempotency: a RUNNING record is findable by
    its key; completion frees the key (the replay store owns it from then
    on); falsy keys never match the default-None majority."""
    reg = AgentTaskRegistry()
    reg.register(AgentTaskRecord(task_id="no-key"))  # idempotency_key=None
    reg.register(AgentTaskRecord(task_id="t-keyed", idempotency_key="idem-1"))

    found = reg.find_running_by_key("idem-1")
    assert found is not None
    assert found.task_id == "t-keyed"
    assert reg.find_running_by_key("other-key") is None
    # None/"" must NOT match the keyless RUNNING record.
    assert reg.find_running_by_key(None) is None
    assert reg.find_running_by_key("") is None

    reg.complete("t-keyed", _result())
    assert reg.find_running_by_key("idem-1") is None


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


def test_get_snapshot_and_list_active_snapshots_round_trip():
    """Locked RPC readers: serialized views, RUNNING-only listing, no
    agent_ref leakage."""
    reg = AgentTaskRegistry()
    reg.register(AgentTaskRecord(task_id="a", goal="g", agent_ref=_FakeAgent()))
    reg.register(AgentTaskRecord(task_id="b"))
    assert reg.complete("b", _result()) is True

    snap = reg.get_snapshot("a")
    assert snap["task_id"] == "a"
    assert snap["status"] == "running"
    assert "agent_ref" not in snap
    assert reg.get_snapshot("b")["status"] == "succeeded"
    assert reg.get_snapshot("missing") is None

    active = reg.list_active_snapshots()
    assert [s["task_id"] for s in active] == ["a"]
    assert active[0]["status"] == "running"


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


def test_snapshot_includes_trace_id_only_when_set():
    """Same absent-when-None additive rule as session_id (Task 2.1 §12):
    pre-trace records keep the exact pre-trace snapshot shape."""
    assert "trace_id" not in AgentTaskRecord(task_id="t").snapshot()
    assert AgentTaskRecord(task_id="t", trace_id="tr-1").snapshot()["trace_id"] == "tr-1"


def test_snapshot_includes_q5_parity_fields_only_when_set():
    """Same absent-when-unset rule for the Q5 parity trio (label/tools/error):
    records that predate the fields keep the exact prior snapshot shape."""
    bare = AgentTaskRecord(task_id="t").snapshot()
    assert "label" not in bare
    assert "tools" not in bare
    assert "error" not in bare

    rich = AgentTaskRecord(
        task_id="t", label="do x", tools=["bash", "web_search"], error="boom"
    )
    snap = rich.snapshot()
    assert snap["label"] == "do x"
    assert snap["tools"] == ["bash", "web_search"]
    assert snap["error"] == "boom"
    # The snapshot owns a copy — later live mutation can't tear it.
    rich.tools.append("late")
    assert snap["tools"] == ["bash", "web_search"]


def test_ledger_line_carries_q5_parity_fields(tmp_path, monkeypatch):
    """End-to-end: register with a label, accumulate tools via
    update_progress, fail the result — the persisted _tasks.jsonl line
    carries label/tools/error alongside the pre-existing keys."""
    import json

    _patch_home(monkeypatch, tmp_path)
    reg = AgentTaskRegistry()
    reg.register(
        AgentTaskRecord(task_id="sa-0-aaaa", session_id="sess-1", goal="g", label="g")
    )
    reg.update_progress("sa-0-aaaa", tool_count=1, last_tool="bash")
    reg.update_progress("sa-0-aaaa", tool_count=2, last_tool="web_search")
    assert reg.complete("sa-0-aaaa", _result(Status.FAILED)) is True

    lines = _tasks_file(tmp_path, "sess-1").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["label"] == "g"
    assert entry["tools"] == ["bash", "web_search"]
    assert entry["error"] == "boom"
    assert entry["last_tool"] == "web_search"  # pre-existing keys untouched
    assert entry["status"] == "failed"


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
# observer seam (Task 2.2 push events)
# ---------------------------------------------------------------------------

def test_observer_receives_started_and_completed_snapshots():
    reg = AgentTaskRegistry()
    events = []
    reg.set_observer(lambda event, snap: events.append((event, snap)))

    reg.register(AgentTaskRecord(task_id="t-obs", goal="watch"))
    assert [e for e, _ in events] == ["started"]
    assert events[0][1]["task_id"] == "t-obs"
    assert events[0][1]["status"] == "running"

    assert reg.complete("t-obs", _result()) is True
    assert [e for e, _ in events] == ["started", "completed"]
    assert events[1][1]["status"] == "succeeded"
    assert events[1][1]["result"]["outputs"] == {"output": "ok"}


def test_observer_exception_never_breaks_register_or_complete():
    """set_observer guard: an observer bug must never break the ledger."""
    reg = AgentTaskRegistry()

    def _boom(event, snap):
        raise RuntimeError("observer bug")

    reg.set_observer(_boom)
    rec = AgentTaskRecord(task_id="t-obs-boom")
    reg.register(rec)  # must not raise
    assert reg.get("t-obs-boom") is rec
    assert reg.complete("t-obs-boom", _result()) is True  # must not raise
    assert reg.get("t-obs-boom").status is TaskStatus.SUCCEEDED


def test_rejected_complete_does_not_notify():
    """Only a SUCCESSFUL complete() notifies — a late duplicate or an unknown
    task_id must never re-emit task.completed downstream."""
    reg = AgentTaskRegistry()
    events = []
    reg.register(AgentTaskRecord(task_id="t-obs-dup"))
    reg.set_observer(lambda event, snap: events.append(event))

    assert reg.complete("t-obs-dup", _result()) is True
    assert reg.complete("t-obs-dup", _result(Status.FAILED)) is False
    assert reg.complete("missing", _result()) is False
    assert events == ["completed"]


def test_set_observer_none_detaches():
    reg = AgentTaskRegistry()
    events = []
    reg.set_observer(lambda event, snap: events.append(event))
    reg.set_observer(None)
    reg.register(AgentTaskRecord(task_id="t-obs-off"))
    reg.complete("t-obs-off", _result())
    assert events == []


# ---------------------------------------------------------------------------
# singleton
# ---------------------------------------------------------------------------

def test_get_registry_singleton():
    assert get_registry() is get_registry()
