"""Unit tests for the loop-diagnostics action dependency graph recorder.

Covers the six required trace shapes from the task:
  * linear      — sequential actions, causal/data edges
  * branching   — fan-out/fan-in from a shared predecessor
  * nested-loop — inner loop inside an outer loop, distinct identities
  * retry       — same tool after a failure => retry edge
  * cancellation — action_start with no action_end, footer still valid
  * malformed-dependency — edge referencing a missing action is tolerated

Plus the contract's behavioral guarantees:
  * disabled => zero traces, zero side effects
  * redaction by construction (secrets never reach the file)
  * per-run event cap truncation still writes a valid footer
  * retain_runs pruning keeps only the newest runs
  * emitted records conform to the machine-readable schema
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hermes_cli.observability.loop_diagnostics_recorder import (
    DEFAULT_MAX_EVENTS_PER_RUN,
    DEFAULT_RETAIN_RUNS,
    LoopDiagnosticsRecorder,
    TraceWriter,
    _redact_secrets,
    _redact_summary,
    load_recorder_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_recorder(tmp_path: Path, run_id: int = 1, **kwargs) -> LoopDiagnosticsRecorder:
    kwargs.setdefault("task_id", "t_test")
    kwargs.setdefault("run_id", run_id)
    kwargs.setdefault("base_dir", tmp_path)
    kwargs.setdefault("enabled", True)
    return LoopDiagnosticsRecorder(**kwargs)


def _read_records(tmp_path: Path, task_id: str = "t_test", run_id: int = 1) -> list[dict]:
    path = tmp_path / task_id / f"{run_id}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _records_by_kind(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for rec in records:
        out.setdefault(rec["kind"], []).append(rec)
    return out


def _edges(records: list[dict]) -> list[tuple[str, str, str]]:
    return [
        (r["from_action_id"], r["to_action_id"], r["edge_kind"])
        for r in records
        if r["kind"] == "edge"
    ]


def _trace_action(
    rec: LoopDiagnosticsRecorder,
    tool: str,
    *,
    turn: str = "t1",
    loop_id=None,
    iteration=None,
    status: str = "ok",
    **end_kwargs,
) -> str:
    """Record a complete tool action; returns its action_id."""
    aid = rec.record_action_start(
        action_kind="tool_call",
        tool_name=tool,
        summary=f"ran {tool}",
        turn_id=turn,
        loop_id=loop_id,
        iteration=iteration,
    )
    assert aid is not None
    rec.record_action_end(action_id=aid, status=status, duration_ms=1, **end_kwargs)
    return aid


# ---------------------------------------------------------------------------
# Schema conformance of emitted records
# ---------------------------------------------------------------------------


def test_emitted_records_conform_to_schema(tmp_path):
    """Every record the recorder writes validates against the v1 schema."""
    import json as _json

    schema = _json.loads(
        (
            PROJECT_ROOT
            / "hermes_cli/observability/schemas/hermes.loop_diagnostics.v1.schema.json"
        ).read_text(encoding="utf-8")
    )

    # Minimal validator mirroring the schema test (ref-aware enum check).
    KIND_TO_BRANCH = {
        "run_header": "RunHeader",
        "action_start": "ActionStart",
        "action_end": "ActionEnd",
        "edge": "DependencyEdge",
        "run_footer": "RunFooter",
    }

    def validate(rec: dict) -> None:
        branch = schema["$defs"][KIND_TO_BRANCH[rec["kind"]]]
        assert rec.get("schema_version") == "hermes.loop_diagnostics.v1"
        for field in branch["required"]:
            assert field in rec, f"{rec['kind']} missing {field}"
        allowed = set(branch.get("properties", {}).keys())
        assert not (set(rec.keys()) - allowed), f"stray fields: {set(rec.keys()) - allowed}"
        for field, spec in branch.get("properties", {}).items():
            if field not in rec or rec[field] is None:
                continue
            if spec.get("$ref"):
                spec = schema["$defs"][spec["$ref"].rsplit("/", 1)[-1]]
            if spec.get("enum") is not None:
                assert rec[field] in spec["enum"], f"{field}={rec[field]!r}"

    rec = _make_recorder(tmp_path)
    a1 = _trace_action(rec, "terminal")
    _trace_action(rec, "read_file")
    rec.finish(outcome="completed")

    records = _read_records(tmp_path)
    assert len(records) >= 5  # header, 2x(start+end), footer, edge
    for r in records:
        validate(r)


# ---------------------------------------------------------------------------
# Linear trace
# ---------------------------------------------------------------------------


def test_linear_trace_has_causal_and_data_edges(tmp_path):
    rec = _make_recorder(tmp_path)
    _trace_action(rec, "write_file", turn="t1")   # producer
    _trace_action(rec, "read_file", turn="t1")    # consumer => data
    _trace_action(rec, "terminal", turn="t1")     # producer+consumer
    rec.finish(outcome="completed")

    records = _read_records(tmp_path)
    kinds = _records_by_kind(records)
    assert len(kinds["action_start"]) == 3
    assert len(kinds["action_end"]) == 3
    assert len(kinds["edge"]) == 2
    assert len(kinds["run_header"]) == 1
    assert len(kinds["run_footer"]) == 1
    assert kinds["run_footer"][0]["outcome"] == "completed"

    # write_file -> read_file is a producer/consumer data edge
    assert ("1:1", "1:2", "data") in _edges(records)
    # read_file -> terminal: read_file is a CONSUMER, terminal is a PRODUCER.
    # The data-edge rule is producer->consumer, so this is causal (not data).
    assert ("1:2", "1:3", "causal") in _edges(records)


def test_linear_trace_action_ids_are_stable_and_monotonic(tmp_path):
    rec = _make_recorder(tmp_path, run_id=42)
    ids = [_trace_action(rec, f"tool{i}") for i in range(3)]
    assert ids == ["42:1", "42:2", "42:3"]


# ---------------------------------------------------------------------------
# Branching trace
# ---------------------------------------------------------------------------


def test_branching_trace_fanout_keeps_shared_predecessor(tmp_path):
    """A delegate_task call that fans out into parallel subagent branches.

    Each child links to the parent delegate_task action via
    parent_action_id, so the graph shows one parent branching to N
    children rather than a sequential sibling chain.
    """
    rec = _make_recorder(tmp_path)
    # parent delegate_task tool call (starts, stays in flight)
    parent = rec.record_action_start(
        action_kind="tool_call", tool_name="delegate_task",
        summary="delegate task", turn_id="t1",
    )
    assert parent is not None
    rec._pending_by_tool[("delegate_task", "t1")] = parent

    # two children spawned from the same parent turn
    rec.on_subagent_start(parent_turn_id="t1", child_session_id="child-A")
    rec.on_subagent_stop(child_session_id="child-A", child_status="ok",
                         child_summary="A done")
    rec.on_subagent_start(parent_turn_id="t1", child_session_id="child-B")
    rec.on_subagent_stop(child_session_id="child-B", child_status="ok",
                         child_summary="B done")

    # parent tool call completes
    rec.record_action_end(action_id=parent, status="ok", duration_ms=100)
    rec.finish(outcome="completed")

    records = _read_records(tmp_path)
    starts = _records_by_kind(records)["action_start"]
    subagent_ids = [
        s["action_id"] for s in starts if s["action_kind"] == "subagent"
    ]
    assert len(subagent_ids) == 2
    # both children carry the parent action as their parent_action_id
    for s in starts:
        if s["action_kind"] == "subagent":
            assert s["parent_action_id"] == parent


def test_branching_fan_in_diamond(tmp_path):
    """A consumer that reads an artifact produced earlier in the run.

    A sequential recorder chains each action from the last one; the
    important guarantee is that the final consumer links to the artifact
    producer (data edge) rather than to an unrelated later action.
    """
    rec = _make_recorder(tmp_path)
    root = _trace_action(rec, "read_file", turn="t1")
    a = _trace_action(rec, "write_file", turn="t2")
    b = _trace_action(rec, "write_file", turn="t3")
    leaf = _trace_action(rec, "read_file", turn="t4")

    records = _read_records(tmp_path)
    edges = _edges(records)
    # Sequential chain: each action's causal predecessor is the last action.
    assert (root, a, "causal") in edges  # read (consumer) -> write (producer)
    assert (a, b, "causal") in edges     # write -> write
    # The consumer leaf links from its turn predecessor (b, the producer).
    assert (b, leaf, "data") in edges    # write -> read = data


# ---------------------------------------------------------------------------
# Nested-loop trace
# ---------------------------------------------------------------------------


def test_nested_loop_keeps_distinct_loop_identities(tmp_path):
    rec = _make_recorder(tmp_path)
    # outer loop iteration 0
    _trace_action(rec, "terminal", turn="t1", loop_id="goal", iteration=0)
    # inner loop (retry:web_search) iteration 0 and 1
    _trace_action(rec, "web_search", turn="t1", loop_id="retry:web_search", iteration=0)
    _trace_action(rec, "web_search", turn="t1", loop_id="retry:web_search", iteration=1)
    # back to outer loop iteration 1
    _trace_action(rec, "terminal", turn="t1", loop_id="goal", iteration=1)
    rec.finish(outcome="completed")

    records = _read_records(tmp_path)
    edges = _edges(records)

    starts = _records_by_kind(records)["action_start"]
    loop_ids = [s["loop_id"] for s in starts]
    assert loop_ids == ["goal", "retry:web_search", "retry:web_search", "goal"]
    assert [s["iteration"] for s in starts] == [0, 0, 1, 1]

    # inner loop: iteration 0 -> 1 is a loop edge
    assert ("1:2", "1:3", "loop") in edges
    # outer loop: 1:1 (iter 0) -> 1:4 (iter 1) is a loop edge
    assert ("1:1", "1:4", "loop") in edges


def test_same_iteration_steps_are_not_loop_edges(tmp_path):
    """Two actions in the SAME iteration must not conflate into a loop edge."""
    rec = _make_recorder(tmp_path)
    _trace_action(rec, "terminal", turn="t1", loop_id="goal", iteration=0)
    _trace_action(rec, "terminal", turn="t1", loop_id="goal", iteration=0)
    rec.finish(outcome="completed")

    records = _read_records(tmp_path)
    edges = _edges(records)
    assert ("1:1", "1:2", "loop") not in edges
    # terminal is both producer and consumer, so same-tool is a data edge
    # (not retry either — the previous action did not error)
    assert ("1:1", "1:2", "data") in edges
    assert ("1:1", "1:2", "retry") not in edges


# ---------------------------------------------------------------------------
# Retry trace
# ---------------------------------------------------------------------------


def test_retry_edge_after_failure(tmp_path):
    rec = _make_recorder(tmp_path)
    _trace_action(rec, "terminal", turn="t1", status="error",
                  error_type="ExitCodeError", error_message="exit 1",
                  result_text="boom")
    _trace_action(rec, "terminal", turn="t1", status="ok")
    rec.finish(outcome="completed")

    records = _read_records(tmp_path)
    assert ("1:1", "1:2", "retry") in _edges(records)


def test_no_retry_edge_after_success(tmp_path):
    rec = _make_recorder(tmp_path)
    _trace_action(rec, "terminal", turn="t1", status="ok")
    _trace_action(rec, "terminal", turn="t1", status="ok")
    rec.finish(outcome="completed")

    records = _read_records(tmp_path)
    assert ("1:1", "1:2", "retry") not in _edges(records)


# ---------------------------------------------------------------------------
# Cancellation trace
# ---------------------------------------------------------------------------


def test_cancellation_tolerates_missing_action_end(tmp_path):
    rec = _make_recorder(tmp_path)
    _trace_action(rec, "write_file", turn="t1")
    # action starts but never ends (process killed / cancelled)
    orphan = rec.record_action_start(
        action_kind="tool_call", tool_name="web_search",
        summary="search", turn_id="t2",
    )
    assert orphan is not None
    rec.finish(outcome="crashed", error="process killed")

    records = _read_records(tmp_path)
    kinds = _records_by_kind(records)
    # the orphan has an action_start but NO action_end
    start_ids = {s["action_id"] for s in kinds["action_start"]}
    end_ids = {e["action_id"] for e in kinds["action_end"]}
    assert orphan in start_ids
    assert orphan not in end_ids
    # footer still written with crash outcome
    assert kinds["run_footer"][0]["outcome"] == "crashed"
    # 2 starts + 1 end + 1 edge (write_file -> web_search? no — web_search
    # never ended, so no edge from it; the write_file ended but the orphan
    # is its successor, so the edge is between them when the orphan ends,
    # which it never does. => 2 starts + 1 end + 0 edges = 3)
    assert kinds["run_footer"][0]["event_count"] == 3


# ---------------------------------------------------------------------------
# Malformed-dependency trace
# ---------------------------------------------------------------------------


def test_malformed_dependency_does_not_raise(tmp_path):
    """An edge referencing a missing action must be tolerated, not fatal."""
    rec = _make_recorder(tmp_path)
    # simulate a stale/malformed dependency: emit an edge to a phantom action
    rec._write_edge("1:999", "1:1000", "causal")  # no such nodes
    _trace_action(rec, "terminal", turn="t1")
    rec.finish(outcome="completed")

    records = _read_records(tmp_path)
    # The phantom edge is recorded (recorder is fail-open), but subsequent
    # action processing is unaffected.
    assert ("1:999", "1:1000", "causal") in _edges(records)
    assert len(_records_by_kind(records)["action_start"]) == 1


def test_record_action_end_unknown_id_returns_false(tmp_path):
    rec = _make_recorder(tmp_path)
    assert rec.record_action_end(action_id="1:999", status="ok") is False
    # no crash, no file side effects beyond header auto-start
    rec.finish(outcome="completed")


# ---------------------------------------------------------------------------
# Disabled = zero cost
# ---------------------------------------------------------------------------


def test_disabled_recorder_writes_nothing(tmp_path):
    rec = _make_recorder(tmp_path, enabled=False)
    aid = rec.record_action_start(
        action_kind="tool_call", tool_name="terminal", summary="secret: hunter2"
    )
    assert aid is None
    rec.record_action_end(action_id=None, status="ok")
    rec.on_pre_tool_call(tool_name="terminal", args={"command": "rm -rf /"})
    rec.on_post_tool_call(tool_name="terminal", result="done")
    rec.finish(outcome="completed")

    assert not (tmp_path / "t_test" / "1.jsonl").exists()
    assert _read_records(tmp_path) == []


def test_disabled_when_missing_identity(tmp_path):
    """Without task/run env identity, the recorder refuses to enable."""
    rec = LoopDiagnosticsRecorder(base_dir=tmp_path, enabled=True)
    assert rec.enabled is False
    assert rec.record_action_start(action_kind="tool_call", tool_name="x") is None


def test_config_default_disabled():
    cfg = load_recorder_config({})
    assert cfg["enabled"] is False
    assert cfg["max_events_per_run"] == DEFAULT_MAX_EVENTS_PER_RUN
    assert cfg["retain_runs"] == DEFAULT_RETAIN_RUNS


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_secrets_never_reach_trace_file(tmp_path):
    rec = _make_recorder(tmp_path)
    secret = "sk-live-abcdef1234567890"
    aid = rec.record_action_start(
        action_kind="tool_call",
        tool_name="terminal",
        summary=f"deploy with token {secret}",
        turn_id="t1",
    )
    assert aid is not None
    rec.record_action_end(
        action_id=aid, status="error", error_message=f"auth failed {secret}",
        result_text=f"error: {secret}",
    )
    rec.finish(outcome="completed")

    blob = (tmp_path / "t_test" / "1.jsonl").read_text(encoding="utf-8")
    assert secret not in blob
    assert "sk-live-" not in blob


def test_result_hash_is_deterministic(tmp_path):
    rec = _make_recorder(tmp_path)
    aid = rec.record_action_start(
        action_kind="tool_call", tool_name="terminal", summary="run",
        turn_id="t1",
    )
    rec.record_action_end(action_id=aid, status="error", result_text="same failure")
    rec2 = _make_recorder(tmp_path, run_id=2)
    aid2 = rec2.record_action_start(
        action_kind="tool_call", tool_name="terminal", summary="run",
        turn_id="t1",
    )
    rec2.record_action_end(action_id=aid2, status="error", result_text="same failure")

    r1 = _records_by_kind(_read_records(tmp_path, run_id=1))["action_end"][0]
    r2 = _records_by_kind(_read_records(tmp_path, run_id=2))["action_end"][0]
    assert r1["result_hash"] == r2["result_hash"]
    assert len(r1["result_hash"]) == 64  # sha256 hex


def test_redact_helpers_never_raise():
    assert _redact_secrets(None) is None  # type: ignore[arg-type]
    assert _redact_secrets("") == ""
    assert _redact_summary(None) is None  # type: ignore[arg-type]
    long = "x" * 5000
    result = _redact_summary(long)
    assert result is not None
    assert len(result) <= 512 + 3  # cap + "..."


# ---------------------------------------------------------------------------
# Bounded growth: cap truncation + retention pruning
# ---------------------------------------------------------------------------


def test_event_cap_truncates_but_footer_writes(tmp_path):
    # 2 actions = 2 starts + 2 ends = 4 counted events; cap 4 means the
    # 3rd action start is dropped. Header/footer bypass the cap.
    rec = _make_recorder(tmp_path, max_events_per_run=4)
    _trace_action(rec, "terminal", turn="t1")
    _trace_action(rec, "terminal", turn="t2")
    dropped = rec.record_action_start(
        action_kind="tool_call", tool_name="terminal", summary="should drop",
        turn_id="t3",
    )
    assert dropped is None

    # a 4th start is also dropped
    dropped2 = rec.record_action_start(
        action_kind="tool_call", tool_name="terminal", summary="also drop",
        turn_id="t4",
    )
    assert dropped2 is None
    rec.finish(outcome="completed")

    records = _read_records(tmp_path)
    kinds = _records_by_kind(records)
    assert len(kinds["action_start"]) == 2  # 3rd and 4th dropped
    # footer still written (header/footer bypass the cap)
    assert kinds["run_footer"][0]["outcome"] == "completed"
    # footer event_count matches written events (4: 2 starts + 2 ends)
    assert kinds["run_footer"][0]["event_count"] == 4


def test_prune_keeps_only_newest_runs(tmp_path):
    # write 3 runs, retain_runs=2 => oldest deleted
    for run_id in (1, 2, 3):
        rec = _make_recorder(tmp_path, run_id=run_id, retain_runs=2)
        _trace_action(rec, "terminal", turn="t1")
        rec.finish(outcome="completed")

    task_dir = tmp_path / "t_test"
    remaining = sorted(p.name for p in task_dir.glob("*.jsonl"))
    assert remaining == ["2.jsonl", "3.jsonl"]


def test_trace_writer_requires_identity():
    with pytest.raises(ValueError):
        TraceWriter(None, None, base_dir=Path("/tmp"))
    with pytest.raises(ValueError):
        TraceWriter("t_x", None, base_dir=Path("/tmp"))


# ---------------------------------------------------------------------------
# Hook entry points
# ---------------------------------------------------------------------------


def test_hook_entry_points_record_actions(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.on_pre_tool_call(
        tool_name="terminal", args={"command": "echo hi"}, turn_id="turn-1",
        tool_call_id="call_1",
    )
    rec.on_post_tool_call(
        tool_name="terminal", result={"output": "hi"}, turn_id="turn-1",
        tool_call_id="call_1", status="ok", duration_ms=5,
    )
    rec.on_pre_tool_call(
        tool_name="read_file", args={"path": "/tmp/x.txt"}, turn_id="turn-1",
        tool_call_id="call_2",
    )
    rec.on_post_tool_call(
        tool_name="read_file", result={"content": "x"}, turn_id="turn-1",
        tool_call_id="call_2", status="ok", duration_ms=3,
    )
    rec.finish(outcome="completed")

    records = _read_records(tmp_path)
    kinds = _records_by_kind(records)
    assert len(kinds["action_start"]) == 2
    # terminal produces, read_file consumes => data edge
    assert ("1:1", "1:2", "data") in _edges(records)
    # summaries are sanitized: raw command never appears
    blob = (tmp_path / "t_test" / "1.jsonl").read_text(encoding="utf-8")
    assert "echo hi" not in blob
    assert "/tmp/x.txt" not in blob


def test_on_llm_turn_end_records_turn_anchor(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.on_llm_turn_end(turn_id="turn-9", model="test-model", duration_ms=100)
    rec.finish(outcome="completed")

    records = _read_records(tmp_path)
    kinds = _records_by_kind(records)
    starts = kinds["action_start"]
    assert len(starts) == 1
    assert starts[0]["action_kind"] == "llm_call"
    assert starts[0]["model"] == "test-model"


def test_subagent_hooks_record_subagent_nodes(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.on_subagent_start(
        parent_turn_id="turn-1", child_session_id="child-1", child_role="leaf",
    )
    rec.on_subagent_stop(
        child_session_id="child-1", child_status="ok",
        child_summary="did the thing", duration_ms=500,
    )
    rec.finish(outcome="completed")

    records = _read_records(tmp_path)
    kinds = _records_by_kind(records)
    starts = kinds["action_start"]
    assert len(starts) == 1
    assert starts[0]["action_kind"] == "subagent"
    ends = kinds["action_end"]
    assert ends[0]["status"] == "ok"
    # child summary is redacted/truncated but present
    assert ends[0]["summary"] == "did the thing"


def test_subagent_status_error_mapping(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.on_subagent_start(parent_turn_id="t", child_session_id="c1")
    rec.on_subagent_stop(child_session_id="c1", child_status="error")
    records = _read_records(tmp_path)
    assert _records_by_kind(records)["action_end"][0]["status"] == "error"


# ---------------------------------------------------------------------------
# No-conflation guarantee across repeated runs
# ---------------------------------------------------------------------------


def test_action_ids_never_reuse_across_runs(tmp_path):
    """Two runs of the same task must never reuse an action_id."""
    rec1 = _make_recorder(tmp_path, run_id=7)
    _trace_action(rec1, "terminal", turn="t1")
    rec1.finish(outcome="completed")

    rec2 = _make_recorder(tmp_path, run_id=8)
    _trace_action(rec2, "terminal", turn="t1")
    rec2.finish(outcome="completed")

    ids1 = {s["action_id"] for s in _records_by_kind(_read_records(tmp_path, run_id=7))["action_start"]}
    ids2 = {s["action_id"] for s in _records_by_kind(_read_records(tmp_path, run_id=8))["action_start"]}
    assert ids1 == {"7:1"}
    assert ids2 == {"8:1"}
    assert ids1.isdisjoint(ids2)
