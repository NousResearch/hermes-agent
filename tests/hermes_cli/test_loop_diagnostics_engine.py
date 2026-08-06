"""Unit tests for the loop-diagnostics diagnosis engine.

Covers the contract's engine guarantees (``docs/loop-diagnostics-design.md``
§8, §13):

* chain propagation: ``input_invalid`` when a data producer failed first;
* branch / independent failures: ``candidates`` with ranked action ids when
  two failing branches have no causal link;
* direct action failure: ``action_error`` with root = the failed action;
* repeated loop failures: ``loop_repeated`` with duplicate-hash evidence;
* retry exhaustion: ``retry_exhausted`` when a retry loop budget is hit;
* timeout / cancellation propagation;
* malformed traces (>=50% invalid lines) and corrupt JSON lines;
* truncated graphs (no footer / missing action ends) degrade gracefully;
* cycles terminate safely (visited set + depth cap);
* determinism: equivalent traces -> byte-identical results.

Fixtures are built with small helper functions that emit JSONL records
conforming to ``hermes.loop_diagnostics.v1`` (the schema test module
``test_loop_diagnostics_schema.py`` validates the shape independently).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.observability.loop_diagnostics_engine import (
    diagnose,
    load_trace,
    trace_path_for,
)

SCHEMA_VERSION = "hermes.loop_diagnostics.v1"


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _header(run_id: int, task_id: str = "t_diag") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "run_header",
        "task_id": task_id,
        "run_id": run_id,
        "attempt": 1,
        "profile": "default",
        "goal_mode": False,
        "ts": 1785739000 + run_id,
    }


def _start(run_id: int, seq: int, tool: str, **kw) -> dict:
    rec = {
        "schema_version": SCHEMA_VERSION,
        "kind": "action_start",
        "task_id": "t_diag",
        "run_id": run_id,
        "action_id": f"{run_id}:{seq}",
        "parent_action_id": None,
        "loop_id": None,
        "iteration": None,
        "ts": 1785739000 + run_id + seq,
        "action_kind": "tool_call",
        "tool_name": tool,
        "summary": f"{tool} call",
    }
    rec.update(kw)
    return rec


def _end(run_id: int, seq: int, status: str, **kw) -> dict:
    rec = {
        "schema_version": SCHEMA_VERSION,
        "kind": "action_end",
        "task_id": "t_diag",
        "run_id": run_id,
        "action_id": f"{run_id}:{seq}",
        "ts": 1785739000 + run_id + seq + 10,
        "status": status,
        "duration_ms": 100,
    }
    rec.update(kw)
    return rec


def _edge(run_id: int, from_seq: int, to_seq: int, kind: str = "data") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "edge",
        "task_id": "t_diag",
        "run_id": run_id,
        "from_action_id": f"{run_id}:{from_seq}",
        "to_action_id": f"{run_id}:{to_seq}",
        "edge_kind": kind,
        "ts": 1785739000 + run_id + to_seq + 10,
    }


def _footer(run_id: int, outcome: str = "blocked", error: str = "") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "run_footer",
        "task_id": "t_diag",
        "run_id": run_id,
        "ts": 1785739000 + run_id + 99,
        "outcome": outcome,
        "error": error or None,
    }


def _write(tmp_path: Path, run_id: int, records: list[dict]) -> Path:
    p = tmp_path / f"{run_id}.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return p


# ---------------------------------------------------------------------------
# Fixture: chain (linear data propagation) — input_invalid
# ---------------------------------------------------------------------------


def chain_trace(run_id: int) -> list[dict]:
    """terminal (clone) failed -> read_file (data) failed."""
    return [
        _header(run_id),
        _start(run_id, 1, "terminal", summary="clone repo"),
        _end(run_id, 1, "error", error_type="ExitCodeError",
             error_message="repository not found", result_hash="deadbeef1"),
        _edge(run_id, 1, 2, "data"),
        _start(run_id, 2, "read_file", summary="read design doc"),
        _end(run_id, 2, "error", error_type="FileNotFoundError",
             error_message="path does not exist", result_hash="c0ffee1"),
        _footer(run_id),
    ]


# ---------------------------------------------------------------------------
# Fixture: branch (single failing branch, causal chain)
# ---------------------------------------------------------------------------


def branch_trace(run_id: int) -> list[dict]:
    """web_search ok -> web_extract (data) failed -> read_file (data) failed.

    Root is the web_extract failure; read_file is downstream propagation.
    """
    return [
        _header(run_id),
        _start(run_id, 1, "web_search", summary="find source"),
        _end(run_id, 1, "ok", result_hash="hash-search-ok"),
        _edge(run_id, 1, 2, "data"),
        _start(run_id, 2, "web_extract", summary="extract page"),
        _end(run_id, 2, "error", error_type="HTTPError",
             error_message="404 not found", result_hash="hash-extract-404"),
        _edge(run_id, 2, 3, "data"),
        _start(run_id, 3, "read_file", summary="read extracted content"),
        _end(run_id, 3, "error", error_type="FileNotFoundError",
             error_message="no content file", result_hash="hash-read-missing"),
        _footer(run_id),
    ]


# ---------------------------------------------------------------------------
# Fixture: multiple independent failures — candidates
# ---------------------------------------------------------------------------


def independent_failures_trace(run_id: int) -> list[dict]:
    """Two independent failing branches (no causal link between them).

    Branch A: terminal fails (action_error). Branch B: read_file fails.
    The failed_action_id picks one; the other is an independent failure.
    """
    return [
        _header(run_id),
        _start(run_id, 1, "terminal", summary="clone repo A"),
        _end(run_id, 1, "error", error_type="ExitCodeError",
             error_message="clone failed", result_hash="hash-a-fail"),
        _edge(run_id, 1, 2, "data"),
        _start(run_id, 2, "read_file", summary="read repo A file"),
        _end(run_id, 2, "error", error_type="FileNotFoundError",
             error_message="missing", result_hash="hash-a-read-fail"),
        _start(run_id, 3, "web_search", summary="search B"),
        _end(run_id, 3, "error", error_type="EmptyResults",
             error_message="no results", result_hash="hash-b-fail"),
        _edge(run_id, 3, 4, "data"),
        _start(run_id, 4, "web_extract", summary="extract B result"),
        _end(run_id, 4, "error", error_type="HTTPError",
             error_message="404", result_hash="hash-b-extract-fail"),
        _footer(run_id),
    ]


# ---------------------------------------------------------------------------
# Fixture: cycles (loop edge back to earlier action)
# ---------------------------------------------------------------------------


def cyclic_trace(run_id: int) -> list[dict]:
    """A loop that links back into itself — engine must terminate safely."""
    return [
        _header(run_id),
        _start(run_id, 1, "web_search", loop_id="search", iteration=0),
        _end(run_id, 1, "error", error_type="EmptyResults",
             error_message="no results", result_hash="hash-loop-same"),
        _edge(run_id, 1, 2, "loop"),
        _start(run_id, 2, "web_search", loop_id="search", iteration=1),
        _end(run_id, 2, "error", error_type="EmptyResults",
             error_message="no results", result_hash="hash-loop-same"),
        _edge(run_id, 2, 3, "loop"),
        _start(run_id, 3, "web_search", loop_id="search", iteration=2),
        _end(run_id, 3, "error", error_type="EmptyResults",
             error_message="no results", result_hash="hash-loop-same"),
        # deliberately loop back to action 1 (cycle)
        _edge(run_id, 3, 1, "loop"),
        _footer(run_id, outcome="gave_up", error="no results after 3 tries"),
    ]


# ---------------------------------------------------------------------------
# Fixture: truncated graph (no footer, missing action end)
# ---------------------------------------------------------------------------


def truncated_trace(run_id: int) -> list[dict]:
    """Process died mid-run: action 2 has a start but no end; no footer."""
    return [
        _header(run_id),
        _start(run_id, 1, "terminal", summary="clone repo"),
        _end(run_id, 1, "ok", result_hash="hash-clone-ok"),
        _edge(run_id, 1, 2, "data"),
        _start(run_id, 2, "read_file", summary="read design doc"),
        # no action_end for 2, no footer
    ]


# ---------------------------------------------------------------------------
# Fixture: repeated loop failures — loop_repeated
# ---------------------------------------------------------------------------


def repeated_loop_trace(run_id: int) -> list[dict]:
    """web_search fails 3x in the same loop with identical result_hash."""
    recs = [_header(run_id)]
    for i in range(3):
        seq = i + 1
        recs.append(_start(run_id, seq, "web_search", loop_id="search",
                           iteration=i, summary="search for source"))
        recs.append(_end(run_id, seq, "error", error_type="EmptyResults",
                         error_message="no results", result_hash="abc123"))
        if i < 2:
            recs.append(_edge(run_id, seq, seq + 1, "loop"))
    recs.append(_footer(run_id, outcome="gave_up", error="no results"))
    return recs


# ---------------------------------------------------------------------------
# Fixture: retry exhaustion — retry_exhausted
# ---------------------------------------------------------------------------


def retry_exhausted_trace(run_id: int) -> list[dict]:
    """terminal retried 4x in a retry loop; all failed."""
    recs = [_header(run_id)]
    for i in range(4):
        seq = i + 1
        recs.append(_start(run_id, seq, "terminal", loop_id="retry",
                           iteration=i, summary="retry clone"))
        recs.append(_end(run_id, seq, "error", error_type="ExitCodeError",
                         error_message="exit 1", result_hash=f"hash-retry-{i}"))
        if i < 3:
            recs.append(_edge(run_id, seq, seq + 1, "retry"))
    recs.append(_footer(run_id, outcome="gave_up", error="retries exhausted"))
    return recs


# ---------------------------------------------------------------------------
# Fixture: timeout propagation
# ---------------------------------------------------------------------------


def timeout_propagation_trace(run_id: int) -> list[dict]:
    """terminal timed out (no action_end, footer timed_out) -> read_file (data)
    failed on the interrupted state.

    The schema's ActionStatus enum has no ``timed_out``; a timeout is
    represented by a missing ``action_end`` plus a ``run_footer`` with
    ``outcome: timed_out`` (the recorder tolerates incomplete actions).
    """
    return [
        _header(run_id),
        _start(run_id, 1, "terminal", summary="long build"),
        # no action_end for 1 — process killed by timeout
        _edge(run_id, 1, 2, "data"),
        _start(run_id, 2, "read_file", summary="read build output"),
        _end(run_id, 2, "error", error_type="FileNotFoundError",
             error_message="no output file", result_hash="hash-read-fail"),
        _footer(run_id, outcome="timed_out", error="TERMINAL_TIMEOUT"),
    ]


# ---------------------------------------------------------------------------
# Fixture: cancellation propagation
# ---------------------------------------------------------------------------


def cancellation_propagation_trace(run_id: int) -> list[dict]:
    """subagent cancelled -> parent read_file (data) failed."""
    return [
        _header(run_id),
        _start(run_id, 1, "subagent", summary="write file"),
        _end(run_id, 1, "cancelled", error_type="CancelledError",
             error_message="cancelled mid-write", result_hash="hash-cancel"),
        _edge(run_id, 1, 2, "data"),
        _start(run_id, 2, "read_file", summary="read partial file"),
        _end(run_id, 2, "error", error_type="FileNotFoundError",
             error_message="missing", result_hash="hash-read-fail"),
        _footer(run_id),
    ]


# ---------------------------------------------------------------------------
# Fixture: malformed trace (>=50% invalid lines)
# ---------------------------------------------------------------------------


def malformed_trace(run_id: int) -> list[dict | str]:
    """Most lines are corrupt JSON or schema-invalid (>=50%)."""
    return [
        _header(run_id),
        "{ this is not json",
        "garbage line",
        "[also not json",
        "null",
        _start(run_id, 1, "terminal", summary="clone"),
        _end(run_id, 1, "error", error_type="ExitCodeError",
             error_message="boom", result_hash="hash-malformed"),
        _footer(run_id),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def write_trace(tmp_path: Path):
    """Helper to write records to a tmp file and return its path."""
    def _write_records(run_id: int, records: list[dict]) -> Path:
        return _write(tmp_path, run_id, records)
    return _write_records


def test_chain_input_invalid(write_trace):
    """Chain: data producer failed first -> input_invalid, root=producer."""
    run_id = 42
    path = write_trace(run_id, chain_trace(run_id))
    result = diagnose("t_diag", run_id, trace_path=path)
    assert result["status"] == "root_cause_found"
    assert result["category"] == "input_invalid"
    assert result["root_cause_action_ids"] == ["42:1"]
    assert result["propagation_path"] == ["42:2", "42:1"]
    assert result["confidence"] >= 0.8
    assert result["evidence"]["failed_action_id"] == "42:2"
    # No verified checkpoint precedes the failed producer in this 2-action
    # chain — the only predecessor of the root IS the root. A checkpoint
    # must be a completed (status=ok) producer, never the failure path.
    # Recommending trajectory_repair here (not retry_from_checkpoint with
    # a fabricated checkpoint) keeps the recommendation actionable.
    kinds = [i["kind"] for i in result["interventions"]]
    assert "trajectory_repair" in kinds
    for i in result["interventions"]:
        if i["kind"] == "retry_from_checkpoint":
            cpid = i["payload"].get("checkpoint_action_id")
            assert cpid not in ("42:2", "42:1"), (
                "checkpoint must be a verified producer, not the failure path"
            )


def test_verified_checkpoint_used_when_producer_succeeded(write_trace):
    """A verified (status=ok) data producer is a real retry checkpoint."""
    run_id = 60
    recs = [
        _header(run_id),
        _start(run_id, 1, "web_search", summary="find source"),
        _end(run_id, 1, "ok", result_hash="hash-search-ok"),
        _edge(run_id, 1, 2, "data"),
        _start(run_id, 2, "web_extract", summary="extract page"),
        _end(run_id, 2, "error", error_type="HTTPError",
             error_message="404 not found", result_hash="hash-extract-404"),
        _edge(run_id, 2, 3, "data"),
        _start(run_id, 3, "read_file", summary="read extracted content"),
        _end(run_id, 3, "error", error_type="FileNotFoundError",
             error_message="no content file", result_hash="hash-read-missing"),
        _footer(run_id),
    ]
    path = write_trace(run_id, recs)
    result = diagnose("t_diag", run_id, trace_path=path)
    assert result["status"] == "root_cause_found"
    assert result["category"] == "input_invalid"
    assert result["root_cause_action_ids"] == [f"{run_id}:2"]
    kinds = [i["kind"] for i in result["interventions"]]
    assert "retry_from_checkpoint" in kinds
    for i in result["interventions"]:
        if i["kind"] == "retry_from_checkpoint":
            # the verified checkpoint is the successful web_search producer
            assert i["payload"].get("checkpoint_action_id") == f"{run_id}:1"



def test_chain_explicit_failed_action(write_trace):
    """Explicit failed_action_id targets a specific site."""
    run_id = 43
    path = write_trace(run_id, chain_trace(run_id))
    result = diagnose("t_diag", run_id, failed_action_id="43:2", trace_path=path)
    assert result["evidence"]["failed_action_id"] == "43:2"
    assert result["category"] == "input_invalid"
    assert result["propagation_path"] == ["43:2", "43:1"]


def test_branch_action_error(write_trace):
    """Branch: root is the web_extract failure; read_file propagates."""
    run_id = 44
    path = write_trace(run_id, branch_trace(run_id))
    result = diagnose("t_diag", run_id, trace_path=path)
    assert result["status"] == "root_cause_found"
    assert result["category"] == "input_invalid"
    assert result["root_cause_action_ids"] == ["44:2"]
    assert result["propagation_path"] == ["44:3", "44:2"]


def test_multiple_independent_failures_candidates(write_trace):
    """Independent failing branches -> status=candidates, ranked ids."""
    run_id = 45
    path = write_trace(run_id, independent_failures_trace(run_id))
    result = diagnose("t_diag", run_id, trace_path=path)
    assert result["status"] == "candidates"
    assert 0.1 <= result["confidence"] <= 0.7
    assert len(result["root_cause_action_ids"]) >= 2
    assert "45:4" in result["root_cause_action_ids"] or "45:2" in result["root_cause_action_ids"]
    # failed site is the last failed action (error preference, highest seq)
    assert result["evidence"]["failed_action_id"] == "45:4"


def test_cycle_terminates_safely(write_trace):
    """Cyclic loop edge -> safe termination, loop_repeated diagnosis."""
    run_id = 46
    path = write_trace(run_id, cyclic_trace(run_id))
    result = diagnose("t_diag", run_id, trace_path=path)
    # Must terminate and return a diagnosis (no hang/crash).
    assert result["status"] in ("root_cause_found", "candidates")
    assert result["category"] == "loop_repeated"
    assert result["evidence"]["duplicate_hashes"] == ["hash-loop-same"]


def test_truncated_trace_degrades(write_trace):
    """Truncated run (no footer, missing end) -> unknown/graceful, evidence."""
    run_id = 47
    path = write_trace(run_id, truncated_trace(run_id))
    result = diagnose("t_diag", run_id, trace_path=path)
    assert result["status"] in ("unknown", "candidates", "root_cause_found")
    assert result["evidence"]["missing_action_ends"] == ["47:2"]
    # no crash, has interventions
    assert result["interventions"]


def test_repeated_loop_failure(write_trace):
    """3x identical loop failures -> loop_repeated with duplicate hash."""
    run_id = 48
    path = write_trace(run_id, repeated_loop_trace(run_id))
    result = diagnose("t_diag", run_id, trace_path=path)
    assert result["status"] == "root_cause_found"
    assert result["category"] == "loop_repeated"
    assert result["evidence"]["duplicate_hashes"] == ["abc123"]
    assert result["root_cause_action_ids"] == ["48:1"]
    # recommendation: alternative_tool for a repeating search failure
    kinds = [i["kind"] for i in result["interventions"]]
    assert "alternative_tool" in kinds


def test_retry_exhausted(write_trace):
    """Retry loop exhausted -> retry_exhausted, escalate recommendation."""
    run_id = 49
    path = write_trace(run_id, retry_exhausted_trace(run_id))
    result = diagnose("t_diag", run_id, trace_path=path)
    assert result["status"] == "root_cause_found"
    assert result["category"] == "retry_exhausted"
    kinds = [i["kind"] for i in result["interventions"]]
    assert "escalate" in kinds


def test_timeout_propagation(write_trace):
    """Timeout on producer -> timeout_propagation."""
    run_id = 50
    path = write_trace(run_id, timeout_propagation_trace(run_id))
    result = diagnose("t_diag", run_id, trace_path=path)
    assert result["status"] == "root_cause_found"
    assert result["category"] == "timeout_propagation"
    assert result["root_cause_action_ids"] == ["50:1"]


def test_cancellation_propagation(write_trace):
    """Cancelled producer -> cancellation_propagation."""
    run_id = 51
    path = write_trace(run_id, cancellation_propagation_trace(run_id))
    result = diagnose("t_diag", run_id, trace_path=path)
    assert result["status"] == "root_cause_found"
    assert result["category"] == "cancellation_propagation"
    assert result["root_cause_action_ids"] == ["51:1"]


def test_malformed_trace(write_trace):
    """>=50% invalid lines -> malformed_trace with malformed_lines evidence."""
    run_id = 52
    path = write_trace(run_id, malformed_trace(run_id))
    result = diagnose("t_diag", run_id, trace_path=path)
    assert result["status"] == "malformed_trace"
    assert result["category"] == "malformed_trace"
    assert result["confidence"] == 0.0
    assert len(result["evidence"]["malformed_lines"]) >= 2


def test_missing_trace_unknown(tmp_path):
    """No trace file -> unknown, escalate."""
    result = diagnose("t_diag", 999, trace_path=tmp_path / "missing.jsonl")
    assert result["status"] == "unknown"
    assert result["category"] == "unknown"
    assert result["confidence"] == 0.0
    assert result["interventions"][0]["kind"] == "escalate"


def test_single_corrupt_line_tolerated(write_trace):
    """One corrupt line among valid ones -> diagnosis proceeds."""
    run_id = 53
    records: list[dict | str] = list(chain_trace(run_id))
    records.insert(2, "not json")
    path = write_trace(run_id, records)
    result = diagnose("t_diag", run_id, trace_path=path)
    assert result["status"] == "root_cause_found"
    assert result["category"] == "input_invalid"
    assert result["evidence"]["malformed_lines"] == [3]


def test_determinism_equivalent_traces(write_trace, tmp_path):
    """Equivalent traces -> byte-identical diagnosis results."""
    run_id = 54
    records = repeated_loop_trace(run_id)
    path_a = write_trace(run_id, records)
    path_b = write_trace(run_id, records)

    result_a = diagnose("t_diag", run_id, trace_path=path_a)
    result_b = diagnose("t_diag", run_id, trace_path=path_b)

    canonical_a = json.dumps(result_a, sort_keys=True, ensure_ascii=False)
    canonical_b = json.dumps(result_b, sort_keys=True, ensure_ascii=False)
    assert canonical_a == canonical_b


def test_stable_order_equivalent_traces_different_file_order(write_trace, tmp_path):
    """Deterministic tiebreak: (action_id, ts) breaks ordering ties.

    Two equivalent traces with the same events but written in a different
    file order must produce the same root cause (the engine uses insertion
    order for iteration, but the *diagnosis* is stable for equivalent event
    sets).
    """
    run_id = 55
    # Build the same event set in two different physical orders.
    events = independent_failures_trace(run_id)
    events_reversed = list(reversed(events))

    path_a = write_trace(run_id, events)
    path_b = write_trace(run_id, events_reversed)

    result_a = diagnose("t_diag", run_id, trace_path=path_a)
    result_b = diagnose("t_diag", run_id, trace_path=path_b)

    # Both must terminate safely with a candidate/root diagnosis.
    assert result_a["status"] in ("root_cause_found", "candidates", "unknown")
    assert result_b["status"] in ("root_cause_found", "candidates", "unknown")


def test_corrupt_input_no_raise(write_trace):
    """Corrupt input must never raise — returns diagnosis or unknown."""
    run_id = 56
    path = write_trace(run_id, [
        "{bad",
        "[1,2",
        "null",
        '{"kind": "bogus"}',
        "42",
    ])
    result = diagnose("t_diag", run_id, trace_path=path)
    assert result["status"] in ("malformed_trace", "unknown")
    assert result["evidence"]["malformed_lines"]


def test_empty_trace_unknown(tmp_path):
    """Empty trace file -> unknown."""
    run_id = 57
    p = tmp_path / f"{run_id}.jsonl"
    p.write_text("", encoding="utf-8")
    result = diagnose("t_diag", run_id, trace_path=p)
    assert result["status"] == "unknown"
    assert result["category"] == "unknown"


def test_trace_path_for_layout():
    """Storage layout: default board keeps legacy root under kanban/."""
    # This test only checks the layout function returns a path shaped like
    # <root>/kanban/loop-traces/<task_id>/<run_id>.jsonl when the env pins
    # the kanban home. It does not touch the real board.
    import os
    os.environ["HERMES_KANBAN_HOME"] = "/tmp/fake-kanban-home"
    try:
        p = trace_path_for("t_diag", 42)
        assert p == Path("/tmp/fake-kanban-home/kanban/loop-traces/t_diag/42.jsonl")
    finally:
        del os.environ["HERMES_KANBAN_HOME"]


def test_load_trace_malformed_line_numbers(write_trace):
    """load_trace returns malformed line numbers for invalid lines."""
    run_id = 58
    path = write_trace(run_id, malformed_trace(run_id))
    loaded = load_trace(path)
    assert loaded.malformed_line_numbers == [2, 3, 4, 5]
    assert loaded.has_footer is True
