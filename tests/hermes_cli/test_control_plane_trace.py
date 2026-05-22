"""Hermetic tests for CP-S0 control-plane event trace instrumentation.

These tests pin the two incident shapes the supervisor asked for:

* Kanban ACK loss — a ``GO`` verdict that never reaches the requester
  because the gateway delivery path has no live runner / subscription /
  target.
* Stale lane/compaction contamination — a stale source candidate that
  conflicts with the current lane/context authority.

Everything here is read-only and deterministic: a frozen clock is
injected so traces are byte-stable, and no DB / gateway code is touched.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli import control_plane_trace as cpt


def _frozen_clock(start: int = 1_700_000_000):
    """A deterministic monotonic clock: each call advances one second."""
    state = {"t": start}

    def _tick() -> int:
        t = state["t"]
        state["t"] += 1
        return t

    return _tick


# --------------------------------------------------------------------------
# Event primitives
# --------------------------------------------------------------------------

def test_event_kinds_cover_all_targets():
    assert cpt.EVENT_KINDS == (
        "task_verdict_recorded",
        "ack_requested",
        "ack_failed",
        "resume_packet_hydrated",
        "lane_contract_conflict_detected",
        "task_status_transition",
        "review_todo_stalled_due_parent_blocked",
        "final_ack_missing",
    )


def test_recorded_event_is_json_serializable_with_stable_fields():
    trace = cpt.EventTrace(lane="#hermes-main", clock=_frozen_clock())
    ev = trace.task_verdict_recorded("t_5aa48775", verdict="GO", run_id=7)

    d = ev.to_dict()
    assert d["seq"] == 1
    assert d["kind"] == "task_verdict_recorded"
    assert d["ts"] == 1_700_000_000
    assert d["dry_run"] is True
    assert d["lane"] == "#hermes-main"
    assert d["payload"]["verdict"] == "GO"
    assert d["payload"]["run_id"] == 7
    # round-trips through JSON unchanged
    assert json.loads(json.dumps(d)) == d


def test_unknown_event_kind_is_rejected():
    trace = cpt.EventTrace(clock=_frozen_clock())
    with pytest.raises(ValueError):
        trace.record("not_a_real_kind", {})


def test_seq_is_monotonic_across_event_kinds():
    trace = cpt.EventTrace(clock=_frozen_clock())
    trace.task_verdict_recorded("t1", verdict="GO")
    trace.ack_requested("t1", target="discord:1497895797579190357")
    trace.ack_failed("t1", ack_error="no_target")
    assert [e.seq for e in trace.events] == [1, 2, 3]


# --------------------------------------------------------------------------
# Incident 1 — Kanban ACK loss
# --------------------------------------------------------------------------

def test_ack_loss_trace_separates_go_verdict_from_failed_ack():
    trace = cpt.build_ack_loss_trace(
        "t_5aa48775",
        ack_error="no_live_gateway_runner",
        lane="#hermes-main",
        chat_id="1497895797579190357",
        clock=_frozen_clock(),
    )
    kinds = [e.kind for e in trace.events]
    assert kinds == ["task_verdict_recorded", "ack_requested", "ack_failed"]

    verdict_ev, _req_ev, ack_ev = trace.events

    # The task verdict says GO and carries no ack status — the verdict
    # being GO must be observable independently of delivery success.
    assert verdict_ev.payload["verdict"] == "GO"
    assert "ack_status" not in verdict_ev.payload

    # The ACK is a separate, failed fact.
    assert ack_ev.payload["ack_status"] == "FAILED"
    assert ack_ev.payload["ack_error"] == "no_live_gateway_runner"
    assert ack_ev.payload["task_verdict"] == "GO"


@pytest.mark.parametrize(
    "ack_error",
    ["no_live_gateway_runner", "no_subscription", "no_target"],
)
def test_ack_loss_trace_reproduces_each_known_ack_error(ack_error):
    trace = cpt.build_ack_loss_trace("t_x", ack_error=ack_error, clock=_frozen_clock())
    ack_ev = trace.events[-1]
    assert ack_ev.payload["ack_error"] == ack_error
    assert ack_error in cpt.KNOWN_ACK_ERRORS


def test_ack_loss_trace_supports_pending_status():
    trace = cpt.build_ack_loss_trace(
        "t_x", ack_error="no_subscription", ack_status="PENDING", clock=_frozen_clock()
    )
    assert trace.events[-1].payload["ack_status"] == "PENDING"


def test_ack_failed_rejects_non_ack_status():
    trace = cpt.EventTrace(clock=_frozen_clock())
    with pytest.raises(ValueError):
        trace.ack_failed("t1", ack_error="no_target", ack_status="DELIVERED")


# --------------------------------------------------------------------------
# Incident 2 — stale lane / compaction contamination
# --------------------------------------------------------------------------

def test_lane_conflict_trace_records_stale_source_vs_current_authority():
    trace = cpt.build_lane_conflict_trace(
        stale_source="compaction-summary@2026-05-01",
        stale_kind="compaction",
        current_lane="#hermes-main",
        current_authority="LaneContract:#hermes-main",
        clock=_frozen_clock(),
    )
    assert [e.kind for e in trace.events] == ["lane_contract_conflict_detected"]

    ev = trace.events[0]
    assert ev.payload["stale_source"] == "compaction-summary@2026-05-01"
    assert ev.payload["stale_kind"] == "compaction"
    assert ev.payload["current_lane"] == "#hermes-main"
    assert ev.payload["current_authority"] == "LaneContract:#hermes-main"
    # observe-only: the trace never claims it resolved the conflict
    assert ev.payload["resolution"] == "observed_only"


@pytest.mark.parametrize("stale_kind", ["compaction", "todo", "lane-contamination"])
def test_lane_conflict_trace_covers_each_stale_kind(stale_kind):
    trace = cpt.build_lane_conflict_trace(
        stale_source="src",
        stale_kind=stale_kind,
        current_lane="#hermes-main",
        current_authority="LaneContract:#hermes-main",
        clock=_frozen_clock(),
    )
    assert trace.events[0].payload["stale_kind"] == stale_kind
    assert stale_kind in cpt.KNOWN_STALE_KINDS


def test_lane_conflict_trace_rejects_unknown_stale_kind():
    with pytest.raises(ValueError):
        cpt.build_lane_conflict_trace(
            stale_source="src",
            stale_kind="bogus",
            current_lane="#hermes-main",
            current_authority="auth",
            clock=_frozen_clock(),
        )


# --------------------------------------------------------------------------
# Trace serialization / read-only guarantees
# --------------------------------------------------------------------------

def test_trace_to_json_is_deterministic_and_round_trips():
    trace = cpt.build_ack_loss_trace(
        "t_5aa48775", ack_error="no_target", clock=_frozen_clock()
    )
    blob = trace.to_json()
    parsed = json.loads(blob)
    assert [e["kind"] for e in parsed] == [
        "task_verdict_recorded",
        "ack_requested",
        "ack_failed",
    ]
    # all events flagged dry_run — Slice 0 emits no live side effects
    assert all(e["dry_run"] is True for e in parsed)


def test_trace_can_be_explicitly_logged_to_jsonl(tmp_path):
    trace = cpt.build_ack_loss_trace(
        "t_5aa48775", ack_error="no_target", clock=_frozen_clock()
    )
    out = tmp_path / "cp" / "trace.jsonl"

    written = trace.write_jsonl(out)

    assert written == 3
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["kind"] for row in rows] == [
        "task_verdict_recorded",
        "ack_requested",
        "ack_failed",
    ]
    assert all(row["dry_run"] is True for row in rows)


def test_resume_packet_hydrated_event_is_emittable():
    trace = cpt.EventTrace(clock=_frozen_clock())
    ev = trace.resume_packet_hydrated(
        "t_5aa48775", source="task_runs", fields=["run_id", "summary"]
    )
    assert ev.kind == "resume_packet_hydrated"
    assert ev.payload["source"] == "task_runs"
    assert ev.payload["fields"] == ["run_id", "summary"]


# --------------------------------------------------------------------------
# Incident 3 — review-required parent blocks a pre-created review child
# --------------------------------------------------------------------------

def test_review_handoff_good_transition_is_not_an_anomaly():
    trace = cpt.EventTrace(clock=_frozen_clock())
    ev = trace.task_status_transition(
        "t_impl",
        transition="implementation_running→done_GO_for_review",
        child_review_task_id="t_review",
    )
    assert ev.kind == "task_status_transition"
    assert ev.payload["anomaly"] is False
    assert ev.payload["expected_transition"] == "implementation_running→done_GO_for_review"


def test_review_handoff_blocked_transition_is_an_anomaly():
    trace = cpt.EventTrace(clock=_frozen_clock())
    ev = trace.task_status_transition(
        "t_impl",
        transition="implementation_running→blocked_review_required",
        child_review_task_id="t_review",
    )
    assert ev.payload["anomaly"] is True
    assert ev.payload["child_review_task_id"] == "t_review"


def test_review_todo_stall_event_records_parent_blocker():
    trace = cpt.EventTrace(clock=_frozen_clock())
    ev = trace.review_todo_stalled_due_parent_blocked(
        "t_review", parent_task_id="t_impl"
    )
    assert ev.kind == "review_todo_stalled_due_parent_blocked"
    assert ev.payload == {
        "review_task_id": "t_review",
        "parent_task_id": "t_impl",
        "reason": "parent_blocked_review_required",
        "anomaly": True,
    }


def test_final_ack_missing_event_keeps_task_verdict_separate():
    trace = cpt.EventTrace(clock=_frozen_clock())
    ev = trace.final_ack_missing(
        "t_impl", task_verdict="GO-for-review-not-promoted", return_to="#shaman"
    )
    assert ev.kind == "final_ack_missing"
    assert ev.payload["task_verdict"] == "GO-for-review-not-promoted"
    assert ev.payload["ack_status"] == "MISSING"
    assert ev.payload["return_to"] == "#shaman"


def test_review_handoff_anomaly_trace_covers_state_machine_chain():
    trace = cpt.build_review_handoff_anomaly_trace(
        implementation_task_id="t_impl",
        review_task_id="t_review",
        return_to="#shaman",
        clock=_frozen_clock(),
    )
    assert [e.kind for e in trace.events] == [
        "task_status_transition",
        "review_todo_stalled_due_parent_blocked",
        "final_ack_missing",
    ]
    assert trace.events[0].payload["transition"] == (
        "implementation_running→blocked_review_required"
    )
    assert trace.events[1].payload["parent_task_id"] == "t_impl"
    assert trace.events[2].payload["return_to"] == "#shaman"


def test_unknown_review_handoff_vocab_is_rejected():
    trace = cpt.EventTrace(clock=_frozen_clock())
    with pytest.raises(ValueError):
        trace.task_status_transition("t_impl", transition="implementation_running→done")
    with pytest.raises(ValueError):
        trace.review_todo_stalled_due_parent_blocked("t_review", parent_task_id="t_impl", reason="other")
