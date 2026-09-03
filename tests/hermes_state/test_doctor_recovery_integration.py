"""End-to-end Stage 2A: doctor arms recovery, core consumes it, doctor verifies.

This is the seam the two halves of Epic 5 meet at. The doctor never ends a
live session itself — it only arms ``_turn_boundary_rollover_pending`` with a
recovery key — so the contract that matters is that the *installed core*
consumes exactly that marker and produces a continuation the doctor can then
read back by the same key.
"""

import importlib.util
import json
import os
from pathlib import Path
import pwd
import sys

import pytest

from hermes_state import SessionDB
from session_rollover import RECOVERY_END_REASON, TurnBoundaryRollover


def _doctor_path() -> Path:
    """Locate the ops script without trusting the test sandbox's HOME.

    conftest rewires HOME/HERMES_HOME to a tempdir, so ``Path.home()`` points
    at the sandbox. The real account home comes from the password database.
    """
    override = os.environ.get("HERMES_SESSION_DOCTOR", "").strip()
    if override:
        return Path(override)
    return (
        Path(pwd.getpwuid(os.getuid()).pw_dir)
        / ".hermes" / "scripts" / "hermes-session-doctor.py"
    )


DOCTOR = _doctor_path()


def load_doctor():
    if not DOCTOR.exists():  # pragma: no cover - local ops script only
        pytest.skip(f"doctor script not installed at {DOCTOR}")
    spec = importlib.util.spec_from_file_location("hermes_session_doctor", DOCTOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The exact incident counters (12h, 1,123 calls, 142,130,176 cache-read,
# 16 compactions, iteration budget exhausted).
INCIDENT_LIFECYCLE = {
    "state": "draining",
    "state_entered_at": 1_000.0,
    "last_progress_at": 1_000.0,
    "updated_at": 1_000.0,
    "context_utilization": 0.74,
    "remaining_context_tokens": 70_000,
    "reserved_headroom_tokens": 36_000,
    "remaining_iterations": 0,
    "in_flight_workers": 0,
    "active_tool_call": False,
    "api_calls": 1_123,
    "cache_read_tokens": 142_130_176,
    "compactions": 16,
}


def test_doctor_request_is_consumed_by_core_and_reads_back_as_recovered(tmp_path: Path):
    doctor = load_doctor()
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session(
        "stalled", source="cli", session_key="tui:1", model_config={
            "provider": "openrouter",
            "turn_boundary_lifecycle": dict(INCIDENT_LIFECYCLE),
        },
    )
    db.append_message("stalled", "user", "long running integration")

    # 1. No blockers: nothing risky is in flight.
    assert doctor.recovery_blockers(db_path, "stalled", now=5_000.0) == []

    # 2. The doctor requests exactly one rollover; a second request is a no-op.
    first = doctor.recover_stalled_session(
        db_path, "stalled", label="Oh My Feed", idempotency_key="k1",
        core_supported=True, dry_run=False,
    )
    second = doctor.recover_stalled_session(
        db_path, "stalled", label="Oh My Feed", idempotency_key="k2",
        core_supported=True, dry_run=False,
    )
    assert first.status == "armed"
    assert second.status == "already_armed"

    # 3. Read-back before the core has acted is pending, never an alert.
    early = doctor.verify_recovery_readback(
        db_path, "stalled", "k1", [], now=5_000.0, deadline_seconds=1_800
    )
    assert early.status == "pending"
    assert early.issue is None

    # 4. The CORE consumes the doctor's marker at its next turn boundary.
    db = SessionDB(db_path=db_path)
    child = TurnBoundaryRollover(db).adopt_at_turn_boundary("stalled", active_work=False)
    assert child

    old = db.get_session("stalled")
    assert old is not None and old["end_reason"] == RECOVERY_END_REASON
    child_row = db.get_session(child)
    assert child_row is not None
    child_config = json.loads(child_row["model_config"])
    # Runtime settings survive; the consumed marker does not.
    assert child_config["provider"] == "openrouter"
    assert "_turn_boundary_rollover_pending" not in child_config
    assert child_config["turn_boundary_handoff"]["recovery_key"] == "k1"
    assert db.get_messages_as_conversation(child) == []

    # 5. The doctor now verifies all three read-back conditions.
    agents = [{
        "agent_status": "working",
        "agent_session": {"value": child},
        "pane_id": "w1T:p1",
    }]
    result = doctor.verify_recovery_readback(
        db_path, "stalled", "k1", agents, now=6_000.0, deadline_seconds=1_800
    )
    assert result.status == "recovered"
    assert result.continuation_session_id == child
    assert result.evidence["old_end_reason"] == RECOVERY_END_REASON
    assert result.evidence["new_session_status"] == "working"
    assert result.issue is None

    # 6. A second adoption cannot produce a duplicate continuation.
    assert TurnBoundaryRollover(db).adopt_at_turn_boundary(
        "stalled", active_work=False
    ) is None


def test_doctor_holds_off_while_the_core_still_has_active_work(tmp_path: Path):
    doctor = load_doctor()
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    lifecycle = dict(INCIDENT_LIFECYCLE, active_tool_call=True)
    db.create_session("busy", source="cli", model_config={
        "turn_boundary_lifecycle": lifecycle
    })

    blockers = doctor.recovery_blockers(db_path, "busy", now=5_000.0)

    assert "실행 중 tool call" in blockers
    # A blocked session is never armed, so the core sees no pending marker.
    row = db.get_session("busy")
    assert row is not None
    assert "_turn_boundary_rollover_pending" not in json.loads(row["model_config"])
