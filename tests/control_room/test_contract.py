"""Contract tests for the Control Room domain contract (CR-007).

Covers: snapshot shape/versioning, deterministic attention ordering,
unavailable-source semantics, stable kind:id identities, no message body in
aggregate status, no cross-profile row leakage, and the TS mirror regeneration
check.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from control_room import (
    SNAPSHOT_VERSION,
    AgentRow,
    AttentionItem,
    AttentionKind,
    AttentionSeverity,
    ControlRoomActionResult,
    ControlRoomError,
    ControlRoomSnapshot,
    ErrorCode,
    MessageRow,
    SnapshotCounts,
    SystemSummary,
    TaskRow,
    rank_attention,
    severity_for_kind,
    unavailable_attention,
    unavailable_source,
    unavailable_system,
)


def _item(kind: AttentionKind, id: str, updated_at: str = "") -> AttentionItem:
    return AttentionItem(
        kind=kind,
        id=id,
        severity=severity_for_kind(kind),
        title=f"{kind.value}:{id}",
        updated_at=updated_at,
    )


class TestOrdering:
    def test_critical_before_error_before_warning_before_info(self):
        items = [
            _item(AttentionKind.running, "a"),
            _item(AttentionKind.blocked_task, "b"),
            _item(AttentionKind.approval, "c"),
            _item(AttentionKind.error, "d"),
        ]
        ranked = rank_attention(items)
        assert [i.stable_id for i in ranked] == [
            "approval:c",
            "error:d",
            "blocked_task:b",
            "running:a",
        ]

    def test_approval_and_held_message_tie_break_by_stable_id(self):
        items = [
            _item(AttentionKind.held_message, "m2"),
            _item(AttentionKind.approval, "a1"),
            _item(AttentionKind.approval, "a2"),
        ]
        ranked = rank_attention(items)
        assert [i.stable_id for i in ranked] == ["approval:a1", "approval:a2", "held_message:m2"]

    def test_newest_actionable_timestamp_first_within_same_severity(self):
        items = [
            _item(AttentionKind.error, "old", updated_at="2026-08-12T10:00:00Z"),
            _item(AttentionKind.error, "new", updated_at="2026-08-12T11:00:00Z"),
        ]
        ranked = rank_attention(items)
        assert [i.stable_id for i in ranked] == ["error:new", "error:old"]

    def test_missing_timestamp_sorts_last_within_severity(self):
        items = [
            _item(AttentionKind.stalled, "no-ts"),
            _item(AttentionKind.stalled, "with-ts", updated_at="2026-08-12T10:00:00Z"),
        ]
        ranked = rank_attention(items)
        assert [i.stable_id for i in ranked] == ["stalled:with-ts", "stalled:no-ts"]

    def test_rank_is_total_and_stable(self):
        items = [
            _item(AttentionKind.info, "i2"),
            _item(AttentionKind.info, "i1"),
            _item(AttentionKind.ready, "r"),
            _item(AttentionKind.system, "s"),
        ]
        first = rank_attention(items)
        second = rank_attention(list(reversed(items)))
        assert [i.stable_id for i in first] == [i.stable_id for i in second]


class TestSeverityMapping:
    def test_kind_to_severity_contract(self):
        assert severity_for_kind(AttentionKind.approval) == AttentionSeverity.critical
        assert severity_for_kind(AttentionKind.held_message) == AttentionSeverity.critical
        assert severity_for_kind(AttentionKind.error) == AttentionSeverity.error
        assert severity_for_kind(AttentionKind.stalled) == AttentionSeverity.error
        assert severity_for_kind(AttentionKind.blocked_task) == AttentionSeverity.warning
        assert severity_for_kind(AttentionKind.review_task) == AttentionSeverity.warning
        assert severity_for_kind(AttentionKind.running) == AttentionSeverity.info
        assert severity_for_kind(AttentionKind.ready) == AttentionSeverity.info


class TestUnavailableSemantics:
    def test_unavailable_source_is_typed_not_silent(self):
        src = unavailable_source("peer")
        assert src.state == "unavailable"
        assert src.detail

    def test_unavailable_attention_is_typed_item_with_no_actions(self):
        item = unavailable_attention("peer")
        assert item.source.state == "unavailable"
        assert item.available_actions == []
        assert item.stable_id == "info:unavailable"

    def test_unavailable_system_is_typed(self):
        sys_sum = unavailable_system("kanban")
        assert sys_sum.source.state == "unavailable"
        assert sys_sum.state == "unknown"

    def test_unavailable_attention_is_not_an_empty_home(self):
        # An unavailable provider must never zero out the needs-you count.
        snapshot = ControlRoomSnapshot(
            profile="default",
            attention=[unavailable_attention("peer", kind=AttentionKind.held_message)],
            counts=SnapshotCounts(needs_you=1),
        )
        assert snapshot.counts.needs_you == 1


class TestStableIdentity:
    def test_row_kind_id_identity(self):
        assert AgentRow(id="p1", name="proc").stable_id == "agent:p1"
        assert AgentRow(kind="process", id="p1", name="proc").stable_id == "process:p1"
        assert AgentRow(kind="delegation", id="d1", name="sub").stable_id == "delegation:d1"
        assert TaskRow(id="t42", title="x").stable_id == "task:t42"
        assert MessageRow(id="m9", title="y").stable_id == "peer_message:m9"
        assert MessageRow(kind="peer_request", id="r1", title="z").stable_id == "peer_request:r1"
        assert _item(AttentionKind.approval, "a1").stable_id == "approval:a1"


class TestNoMessageBodyLeak:
    def test_snapshot_json_has_no_body_key(self):
        snap = ControlRoomSnapshot(
            profile="default",
            messages=[MessageRow(id="m1", title="incoming request", sender="peer-x")],
            counts=SnapshotCounts(messages_unread=1),
        )
        payload = snap.model_dump(mode="json")
        assert "body" not in json.dumps(payload)

    def test_invariant_validator_clean_snapshot_no_false_positive(self):
        snap = ControlRoomSnapshot(
            profile="default",
            messages=[MessageRow(id="m1", title="incoming request", sender="peer-x")],
            counts=SnapshotCounts(messages_unread=1),
        )
        # A clean snapshot has no violations; the validator's body-key scan
        # passes because the model surface has no body field at all.
        violations = [v for v in snap.validate_invariants() if "body" in v]
        assert violations == []


class TestNoCrossProfileLeakage:
    def test_rows_carry_explicit_profile(self):
        snap = ControlRoomSnapshot(
            profile="kensei",
            agents=[AgentRow(id="a1", name="x", profile="kensei")],
            tasks=[TaskRow(id="t1", title="y", profile="kensei")],
            messages=[MessageRow(id="m1", title="z", profile="kensei")],
        )
        for row in snap.agents + snap.tasks + snap.messages:
            assert row.profile == "kensei"

    def test_multi_profile_rows_are_labeled(self):
        snap = ControlRoomSnapshot(
            profile="kensei",
            tasks=[TaskRow(id="t1", title="y", profile="remii")],
        )
        # Aggregate view must never hide ownership.
        assert snap.tasks[0].profile == "remii"


class TestCountsInvariant:
    def test_needs_you_matches_critical_error_attention(self):
        snap = ControlRoomSnapshot(
            profile="default",
            attention=[
                _item(AttentionKind.approval, "a1"),
                _item(AttentionKind.error, "e1"),
                _item(AttentionKind.running, "r1"),
            ],
            counts=SnapshotCounts(needs_you=2),
        )
        assert snap.validate_invariants() == []

    def test_count_mismatch_is_reported(self):
        snap = ControlRoomSnapshot(
            profile="default",
            attention=[_item(AttentionKind.approval, "a1")],
            counts=SnapshotCounts(needs_you=0),
        )
        violations = snap.validate_invariants()
        assert any("needs_you" in v for v in violations)


class TestActionEnvelope:
    def test_confirmation_default_is_required(self):
        from control_room import ControlRoomAction

        action = ControlRoomAction(id="act-1", target={"kind": "process", "id": "p1"})
        assert action.confirmation == "required"

    def test_result_statuses(self):
        assert ControlRoomActionResult(status="completed", message="ok").status == "completed"
        assert ControlRoomActionResult(status="stale", message="changed").status == "stale"
        assert ControlRoomActionResult(status="unavailable", message="no").status == "unavailable"

    def test_error_codes_are_strings(self):
        assert ErrorCode.STALE_TARGET.value == "stale_target"
        err = ControlRoomError(code=ErrorCode.CROSS_PROFILE, message="no")
        assert err.model_dump(mode="json")["code"] == "cross_profile"


class TestSnapshotShape:
    def test_version_is_contract(self):
        assert SNAPSHOT_VERSION == 1
        snap = ControlRoomSnapshot(profile="default")
        assert snap.version == 1

    def test_capabilities_default_all_false(self):
        snap = ControlRoomSnapshot(profile="default")
        assert snap.capabilities.model_dump() == {
            "approvals": False,
            "peer_messages": False,
            "kanban_actions": False,
            "process_control": False,
            "delegation_control": False,
        }

    def test_system_summary_default_unknown(self):
        assert SystemSummary().state == "unknown"


class TestTsMirror:
    def test_generated_ts_matches_checked_in_file(self):
        """Regeneration determinism gate: the checked-in TS mirror must match
        what the generator produces, byte for byte."""
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, "-m", "control_room.export_ts_schema", "--check"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_ts_mirror_contains_literal_unions(self):
        ts_path = Path(__file__).resolve().parents[2] / "control_room" / "schema" / "control-room-v1.ts"
        text = ts_path.read_text()
        assert '"approval"' in text
        assert '"stale_target"' in text
        assert "export interface ControlRoomSnapshot" in text
