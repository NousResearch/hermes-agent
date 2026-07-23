from __future__ import annotations

import json
from pathlib import Path

from hermes_cli.kanban_label_dispatch import (
    CONTROL_FLAGS,
    DISPATCH_LABELS,
    REQUIRED_PACKET_FIELDS,
    build_dry_run_report,
    main,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "kanban_label_dispatch"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def assert_dry_run_guards(report: dict) -> None:
    assert report["mode"] == "dry_run_only"
    assert report["dry_run_only"] is True
    assert report["controlled_writeback_enabled"] is False
    assert report["auto_merge_enabled"] is False
    assert report["auto_close_enabled"] is False
    assert report["runtime_execution_allowed"] is False
    assert report["full_unattended_ready"] is False
    routing = report["routing"]
    assert routing["will_send_to_actor"] is False
    assert routing["will_write_github_comment"] is False
    assert routing["will_modify_labels"] is False
    assert routing["will_move_kanban"] is False
    assert routing["will_merge_pr"] is False
    assert routing["will_close_issue"] is False


def assert_required_packet_fields(packet: dict) -> None:
    missing = [field for field in REQUIRED_PACKET_FIELDS if field not in packet]
    assert missing == []


def test_dispatch_007_issue_generates_007_packet() -> None:
    report = build_dry_run_report(load_fixture("dispatch_007_issue.json"))

    assert_dry_run_guards(report)
    assert report["routing"]["dispatch_type"] == "dispatch_007"
    assert report["routing"]["blocked"] is False
    packet = report["packet"]
    assert_required_packet_fields(packet)
    assert packet["packet_type"] == "007_task_packet"
    assert packet["assigned_agent"] == "007"
    assert packet["from_gm"] == "GM2"


def test_dispatch_audit_pr_generates_audit_packet() -> None:
    report = build_dry_run_report(load_fixture("dispatch_audit_pr.json"))

    assert_dry_run_guards(report)
    assert report["routing"]["dispatch_type"] == "dispatch_audit"
    packet = report["packet"]
    assert_required_packet_fields(packet)
    assert packet["packet_type"] == "audit_packet"
    assert packet["assigned_agent"] == "八府巡按"


def test_dispatch_closeout_issue_generates_gm2_closeout_packet() -> None:
    report = build_dry_run_report(load_fixture("dispatch_closeout_issue.json"))

    assert_dry_run_guards(report)
    assert report["routing"]["dispatch_type"] == "dispatch_closeout"
    packet = report["packet"]
    assert_required_packet_fields(packet)
    assert packet["packet_type"] == "gm2_closeout_packet"
    assert packet["assigned_agent"] == "GM2"


def test_multiple_dispatch_labels_create_conflict_failure_packet() -> None:
    report = build_dry_run_report(load_fixture("conflict_issue.json"))

    assert_dry_run_guards(report)
    routing = report["routing"]
    assert routing["dispatch_type"] is None
    assert routing["conflict"] is True
    assert routing["blocked"] is True
    assert routing["reason"] == "multiple_dispatch_labels_matched"
    assert set(routing["matched_dispatch_labels"]) == {"dispatch_007", "dispatch_audit"}
    packet = report["packet"]
    assert_required_packet_fields(packet)
    assert packet["packet_type"] == "failure_packet"
    assert packet["current_state"] == "blocked"
    assert packet["assigned_agent"] == "GM2"


def test_control_flag_blocks_dispatch_even_with_valid_dispatch_label() -> None:
    report = build_dry_run_report(load_fixture("control_flag_pr.json"))

    assert_dry_run_guards(report)
    routing = report["routing"]
    assert routing["dispatch_type"] is None
    assert routing["blocked"] is True
    assert routing["reason"] == "control_flag_present"
    assert routing["control_flags"] == ["gm2:needs-info"]
    assert routing["matched_dispatch_labels"] == {"dispatch_007": ["gm2:dispatch-007"]}
    packet = report["packet"]
    assert_required_packet_fields(packet)
    assert packet["packet_type"] == "failure_packet"
    assert packet["current_state"] == "blocked"


def test_supported_label_taxonomy_matches_mvp_contract() -> None:
    assert DISPATCH_LABELS == {
        "dispatch_007": {"gm2:dispatch-007", "gm2:build", "gm2:implement", "gm2:fix"},
        "dispatch_audit": {"gm2:dispatch-audit", "gm2:audit", "gm2:review", "gm2:verify"},
        "dispatch_closeout": {"gm2:dispatch-closeout", "gm2:closeout", "gm2:handoff", "gm2:summary"},
    }
    assert CONTROL_FLAGS == {
        "gm2:blocked",
        "gm2:needs-info",
        "gm2:no-ack",
        "gm2:timeout",
        "gm2:audit-missing",
        "gm2:audit-fail",
    }


def test_cli_emits_dry_run_preview(capsys) -> None:
    assert main([str(FIXTURES / "dispatch_007_issue.json"), "--pretty"]) == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["mode"] == "dry_run_only"
    assert report["packet"]["assigned_agent"] == "007"
