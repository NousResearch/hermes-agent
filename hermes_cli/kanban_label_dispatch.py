"""Dry-run label router for Kanban task packet generation.

This module intentionally has no GitHub/Kanban side effects.  It accepts a
pre-fetched issue/PR-like record, classifies labels, and returns a preview
report that can be audited before any controlled writeback is introduced.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DISPATCH_LABELS: dict[str, set[str]] = {
    "dispatch_007": {
        "gm2:dispatch-007",
        "gm2:build",
        "gm2:implement",
        "gm2:fix",
    },
    "dispatch_audit": {
        "gm2:dispatch-audit",
        "gm2:audit",
        "gm2:review",
        "gm2:verify",
    },
    "dispatch_closeout": {
        "gm2:dispatch-closeout",
        "gm2:closeout",
        "gm2:handoff",
        "gm2:summary",
    },
}

CONTROL_FLAGS: set[str] = {
    "gm2:blocked",
    "gm2:needs-info",
    "gm2:no-ack",
    "gm2:timeout",
    "gm2:audit-missing",
    "gm2:audit-fail",
}

ACTOR_BY_DISPATCH: dict[str, str] = {
    "dispatch_007": "007",
    "dispatch_audit": "八府巡按",
    "dispatch_closeout": "GM2",
}

PACKET_TYPE_BY_DISPATCH: dict[str, str] = {
    "dispatch_007": "007_task_packet",
    "dispatch_audit": "audit_packet",
    "dispatch_closeout": "gm2_closeout_packet",
}

DRY_RUN_GUARDS: dict[str, bool] = {
    "dry_run_only": True,
    "controlled_writeback_enabled": False,
    "auto_merge_enabled": False,
    "auto_close_enabled": False,
    "runtime_execution_allowed": False,
    "full_unattended_ready": False,
}

REQUIRED_PACKET_FIELDS: tuple[str, ...] = (
    "from_gm",
    "source_name",
    "project_id",
    "task_id",
    "priority",
    "formal_record",
    "current_state",
    "assigned_agent",
    "objective",
    "allowed_scope",
    "forbidden_actions",
    "acceptance",
    "evidence_required",
    "stop_conditions",
    "next_report_expected",
)

DEFAULT_ALLOWED_SCOPE: list[str] = [
    "read source issue/pr record",
    "generate dry-run actor packet",
    "produce preview/report only",
]

DEFAULT_FORBIDDEN_ACTIONS: list[str] = [
    "send actor task",
    "write GitHub comment",
    "modify labels",
    "move Kanban card",
    "merge PR",
    "close issue",
    "enable runtime execution",
    "touch production/payment/credits/webhook/customer data/secret",
    "claim full_unattended_ready",
]

DEFAULT_ACCEPTANCE: list[str] = [
    "packet includes all required fields",
    "dry-run report lists routing decision and matched labels",
    "no external writeback or actor dispatch occurs",
]

DEFAULT_EVIDENCE_REQUIRED: list[str] = [
    "dry-run preview JSON",
    "routing decision",
    "matched dispatch labels and control flags",
]

DEFAULT_STOP_CONDITIONS: list[str] = [
    "any control flag is present",
    "multiple dispatch classes match",
    "source record lacks enough identity fields to form a formal record",
]


@dataclass(frozen=True)
class LabelRoute:
    dispatch_type: str | None
    matched_dispatch_labels: dict[str, list[str]]
    control_flags: list[str]
    conflict: bool
    blocked: bool
    reason: str


def _normalise_labels(labels: list[Any]) -> list[str]:
    normalised: list[str] = []
    for label in labels:
        if isinstance(label, str):
            name = label
        elif isinstance(label, dict):
            name = str(label.get("name", ""))
        else:
            name = str(label)
        name = name.strip()
        if name:
            normalised.append(name)
    return normalised


def classify_labels(labels: list[Any]) -> LabelRoute:
    """Classify labels into one dispatch route, conflict, or blocked state."""

    label_set = set(_normalise_labels(labels))
    matched = {
        dispatch: sorted(label_set & dispatch_labels)
        for dispatch, dispatch_labels in DISPATCH_LABELS.items()
        if label_set & dispatch_labels
    }
    controls = sorted(label_set & CONTROL_FLAGS)

    if controls:
        return LabelRoute(
            dispatch_type=None,
            matched_dispatch_labels=matched,
            control_flags=controls,
            conflict=False,
            blocked=True,
            reason="control_flag_present",
        )

    if len(matched) > 1:
        return LabelRoute(
            dispatch_type=None,
            matched_dispatch_labels=matched,
            control_flags=[],
            conflict=True,
            blocked=True,
            reason="multiple_dispatch_labels_matched",
        )

    if len(matched) == 1:
        dispatch_type = next(iter(matched))
        return LabelRoute(
            dispatch_type=dispatch_type,
            matched_dispatch_labels=matched,
            control_flags=[],
            conflict=False,
            blocked=False,
            reason="dispatch_label_matched",
        )

    return LabelRoute(
        dispatch_type=None,
        matched_dispatch_labels={},
        control_flags=[],
        conflict=False,
        blocked=False,
        reason="no_dispatch_label_matched",
    )


def _source_name(record: dict[str, Any]) -> str:
    if record.get("source_name"):
        return str(record["source_name"])
    repo = record.get("repo") or record.get("repository") or "unknown/repo"
    number = record.get("number") or record.get("id") or "unknown"
    source_type = record.get("source_type") or record.get("type") or "issue"
    return f"{repo}#{number} ({source_type})"


def _task_id(record: dict[str, Any]) -> str:
    if record.get("task_id"):
        return str(record["task_id"])
    source_type = record.get("source_type") or record.get("type") or "issue"
    number = record.get("number") or record.get("id") or "unknown"
    return f"{source_type}-{number}"


def _formal_record(record: dict[str, Any]) -> str:
    return str(record.get("formal_record") or record.get("html_url") or record.get("url") or _source_name(record))


def _objective(record: dict[str, Any], route: LabelRoute) -> str:
    if record.get("objective"):
        return str(record["objective"])
    title = str(record.get("title") or "Untitled GitHub task")
    body = str(record.get("body") or "").strip()
    prefix = {
        "dispatch_007": "Execute bounded implementation/build/fix task",
        "dispatch_audit": "Audit and verify the formal record",
        "dispatch_closeout": "Prepare closeout/handoff/summary packet",
    }.get(route.dispatch_type or "", "Stop automatic dispatch and surface failure state")
    if body:
        return f"{prefix}: {title}\n\nSource summary:\n{body[:1200]}"
    return f"{prefix}: {title}"


def build_packet(record: dict[str, Any], route: LabelRoute) -> dict[str, Any] | None:
    """Build an actor/failure packet for a routed record.

    Returns ``None`` for records with no dispatch label and no failure state.
    """

    if not route.dispatch_type and not route.blocked:
        return None

    if route.blocked:
        assigned_agent = "GM2"
        packet_type = "failure_packet"
        current_state = "blocked"
    else:
        assigned_agent = ACTOR_BY_DISPATCH[route.dispatch_type or ""]
        packet_type = PACKET_TYPE_BY_DISPATCH[route.dispatch_type or ""]
        current_state = str(record.get("current_state") or "ready_for_dry_run_dispatch")

    packet = {
        "packet_type": packet_type,
        "from_gm": str(record.get("from_gm") or "GM2"),
        "source_name": _source_name(record),
        "project_id": str(record.get("project_id") or record.get("repo") or "unknown-project"),
        "task_id": _task_id(record),
        "priority": str(record.get("priority") or "P2"),
        "formal_record": _formal_record(record),
        "current_state": current_state,
        "assigned_agent": assigned_agent,
        "objective": _objective(record, route),
        "allowed_scope": list(record.get("allowed_scope") or DEFAULT_ALLOWED_SCOPE),
        "forbidden_actions": list(record.get("forbidden_actions") or DEFAULT_FORBIDDEN_ACTIONS),
        "acceptance": list(record.get("acceptance") or DEFAULT_ACCEPTANCE),
        "evidence_required": list(record.get("evidence_required") or DEFAULT_EVIDENCE_REQUIRED),
        "stop_conditions": list(record.get("stop_conditions") or DEFAULT_STOP_CONDITIONS),
        "next_report_expected": str(record.get("next_report_expected") or "dry-run preview only; await GM2/八府巡按 gate"),
    }

    missing = [field for field in REQUIRED_PACKET_FIELDS if field not in packet or packet[field] in (None, "")]
    if missing:
        raise ValueError(f"packet missing required fields: {', '.join(missing)}")
    return packet


def build_dry_run_report(record: dict[str, Any]) -> dict[str, Any]:
    labels = _normalise_labels(list(record.get("labels") or []))
    route = classify_labels(labels)
    packet = build_packet(record, route)

    return {
        "mode": "dry_run_only",
        **DRY_RUN_GUARDS,
        "source_name": _source_name(record),
        "task_id": _task_id(record),
        "labels": labels,
        "routing": {
            "dispatch_type": route.dispatch_type,
            "reason": route.reason,
            "matched_dispatch_labels": route.matched_dispatch_labels,
            "control_flags": route.control_flags,
            "conflict": route.conflict,
            "blocked": route.blocked,
            "will_send_to_actor": False,
            "will_write_github_comment": False,
            "will_modify_labels": False,
            "will_move_kanban": False,
            "will_merge_pr": False,
            "will_close_issue": False,
        },
        "packet": packet,
        "required_packet_fields": list(REQUIRED_PACKET_FIELDS),
    }


def load_record(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("source record must be a JSON object")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run GitHub label to Kanban actor packet router")
    parser.add_argument("record", help="Path to issue/PR JSON record")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args(argv)

    report = build_dry_run_report(load_record(args.record))
    indent = 2 if args.pretty else None
    print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
