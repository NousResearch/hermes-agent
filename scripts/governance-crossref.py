#!/usr/bin/env python3
"""Project canonical governance finding events into current JSON and closure views.

This is the existing governance-crossref cron seam.  It no longer reads the
obsolete profile-score/self-evaluation files: the append-only activity ledger is
the only source of current finding truth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes_cli import governance_findings  # noqa: E402
from hermes_cli import profile_activity_ledger as ledger  # noqa: E402

HERMES_HOME = Path(os.environ.get("HERMES_HOME", "/home/kensei/.hermes"))
LOGBOARD = HERMES_HOME / "governance" / "logboard"
OVERDUE_DAYS = 7
_SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def _age_days(source_observed_at: int, now: int) -> int:
    return max(0, (int(now) - int(source_observed_at)) // 86400)


def _age_bucket(age_days: int) -> str:
    if age_days < 7:
        return "0-6d"
    if age_days < 30:
        return "7-29d"
    return "30d+"


def _finding_key(item: Mapping[str, Any]) -> str:
    return str(item.get("dedupe_key") or item.get("finding_id") or "")


def _normalise_previous(previous: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if not isinstance(previous, Mapping) or previous.get("schema_version") != 1:
        return {}
    rows = previous.get("findings")
    if not isinstance(rows, list):
        return {}
    return {
        _finding_key(row): row
        for row in rows
        if isinstance(row, Mapping) and _finding_key(row)
    }


def _missing(item: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    if not item.get("evidence_refs"):
        missing.append("evidence")
    if not item.get("owner"):
        missing.append("owner")
    if not item.get("task_id"):
        missing.append("action_task")
    if item.get("state") in {"resolved", "dismissed", "risk_accepted"} and not item.get("resolution_ref"):
        missing.append("resolution_proof")
    return missing


def _next_action(item: Mapping[str, Any], missing: list[str]) -> str:
    if item.get("state") == "resolved" and "resolution_proof" not in missing:
        return "closed"
    if "owner" in missing:
        return "assign owner"
    if "action_task" in missing:
        return "create action task"
    if "evidence" in missing:
        return "attach evidence"
    if "resolution_proof" in missing:
        return "attach resolution proof"
    if item.get("state") == "open":
        return "owner action required"
    return "monitor"


def _current_items(events: Iterable[Mapping[str, Any]], now: int) -> list[dict[str, Any]]:
    reduced = governance_findings.reduce_findings(events)
    items: list[dict[str, Any]] = []
    for key, raw in reduced.items():
        item = dict(raw)
        item["age_days"] = _age_days(item["source_observed_at"], now)
        item["overdue"] = item["state"] == "open" and item["age_days"] >= OVERDUE_DAYS
        item["missing"] = _missing(item)
        item["next_action"] = _next_action(item, item["missing"])
        item["finding_key"] = key
        items.append(item)
    items.sort(
        key=lambda item: (
            -_SEVERITY_RANK.get(str(item.get("severity")), -1),
            -int(item.get("source_observed_at") or 0),
            str(item.get("finding_key") or ""),
        )
    )
    return items


def _attention_reason(item: Mapping[str, Any], old: Mapping[str, Any] | None) -> str | None:
    if old is None:
        return "new"
    old_rank = _SEVERITY_RANK.get(str(old.get("severity")), -1)
    new_rank = _SEVERITY_RANK.get(str(item.get("severity")), -1)
    if new_rank > old_rank or (
        old.get("state") in {"resolved", "dismissed", "risk_accepted"}
        and item.get("state") == "open"
    ):
        return "worsened"
    if item.get("overdue") and not old.get("overdue"):
        return "overdue"
    if item.get("missing"):
        return "human_decision"
    return None


def _weekly_metrics(events: Iterable[Mapping[str, Any]], items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_key: dict[str, list[Mapping[str, Any]]] = {}
    terminal_counts = Counter()
    for event in events:
        event_type = str(event.get("event_type") or "")
        if event_type not in governance_findings.FINDING_EVENT_TYPES:
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        key = str(payload.get("dedupe_key") or "")
        if not key:
            continue
        by_key.setdefault(key, []).append(event)
        if event_type in {"governance.finding.resolved", "governance.finding.dismissed"}:
            terminal_counts[event_type] += 1

    owner_durations: list[int] = []
    resolution_durations: list[int] = []
    recurring = 0
    for key, history in by_key.items():
        ordered = sorted(history, key=lambda event: (
            int((event.get("payload") or {}).get("source_observed_at") or event.get("occurred_at") or 0),
            str(event.get("event_id") or ""),
        ))
        first_time = int((ordered[0].get("payload") or {}).get("source_observed_at") or ordered[0].get("occurred_at") or 0)
        opened_count = sum(1 for event in ordered if event.get("event_type") == "governance.finding.opened")
        if opened_count > 1:
            recurring += 1
        owner_event = next(
            (
                event for event in ordered
                if isinstance(event.get("payload"), Mapping) and event["payload"].get("owner")
            ),
            None,
        )
        if owner_event is not None:
            owner_time = int(owner_event["payload"].get("source_observed_at") or owner_event.get("occurred_at") or first_time)
            owner_durations.append(max(0, owner_time - first_time))
        resolution_event = next(
            (
                event for event in ordered
                if event.get("event_type") == "governance.finding.resolved"
                and isinstance(event.get("payload"), Mapping)
                and event["payload"].get("resolution_ref")
            ),
            None,
        )
        if resolution_event is not None:
            resolution_time = int(resolution_event["payload"].get("source_observed_at") or resolution_event.get("occurred_at") or first_time)
            resolution_durations.append(max(0, resolution_time - first_time))

    terminal_total = terminal_counts["governance.finding.resolved"] + terminal_counts["governance.finding.dismissed"]
    return {
        "false_positive_rate": (
            round(terminal_counts["governance.finding.dismissed"] / terminal_total, 4)
            if terminal_total else None
        ),
        "false_positive_definition": "dismissed findings / resolved-or-dismissed findings",
        "time_to_owner_days": (
            round(sum(owner_durations) / len(owner_durations) / 86400, 2)
            if owner_durations else None
        ),
        "time_to_verified_resolution_days": (
            round(sum(resolution_durations) / len(resolution_durations) / 86400, 2)
            if resolution_durations else None
        ),
        "recurring_findings": recurring,
        "human_decisions_waiting": sum(1 for item in items if item.get("missing")),
    }


def project_events(
    events: Iterable[Mapping[str, Any]],
    *,
    now: int | None = None,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic current projection and attention signal."""
    reference_time = int(time.time()) if now is None else int(now)
    event_list = list(events)
    items = _current_items(event_list, reference_time)
    old_by_key = _normalise_previous(previous)
    attention: list[dict[str, Any]] = []
    for item in items:
        reason = _attention_reason(item, old_by_key.get(item["finding_key"]))
        if reason:
            attention.append({"finding_key": item["finding_key"], "reason": reason})

    groups = {
        "severity": dict(sorted(Counter(str(item["severity"]) for item in items).items())),
        "owner": dict(sorted(Counter(str(item.get("owner") or "unassigned") for item in items).items())),
        "state": dict(sorted(Counter(str(item["state"]) for item in items).items())),
        "age": dict(sorted(Counter(_age_bucket(int(item["age_days"])) for item in items).items())),
    }
    return {
        "schema_version": 1,
        "findings": items,
        "groups": groups,
        "metrics": _weekly_metrics(event_list, items),
        "attention": attention,
        "emit": bool(attention),
    }


def render_projection(projection: Mapping[str, Any]) -> str:
    """Render the JSON projection with stable formatting and no timestamps."""
    return json.dumps(projection, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def render_closure_view(projection: Mapping[str, Any]) -> str:
    """Render the concise weekly Denji closure report."""
    lines = [
        "Finding | Evidence | Owner | Age | Action task | State | Resolution proof | Next action"
    ]
    for item in projection.get("findings", []):
        evidence = "; ".join(item.get("evidence_refs") or []) or "MISSING"
        lines.append(
            " | ".join(
                [
                    str(item.get("finding_id") or ""),
                    evidence,
                    str(item.get("owner") or "MISSING"),
                    f"{item.get('age_days', 0)}d",
                    str(item.get("task_id") or "MISSING"),
                    str(item.get("state") or ""),
                    str(item.get("resolution_ref") or "MISSING"),
                    str(item.get("next_action") or "monitor"),
                ]
            )
        )
    metrics = projection.get("metrics") or {}
    lines.extend(
        [
            "",
            "Weekly metrics",
            f"False-positive rate | {metrics.get('false_positive_rate') if metrics.get('false_positive_rate') is not None else 'insufficient evidence'}",
            f"Time to owner (days) | {metrics.get('time_to_owner_days') if metrics.get('time_to_owner_days') is not None else 'insufficient evidence'}",
            f"Time to verified resolution (days) | {metrics.get('time_to_verified_resolution_days') if metrics.get('time_to_verified_resolution_days') is not None else 'insufficient evidence'}",
            f"Recurring findings | {metrics.get('recurring_findings', 0)}",
            f"Human decisions waiting | {metrics.get('human_decisions_waiting', 0)}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_projection(
    output_dir: Path,
    events: Iterable[Mapping[str, Any]],
    *,
    now: int | None = None,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the deterministic JSON and Markdown views and return the projection."""
    projection = project_events(events, now=now, previous=previous)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "current-findings.json").write_text(
        render_projection(projection), encoding="utf-8"
    )
    (output_dir / "denji-closure-view.md").write_text(
        render_closure_view(projection), encoding="utf-8"
    )
    return projection


def load_current_projection(path: Path) -> dict[str, Any] | None:
    """Read only this projector's schema; ignore legacy files as authority."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and data.get("schema_version") == 1 else None


def load_canonical_events() -> list[dict[str, Any]]:
    return ledger.query_events(event_types=sorted(governance_findings.FINDING_EVENT_TYPES))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Kept as a compatibility positional so old scheduler invocations do not
    # crash; it is deliberately not read and cannot become current truth.
    parser.add_argument("_legacy_input", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=LOGBOARD)
    parser.add_argument("--now", type=int, default=None)
    args = parser.parse_args(argv)

    events = load_canonical_events()
    previous = load_current_projection(args.output_dir / "current-findings.json")
    projection = write_projection(args.output_dir, events, now=args.now, previous=previous)
    if not projection["emit"]:
        print("[SILENT]")
        return 0

    by_reason = Counter(item["reason"] for item in projection["attention"])
    summary = ", ".join(f"{key}={by_reason[key]}" for key in sorted(by_reason))
    print(f"Governance findings require attention: {len(projection['attention'])} ({summary})")
    print(f"Projection: {args.output_dir / 'current-findings.json'}")
    print(f"Closure view: {args.output_dir / 'denji-closure-view.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
