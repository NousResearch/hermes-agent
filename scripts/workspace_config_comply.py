#!/usr/bin/env python3
"""Compliance checker for workspace config governance issue files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "issue_id",
    "phase",
    "goal",
    "status",
    "done_percent",
    "remaining_percent",
    "evidence",
}
COMPLETE_STATUSES = {"complete", "completed", "done", "verified"}
PENDING_EVIDENCE = {"", "pending", "none", "n/a", "not_applicable"}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _parse_percent(value: str) -> int | None:
    cleaned = value.strip().strip("%")
    try:
        parsed = int(cleaned)
    except ValueError:
        return None
    if parsed < 0 or parsed > 100:
        return None
    return parsed


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _parse_issue_rows(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.splitlines() if line.strip().startswith("|")]
    rows: list[dict[str, str]] = []
    headers: list[str] = []
    for line in lines:
        cells = _split_row(line)
        if not cells:
            continue
        normalized = [cell.lower().replace(" ", "_") for cell in cells]
        if "issue_id" in normalized and "done_percent" in normalized:
            headers = normalized
            continue
        if not headers:
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if len(cells) != len(headers):
            continue
        row = dict(zip(headers, cells))
        if row.get("issue_id", ""):
            rows.append(row)
    return rows


def _issue_ok(row: dict[str, str]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    missing = sorted(column for column in REQUIRED_COLUMNS if not row.get(column))
    if missing:
        reasons.append(f"missing columns: {', '.join(missing)}")

    done = _parse_percent(row.get("done_percent", ""))
    remaining = _parse_percent(row.get("remaining_percent", ""))
    if done != 100:
        reasons.append("done_percent is not 100")
    if remaining != 0:
        reasons.append("remaining_percent is not 0")

    status = row.get("status", "").strip().lower()
    if status not in COMPLETE_STATUSES:
        reasons.append("status is not complete")

    evidence = row.get("evidence", "").strip().lower()
    if evidence in PENDING_EVIDENCE:
        reasons.append("evidence is pending")

    return not reasons, reasons


def inspect_issue_file(issue_file: str | Path) -> dict[str, Any]:
    path = Path(issue_file).expanduser().resolve()
    rows = _parse_issue_rows(_read_text(path))
    issues: list[dict[str, Any]] = []
    for row in rows:
        ok, reasons = _issue_ok(row)
        done = _parse_percent(row.get("done_percent", "")) or 0
        remaining = _parse_percent(row.get("remaining_percent", "")) or 0
        issues.append(
            {
                "issue_id": row.get("issue_id", ""),
                "phase": row.get("phase", ""),
                "goal": row.get("goal", ""),
                "status": row.get("status", ""),
                "done_percent": done,
                "remaining_percent": remaining,
                "evidence": row.get("evidence", ""),
                "ok": ok,
                "reasons": reasons,
            }
        )

    issue_count = len(issues)
    done_total = sum(issue["done_percent"] for issue in issues)
    remaining_total = sum(issue["remaining_percent"] for issue in issues)
    done_percent = round(done_total / issue_count, 2) if issue_count else 0.0
    remaining_percent = round(remaining_total / issue_count, 2) if issue_count else 100.0
    ok = bool(issues) and all(issue["ok"] for issue in issues)
    return {
        "issue_file": str(path),
        "ok": ok,
        "summary": {
            "issues": issue_count,
            "complete": sum(1 for issue in issues if issue["ok"]),
            "incomplete": sum(1 for issue in issues if not issue["ok"]),
            "done_percent": done_percent,
            "remaining_percent": remaining_percent,
        },
        "issues": issues,
    }


def render_report(report: dict[str, Any], fmt: str = "text") -> str:
    if fmt == "json":
        return json.dumps(report, ensure_ascii=False, indent=2) + "\n"

    lines = [
        "Workspace config governance comply",
        f"Issue file: {report['issue_file']}",
        (
            f"Summary: {report['summary']['complete']}/{report['summary']['issues']} complete, "
            f"{report['summary']['done_percent']}% done, "
            f"{report['summary']['remaining_percent']}% remaining"
        ),
        "",
        "| Phase | Issue | รายละเอียด | ทำได้ % | เหลือ % | หลักฐานตรวจ | สถานะ |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for issue in report["issues"]:
        status = issue["status"] if issue["ok"] else "blocked"
        lines.append(
            f"| {issue['phase']} | {issue['issue_id']} | {issue['goal']} | "
            f"{issue['done_percent']} | {issue['remaining_percent']} | "
            f"{issue['evidence']} | {status} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue-file", required=True, help="Markdown issue file to inspect.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    args = parser.parse_args(argv)

    report = inspect_issue_file(args.issue_file)
    print(render_report(report, args.format), end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
