#!/usr/bin/env python3
"""Deterministic closeout verifier for targeted repo-review queue rows.

This verifier is intended for fold-in / wave closeout checks where a report claims
completion for specific queue rows. It fail-closes when targeted rows are missing,
out-of-parity across queue truth surfaces, or non-terminal.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_PRIMARY_PATH = "reports/github_repo_review_queue_openclaw_patterns_2026-03-20.md"
DEFAULT_NORMALIZED_PATH = "reports/github_repo_review_queue_openclaw_patterns_2026-03-20_NORMALIZED.md"
DEFAULT_TERMINAL_STATUSES = ["done", "folded"]

ROW_HEADING_RE = re.compile(r"^###\s+(.+?)\s*$")
STATUS_RE = re.compile(r"^- Status:\s+(.+?)\s*$", re.MULTILINE)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_repo_path(repo_root: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    else:
        p = p.resolve()
    return p


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n")).strip()
    return normalized + "\n"


def collect_row_blocks(text: str) -> Dict[str, str]:
    lines = normalize_text(text).splitlines()
    blocks: Dict[str, str] = {}
    indices: List[Tuple[str, int]] = []

    for idx, line in enumerate(lines):
        m = ROW_HEADING_RE.match(line)
        if m:
            indices.append((m.group(1).strip(), idx))

    for i, (row_id, start) in enumerate(indices):
        end = indices[i + 1][1] if i + 1 < len(indices) else len(lines)
        while end > start and not lines[end - 1].strip():
            end -= 1
        blocks[row_id] = "\n".join(lines[start:end]).strip() + "\n"

    return blocks


def extract_field(pattern: re.Pattern[str], block: str) -> Optional[str]:
    m = pattern.search(block)
    if not m:
        return None
    return m.group(1).strip()


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        token = item.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def evaluate(
    *,
    repo_root: Path,
    primary_path: Path,
    normalized_path: Path,
    target_rows: List[str],
    terminal_statuses: List[str],
    report_path: Optional[Path],
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    parity_failures: List[Dict[str, Any]] = []

    if report_path is not None:
        report_exists = report_path.exists() and report_path.is_file()
        checks.append(
            {
                "check": "closeout_report_path_exists",
                "ok": report_exists,
                "report_path": str(report_path),
            }
        )
        if not report_exists:
            parity_failures.append(
                {
                    "code": "closeout_report_missing",
                    "report_path": str(report_path),
                }
            )

    if not target_rows:
        checks.append(
            {
                "check": "target_rows_nonempty",
                "ok": False,
                "target_rows": [],
            }
        )
        parity_failures.append({"code": "target_rows_missing"})
        decision = "BLOCK"
        return {
            "schema": "clawd.repo_review_queue_closeout_verifier.v1",
            "generated_at": now_iso(),
            "decision": decision,
            "block_reason": parity_failures[0]["code"],
            "primary_path": str(primary_path),
            "normalized_path": str(normalized_path),
            "report_path": str(report_path) if report_path is not None else None,
            "target_rows": target_rows,
            "terminal_statuses": terminal_statuses,
            "checks": checks,
            "parity_failures": parity_failures,
        }

    missing_surfaces = [
        str(path)
        for path in (primary_path, normalized_path)
        if not path.exists() or not path.is_file()
    ]
    checks.append(
        {
            "check": "queue_surface_paths_exist",
            "ok": len(missing_surfaces) == 0,
            "missing": missing_surfaces,
        }
    )
    if missing_surfaces:
        parity_failures.append(
            {
                "code": "queue_surface_missing",
                "missing": missing_surfaces,
            }
        )
        return {
            "schema": "clawd.repo_review_queue_closeout_verifier.v1",
            "generated_at": now_iso(),
            "decision": "BLOCK",
            "block_reason": parity_failures[0]["code"],
            "primary_path": str(primary_path),
            "normalized_path": str(normalized_path),
            "report_path": str(report_path) if report_path is not None else None,
            "target_rows": target_rows,
            "terminal_statuses": terminal_statuses,
            "checks": checks,
            "parity_failures": parity_failures,
        }

    primary_rows = collect_row_blocks(load_text(primary_path))
    normalized_rows = collect_row_blocks(load_text(normalized_path))

    primary_ids = set(primary_rows.keys())
    normalized_ids = set(normalized_rows.keys())
    missing_in_primary = sorted(normalized_ids - primary_ids)
    missing_in_normalized = sorted(primary_ids - normalized_ids)
    row_ids_match = not missing_in_primary and not missing_in_normalized
    checks.append(
        {
            "check": "queue_row_ids_match",
            "ok": row_ids_match,
            "primary_row_count": len(primary_ids),
            "normalized_row_count": len(normalized_ids),
            "missing_in_primary": missing_in_primary,
            "missing_in_normalized": missing_in_normalized,
        }
    )
    if not row_ids_match:
        parity_failures.append(
            {
                "code": "queue_row_id_drift",
                "missing_in_primary": missing_in_primary,
                "missing_in_normalized": missing_in_normalized,
            }
        )

    target_presence_violations: List[Dict[str, Any]] = []
    for row_id in target_rows:
        in_primary = row_id in primary_rows
        in_normalized = row_id in normalized_rows
        if not in_primary or not in_normalized:
            target_presence_violations.append(
                {
                    "row_id": row_id,
                    "present_in_primary": in_primary,
                    "present_in_normalized": in_normalized,
                }
            )
            parity_failures.append(
                {
                    "code": "target_row_missing",
                    "row_id": row_id,
                    "present_in_primary": in_primary,
                    "present_in_normalized": in_normalized,
                }
            )

    checks.append(
        {
            "check": "target_rows_present_in_both_surfaces",
            "ok": len(target_presence_violations) == 0,
            "target_rows": target_rows,
            "violations": target_presence_violations,
        }
    )

    status_drift_rows: List[Dict[str, Any]] = []
    non_terminal_rows: List[Dict[str, Any]] = []

    terminal_status_set = set(terminal_statuses)
    for row_id in target_rows:
        if row_id not in primary_rows or row_id not in normalized_rows:
            continue

        primary_status = extract_field(STATUS_RE, primary_rows[row_id])
        normalized_status = extract_field(STATUS_RE, normalized_rows[row_id])

        if primary_status != normalized_status:
            payload = {
                "row_id": row_id,
                "primary_status": primary_status,
                "normalized_status": normalized_status,
            }
            status_drift_rows.append(payload)
            parity_failures.append({"code": "target_row_status_drift", **payload})
            continue

        if primary_status not in terminal_status_set:
            payload = {
                "row_id": row_id,
                "status": primary_status,
                "terminal_statuses": terminal_statuses,
            }
            non_terminal_rows.append(payload)
            parity_failures.append(
                {
                    "code": "closeout_report_overclaim_completion" if report_path is not None else "target_row_not_terminal",
                    **payload,
                    "report_path": str(report_path) if report_path is not None else None,
                }
            )

    checks.append(
        {
            "check": "target_rows_status_parity",
            "ok": len(status_drift_rows) == 0,
            "violations": status_drift_rows,
        }
    )
    checks.append(
        {
            "check": "target_rows_terminalized",
            "ok": len(non_terminal_rows) == 0,
            "violations": non_terminal_rows,
        }
    )

    decision = "PASS" if len(parity_failures) == 0 else "BLOCK"
    return {
        "schema": "clawd.repo_review_queue_closeout_verifier.v1",
        "generated_at": now_iso(),
        "decision": decision,
        "block_reason": parity_failures[0]["code"] if parity_failures else None,
        "primary_path": str(primary_path),
        "normalized_path": str(normalized_path),
        "report_path": str(report_path) if report_path is not None else None,
        "target_rows": target_rows,
        "terminal_statuses": terminal_statuses,
        "checks": checks,
        "parity_failures": parity_failures,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deterministic repo-review queue closeout verifier")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--primary-path", default=DEFAULT_PRIMARY_PATH, help="Primary queue markdown path")
    ap.add_argument("--normalized-path", default=DEFAULT_NORMALIZED_PATH, help="Normalized queue markdown path")
    ap.add_argument(
        "--target-row",
        action="append",
        dest="target_rows",
        default=None,
        help="Target row ID expected to be terminalized (repeatable)",
    )
    ap.add_argument(
        "--terminal-status",
        action="append",
        dest="terminal_statuses",
        default=None,
        help="Allowed terminal status for target rows (repeatable; default: done, folded)",
    )
    ap.add_argument("--report-path", default=None, help="Optional closeout/fold-in report path tied to this verification run")
    ap.add_argument("--json", action="store_true", help="Pretty-print JSON")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = resolve_repo_path(DEFAULT_REPO_ROOT, str(args.repo_root))
    primary_path = resolve_repo_path(repo_root, str(args.primary_path))
    normalized_path = resolve_repo_path(repo_root, str(args.normalized_path))
    report_path = resolve_repo_path(repo_root, str(args.report_path)) if args.report_path else None

    target_rows = _dedupe_keep_order(list(args.target_rows or []))
    terminal_statuses = _dedupe_keep_order(list(args.terminal_statuses or DEFAULT_TERMINAL_STATUSES))

    payload = evaluate(
        repo_root=repo_root,
        primary_path=primary_path,
        normalized_path=normalized_path,
        target_rows=target_rows,
        terminal_statuses=terminal_statuses,
        report_path=report_path,
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    return 0 if payload.get("decision") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
