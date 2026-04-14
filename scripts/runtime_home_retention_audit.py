#!/usr/bin/env python3
"""runtime_home_retention_audit.py

Non-destructive retention audit for OpenClaw runtime-home artifacts (default: ~/.openclaw).

Design goals:
- fail-closed on missing runtime home or ambiguous rule overlaps,
- no destructive deletion in this helper,
- provide deterministic stale-candidate inventory for operator-reviewed pruning.

Usage:
  python3 scripts/runtime_home_retention_audit.py --json
  python3 scripts/runtime_home_retention_audit.py --runtime-home /tmp/.openclaw --strict --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

SCHEMA_VERSION = "openclaw.runtime_home_retention_audit.v1"
SECONDS_PER_DAY = 86400


@dataclass(frozen=True)
class RetentionRule:
    rule_id: str
    description: str
    patterns: tuple[str, ...]
    retention_days: int
    action: str


DEFAULT_RULES: tuple[RetentionRule, ...] = (
    RetentionRule(
        rule_id="sessions.active_jsonl",
        description="Active session transcripts under per-agent session stores",
        patterns=("agents/*/sessions/*.jsonl",),
        retention_days=14,
        action="archive_then_prune_candidate",
    ),
    RetentionRule(
        rule_id="sessions.archive_payloads",
        description="Session archives produced by watchdog/prune tooling",
        patterns=("_archives/sessions/*",),
        retention_days=21,
        action="prune_archive_candidate",
    ),
    RetentionRule(
        rule_id="media.inbound",
        description="Inbound media/files retained in runtime home",
        patterns=("media/inbound/**/*",),
        retention_days=30,
        action="archive_or_externalize_candidate",
    ),
    RetentionRule(
        rule_id="cron.run_logs",
        description="Cron run JSONL logs",
        patterns=("cron/runs/*.jsonl",),
        retention_days=30,
        action="archive_then_prune_candidate",
    ),
    RetentionRule(
        rule_id="logs.general",
        description="Runtime logs emitted under ~/.openclaw/logs",
        patterns=("logs/**/*.log", "logs/**/*.jsonl"),
        retention_days=14,
        action="prune_candidate",
    ),
)


def _safe_rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _iter_rule_files(runtime_home: Path, rule: RetentionRule) -> List[Path]:
    files: List[Path] = []
    seen: set[Path] = set()
    for pattern in rule.patterns:
        for p in runtime_home.glob(pattern):
            if not p.is_file():
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            files.append(p)
    return files


def build_audit_report(
    runtime_home: Path,
    *,
    now_epoch: int,
    sample_limit: int,
    rules: Sequence[RetentionRule] = DEFAULT_RULES,
) -> Dict[str, Any]:
    ownership: Dict[Path, str] = {}
    overlap_conflicts: List[Dict[str, str]] = []

    rule_reports: List[Dict[str, Any]] = []

    unique_files_scanned: set[Path] = set()
    unique_stale_files: set[Path] = set()

    total_bytes = 0
    total_stale_bytes = 0

    for rule in rules:
        files = _iter_rule_files(runtime_home, rule)
        entries: List[Dict[str, Any]] = []

        for p in files:
            rp = p.resolve()
            owner = ownership.get(rp)
            if owner and owner != rule.rule_id:
                overlap_conflicts.append(
                    {
                        "path": _safe_rel(p, runtime_home),
                        "first_rule": owner,
                        "second_rule": rule.rule_id,
                    }
                )
                continue
            ownership[rp] = rule.rule_id

            st = p.stat()
            age_seconds = max(0, int(now_epoch - int(st.st_mtime)))
            age_days = age_seconds / SECONDS_PER_DAY
            stale = age_seconds > (rule.retention_days * SECONDS_PER_DAY)

            entry = {
                "path": _safe_rel(p, runtime_home),
                "size_bytes": int(st.st_size),
                "mtime_epoch": int(st.st_mtime),
                "age_days": round(age_days, 2),
                "stale_candidate": bool(stale),
            }
            entries.append(entry)

            unique_files_scanned.add(rp)
            total_bytes += int(st.st_size)
            if stale:
                unique_stale_files.add(rp)
                total_stale_bytes += int(st.st_size)

        stale_entries = [e for e in entries if e["stale_candidate"]]
        stale_entries_sorted = sorted(
            stale_entries,
            key=lambda e: (e["mtime_epoch"], e["path"]),
        )

        rule_reports.append(
            {
                "rule_id": rule.rule_id,
                "description": rule.description,
                "patterns": list(rule.patterns),
                "retention_days": rule.retention_days,
                "action": rule.action,
                "matched_files": len(entries),
                "matched_bytes": int(sum(int(e["size_bytes"]) for e in entries)),
                "stale_candidates": len(stale_entries),
                "stale_candidate_bytes": int(sum(int(e["size_bytes"]) for e in stale_entries)),
                "sample_candidates": stale_entries_sorted[:sample_limit],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_epoch": int(now_epoch),
        "runtime_home": str(runtime_home.resolve()),
        "policy": {
            "name": "runtime-home-retention-core-v1",
            "mode": "audit_only_non_destructive",
        },
        "rules": rule_reports,
        "totals": {
            "files_scanned": len(unique_files_scanned),
            "bytes_scanned": int(total_bytes),
            "stale_candidates": len(unique_stale_files),
            "stale_candidate_bytes": int(total_stale_bytes),
        },
        "overlap_conflicts": overlap_conflicts,
    }


def _to_text(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("OpenClaw runtime-home retention audit")
    lines.append("====================================")
    lines.append(f"schema: {report.get('schema_version')}")
    lines.append(f"runtime_home: {report.get('runtime_home')}")
    lines.append(f"mode: {((report.get('policy') or {}).get('mode') or 'unknown')}")
    lines.append("")

    totals = report.get("totals") or {}
    lines.append(
        "totals: "
        f"files_scanned={totals.get('files_scanned', 0)} "
        f"bytes_scanned={totals.get('bytes_scanned', 0)} "
        f"stale_candidates={totals.get('stale_candidates', 0)} "
        f"stale_candidate_bytes={totals.get('stale_candidate_bytes', 0)}"
    )

    conflicts = report.get("overlap_conflicts") or []
    if conflicts:
        lines.append("")
        lines.append(f"BLOCKER: policy overlap conflicts={len(conflicts)}")
        for c in conflicts[:10]:
            lines.append(
                f"  - {c.get('path')} first={c.get('first_rule')} second={c.get('second_rule')}"
            )

    lines.append("")
    lines.append("per-rule summary:")
    for rule in report.get("rules") or []:
        lines.append(
            f"- {rule.get('rule_id')}: matched={rule.get('matched_files')} "
            f"stale={rule.get('stale_candidates')} retention_days={rule.get('retention_days')} "
            f"action={rule.get('action')}"
        )
        for sample in rule.get("sample_candidates") or []:
            lines.append(
                f"    * {sample.get('path')} age_days={sample.get('age_days')} size={sample.get('size_bytes')}"
            )

    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Non-destructive retention audit for ~/.openclaw runtime artifacts")
    ap.add_argument("--runtime-home", default=str(Path.home() / ".openclaw"), help="Runtime home to audit")
    ap.add_argument("--now-epoch", type=int, default=int(time.time()), help="Override current epoch for deterministic runs")
    ap.add_argument("--sample-limit", type=int, default=5, help="Max stale examples per rule")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero when stale candidates are found")
    ap.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    ap.add_argument("--write-json", help="Optional file path to persist JSON report")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    runtime_home = Path(os.path.expanduser(args.runtime_home)).resolve()
    if not runtime_home.exists() or not runtime_home.is_dir():
        blocker = {
            "schema_version": SCHEMA_VERSION,
            "status": "BLOCKER",
            "reason": "runtime_home_missing_or_not_directory",
            "runtime_home": str(runtime_home),
        }
        if args.json:
            print(json.dumps(blocker, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            sys.stdout.write(
                f"BLOCKER: runtime_home_missing_or_not_directory path={runtime_home}\n"
            )
        return 2

    report = build_audit_report(
        runtime_home,
        now_epoch=int(args.now_epoch),
        sample_limit=max(1, int(args.sample_limit)),
    )

    if args.write_json:
        _write_json(Path(args.write_json), report)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        sys.stdout.write(_to_text(report))

    conflicts = report.get("overlap_conflicts") or []
    if conflicts:
        return 2

    stale_total = int(((report.get("totals") or {}).get("stale_candidates") or 0))
    if args.strict and stale_total > 0:
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
