#!/usr/bin/env python3
"""Reconcile learning-ledger JSONL entries to run envelopes (stdlib only).

Reports:
  - linked: entry has run_id that matches an envelope under --runs-root
  - orphan: entry has run_id but no matching envelope found
  - unlinked: entry missing run_id
  - routing / gate quality tallies (from feedback fields)

Never mutates skill corpus. Never promotes canon.

Usage:
  python scripts/reconcile_learning.py --ledger out/neon-genie/learning-ledger.jsonl \\
    --runs-root out/neon-genie --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"FAIL: NG-LEARN-010: ledger line {i}: {exc}") from exc
        if isinstance(obj, dict):
            entries.append(obj)
    return entries


def index_envelopes(runs_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not runs_root.is_dir():
        return found
    for path in runs_root.rglob("run-envelope.json"):
        try:
            env = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rid = env.get("run_id")
        if isinstance(rid, str) and rid:
            found[rid] = path
    return found


def reconcile(
    entries: list[dict[str, Any]],
    envelopes: dict[str, Path],
) -> dict[str, Any]:
    linked: list[dict[str, Any]] = []
    orphan: list[dict[str, Any]] = []
    unlinked: list[dict[str, Any]] = []
    routing = Counter()
    gates_q = Counter()
    classes = Counter()
    auto_apply_violations = 0

    for e in entries:
        classes[str(e.get("observation_class") or "unknown")] += 1
        if e.get("auto_apply_forbidden") is False or e.get("canon_status") not in {
            None,
            "PROPOSED",
            "OBSERVATION",
        }:
            auto_apply_violations += 1
        rq = e.get("routing_quality")
        if rq:
            routing[str(rq)] += 1
        gq = e.get("gate_quality")
        if gq:
            gates_q[str(gq)] += 1

        rid = e.get("run_id") or ""
        if not rid and isinstance(e.get("source_run"), str):
            # tolerate source_run holding run_id
            sr = e["source_run"]
            if sr.startswith("ng_run_"):
                rid = sr

        row = {
            "entry_id": e.get("entry_id"),
            "run_id": rid or None,
            "observation_class": e.get("observation_class"),
            "summary": (e.get("summary") or "")[:120],
        }
        if not rid:
            unlinked.append(row)
        elif rid in envelopes:
            row["envelope"] = str(envelopes[rid])
            linked.append(row)
        else:
            orphan.append(row)

    return {
        "skill": "neon-genie",
        "entries": len(entries),
        "envelopes_indexed": len(envelopes),
        "linked": linked,
        "orphan_run_ids": orphan,
        "unlinked": unlinked,
        "counts": {
            "linked": len(linked),
            "orphan": len(orphan),
            "unlinked": len(unlinked),
        },
        "observation_classes": dict(classes),
        "routing_quality": dict(routing),
        "gate_quality": dict(gates_q),
        "auto_apply_violations": auto_apply_violations,
        "policy": {
            "canon_auto_apply": False,
            "promotion_requires_review": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reconcile learning ledger to run envelopes")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("out/neon-genie/learning-ledger.jsonl"),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("out/neon-genie"),
        help="Directory tree to search for run-envelope.json",
    )
    parser.add_argument("--json", action="store_true", help="JSON report only")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON report path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if orphans, unlinked, or auto_apply violations exist",
    )
    args = parser.parse_args(argv)

    ledger = args.ledger if args.ledger.is_absolute() else Path.cwd() / args.ledger
    runs = args.runs_root if args.runs_root.is_absolute() else Path.cwd() / args.runs_root

    entries = load_jsonl(ledger)
    envelopes = index_envelopes(runs)
    report = reconcile(entries, envelopes)

    if args.out:
        out = args.out if args.out.is_absolute() else Path.cwd() / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        c = report["counts"]
        print("PASS: learning ledger reconcile" if not args.strict else "REPORT: learning ledger reconcile")
        print(f"  entries: {report['entries']}")
        print(f"  linked: {c['linked']}  orphan: {c['orphan']}  unlinked: {c['unlinked']}")
        print(f"  envelopes indexed: {report['envelopes_indexed']}")
        if report["routing_quality"]:
            print(f"  routing_quality: {report['routing_quality']}")
        if report["gate_quality"]:
            print(f"  gate_quality: {report['gate_quality']}")
        if report["auto_apply_violations"]:
            print(
                f"  FAIL risk: auto_apply_violations={report['auto_apply_violations']}",
                file=sys.stderr,
            )

    if args.strict:
        if report["auto_apply_violations"]:
            print("FAIL: NG-LEARN-011: auto_apply or canon policy violation", file=sys.stderr)
            return 1
        # strict on empty ledger is ok
        if report["entries"] and (report["counts"]["orphan"] or report["counts"]["unlinked"]):
            print(
                "FAIL: NG-LEARN-012: unlinked or orphan run_id entries (link with --run-id/--envelope)",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
