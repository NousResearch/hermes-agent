#!/usr/bin/env python3
"""Queue decay checker for ai-evolution-candidate-queue.md.

Spec: dual-agent-evolution-plan-2026q2.md §10.6.

Rules:
- collecting > 14 days without new evidence → mark `stale`
- stale > 14 days → force weekly review裁決 (report only, no auto-sunset)
- pilot 連續 2 次 monthly review 未升格 → auto-sunset candidate

Reads  : ~/wiki/operations/ai-evolution-candidate-queue.md
Outputs: stdout markdown report with recommendations.
Exits  : 0 always (advisory only).
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

QUEUE_PATH_DEFAULT = Path.home() / "wiki" / "operations" / "ai-evolution-candidate-queue.md"
STATUS_PATH_DEFAULT = Path.home() / "wiki" / "operations" / "ai-evolution-status.md"

STALE_AFTER_DAYS = 14
FORCE_REVIEW_AFTER_STALE_DAYS = 14

# Pattern for a queue row:
# | se-xxx | title | Agent | Layer | change | evidence... | gain | risk | proof | status | cadence |
ROW_RE = re.compile(r"^\|\s*(se-\d+)\s*\|", re.IGNORECASE)


def parse_queue_rows(queue_text: str) -> List[Dict[str, str]]:
    """Return list of dicts for each se-* row in the Current candidates table."""
    rows: List[Dict[str, str]] = []
    in_current = False
    for line in queue_text.splitlines():
        if line.strip().lower().startswith("## current candidates"):
            in_current = True
            continue
        if in_current and line.strip().startswith("## "):
            break  # left the section
        if not in_current:
            continue
        if not ROW_RE.match(line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 10:
            continue
        rows.append({
            "raw": line,
            "id": cells[0],
            "title": cells[1],
            "agent": cells[2],
            "layer": cells[3],
            "change_type": cells[4],
            "evidence": cells[5],
            "gain": cells[6],
            "risk": cells[7],
            "required_proof": cells[8],
            "status": cells[9],
            "cadence": cells[10] if len(cells) > 10 else "",
        })
    return rows


def extract_latest_evidence_date(evidence_text: str) -> Optional[date]:
    """Pull the most recent YYYY-MM-DD or YYYY-M-D from evidence text."""
    dates: List[date] = []
    for m in re.finditer(r"(20\d\d)-(\d{1,2})-(\d{1,2})", evidence_text):
        try:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            dates.append(date(y, mo, d))
        except ValueError:
            continue
    return max(dates) if dates else None


def classify_row(row: Dict[str, str], today: date) -> Tuple[str, str]:
    """Return (classification, reason).

    classification ∈ {fresh, stale, force-review, pilot-at-risk, terminal}
    """
    status = row["status"].lower()
    if status in ("accepted", "sunset", "rejected", "deferred"):
        return "terminal", f"already {status}"

    latest = extract_latest_evidence_date(row["evidence"])
    if latest is None:
        return "stale", "no dated evidence found in text"

    age_days = (today - latest).days
    if status == "pilot" and age_days > 30:
        return "pilot-at-risk", f"pilot {age_days}d since last evidence; monthly裁決 due"

    if age_days <= STALE_AFTER_DAYS:
        return "fresh", f"{age_days}d since latest evidence"

    if age_days <= STALE_AFTER_DAYS + FORCE_REVIEW_AFTER_STALE_DAYS:
        return "stale", f"{age_days}d since latest evidence (>{STALE_AFTER_DAYS})"

    return "force-review", f"{age_days}d since latest evidence; weekly裁決 required"


def main() -> int:
    ap = argparse.ArgumentParser(description="Scan evolution queue for decay signals.")
    ap.add_argument("--queue", type=Path, default=QUEUE_PATH_DEFAULT)
    ap.add_argument("--today", type=str, help="Override today's date (YYYY-MM-DD)")
    args = ap.parse_args()

    if args.today:
        today = datetime.strptime(args.today, "%Y-%m-%d").date()
    else:
        today = date.today()

    if not args.queue.exists():
        print(f"ERROR: queue file not found: {args.queue}", file=sys.stderr)
        return 2

    rows = parse_queue_rows(args.queue.read_text(encoding="utf-8"))
    if not rows:
        print("# Queue decay report\n\nNo candidate rows parsed.\n")
        return 0

    buckets: Dict[str, List[Tuple[Dict[str, str], str]]] = {
        "fresh": [], "stale": [], "force-review": [], "pilot-at-risk": [], "terminal": [],
    }
    for row in rows:
        cls, reason = classify_row(row, today)
        buckets[cls].append((row, reason))

    print(f"# Queue decay report ({today.isoformat()})\n")
    print(f"Total rows: {len(rows)}\n")
    print(f"- fresh: {len(buckets['fresh'])}")
    print(f"- stale: {len(buckets['stale'])}")
    print(f"- force-review: {len(buckets['force-review'])}")
    print(f"- pilot-at-risk: {len(buckets['pilot-at-risk'])}")
    print(f"- terminal: {len(buckets['terminal'])}\n")

    for bucket in ("force-review", "pilot-at-risk", "stale"):
        if not buckets[bucket]:
            continue
        print(f"## {bucket} ({len(buckets[bucket])})\n")
        for row, reason in buckets[bucket]:
            print(f"- **{row['id']}** — {row['title'][:80]}")
            print(f"  - status: {row['status']}  |  agent: {row['agent']}")
            print(f"  - reason: {reason}")
            print(f"  - recommended: {recommend(bucket)}\n")

    if not any(buckets[b] for b in ("force-review", "pilot-at-risk", "stale")):
        print("## All clear\n\nNo candidates flagged this pass.\n")

    return 0


def recommend(bucket: str) -> str:
    return {
        "stale": "在下次 weekly review 補新證據；若仍缺，降 priority 或進入 force-review",
        "force-review": "weekly 必須裁決 accept / rewrite / sunset，不可再 collecting",
        "pilot-at-risk": "monthly review 必須決定 accept 或 sunset；pilot 不可永久存在",
    }.get(bucket, "")


if __name__ == "__main__":
    sys.exit(main())
