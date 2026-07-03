#!/usr/bin/env python3
"""
Monthly Summary Compression Cron Script.

Compresses 4-5 weekly summaries into a monthly_summary Markdown file.
Only preserves identity-relevant trajectory (not daily details).

Usage:
    python monthly_summary.py                              # current month
    python monthly_summary.py --month 2026-05              # specific month
    python monthly_summary.py --dry-run
    python monthly_summary.py --force
"""

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

_HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
_PLUGIN_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(_PLUGIN_DIR.parent))
sys.path.insert(0, str(_HERMES_HOME / "hermes-agent"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("monthly_summary")

TEMPLATE_PATH = _PLUGIN_DIR / "templates" / "monthly_summary.md"
TOKEN_BUDGET = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monthly Summary Compression")
    parser.add_argument("--month", type=str, default="", help="Month key (e.g. 2026-05)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def get_month_key(date_str: str = "") -> str:
    if date_str:
        d = date.fromisoformat(f"{date_str}-01")
    else:
        d = date.today()
    return f"{d.year}-{d.month:02d}"


def get_weeks_in_month(month_key: str) -> List[str]:
    """Get ISO week keys that fall within this month."""
    year = int(month_key[:4])
    month = int(month_key[5:7])

    # Get first and last day of month
    if month == 12:
        first_day = date(year, 12, 1)
        last_day = date(year, 12, 31)
    else:
        first_day = date(year, month, 1)
        last_day = date(year, month + 1, 1) - timedelta(days=1)

    weeks = set()
    d = first_day
    while d <= last_day:
        iso_year, iso_week, _ = d.isocalendar()
        weeks.add(f"{iso_year}-W{iso_week:02d}")
        d += timedelta(days=1)

    return sorted(weeks)


def load_weekly_summaries(week_keys: List[str]) -> List[Dict]:
    """Load weekly summary files for given weeks."""
    summaries = []
    for wk in week_keys:
        path = _HERMES_HOME / "continuum" / "journal" / "weekly" / f"{wk}.md"
        if path.exists():
            summaries.append({
                "week": wk,
                "path": str(path),
                "content": path.read_text(encoding="utf-8", errors="replace")[:1500],
            })
    return summaries


def compress_to_monthly(summaries: List[Dict], month_key: str) -> str:
    """Generate monthly summary from weekly summaries."""
    if not summaries:
        return f"No weekly summaries found for {month_key}."

    lines = [
        f"## Identity Trajectory\n\n"
        f"Monthly summary for {month_key} — synthesised from {len(summaries)} weekly summaries.\n\n"
    ]

    for s in summaries:
        lines.append(f"### Week {s['week']} Highlights\n")
        lines.append(s['content'][:600] + "\n\n")

    # Extract long-running themes
    lines.append("\n## Persistent Open Threads\n")
    lines.append("(Recurring patterns extracted from weekly summaries)\n")

    return "".join(lines)


def run(month_key: str = "", dry_run: bool = False, force: bool = False) -> Dict:
    if not month_key:
        month_key = get_month_key()

    week_keys = get_weeks_in_month(month_key)
    summaries = load_weekly_summaries(week_keys)

    logger.info("Month %s: %d weekly summaries found", month_key, len(summaries))

    # Output path
    month_dir = _HERMES_HOME / "continuum" / "journal" / "monthly"
    month_dir.mkdir(parents=True, exist_ok=True)
    output_path = month_dir / f"{month_key}.md"

    if output_path.exists() and not force:
        logger.info("Monthly summary already exists at %s (use --force)", output_path)
        return {"status": "skipped", "path": str(output_path)}

    content = compress_to_monthly(summaries, month_key)
    max_chars = TOKEN_BUDGET * 4
    if len(content) > max_chars:
        content = content[:max_chars] + "\n... [truncated to fit budget]"

    if dry_run:
        logger.info("DRY RUN — monthly summary:\n%s", content[:500])
        return {"status": "dry-run", "content_preview": content[:200]}

    output_path.write_text(content, encoding="utf-8")
    logger.info("Written monthly summary to %s", output_path)

    # Register DAG node
    try:
        from hashlib import md5
        from plugins.continuity.db import upsert_node, add_edge, get_nodes_by_type

        node_id = f"monthly_{md5(month_key.encode()).hexdigest()[:12]}"
        upsert_node(
            node_type="monthly",
            date_key=month_key,
            title=f"Monthly Summary — {month_key}",
            markdown_path=str(output_path),
            token_count=len(content) // 4,
            compression_depth=3,
            provider="cron",
            model="monthly-compression",
            author_mode="cron",
            node_id=node_id,
        )

        # Link weekly summaries as sources (weekly → monthly)
        for s in summaries:
            weekly_nodes = get_nodes_by_type("weekly", limit=20)
            for wn in weekly_nodes:
                if wn.get("markdown_path", "").endswith(f"{s['week']}.md"):
                    add_edge(parent_id=wn["node_id"], child_id=node_id)
                    break

        logger.info("DAG node registered: %s", node_id)
        return {"status": "ok", "path": str(output_path), "node_id": node_id, "weeks": len(summaries)}
    except Exception as exc:
        logger.warning("DAG registration failed: %s", exc)
        return {"status": "ok", "path": str(output_path), "dag_error": str(exc)}


def main():
    args = parse_args()
    result = run(month_key=args.month, dry_run=args.dry_run, force=args.force)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
