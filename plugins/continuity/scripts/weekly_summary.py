#!/usr/bin/env python3
"""
Weekly Summary Compression Cron Script.

Compresses 7 daily journals into a weekly_summary Markdown file.
Registers the weekly node in the DAG and links it to child daily nodes.

Usage:
    python weekly_summary.py                              # current week
    python weekly_summary.py --week 2026-W22              # specific week
    python weekly_summary.py --dry-run
    python weekly_summary.py --force
"""

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

_HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
_PLUGIN_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(_PLUGIN_DIR.parent))
sys.path.insert(0, str(_HERMES_HOME / "hermes-agent"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("weekly_summary")

TEMPLATE_PATH = _PLUGIN_DIR / "templates" / "weekly_summary.md"
TOKEN_BUDGET = 650


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly Summary Compression")
    parser.add_argument("--week", type=str, default="", help="Week key (e.g. 2026-W22)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def get_current_week_dates() -> List[str]:
    """Return ISO dates for Mon-Sun of current week."""
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    return [(monday + timedelta(days=i)).isoformat() for i in range(7)]


def get_week_key(date_str: str = "") -> str:
    if date_str:
        d = date.fromisoformat(date_str)
    else:
        d = date.today()
    iso_year, iso_week, _ = d.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def load_daily_journals(week_dates: List[str]) -> List[Dict]:
    """Load daily journal nodes for given dates."""
    journals = []
    for d in week_dates:
        path = _HERMES_HOME / "continuum" / "journal" / "daily" / f"{d}.md"
        if path.exists():
            journals.append({"date": d, "path": str(path), "content": path.read_text(encoding="utf-8", errors="replace")[:2000]})
    return journals


def compress_to_weekly(journals: List[Dict], week_key: str) -> str:
    """Generate weekly summary content from daily journals."""
    if not journals:
        return f"No daily journals found for week {week_key}."

    lines = [
        f"## Operational Arc\n\n"
        f"Week {week_key} spanned {len(journals)} days with journal entries.\n\n"
    ]

    for j in journals:
        lines.append(f"### {j['date']}\n")
        content = j['content'][:800]
        lines.append(content + "\n")

    lines.append(f"\n## Relational Continuity Arc\n\n")
    for j in journals:
        rel_part = _extract_relational(j['content'])[:600]
        if rel_part:
            lines.append(f"({j['date']}): {rel_part}\n")

    return "\n".join(lines)


def _extract_relational(content: str) -> str:
    """Extract relational section from a daily journal."""
    import re
    for marker in ["## Relational Continuity", "### Relational Continuity", "## Relational"]:
        if marker in content:
            parts = content.split(marker, 1)
            if len(parts) > 1:
                rest = parts[1]
                end = re.search(r"\n## ", rest)
                if end:
                    return rest[:end.start()].strip()
                return rest.strip()[:800]
    return ""


def run(week_key: str = "", dry_run: bool = False, force: bool = False) -> Dict:
    if not week_key:
        week_key = get_week_key()

    week_dates = get_week_dates(week_key)
    journals = load_daily_journals(week_dates)

    logger.info("Week %s: %d daily journals found", week_key, len(journals))

    # Output path
    week_dir = _HERMES_HOME / "continuum" / "journal" / "weekly"
    week_dir.mkdir(parents=True, exist_ok=True)
    output_path = week_dir / f"{week_key}.md"

    if output_path.exists() and not force:
        logger.info("Weekly summary already exists at %s (use --force)", output_path)
        return {"status": "skipped", "path": str(output_path)}

    content = compress_to_weekly(journals, week_key)
    content = _truncate_to_budget(content, TOKEN_BUDGET)

    if dry_run:
        logger.info("DRY RUN — would write:\n%s", content[:500])
        return {"status": "dry-run", "content_preview": content[:200]}

    output_path.write_text(content, encoding="utf-8")
    logger.info("Written weekly summary to %s", output_path)

    # Register DAG node
    try:
        from hashlib import md5
        from plugins.continuity.db import upsert_node, add_edge, get_nodes_by_type

        node_id = f"weekly_{md5(week_key.encode()).hexdigest()[:12]}"
        upsert_node(
            node_type="weekly",
            date_key=week_key,
            title=f"Weekly Summary — {week_key}",
            markdown_path=str(output_path),
            token_count=len(content) // 4,
            compression_depth=2,
            provider="cron",
            model="weekly-compression",
            author_mode="cron",
            node_id=node_id,
        )

        # Link daily journals as sources (daily → weekly)
        for j in journals:
            daily_nodes = get_nodes_by_type("daily", limit=31)
            for dn in daily_nodes:
                if dn.get("markdown_path", "").endswith(f"{j['date']}.md"):
                    add_edge(parent_id=dn["node_id"], child_id=node_id)
                    break

        logger.info("DAG node registered: %s", node_id)
        return {"status": "ok", "path": str(output_path), "node_id": node_id, "journals": len(journals)}
    except Exception as exc:
        logger.warning("DAG registration failed: %s", exc)
        return {"status": "ok", "path": str(output_path), "dag_error": str(exc)}


def get_week_dates(week_key: str) -> List[str]:
    """For a week_key like '2026-W22', return the 7 dates."""
    from datetime import datetime
    try:
        year = int(week_key[:4])
        week = int(week_key.split("W")[1])
        # January 4th is always in week 1
        jan4 = date(year, 1, 4)
        start = jan4 - timedelta(days=jan4.weekday()) + timedelta(weeks=week - 1)
        return [(start + timedelta(days=i)).isoformat() for i in range(7)]
    except Exception:
        return get_current_week_dates()


def _truncate_to_budget(text: str, budget: int) -> str:
    max_chars = budget * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated to fit budget]"


def main():
    args = parse_args()
    result = run(week_key=args.week, dry_run=args.dry_run, force=args.force)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
