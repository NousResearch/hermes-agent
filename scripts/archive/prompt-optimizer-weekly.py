#!/usr/bin/env python3
"""Weekly prompt-optimizer digest — run via Hermes cron."""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

HERMES_HOME = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
METRICS_DB = HERMES_HOME / "plugins" / "prompt-optimizer" / "metrics.db"

def main():
    if not METRICS_DB.exists():
        print("No metrics database yet — nothing to report.")
        return

    conn = sqlite3.connect(str(METRICS_DB))
    try:
        week_start = time.time() - 7 * 86400
        rows = conn.execute(
            "SELECT day, rewrites, avg_quality_before, avg_quality_after, avg_token_delta_pct FROM daily_stats WHERE day >= date(?, 'unixepoch') ORDER BY day",
            (week_start,),
        ).fetchall()

        total = conn.execute("SELECT COUNT(*) FROM rewrites").fetchone()[0]
        avg_delta = conn.execute(
            "SELECT AVG(token_delta_pct) FROM rewrites WHERE approved = 1"
        ).fetchone()[0] or 0.0
        avg_qb = conn.execute(
            "SELECT AVG(quality_before) FROM rewrites"
        ).fetchone()[0] or 0.0
        avg_qa = conn.execute(
            "SELECT AVG(quality_after) FROM rewrites WHERE approved = 1"
        ).fetchone()[0] or 0.0
    finally:
        conn.close()

    lines = [
        "📈 **Prompt Optimizer Weekly Digest**",
        "",
        f"Lifetime rewrites: **{total}**",
        f"Avg quality lift: **{avg_qb:.1f} → {avg_qa:.1f}**",
        f"Avg token delta: **{avg_delta:+.1f}%**",
        "",
    ]
    if rows:
        lines.append("Daily breakdown:")
        for day, rw, qb, qa, td in rows:
            qb_f = f"{qb:.1f}" if qb is not None else "—"
            qa_f = f"{qa:.1f}" if qa is not None else "—"
            td_f = f"{td:+.1f}%" if td is not None else "—"
            lines.append(f"  • {day}: {rw} rewrites | {qb_f}→{qa_f} | {td_f}")
    else:
        lines.append("No rewrites recorded this week.")

    lines.append("")
    lines.append("Tip: Use /prompt-stats --best to see your top improvements.")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
