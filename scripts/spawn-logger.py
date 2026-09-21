#!/usr/bin/env python3
"""

# P13: disabled-staging guard — exit early when cron is disabled
import os as _os, sys as _sys
if _os.environ.get("DRY_RUN") == "1":
    print(f"[DRY_RUN] {_os.path.basename(__file__)}")
    _sys.exit(0)
Sub-Agent Spawn Logger - hooks into delegate_task events to track
ad-hoc sub-agent creation patterns. Logs novel spawns to the central
activity ledger. Denji's weekly profile review reads these events
and surfaces repeated patterns as candidates for permanent profiles.

Usage: Called automatically by delegate_task tool. Can also be run
standalone: spawn-logger.py --query to list recent spawns.
"""

import json
import os
import sqlite3
import datetime as dt
from pathlib import Path
from collections import Counter

HERMES_HOME = Path(os.environ.get("HERMES_HOME", "/home/kensei/.hermes"))
LEDGER_DB = HERMES_HOME / "data" / "activity-ledger.db"
LOGBOARD = HERMES_HOME / "governance" / "logboard"

# Known specialist profiles - spawns matching these are routine, not novel
KNOWN_PROFILES = {
    "default", "gojo", "kensei-review", "octacon", "octacon-frontend",
    "remii-deep", "wesker", "wesker-ops", "ceecee", "dezzy",
    "denji", "denji-monitor", "denji-reviewer", "misa-misa",
    "mrhermagi", "light", "quan", "market-scanner",
}


def classify_goal(goal_text: str) -> str:
    """Classify a sub-agent goal into a category for pattern detection."""
    goal_lower = goal_text.lower()
    if any(w in goal_lower for w in ["performance", "perf", "n+1", "query", "slow", "latency"]):
        return "performance-profiler"
    if any(w in goal_lower for w in ["security", "audit", "vuln", "cve", "injection"]):
        return "security-auditor"
    if any(w in goal_lower for w in ["test", "qa", "quality", "verify", "regression"]):
        return "qa-tester"
    if any(w in goal_lower for w in ["deploy", "infra", "docker", "server", "ci/cd"]):
        return "infra-ops"
    if any(w in goal_lower for w in ["design", "ui", "ux", "component", "layout", "css"]):
        return "design-reviewer"
    if any(w in goal_lower for w in ["research", "scan", "market", "competitor", "paper"]):
        return "researcher"
    if any(w in goal_lower for w in ["content", "draft", "write", "post", "social"]):
        return "content-writer"
    if any(w in goal_lower for w in ["data", "schema", "migration", "database", "db"]):
        return "data-engineer"
    if any(w in goal_lower for w in ["debug", "fix", "bug", "error", "trace"]):
        return "debugger"
    if any(w in goal_lower for w in ["simplify", "clean", "refactor", "code-review"]):
        return "code-cleaner"
    if any(w in goal_lower for w in ["review", "check", "validate", "gate"]):
        return "reviewer"
    return "general-worker"


def log_spawn(profile: str, goal: str, task_id: str = ""):
    """Log a sub-agent spawn event to the activity ledger."""
    category = classify_goal(goal)
    timestamp = dt.datetime.now().isoformat()

    try:
        conn = sqlite3.connect(str(LEDGER_DB))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS delegate_spawns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                profile TEXT NOT NULL,
                goal TEXT NOT NULL,
                category TEXT NOT NULL,
                task_id TEXT DEFAULT ''
            )
        """)
        conn.execute(
            "INSERT INTO delegate_spawns (timestamp, profile, goal, category, task_id) VALUES (?, ?, ?, ?, ?)",
            (timestamp, profile, goal[:500], category, task_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        # Fails-open - logging failure shouldn't block the actual spawn
        print(f"spawn-logger: failed to log spawn: {e}", file=__import__('sys').stderr)


def detect_patterns(days: int = 14) -> list:
    """Detect repeated novel sub-agent categories over the past N days."""
    cutoff = (dt.datetime.now() - dt.timedelta(days=days)).isoformat()
    
    try:
        conn = sqlite3.connect(str(LEDGER_DB))
        cursor = conn.execute(
            "SELECT category, COUNT(*) as cnt FROM delegate_spawns WHERE timestamp > ? GROUP BY category ORDER BY cnt DESC",
            (cutoff,)
        )
        rows = cursor.fetchall()
        conn.close()
    except Exception:
        return []

    patterns = []
    for category, count in rows:
        if count >= 3:  # threshold: 3+ spawns in 14 days
            patterns.append({
                "category": category,
                "count": count,
                "recommendation": f"Consider creating permanent '{category}' profile",
            })
    return patterns


def produce_report(days: int = 14) -> str:
    """Produce a markdown report of spawn patterns for Denji's review."""
    patterns = detect_patterns(days)
    
    if not patterns:
        return ""

    now = dt.datetime.now()
    lines = [
        f"## Sub-Agent Spawn Patterns - {now.strftime('%d/%m/%Y')}",
        "",
        f"Period: past {days} days | Threshold: 3+ spawns per category",
        "",
        "| Category | Spawns | Recommendation |",
        "|----------|--------|---------------|",
    ]
    for p in patterns:
        lines.append(f"| {p['category']} | {p['count']} | {p['recommendation']} |")

    return "\n".join(lines)


def main():
    import sys
    if "--query" in sys.argv:
        patterns = detect_patterns()
        if patterns:
            for p in patterns:
                print(f"{p['category']}: {p['count']} spawns - {p['recommendation']}")
        else:
            print("No novel spawn patterns detected (threshold: 3+ in 14 days)")
        return

    # Default to --report when run with no args (cron no_agent mode)
    if "--report" in sys.argv or len(sys.argv) == 1:
        report = produce_report()
        if report:
            report_path = LOGBOARD / f"spawn-patterns-{dt.datetime.now().strftime('%Y%m%d')}.md"
            report_path.write_text(report)
            print(report)
        else:
            return  # silent — no pattern
        return

    print("Usage: spawn-logger.py [--query | --report]")
    print("  --query   Show current spawn patterns")
    print("  --report  Write pattern report to governance logboard")


if __name__ == "__main__":
    main()
