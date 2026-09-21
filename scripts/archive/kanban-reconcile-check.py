#!/usr/bin/env python3
"""Kanban status-vs-run reconciliation monitor.

Catches the drift class where a task is marked done/completed while every
recorded run failed (the "masquerading as success" failure). Also reports
overlay inconsistency where a task is archived on one board but still shows
done/completed on a profile overlay.

Watchdog/--no-agent contract: prints NOTHING when clean, alert text when drift
is found. Deterministic SQL only, no LLM needed.

Schema reference: ~/.hermes/skills/devops/governance/references/profile-activity-ledger.md
Real columns: task_events.kind (not event_type); task_runs.outcome in
(completed, done, crashed, blocked, reclaimed, unblocked).
"""

import glob
import os
import sqlite3

HERMES = os.path.expanduser("~/.hermes")
SUCCESS = ("completed", "done")
# Review-flow tasks close without producing a successful run, so they are not
# true work-crash drift. Skip these prefixes from the genuine-drift list.
REVIEW_PREFIXES = ("review:", "clean up:", "investigate:")


def board_dbs():
    for path in glob.glob(os.path.join(HERMES, "**", "kanban.db"), recursive=True):
        if "backup" in path or "overlay-migration" in path:
            continue
        yield path


def collect():
    """Dedupe tasks by id across all boards (profiles mirror the same boards)."""
    seen = {}
    for db in board_dbs():
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            tables = {r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"tasks", "task_runs"} <= tables:
                continue
            rows = c.execute(
                "SELECT id, title, status FROM tasks "
                "WHERE status IN ('done','completed','archived')").fetchall()
            for tid, title, status in rows:
                nruns = c.execute(
                    "SELECT count(*) FROM task_runs WHERE task_id=?", (tid,)
                ).fetchone()[0]
                nok = c.execute(
                    "SELECT count(*) FROM task_runs WHERE task_id=? "
                    "AND outcome IN ('completed','done')", (tid,)
                ).fetchone()[0]
                last = c.execute(
                    "SELECT outcome FROM task_runs WHERE task_id=? "
                    "ORDER BY COALESCE(ended_at, started_at) DESC LIMIT 1", (tid,)
                ).fetchone()
                rec = seen.setdefault(tid, {
                    "title": title, "statuses": set(),
                    "nruns": 0, "nok": 0, "last": None})
                rec["statuses"].add(status)
                rec["nruns"] = max(rec["nruns"], nruns)
                rec["nok"] = max(rec["nok"], nok)
                if rec["last"] is None and last:
                    rec["last"] = last[0]
            c.close()
        except sqlite3.Error:
            continue
    return seen


def main():
    seen = collect()

    drift = {tid: r for tid, r in seen.items()
             if r["nruns"] >= 1 and r["nok"] == 0 and "archived" not in r["statuses"]}
    genuine = {tid: r for tid, r in drift.items()
               if not (r["title"] or "").lower().startswith(REVIEW_PREFIXES)}
    inconsistent = {tid: r for tid, r in seen.items()
                    if "archived" in r["statuses"] and (r["statuses"] - {"archived"})}

    if not genuine and not inconsistent:
        return  # silent = healthy

    lines = []
    if genuine:
        lines.append(f"⚠️ Kanban drift: {len(genuine)} task(s) marked done/completed "
                     f"but every run failed (no successful run):")
        for tid, r in sorted(genuine.items(), key=lambda x: x[1]["last"] or ""):
            lines.append(f"  • {tid} [last run: {r['last']}, {r['nruns']} runs] "
                         f"{(r['title'] or '')[:72]}")
    if inconsistent:
        lines.append(f"\nℹ️ Overlay status drift: {len(inconsistent)} task(s) archived on the "
                     f"base board but still done/completed on a profile overlay (cosmetic, "
                     f"overlays not synced).")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
