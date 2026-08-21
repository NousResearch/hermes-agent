#!/usr/bin/env python3
"""agenda.py — Process Hermes agenda SQLite database from CLI.

Usage:
  agenda.py next [--n N] [--domain DOMAIN] [--json]
  agenda.py list [--status STATUS] [--domain DOMAIN] [--limit N] [--json]
  agenda.py add "<title>" [--detail "<detail>"] [--domain DOMAIN] [--kind KIND] [--priority N] [--cooldown N] [--json]
  agenda.py done <id> [--outcome "<outcome>"] [--json]
  agenda.py spark "<idea>" [--observation "<obs>"] [--domain DOMAIN] [--score S] [--confidence C] [--json]
  agenda.py sparks [--status STATUS] [--json]
  agenda.py status [--json]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS agenda (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT,
    kind TEXT,
    title TEXT NOT NULL,
    detail TEXT,
    priority INTEGER DEFAULT 3,
    status TEXT DEFAULT 'pending',
    cooldown_days INTEGER DEFAULT 0,
    last_done TEXT,
    times_done INTEGER DEFAULT 0,
    created TEXT NOT NULL,
    note TEXT,
    surfaced INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    agenda_id INTEGER NOT NULL,
    title TEXT,
    outcome TEXT,
    surfaced INTEGER DEFAULT 0,
    FOREIGN KEY (agenda_id) REFERENCES agenda(id)
);

CREATE TABLE IF NOT EXISTS sparks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation TEXT,
    idea TEXT NOT NULL,
    domain TEXT,
    score REAL,
    confidence REAL,
    decision TEXT,
    status TEXT DEFAULT 'open',
    kill_criteria TEXT,
    created TEXT NOT NULL,
    note TEXT
);

CREATE INDEX IF NOT EXISTS idx_agenda_priority_status ON agenda(priority, status);
CREATE INDEX IF NOT EXISTS idx_agenda_domain_kind ON agenda(domain, kind);
CREATE INDEX IF NOT EXISTS idx_log_agenda_id ON log(agenda_id);
CREATE INDEX IF NOT EXISTS idx_sparks_domain ON sparks(domain);
"""


def _get_default_db_path() -> Path:
    env_override = os.environ.get("HERMES_AGENDA_DB")
    if env_override and env_override.strip():
        return Path(env_override.strip()).expanduser().resolve()

    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "agenda.db"
    except Exception:
        home = Path.home()
        hermes_dir = home / ".hermes"
        if hermes_dir.exists() and hermes_dir.is_dir():
            return hermes_dir / "agenda.db"
        return home / ".hermes" / "agenda.db"


def get_conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or _get_default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    with conn:
        conn.executescript(SCHEMA_SQL)
    return conn


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def add_item(
    title: str,
    *,
    detail: str = "",
    domain: str = "general",
    kind: str = "task",
    priority: int = 3,
    cooldown_days: int = 0,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    created = now_iso()
    status = "recurring" if cooldown_days > 0 else "pending"
    conn = get_conn(db_path)
    with conn:
        cur = conn.execute(
            """
            INSERT INTO agenda (domain, kind, title, detail, priority, status, cooldown_days, created)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (domain, kind, title, detail, priority, status, cooldown_days, created),
        )
        item_id = cur.lastrowid
        cur = conn.execute("SELECT * FROM agenda WHERE id = ?", (item_id,))
        row = dict(cur.fetchone())
    conn.close()
    return row


def list_items(
    *,
    status: Optional[str] = None,
    domain: Optional[str] = None,
    limit: int = 20,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    conn = get_conn(db_path)
    query = "SELECT * FROM agenda WHERE 1=1"
    params: List[Any] = []
    if status:
        query += " AND status = ?"
        params.append(status)
    if domain:
        query += " AND domain = ?"
        params.append(domain)
    query += " ORDER BY priority ASC, created ASC LIMIT ?"
    params.append(limit)

    cur = conn.execute(query, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def next_items(
    n: int = 1,
    *,
    domain: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    conn = get_conn(db_path)
    query = "SELECT * FROM agenda WHERE status IN ('pending', 'recurring')"
    params: List[Any] = []
    if domain:
        query += " AND domain = ?"
        params.append(domain)
    query += " ORDER BY priority ASC, created ASC LIMIT ?"
    params.append(n)

    cur = conn.execute(query, params)
    rows = [dict(r) for r in cur.fetchall()]
    if rows:
        ids = [r["id"] for r in rows]
        placeholders = ",".join("?" for _ in ids)
        with conn:
            conn.execute(
                f"UPDATE agenda SET status = 'active', surfaced = 1 WHERE id IN ({placeholders})",
                ids,
            )
    conn.close()
    return rows


def done_item(
    item_id: int,
    *,
    outcome: str = "",
    db_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    ts = now_iso()
    conn = get_conn(db_path)
    cur = conn.execute("SELECT * FROM agenda WHERE id = ?", (item_id,))
    item = cur.fetchone()
    if not item:
        conn.close()
        return None

    item_dict = dict(item)
    cooldown = item_dict.get("cooldown_days") or 0
    new_status = "recurring" if cooldown > 0 else "done"

    with conn:
        conn.execute(
            """
            UPDATE agenda
            SET status = ?, times_done = COALESCE(times_done, 0) + 1, last_done = ?
            WHERE id = ?
            """,
            (new_status, ts, item_id),
        )
        conn.execute(
            """
            INSERT INTO log (ts, agenda_id, title, outcome)
            VALUES (?, ?, ?, ?)
            """,
            (ts, item_id, item_dict["title"], outcome),
        )
        cur = conn.execute("SELECT * FROM agenda WHERE id = ?", (item_id,))
        updated = dict(cur.fetchone())
    conn.close()
    return updated


def add_spark(
    idea: str,
    *,
    observation: str = "",
    domain: str = "general",
    score: Optional[float] = None,
    confidence: Optional[float] = None,
    kill_criteria: str = "",
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    created = now_iso()
    conn = get_conn(db_path)
    with conn:
        cur = conn.execute(
            """
            INSERT INTO sparks (observation, idea, domain, score, confidence, decision, status, kill_criteria, created)
            VALUES (?, ?, ?, ?, ?, 'open', 'open', ?, ?)
            """,
            (observation, idea, domain, score, confidence, kill_criteria, created),
        )
        spark_id = cur.lastrowid
        cur = conn.execute("SELECT * FROM sparks WHERE id = ?", (spark_id,))
        row = dict(cur.fetchone())
    conn.close()
    return row


def list_sparks(
    *,
    status: Optional[str] = None,
    limit: int = 20,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    conn = get_conn(db_path)
    query = "SELECT * FROM sparks WHERE 1=1"
    params: List[Any] = []
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    cur = conn.execute(query, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_status(db_path: Optional[Path] = None) -> Dict[str, Any]:
    conn = get_conn(db_path)
    cur = conn.execute("SELECT status, COUNT(*) as cnt FROM agenda GROUP BY status")
    by_status = {r["status"]: r["cnt"] for r in cur.fetchall()}

    cur = conn.execute("SELECT domain, COUNT(*) as cnt FROM agenda GROUP BY domain")
    by_domain = {r["domain"]: r["cnt"] for r in cur.fetchall()}

    cur = conn.execute("SELECT COUNT(*) as cnt FROM sparks WHERE status = 'open'")
    open_sparks = cur.fetchone()["cnt"]

    cur = conn.execute("SELECT COUNT(*) as cnt FROM log")
    logged_outcomes = cur.fetchone()["cnt"]
    conn.close()

    return {
        "status_counts": by_status,
        "domain_counts": by_domain,
        "open_sparks": open_sparks,
        "logged_outcomes": logged_outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes Agenda goal and task tracking CLI.")
    parser.add_argument("--db", type=Path, default=None, help="Custom SQLite database path.")
    parser.add_argument("--json", action="store_true", help="Format output as JSON.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # next
    p_next = subparsers.add_parser("next", help="Get next pending/recurring items and mark active.")
    p_next.add_argument("--n", type=int, default=1, help="Number of items to retrieve.")
    p_next.add_argument("--domain", type=str, default=None, help="Filter by domain.")

    # list
    p_list = subparsers.add_parser("list", help="List agenda items.")
    p_list.add_argument("--status", type=str, default=None, help="Filter by status.")
    p_list.add_argument("--domain", type=str, default=None, help="Filter by domain.")
    p_list.add_argument("--limit", type=int, default=20, help="Max items to list.")

    # add
    p_add = subparsers.add_parser("add", help="Add a new agenda item.")
    p_add.add_argument("title", type=str, help="Task title.")
    p_add.add_argument("--detail", type=str, default="", help="Detailed instructions or context.")
    p_add.add_argument("--domain", type=str, default="general", help="Domain area.")
    p_add.add_argument("--kind", type=str, default="task", help="Item kind (e.g. paper, bugfix, experiment).")
    p_add.add_argument("--priority", type=int, default=3, help="Priority (1=highest, 5=lowest).")
    p_add.add_argument("--cooldown", type=int, default=0, help="Recurring cooldown in days.")

    # done
    p_done = subparsers.add_parser("done", help="Mark an agenda item as completed.")
    p_done.add_argument("id", type=int, help="Agenda item ID.")
    p_done.add_argument("--outcome", type=str, default="", help="Outcome or execution summary.")

    # spark
    p_spark = subparsers.add_parser("spark", help="Record a new idea or raw observation.")
    p_spark.add_argument("idea", type=str, help="Idea summary.")
    p_spark.add_argument("--observation", type=str, default="", help="Context or raw observation.")
    p_spark.add_argument("--domain", type=str, default="general", help="Topic domain.")
    p_spark.add_argument("--score", type=float, default=None, help="Heuristic score (0.0-1.0).")
    p_spark.add_argument("--confidence", type=float, default=None, help="Confidence (0.0-1.0).")
    p_spark.add_argument("--kill-criteria", type=str, default="", help="What would falsify this idea?")

    # sparks
    p_sparks = subparsers.add_parser("sparks", help="List captured sparks.")
    p_sparks.add_argument("--status", type=str, default=None, help="Filter sparks by status.")
    p_sparks.add_argument("--limit", type=int, default=20, help="Limit number of sparks.")

    # status
    subparsers.add_parser("status", help="Get summary status statistics.")

    args = parser.parse_args()
    db = args.db

    if args.command == "next":
        items = next_items(args.n, domain=args.domain, db_path=db)
        if args.json:
            print(json.dumps(items, indent=2))
        else:
            if not items:
                print("No pending agenda items found.")
            for it in items:
                print(f"[#{it['id']}] (Priority {it['priority']}) [{it['domain']}/{it['kind']}] {it['title']}")
                if it.get("detail"):
                    print(f"    Detail: {it['detail']}")

    elif args.command == "list":
        items = list_items(status=args.status, domain=args.domain, limit=args.limit, db_path=db)
        if args.json:
            print(json.dumps(items, indent=2))
        else:
            if not items:
                print("No agenda items found matching criteria.")
            for it in items:
                print(f"#{it['id']}: [{it['status']}] (P{it['priority']}) {it['title']}")

    elif args.command == "add":
        item = add_item(
            args.title,
            detail=args.detail,
            domain=args.domain,
            kind=args.kind,
            priority=args.priority,
            cooldown_days=args.cooldown,
            db_path=db,
        )
        if args.json:
            print(json.dumps(item, indent=2))
        else:
            print(f"Added agenda item #{item['id']}: [{item['status']}] {item['title']} (Priority {item['priority']})")

    elif args.command == "done":
        item = done_item(args.id, outcome=args.outcome, db_path=db)
        if not item:
            print(f"Agenda item #{args.id} not found.", file=sys.stderr)
            sys.exit(1)
        if args.json:
            print(json.dumps(item, indent=2))
        else:
            print(f"Marked #{args.id} as {item['status']} (completed {item['times_done']}x).")

    elif args.command == "spark":
        sp = add_spark(
            args.idea,
            observation=args.observation,
            domain=args.domain,
            score=args.score,
            confidence=args.confidence,
            kill_criteria=args.kill_criteria,
            db_path=db,
        )
        if args.json:
            print(json.dumps(sp, indent=2))
        else:
            print(f"Logged spark #{sp['id']}: {sp['idea']}")

    elif args.command == "sparks":
        sparks = list_sparks(status=args.status, limit=args.limit, db_path=db)
        if args.json:
            print(json.dumps(sparks, indent=2))
        else:
            if not sparks:
                print("No sparks found.")
            for s in sparks:
                print(f"#{s['id']}: [{s['status']}] {s['idea']}")

    elif args.command == "status":
        st = get_status(db_path=db)
        if args.json:
            print(json.dumps(st, indent=2))
        else:
            print("Agenda Status Summary:")
            print(f"  Status breakdown: {st['status_counts']}")
            print(f"  Domain breakdown: {st['domain_counts']}")
            print(f"  Open sparks: {st['open_sparks']}")
            print(f"  Completed logs: {st['logged_outcomes']}")


if __name__ == "__main__":
    main()
