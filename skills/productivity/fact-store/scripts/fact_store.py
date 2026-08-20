#!/usr/bin/env python3
"""
Long-term memory — SQLite-backed fact store for Hermes Agent.

Provides: add, search, list, delete, update, cleanup, stats
with metadata tags, categories, and automatic last_used tracking.

DB path: ~/.hermes/facts.db (override with FACT_STORE_DB env var)

Usage:
    python3 fact_store.py add "fact text" --tags tag1 tag2 --category cat
    python3 fact_store.py search "query" [--category cat] [--limit N]
    python3 fact_store.py list [--category cat] [--limit N] [--order last_used]
    python3 fact_store.py delete <ID>
    python3 fact_store.py update <ID> --fact "new text" --tags t1 t2 --category c
    python3 fact_store.py cleanup --days 730
    python3 fact_store.py stats [--json]
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

DB_PATH = os.environ.get("FACT_STORE_DB", os.path.expanduser("~/.hermes/facts.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact TEXT NOT NULL,
    meta_tags TEXT NOT NULL DEFAULT '[]',
    category TEXT DEFAULT NULL,
    date_created TEXT NOT NULL,
    last_used TEXT NOT NULL,
    use_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_facts_meta_tags ON facts(meta_tags);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_facts_last_used ON facts(last_used);
CREATE INDEX IF NOT EXISTS idx_facts_date_created ON facts(date_created);
"""


def get_db():
    """Get database connection with schema initialized."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def add_fact(fact, tags=None, category=None):
    """Add a new fact to the database."""
    if tags is None:
        tags = []
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO facts (fact, meta_tags, category, date_created, last_used) VALUES (?, ?, ?, ?, ?)",
            (fact, json.dumps(tags), category, now, now)
        )
        conn.commit()
        return {"id": cur.lastrowid, "fact": fact, "tags": tags, "category": category, "created": now}
    finally:
        conn.close()


def search_facts(query, limit=20, category=None):
    """Search facts by keyword match in fact text, tags, or category."""
    conn = get_db()
    try:
        # Mark results as used
        now = datetime.now(timezone.utc).isoformat()
        like_query = f"%{query}%"
        params = [like_query, like_query, like_query]
        where_cat = ""
        if category:
            where_cat = " AND category = ?"
            params.append(category)
        
        rows = conn.execute(
            f"""SELECT * FROM facts 
            WHERE (fact LIKE ? OR meta_tags LIKE ? OR category LIKE ?){where_cat}
            ORDER BY last_used DESC LIMIT ?""",
            params + [limit]
        ).fetchall()
        
        result_ids = [r["id"] for r in rows]
        if result_ids:
            conn.execute(
                f"UPDATE facts SET last_used = ?, use_count = use_count + 1 WHERE id IN ({','.join('?' * len(result_ids))})",
                [now] + result_ids
            )
            conn.commit()
        
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_facts(category=None, limit=50, offset=0, order_by="last_used"):
    """List all facts, optionally filtered by category."""
    conn = get_db()
    try:
        where = ""
        params = []
        if category:
            where = "WHERE category = ?"
            params.append(category)
        
        valid_orders = {"last_used", "date_created", "use_count", "id"}
        if order_by not in valid_orders:
            order_by = "last_used"
        
        rows = conn.execute(
            f"SELECT * FROM facts {where} ORDER BY {order_by} DESC LIMIT ? OFFSET ?",
            params + [limit, offset]
        ).fetchall()
        
        total = conn.execute(f"SELECT COUNT(*) as cnt FROM facts {where}", params).fetchone()["cnt"]
        
        return {"total": total, "facts": [dict(r) for r in rows]}
    finally:
        conn.close()


def delete_fact(fact_id):
    """Delete a fact by ID."""
    conn = get_db()
    try:
        cur = conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def update_fact(fact_id, fact=None, tags=None, category=None):
    """Update a fact's text, tags, or category."""
    conn = get_db()
    try:
        sets = []
        params = []
        if fact is not None:
            sets.append("fact = ?")
            params.append(fact)
        if tags is not None:
            sets.append("meta_tags = ?")
            params.append(json.dumps(tags))
        if category is not None:
            sets.append("category = ?")
            params.append(category)
        
        if not sets:
            return None
        
        params.append(fact_id)
        cur = conn.execute(f"UPDATE facts SET {', '.join(sets)} WHERE id = ?", params)
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def cleanup_stale(days=365):
    """Remove facts not used for more than `days` days."""
    conn = get_db()
    try:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cur = conn.execute("DELETE FROM facts WHERE last_used < ?", (cutoff,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def format_output(data, json_output=False):
    """Format output for display."""
    if json_output:
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    if isinstance(data, list):
        lines = []
        for f in data:
            tags = json.loads(f.get("meta_tags", "[]"))
            tag_str = ", ".join(tags) if tags else ""
            cat = f.get("category") or ""
            lines.append(f"[{f['id']}] {f['fact']}")
            if tag_str:
                lines.append(f"     tags: {tag_str}")
            if cat:
                lines.append(f"     cat: {cat}")
            lines.append(f"     created: {f['date_created'][:10]} | last_used: {f['last_used'][:10]} | uses: {f['use_count']}")
        return "\n".join(lines)
    
    if isinstance(data, dict):
        if "total" in data:
            lines = [f"Total: {data['total']} facts"]
            lines.append("")
            lines.append(format_output(data["facts"]))
            return "\n".join(lines)
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    return str(data)


def main():
    parser = argparse.ArgumentParser(description="Long-term memory fact store")
    sub = parser.add_subparsers(dest="command", help="Command")
    
    # add
    p_add = sub.add_parser("add", help="Add a fact")
    p_add.add_argument("fact", help="Fact text")
    p_add.add_argument("--tags", nargs="*", default=[], help="Metadata tags")
    p_add.add_argument("--category", "-c", help="Category")
    
    # search
    p_search = sub.add_parser("search", help="Search facts")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--category", "-c", help="Filter by category")
    p_search.add_argument("--limit", "-n", type=int, default=20, help="Max results")
    
    # list
    p_list = sub.add_parser("list", help="List facts")
    p_list.add_argument("--category", "-c", help="Filter by category")
    p_list.add_argument("--limit", "-n", type=int, default=50, help="Max results")
    p_list.add_argument("--offset", type=int, default=0, help="Offset")
    p_list.add_argument("--order", default="last_used", choices=["last_used", "date_created", "use_count", "id"])
    
    # delete
    p_delete = sub.add_parser("delete", help="Delete a fact by ID")
    p_delete.add_argument("id", type=int, help="Fact ID to delete")
    
    # update
    p_update = sub.add_parser("update", help="Update a fact")
    p_update.add_argument("id", type=int, help="Fact ID")
    p_update.add_argument("--fact", "-f", help="New fact text")
    p_update.add_argument("--tags", nargs="*", help="New tags (replaces all)")
    p_update.add_argument("--category", "-c", help="New category")
    
    # cleanup
    p_cleanup = sub.add_parser("cleanup", help="Remove stale facts")
    p_cleanup.add_argument("--days", "-d", type=int, default=365, help="Remove facts not used for N days")
    
    # stats
    p_stats = sub.add_parser("stats", help="Show database statistics")
    
    # json output flag for all commands
    for p in [p_add, p_search, p_list, p_delete, p_update, p_cleanup, p_stats]:
        p.add_argument("--json", action="store_true", help="JSON output")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "add":
            result = add_fact(args.fact, args.tags, args.category)
            print(format_output(result, args.json))
        
        elif args.command == "search":
            results = search_facts(args.query, limit=args.limit, category=args.category)
            print(format_output(results, args.json))
        
        elif args.command == "list":
            data = list_facts(category=args.category, limit=args.limit, offset=args.offset, order_by=args.order)
            print(format_output(data, args.json))
        
        elif args.command == "delete":
            ok = delete_fact(args.id)
            print(json.dumps({"deleted": ok}))
        
        elif args.command == "update":
            ok = update_fact(args.id, fact=args.fact, tags=args.tags, category=args.category)
            print(json.dumps({"updated": ok}))
        
        elif args.command == "cleanup":
            count = cleanup_stale(args.days)
            print(json.dumps({"removed": count, "older_than_days": args.days}))
        
        elif args.command == "stats":
            conn = get_db()
            try:
                total = conn.execute("SELECT COUNT(*) as cnt FROM facts").fetchone()["cnt"]
                categories = conn.execute("SELECT category, COUNT(*) as cnt FROM facts GROUP BY category").fetchall()
                oldest = conn.execute("SELECT MIN(date_created) as d FROM facts").fetchone()["d"]
                newest = conn.execute("SELECT MAX(date_created) as d FROM facts").fetchone()["d"]
                most_used = conn.execute("SELECT fact, use_count FROM facts ORDER BY use_count DESC LIMIT 5").fetchall()
                stats = {
                    "total_facts": total,
                    "categories": {r["category"] or "(none)": r["cnt"] for r in categories},
                    "oldest": oldest,
                    "newest": newest,
                    "most_used": [dict(r) for r in most_used]
                }
                print(format_output(stats, args.json))
            finally:
                conn.close()
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())