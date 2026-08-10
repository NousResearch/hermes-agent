#!/usr/bin/env python3
"""Search the local Nous Discord archive index.

Usage:
  python3 search.py "query" [--author TEKNIUM] [--channel hermes-agent]
      [--thread "title"] [--since 2026-08-01] [--limit 10] [--raw]

FTS5 syntax applies to the query (AND/OR/phrases). If FTS syntax fails,
falls back to a plain substring search.
"""
import argparse
import os
import sqlite3
import sys

DB_PATH = os.path.expanduser("~/.hermes/nous-archive/nous_archive.db")


def run(query, author=None, channel=None, thread=None, since=None, limit=10, raw=False):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # FTS5 requires escaping double quotes in user terms
    safe_q = query.replace('"', '""')
    sql = (
        "SELECT m.ts, m.author, m.channel, COALESCE(m.thread,''), m.content "
        "FROM messages_fts f JOIN messages m ON m.id = f.rowid "
        "WHERE messages_fts MATCH ? "
    )
    params = [safe_q]
    if author:
        sql += "AND m.author = ? "
        params.append(author)
    if channel:
        sql += "AND m.channel = ? "
        params.append(channel)
    if thread:
        sql += "AND m.thread LIKE ? "
        params.append(f"%{thread}%")
    if since:
        sql += "AND m.ts >= ? "
        params.append(since)
    sql += "ORDER BY m.ts DESC LIMIT ?"
    params.append(limit)

    try:
        rows = cur.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        # FTS syntax error -> fallback substring scan
        sql = (
            "SELECT m.ts, m.author, m.channel, COALESCE(m.thread,''), m.content "
            "FROM messages m WHERE m.content LIKE ? "
        )
        params = [f"%{query}%"]
        if author:
            sql += "AND m.author = ? "
            params.append(author)
        if channel:
            sql += "AND m.channel = ? "
            params.append(channel)
        if thread:
            sql += "AND m.thread LIKE ? "
            params.append(f"%{thread}%")
        if since:
            sql += "AND m.ts >= ? "
            params.append(since)
        sql += "ORDER BY m.ts DESC LIMIT ?"
        params.append(limit)
        rows = cur.execute(sql, params).fetchall()

    for ts, author_, channel_, thread_, content in rows:
        loc = f"#{channel_}"
        if thread_:
            loc += f" / {thread_}"
        if raw:
            print(f"[{ts}] {author_} ({loc})\n{content}\n---")
        else:
            c = " ".join(content.split())
            print(f"[{ts}] {author_} in {loc}: {c[:280]}")
    if not rows:
        print("(no results)")
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--author")
    ap.add_argument("--channel")
    ap.add_argument("--thread")
    ap.add_argument("--since")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()
    run(a.query, a.author, a.channel, a.thread, a.since, a.limit, a.raw)
