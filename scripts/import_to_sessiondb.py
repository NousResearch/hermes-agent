#!/usr/bin/env python3
"""
Import external content into a Hermes-compatible SessionDB for use with session_search.

Supported input formats (auto-detected):
  1. JSONL    — Claude Code conversation export (*.jsonl)
  2. Text     — Plain text file, paragraphs become messages
  3. Markdown — .md files, ## headings become sessions, paragraphs become messages
  4. Dir      — Directory of files, each file becomes a session (paragraphs → messages)

Output: SQLite DB matching Hermes state.db schema with FTS5 indexes.

Usage:
  python3 import_to_sessiondb.py input.jsonl -o sessions.db
  python3 import_to_sessiondb.py article.md -o sessions.db
  python3 import_to_sessiondb.py notes.txt -o sessions.db -s wiki:notes
  python3 import_to_sessiondb.py docs/ -o sessions.db -s my-docs

Requirements: Python 3.10+, stdlib only (sqlite3).
"""
import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, source TEXT, model TEXT DEFAULT 'import',
    started_at REAL, message_count INTEGER DEFAULT 0, title TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
    content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT,
    timestamp REAL, active INTEGER DEFAULT 1, compacted INTEGER DEFAULT 0,
    observed INTEGER DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content, tokenize='porter unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(content, tokenize='trigram');
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, COALESCE(new.content,''));
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (new.id, COALESCE(new.content,''));
END;
"""


def detect_format(path: str) -> str:
    """Auto-detect input format."""
    p = Path(path)
    if p.is_dir():
        return "dir"
    if p.suffix == ".jsonl":
        return "jsonl"
    if p.suffix in (".md", ".markdown"):
        return "markdown"
    return "text"


def import_jsonl(path: str, conn: sqlite3.Connection, source_tag: str,
                 next_id: int, now: float) -> tuple:
    """Import Claude Code JSONL export."""
    session_count = 0
    msg_count = 0

    for jf in sorted(Path(path).glob("*.jsonl") if Path(path).is_dir() else [Path(path)]):
        sid = jf.stem
        try:
            with open(jf) as f:
                lines = [json.loads(line) for line in f if line.strip()]
        except Exception:
            continue
        if not lines:
            continue

        msgs = []
        for msg in lines:
            claude_type = msg.get("type", "unknown")
            role = {"user": "user", "assistant": "assistant"}.get(claude_type)
            if role is None:
                continue
            content = ""
            nested = msg.get("message", {})
            if isinstance(nested, dict):
                raw = nested.get("content", "")
                if isinstance(raw, str):
                    content = raw
                elif isinstance(raw, list):
                    parts = []
                    for item in raw:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
                    content = "\n".join(parts)
            if content.strip():
                msgs.append((role, content))

        if not msgs:
            continue

        title = None
        for role, content in msgs:
            if role == "user" and content.strip():
                title = content.strip()[:200]
                break

        conn.execute(
            "INSERT INTO sessions (id, source, model, started_at, message_count, title) "
            "VALUES (?,?,?,?,?,?)",
            (sid, source_tag, "claude-sonnet-4", now, len(msgs), title))

        for i, (role, content) in enumerate(msgs):
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, timestamp, active, compacted) "
                "VALUES (?,?,?,?,?,1,0)",
                (next_id + i, sid, role, content, now + i * 0.001))

        next_id += len(msgs)
        session_count += 1
        msg_count += len(msgs)

    return session_count, msg_count, next_id


def import_text(path: str, conn: sqlite3.Connection, source_tag: str,
                next_id: int, now: float) -> tuple:
    """Import plain text file — paragraphs become alternating user/assistant messages.

    Empty lines separate messages. First paragraph is 'user', then alternates.
    """
    with open(path) as f:
        text = f.read()

    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        return 0, 0, next_id

    sid = f"text-{Path(path).stem}"
    title = paragraphs[0][:200] if paragraphs else Path(path).stem

    conn.execute(
        "INSERT INTO sessions (id, source, model, started_at, message_count, title) "
        "VALUES (?,?,?,?,?,?)",
        (sid, source_tag, "import", now, len(paragraphs), title))

    roles = ["user", "assistant"]
    for i, para in enumerate(paragraphs):
        role = roles[i % 2]
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp, active, compacted) "
            "VALUES (?,?,?,?,?,1,0)",
            (next_id + i, sid, role, para, now + i * 0.001))

    return 1, len(paragraphs), next_id + len(paragraphs)


def import_markdown(path: str, conn: sqlite3.Connection, source_tag: str,
                    next_id: int, now: float) -> tuple:
    """Import Markdown — ## headings become sessions, paragraphs become messages."""
    with open(path) as f:
        text = f.read()

    # Split by ## headings
    sections = re.split(r'\n(?=## )', text)
    session_count = 0
    msg_count = 0
    base_name = Path(path).stem

    for si, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        # Extract heading and body
        heading_match = re.match(r'^## (.+)', section)
        heading = heading_match.group(1).strip() if heading_match else f"Section {si+1}"
        body = section[heading_match.end():].strip() if heading_match else section

        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', body) if p.strip()]
        if not paragraphs:
            continue

        sid = f"{base_name}-{si+1:03d}"
        conn.execute(
            "INSERT INTO sessions (id, source, model, started_at, message_count, title) "
            "VALUES (?,?,?,?,?,?)",
            (sid, source_tag, "import", now, len(paragraphs), heading))

        roles = ["user", "assistant"]
        for i, para in enumerate(paragraphs):
            role = roles[i % 2]
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, timestamp, active, compacted) "
                "VALUES (?,?,?,?,?,1,0)",
                (next_id, sid, role, para, now + i * 0.001))
            next_id += 1

        session_count += 1
        msg_count += len(paragraphs)

    return session_count, msg_count, next_id


def import_dir(path: str, conn: sqlite3.Connection, source_tag: str,
               next_id: int, now: float) -> tuple:
    """Import directory — each file becomes a session (paragraphs → messages)."""
    session_count = 0
    msg_count = 0
    dir_path = Path(path)

    for fpath in sorted(dir_path.iterdir()):
        if not fpath.is_file():
            continue
        if fpath.suffix not in ('.txt', '.md', '.rst', ''):
            continue

        try:
            with open(fpath) as f:
                text = f.read()
        except Exception:
            continue

        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        if not paragraphs:
            continue

        sid = f"{dir_path.name}-{fpath.stem}"
        title = paragraphs[0][:200]

        conn.execute(
            "INSERT INTO sessions (id, source, model, started_at, message_count, title) "
            "VALUES (?,?,?,?,?,?)",
            (sid, source_tag, "import", now, len(paragraphs), title))

        roles = ["user", "assistant"]
        for i, para in enumerate(paragraphs):
            role = roles[i % 2]
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, timestamp, active, compacted) "
                "VALUES (?,?,?,?,?,1,0)",
                (next_id, sid, role, para, now + i * 0.001))
            next_id += 1

        session_count += 1
        msg_count += len(paragraphs)

    return session_count, msg_count, next_id


def main():
    parser = argparse.ArgumentParser(
        description="Import external content into Hermes-compatible SessionDB")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", default="session_import.db",
                        help="Output SQLite DB path (default: session_import.db)")
    parser.add_argument("-s", "--source", default="import",
                        help="Source tag for sessions (default: 'import')")
    parser.add_argument("-f", "--format", choices=["jsonl", "text", "markdown", "dir"],
                        help="Input format (auto-detected if omitted)")
    args = parser.parse_args()

    fmt = args.format or detect_format(args.input)
    print(f"Format: {fmt}")
    print(f"Source: {args.source}")
    print(f"Output: {args.output}")

    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()

    conn = sqlite3.connect(str(output_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.commit()

    next_id = 1
    now = time.time()

    importers = {
        "jsonl": import_jsonl,
        "text": import_text,
        "markdown": import_markdown,
        "dir": import_dir,
    }

    importer = importers.get(fmt)
    if not importer:
        print(f"Unknown format: {fmt}", file=sys.stderr)
        sys.exit(1)

    sessions, msgs, _ = importer(args.input, conn, args.source, next_id, now)
    conn.commit()

    # Optimize FTS
    conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('optimize')")
    conn.execute("INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('optimize')")
    conn.commit()

    size_kb = output_path.stat().st_size / 1024
    print(f"\nImported: {sessions} sessions, {msgs} messages → {output_path} ({size_kb:.0f} KB)")

    # Verify
    cur = conn.cursor()
    cur.execute("SELECT role, COUNT(*) FROM messages GROUP BY role")
    for r in cur.fetchall():
        print(f"  {r[0]}: {r[1]}")
    conn.close()


if __name__ == "__main__":
    main()
