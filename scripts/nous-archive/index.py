#!/usr/bin/env python3
"""Build/refresh a local SQLite FTS5 index of the Nous Discord archive.

Source: ~/repos/nous-discord-archive/archives/
Index:  ~/.hermes/nous-archive/nous_archive.db

Incremental: files whose (mtime, size) are unchanged are skipped.
Usage: python3 index.py [--force]
"""
import argparse
import json
import os
import re
import sqlite3
import sys

ARCHIVE_ROOT = os.path.expanduser("~/repos/nous-discord-archive/archives")
DB_PATH = os.path.expanduser("~/.hermes/nous-archive/nous_archive.db")
META_TABLE = "file_meta"
MSG_TABLE = "messages"

MSG_HEADER_RE = re.compile(r"^\[([^\]]+)\] (.+) \(id=(\d+)\)$")

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {MSG_TABLE} (
    id INTEGER PRIMARY KEY,
    file_path TEXT NOT NULL,
    channel TEXT NOT NULL,
    thread TEXT,
    ts TEXT,
    author TEXT,
    msg_id TEXT,
    content TEXT
);
CREATE TABLE IF NOT EXISTS {META_TABLE} (
    path TEXT PRIMARY KEY,
    mtime_ns INTEGER,
    size INTEGER
);
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, author, channel, thread,
    content='{MSG_TABLE}', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON {MSG_TABLE} BEGIN
    INSERT INTO messages_fts(rowid, content, author, channel, thread)
    VALUES (new.id, new.content, new.author, new.channel, new.thread);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON {MSG_TABLE} BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, author, channel, thread)
    VALUES ('delete', old.id, old.content, old.author, old.channel, old.thread);
END;
"""


def channel_thread_for(path: str):
    """Map an archive file path to (channel, thread)."""
    rel = os.path.relpath(path, ARCHIVE_ROOT)
    parts = rel.split(os.sep)
    if len(parts) == 1:
        return parts[0][:-4], None  # strip .txt
    # forum thread: <dir>/<threadid>-<title>.txt
    dirname = parts[0]
    stem = parts[1][:-4]
    m = re.match(r"^\d+-(.+)$", stem)
    if m:
        return dirname, m.group(1)
    return dirname, stem


def parse_file(path: str):
    """Yield (channel, thread, ts, author, msg_id, content) tuples."""
    channel, thread = channel_thread_for(path)
    messages = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    cur_ts = cur_author = cur_msg = None
    cur_content = []
    in_msg = False

    def flush():
        nonlocal in_msg
        if in_msg and cur_msg is not None:
            text = "\n".join(cur_content).strip()
            if text:
                messages.append((channel, thread, cur_ts, cur_author, cur_msg, text))
        in_msg = False

    for line in lines:
        if line.startswith("#"):
            flush()
            continue
        m = MSG_HEADER_RE.match(line.rstrip("\n"))
        if m:
            flush()
            cur_ts, cur_author, cur_msg = m.group(1), m.group(2), m.group(3)
            cur_content = []
            in_msg = True
        elif in_msg:
            # content lines are indented 4 spaces
            cur_content.append(line[4:] if line.startswith("    ") else line.rstrip("\n"))
        else:
            flush()
    flush()
    return messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild everything")
    args = ap.parse_args()

    if not os.path.isdir(ARCHIVE_ROOT):
        print(f"FATAL: archive root missing: {ARCHIVE_ROOT}")
        sys.exit(1)

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)

    if args.force:
        con.execute(f"DELETE FROM {MSG_TABLE}")
        con.execute(f"DELETE FROM {META_TABLE}")

    cur = con.cursor()
    cur.execute(f"SELECT path, mtime_ns, size FROM {META_TABLE}")
    meta = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    changed = skipped = total_msgs = 0
    for root, _dirs, files in os.walk(ARCHIVE_ROOT):
        for fn in sorted(files):
            if not fn.endswith(".txt"):
                continue
            path = os.path.join(root, fn)
            st = os.stat(path)
            key = (st.st_mtime_ns, st.st_size)
            if not args.force and meta.get(path) == key:
                skipped += 1
                continue
            try:
                msgs = parse_file(path)
            except Exception as e:
                print(f"  WARN parse {path}: {e}")
                continue
            con.execute(f"DELETE FROM {MSG_TABLE} WHERE file_path = ?", (path,))
            con.executemany(
                f"INSERT INTO {MSG_TABLE} (file_path, channel, thread, ts, author, msg_id, content) "
                f"VALUES (?,?,?,?,?,?,?)",
                [(path, *m) for m in msgs],
            )
            cur.execute(
                f"INSERT OR REPLACE INTO {META_TABLE} (path, mtime_ns, size) VALUES (?,?,?)",
                (path, st.st_mtime_ns, st.st_size),
            )
            changed += 1
            total_msgs += len(msgs)

    con.commit()
    cur.execute(f"SELECT COUNT(*) FROM {MSG_TABLE}")
    n = cur.fetchone()[0]
    print(f"indexed: {changed} files changed, {skipped} skipped, {total_msgs} new msgs")
    print(f"total messages in index: {n}")
    con.close()


if __name__ == "__main__":
    main()
