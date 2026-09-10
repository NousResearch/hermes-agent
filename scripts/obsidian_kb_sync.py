#!/usr/bin/env python3
"""
obsidian_kb_sync.py — Cross-platform helper for Obsidian Vault KB sync.

Handles:
  - Schema creation (FTS5 trigram tokenizer)
  - FTS writes (index markdown files)
  - mtime reconciliation (track file modification times)
  - Deleted-file cleanup (remove entries for missing files)

Usage:
  python3 scripts/obsidian_kb_sync.py <vault_path> <db_path> [--reindex]

  <vault_path>  Path to the Obsidian vault root.
  <db_path>     Path to the SQLite database file (created if nonexistent).
  --reindex     Force full reindex instead of incremental update.

Output:
  JSON summary on stdout: {"added": N, "updated": M, "removed": K, "total": T}
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone


# ── Schema ────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    module      TEXT NOT NULL,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    mtime       REAL NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title,
    content,
    tokenize='trigram',
    content=documents,
    content_rowid='rowid'
);

CREATE INDEX IF NOT EXISTS idx_documents_mtime ON documents(mtime);
CREATE INDEX IF NOT EXISTS idx_documents_module ON documents(module);
"""


# ── Helpers ───────────────────────────────────────────────────────────────

def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse YAML frontmatter from markdown text. Returns (metadata, body)."""
    stripped = text.lstrip('\ufeff')  # strip BOM
    if not stripped.startswith('---'):
        return {}, stripped
    end = stripped.find('---', 3)
    if end == -1:
        return {}, stripped
    fm_block = stripped[3:end].strip()
    body = stripped[end + 3:].strip()
    metadata: dict[str, str] = {}
    for line in fm_block.split('\n'):
        m = re.match(r'^(\w[\w\s-]*?)\s*:\s*(.+)$', line)
        if m:
            key = m.group(1).strip().lower()
            val = m.group(2).strip().strip('"\'')
            metadata[key] = val
    return metadata, body


def extract_title(metadata: dict, body: str, filename: str) -> str:
    """Extract the best title from frontmatter, first H1, or filename."""
    for key in ('title', 'alias'):
        if key in metadata and metadata[key]:
            return metadata[key]
    h1 = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
    if h1:
        return h1.group(1).strip()
    return os.path.splitext(filename)[0]


def extract_headings(body: str) -> str:
    """Extract H1/H2 headings for section-based indexing."""
    headings = re.findall(r'^(#{1,2})\s+(.+)$', body, re.MULTILINE)
    return '\n'.join(h[1].strip() for h in headings)


def get_module(file_path: str, vault_root: str) -> str:
    """Derive module name from directory relative to vault root."""
    rel = os.path.relpath(os.path.dirname(file_path), vault_root)
    if rel == '.':
        return os.path.splitext(os.path.basename(file_path))[0]
    return rel.replace(os.sep, '/')


def walk_markdown_files(vault_root: str) -> list[str]:
    """Recursively find all .md files under vault_root."""
    results: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(vault_root):
        # Skip hidden directories
        if any(part.startswith('.') for part in dirpath.split(os.sep)):
            continue
        for fn in filenames:
            if fn.endswith('.md'):
                results.append(os.path.join(dirpath, fn))
    return results


# ── Core Operations ───────────────────────────────────────────────────────

def init_db(conn: sqlite3.Connection) -> None:
    """Create schema if not exists."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def get_stored_mtimes(conn: sqlite3.Connection) -> dict[str, float]:
    """Return {doc_id: mtime} for all stored documents."""
    rows = conn.execute("SELECT doc_id, mtime FROM documents").fetchall()
    return {r[0]: r[1] for r in rows}


def index_file(
    conn: sqlite3.Connection,
    file_path: str,
    vault_root: str,
) -> str | None:
    """Index a single markdown file. Returns doc_id or None on failure."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            raw = f.read()
    except (OSError, PermissionError) as e:
        print(f"  [warn] cannot read {file_path}: {e}", file=sys.stderr)
        return None

    metadata, body = parse_frontmatter(raw)
    filename = os.path.basename(file_path)
    doc_id = os.path.relpath(file_path, vault_root).replace(os.sep, '/')
    title = extract_title(metadata, body, filename)
    module = get_module(file_path, vault_root)
    headings = extract_headings(body)

    # Build content: prefer headings + body for search relevance
    content = f"{headings}\n\n{body}" if headings else body

    mtime = os.path.getmtime(file_path)
    conn.execute(
        """INSERT OR REPLACE INTO documents
           (doc_id, module, title, content, file_path, mtime)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (doc_id, module, title, content, file_path, mtime),
    )
    return doc_id


def sync_fts(conn: sqlite3.Connection) -> None:
    """Rebuild FTS5 index from the documents table."""
    conn.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
    conn.commit()


def remove_deleted(conn: sqlite3.Connection, active_doc_ids: set[str]) -> int:
    """Remove docs for files that no longer exist. Returns count removed."""
    stored = conn.execute("SELECT doc_id FROM documents").fetchall()
    to_delete = [r[0] for r in stored if r[0] not in active_doc_ids]
    for doc_id in to_delete:
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    conn.commit()
    return len(to_delete)


# ── Main ──────────────────────────────────────────────────────────────────

def run_sync(vault_root: str, db_path: str, reindex: bool = False) -> dict:
    """Run the full sync pipeline. Returns summary dict."""
    vault_root = os.path.abspath(vault_root)
    db_path = os.path.abspath(db_path)

    if not os.path.isdir(vault_root):
        raise FileNotFoundError(f"Vault path not found: {vault_root}")

    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
    conn = sqlite3.connect(db_path)
    init_db(conn)

    # Phase 1: scan files
    markdown_files = walk_markdown_files(vault_root)

    # Phase 2: mtime reconciliation
    stored_mtimes = get_stored_mtimes(conn) if not reindex else {}
    active_doc_ids: set[str] = set()
    added = 0
    updated = 0
    skipped = 0

    for file_path in markdown_files:
        rel = os.path.relpath(file_path, vault_root).replace(os.sep, '/')
        active_doc_ids.add(rel)
        current_mtime = os.path.getmtime(file_path)

        if not reindex and rel in stored_mtimes:
            if abs(current_mtime - stored_mtimes[rel]) < 0.001:
                skipped += 1
                continue
            # mtime changed — update
            result = index_file(conn, file_path, vault_root)
            if result:
                updated += 1
        else:
            # New file
            result = index_file(conn, file_path, vault_root)
            if result:
                added += 1

    # Phase 3: remove deleted files
    removed = remove_deleted(conn, active_doc_ids)

    # Phase 4: rebuild FTS index
    sync_fts(conn)

    total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()

    return {
        "added": added,
        "updated": updated,
        "removed": removed,
        "skipped": skipped,
        "total": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index Obsidian vault markdown files into a SQLite FTS5 knowledge base."
    )
    parser.add_argument("vault_path", help="Path to the Obsidian vault root")
    parser.add_argument("db_path", help="Path to the SQLite database file")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force full reindex (ignore mtime cache)",
    )
    args = parser.parse_args()

    try:
        summary = run_sync(args.vault_path, args.db_path, reindex=args.reindex)
        print(json.dumps(summary))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()