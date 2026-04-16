#!/usr/bin/env python3
"""Hermes Memory MCP Server — exposes Hermes memory stores to any MCP client (Claude Code, Cursor, etc.) via stdio."""
# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp[cli]>=1.2.0"]
# ///

from __future__ import annotations

import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

HERMES_DIR = Path.home() / ".hermes"
MEMORIES_DIR = HERMES_DIR / "memories"
STATE_DB = HERMES_DIR / "state.db"

MEMORY_FILE = MEMORIES_DIR / "MEMORY.md"
USER_FILE = MEMORIES_DIR / "USER.md"

CHAR_LIMIT_MEMORY = 2_200
CHAR_LIMIT_USER = 1_375
SECTION_SEP = "\n§\n"

mcp = FastMCP("hermes-memory")


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except PermissionError:
        return f"[Error] Permission denied: {path}"


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def read_memory(store: str = "all") -> str:
    """Read Hermes memory content.

    Args:
        store: Which store to read — "memory", "user", or "all" (default).
    """
    parts: list[str] = []
    if store in ("all", "memory"):
        content = _read_file(MEMORY_FILE)
        parts.append(f"=== MEMORY.md ({len(content)} chars) ===\n{content}")
    if store in ("all", "user"):
        content = _read_file(USER_FILE)
        parts.append(f"=== USER.md ({len(content)} chars) ===\n{content}")
    if not parts:
        return f"[Error] Unknown store '{store}'. Use 'memory', 'user', or 'all'."
    return "\n\n".join(parts)


@mcp.tool()
def add_memory_entry(
    store: str,
    entry: str,
    old_text: str | None = None,
) -> str:
    """Add or replace an entry in a Hermes memory store.

    Args:
        store: Target store — "memory" or "user".
        entry: The new text to insert.
        old_text: If provided, find this substring and replace it with `entry`.
                  If not provided, append `entry` as a new section (§-separated).
    """
    file_map = {"memory": MEMORY_FILE, "user": USER_FILE}
    limit_map = {"memory": CHAR_LIMIT_MEMORY, "user": CHAR_LIMIT_USER}

    if store not in file_map:
        return f"[Error] Unknown store '{store}'. Use 'memory' or 'user'."

    path = file_map[store]
    limit = limit_map[store]
    current = _read_file(path)

    if old_text:
        if old_text not in current:
            return f"[Error] old_text not found in {path.name}. No changes made."
        new_content = current.replace(old_text, entry, 1)
    else:
        if current and not current.endswith("\n"):
            new_content = current + SECTION_SEP + entry
        elif current:
            new_content = current.rstrip("\n") + SECTION_SEP + entry
        else:
            new_content = entry

    if len(new_content) > limit:
        return (
            f"[Error] Would exceed char limit: {len(new_content)}/{limit}. "
            f"Current: {len(current)} chars. Entry: {len(entry)} chars. "
            "Remove old entries first or shorten the new entry."
        )

    try:
        _write_file(path, new_content)
    except PermissionError:
        return f"[Error] Permission denied writing to {path}"
    except OSError as exc:
        return f"[Error] Failed to write {path}: {exc}"

    return f"OK — {path.name} updated. Now {len(new_content)} chars ({limit - len(new_content)} remaining)."


@mcp.tool()
def memory_status() -> str:
    """Return current memory usage (char count / limit) for all stores."""
    lines: list[str] = []
    for label, path, limit in [
        ("MEMORY.md", MEMORY_FILE, CHAR_LIMIT_MEMORY),
        ("USER.md", USER_FILE, CHAR_LIMIT_USER),
    ]:
        content = _read_file(path)
        chars = len(content)
        sections = content.count("§") + 1 if content else 0
        pct = chars / limit * 100 if limit else 0
        lines.append(
            f"{label}: {chars:,}/{limit:,} chars ({pct:.1f}%) — {sections} sections"
        )
    return "\n".join(lines)


@mcp.tool()
def session_search(
    query: str,
    limit: int = 20,
    source: str | None = None,
) -> str:
    """Search past Hermes sessions using FTS5 full-text search.

    Args:
        query: FTS5 search query (supports AND, OR, NOT, phrases in quotes).
        limit: Max results to return (default 20, max 100).
        source: Optional filter by session source (e.g. "claude-code", "hermes").
    """
    if not STATE_DB.exists():
        return f"[Error] state.db not found at {STATE_DB}"

    limit = min(max(1, limit), 100)

    sql = """
        SELECT
            s.id,
            s.source,
            s.model,
            s.title,
            datetime(s.started_at, 'unixepoch', 'localtime') AS started,
            s.message_count,
            s.estimated_cost_usd,
            snippet(messages_fts, 0, '>>>', '<<<', '...', 48) AS snippet
        FROM messages_fts
        JOIN messages m ON m.id = messages_fts.rowid
        JOIN sessions s ON s.id = m.session_id
        WHERE messages_fts MATCH ?
    """
    params: list = [query]

    if source:
        sql += " AND s.source = ?"
        params.append(source)

    sql += " ORDER BY s.started_at DESC LIMIT ?"
    params.append(limit)

    try:
        conn = sqlite3.connect(
            f"file:{STATE_DB}?mode=ro",
            uri=True,
            timeout=5,
        )
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
    except sqlite3.OperationalError as exc:
        if "database is locked" in str(exc):
            return "[Error] state.db is locked by another process. Try again shortly."
        return f"[Error] SQLite error: {exc}"
    except sqlite3.DatabaseError as exc:
        return f"[Error] Database error: {exc}"

    if not rows:
        return f"No results for query: {query}"

    parts: list[str] = [f"Found {len(rows)} result(s) for '{query}':\n"]
    for r in rows:
        cost = f"${r['estimated_cost_usd']:.4f}" if r["estimated_cost_usd"] else "n/a"
        parts.append(
            f"- [{r['started']}] {r['title'] or '(untitled)'}\n"
            f"  source={r['source']} model={r['model']} msgs={r['message_count']} cost={cost}\n"
            f"  id={r['id']}\n"
            f"  > {r['snippet']}"
        )
    return "\n".join(parts)


if __name__ == "__main__":
    mcp.run(transport="stdio")
