"""Hermes Docs MCP Server — Let Hermes search its own documentation.

An MCP (Model Context Protocol) server that indexes the Hermes Agent
documentation and provides tools to search, navigate, and read docs
directly from within any MCP-compatible client — including Hermes itself.

MIT License — see LICENSE
"""

from __future__ import annotations

import json
import os
import sqlite3
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from mcp.server import MCPServer, stdio_server
    from mcp.types import CallToolRequestResult, JSONRPCError, TextContent
except ImportError:
    msg = "MCP SDK not installed. Run: pip install mcp"
    raise ImportError(msg) from None

# ── Config ──────────────────────────────────────────────────────────────

@dataclass
class DocPage:
    """A single documentation page."""
    slug: str
    title: str
    section: str
    content: str
    word_count: int = 0


class DocsIndex:
    """SQLite-backed documentation index with FTS5 search."""

    DB_FILENAME = "hermes_docs_index.sqlite"

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._open()

    def _open(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='docs'"
        )
        if cur.fetchone():
            return
        self._conn.executescript(
            """
            CREATE VIRTUAL TABLE docs USING fts5(
                slug UNINDEXED,
                title,
                section UNINDEXED,
                content,
                tokenize='porter unicode61'
            );
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        self._conn.commit()

    def rebuild(self, pages: list[DocPage]) -> int:
        """Drop and re-index all pages. Returns count."""
        self._conn.execute("DELETE FROM docs")
        for p in pages:
            p.word_count = len(p.content.split())
            self._conn.execute(
                "INSERT INTO docs (slug, title, section, content) VALUES (?, ?, ?, ?)",
                (p.slug, p.title, p.section, p.content),
            )
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('page_count', ?)",
            (str(len(pages)),),
        )
        self._conn.commit()
        return len(pages)

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """FTS5 search returning {slug, title, section, snippet}."""
        try:
            rows = self._conn.execute(
                """
                SELECT slug, title, section,
                       snippet(docs, 3, '<<', '>>', '...', 32) AS snippet_text,
                       rank
                FROM docs
                WHERE docs MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # Treat unsanitary FTS5 syntax as a plain phrase search
            phrase = " ".join(query.split())
            rows = self._conn.execute(
                """
                SELECT slug, title, section,
                       snippet(docs, 3, '<<', '>>', '...', 32) AS snippet_text,
                       rank
                FROM docs
                WHERE docs MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (f'"{phrase}"', limit),
            ).fetchall()

        return [
            {
                "slug": r["slug"],
                "title": r["title"],
                "section": r["section"],
                "snippet": r["snippet_text"],
            }
            for r in rows
        ]

    def get_page(self, slug: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT slug, title, section, content FROM docs WHERE slug = ?",
            (slug,),
        ).fetchone()
        if row is None:
            return None
        return {
            "slug": row["slug"],
            "title": row["title"],
            "section": row["section"],
            "content": row["content"],
        }

    def list_sections(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT section, COUNT(*) AS page_count FROM docs GROUP BY section ORDER BY section"
        ).fetchall()
        return [{"section": r["section"], "page_count": r["page_count"]} for r in rows]

    def list_pages_in_section(self, section: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT slug, title FROM docs WHERE section = ? ORDER BY title",
            (section,),
        ).fetchall()
        return [{"slug": r["slug"], "title": r["title"]} for r in rows]

    def stats(self) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT COUNT(*) AS total_pages, SUM(LENGTH(content)) AS total_chars FROM docs"
        ).fetchone()
        meta = self._conn.execute("SELECT key, value FROM meta").fetchall()
        return {
            "total_pages": row["total_pages"],
            "total_chars": row["total_chars"] or 0,
            "meta": {r["key"]: r["value"] for r in meta},
        }

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    def __del__(self) -> None:
        if self._conn:
            self.close()


# ── MCP Server ──────────────────────────────────────────────────────────

def _get_index_path() -> Path:
    """Resolve the index database path from env or default."""
    env_path = os.environ.get("HERMES_DOCS_INDEX_PATH")
    if env_path:
        return Path(env_path)
    # Default: ~/.hermes/hermes_docs_index.sqlite so rebuild and server
    # use the same database regardless of working directory.
    return Path.home() / ".hermes" / DocsIndex.DB_FILENAME


def _build_tools(index: DocsIndex) -> list:
    """Return the MCP tool definitions."""

    def search_docs(query: str, limit: int = 10) -> str:
        """Search the Hermes documentation for pages matching the query.

        Uses full-text search with stemming support. Returns up to `limit` results
        with titles, sections, and relevant snippets.
        """
        results = index.search(query, limit=limit)
        if not results:
            return json.dumps({"results": [], "query": query, "total": 0})
        return json.dumps(
            {
                "results": results,
                "query": query,
                "total": len(results),
            },
            indent=2,
        )

    def read_page(slug: str) -> str:
        """Read the full content of a documentation page by its slug.

        The slug is the path component of the doc URL. Examples:
        'user-guide/configuration', 'getting-started/installation',
        'features/mcp'
        """
        page = index.get_page(slug)
        if page is None:
            return json.dumps({"error": f"Page '{slug}' not found"})
        return json.dumps(
            {
                "slug": page["slug"],
                "title": page["title"],
                "section": page["section"],
                "content": page["content"],
            },
            indent=2,
        )

    def list_sections() -> str:
        """List all documentation sections (categories) with their page counts."""
        sections = index.list_sections()
        return json.dumps({"sections": sections}, indent=2)

    def list_pages(section: str) -> str:
        """List all documentation pages within a given section.

        Pass the section name as it appears in list_sections output,
        for example: 'Getting Started', 'User Guide', 'Features', 'Reference'.
        """
        pages = index.list_pages_in_section(section)
        return json.dumps({"section": section, "pages": pages}, indent=2)

    def doc_stats() -> str:
        """Get index statistics: total pages, characters indexed, metadata."""
        stats = index.stats()
        return json.dumps(stats, indent=2)

    return [search_docs, read_page, list_sections, list_pages, doc_stats]


def create_server(index: DocsIndex) -> MCPServer:
    """Create the MCP server instance."""
    server = MCPServer(
        name="hermes-docs",
        version="1.0.0",
        instructions=textwrap.dedent("""\
            Hermes Documentation Server — search, navigate, and read
            the complete Hermes Agent documentation.

            Tools:
            - search_docs(query, limit=10) — Full-text search across all docs
            - read_page(slug) — Read a full documentation page by slug
            - list_sections() — List all doc sections with page counts
            - list_pages(section) — List pages within a section
            - doc_stats() — Index statistics

            Use search_docs first to find relevant pages, then read_page
            to get the full content. Use list_sections to browse.
        """),
    )

    for tool_fn in _build_tools(index):
        server.add_tool(tool_fn)

    return server


# ── CLI Entry Points ────────────────────────────────────────────────────

def run_server() -> None:
    """Run the MCP server over stdio (the standard MCP transport)."""
    index = DocsIndex(_get_index_path())
    server = create_server(index)
    stdio_server(server)


def rebuild_index() -> None:
    """CLI entry point: fetch docs from the Hermes docs site and rebuild the index."""
    import sys

    # Try to use the indexer if available
    try:
        from scripts.fetch_hermes_docs import fetch_all_docs
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        try:
            from scripts.fetch_hermes_docs import fetch_all_docs  # type: ignore[import-not-found]
        except ImportError:
            print(
                "Index builder not available. "
                "Run: pip install -e .[index]  or  python -m scripts.fetch_hermes_docs"
            )
            sys.exit(1)

    index = DocsIndex(_get_index_path())
    pages = fetch_all_docs()
    count = index.rebuild(pages)
    print(f"Index rebuilt: {count} pages indexed at {index.db_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "rebuild":
        rebuild_index()
    else:
        run_server()
