"""MCP stdio server exposing local Teams chat context."""

from __future__ import annotations

from typing import Any

from plugins.teams_context.store import TeamsContextStore


def _format_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for row in rows:
        formatted.append(
            {
                "source_id": row.get("source_id") or row.get("chat_id"),
                "source_type": row.get("source_type") or "channel",
                "source_label": row.get("source_label"),
                "chat_id": row.get("chat_id") or row.get("source_id"),
                "message_id": row.get("message_id") or row.get("item_id"),
                "item_id": row.get("item_id") or row.get("message_id"),
                "sender": row.get("sender_name"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
                "text": row.get("text"),
                "web_url": row.get("web_url"),
                "meeting_id": row.get("meeting_id"),
                "chunk_index": row.get("chunk_index"),
                "metadata": row.get("metadata") or {},
            }
        )
    return formatted


try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - exercised only without mcp extra
    FastMCP = None  # type: ignore[assignment]


class _MissingMCP:
    def tool(self):
        def decorator(fn):
            return fn

        return decorator

    def run(self) -> None:
        raise SystemExit(
            "The Teams context MCP server requires the Hermes mcp extra. "
            "Run: uv sync --extra mcp"
        )


mcp = FastMCP("teams_context") if FastMCP is not None else _MissingMCP()


@mcp.tool()
def search_teams_context(
    query: str,
    chat_id: str | None = None,
    source_id: str | None = None,
    source_type: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Search local TeamContext channel messages and meeting KB chunks."""
    rows = TeamsContextStore().unified_search(
        query,
        source_id=source_id or chat_id,
        source_type=source_type,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )
    return {"results": _format_rows(rows)}


@mcp.tool()
def get_teams_thread(
    chat_id: str,
    message_id: str,
    before: int = 10,
    after: int = 10,
) -> dict[str, Any]:
    """Return messages around a captured Teams message."""
    rows = TeamsContextStore().thread(chat_id, message_id, before=before, after=after)
    return {"messages": _format_rows(rows)}


@mcp.tool()
def get_meeting_context(meeting_id: str, limit: int = 50) -> dict[str, Any]:
    """Return captured Teams chat messages associated with a meeting id."""
    rows = TeamsContextStore().meeting_context(meeting_id, limit=limit)
    return {"messages": _format_rows(rows)}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
