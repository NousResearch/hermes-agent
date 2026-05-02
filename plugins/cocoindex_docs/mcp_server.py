"""MCP stdio server exposing Hermes local document search."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from search import search_documents as run_document_search


mcp = FastMCP(
    "cocoindex_docs",
    instructions=(
        "Searches the user's local ~/Dokumente knowledge index. Results include "
        "semantic matches plus relative paths, topics, folder context, and excerpts."
    ),
)


@mcp.tool(
    description=(
        "Search local documents indexed from ~/Dokumente. Use topic to narrow by "
        "top-level folder, folder path, or path fragment. Returns file paths and excerpts."
    )
)
def search_documents(
    query: str,
    topic: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Search the local document index."""
    return run_document_search(query, topic=topic, limit=limit)


if __name__ == "__main__":
    mcp.run(transport="stdio")
