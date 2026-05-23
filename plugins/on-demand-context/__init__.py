"""
on-demand-context — Hermes Plugin
====================================

A plugin that defers context injection to save tokens.

Instead of injecting all available context documents upfront (which wastes
tokens on every turn), this plugin:

1. On the first turn of a conversation, injects a lightweight index
   (~400 bytes) listing available context documents.
2. Provides a ``load_context()`` tool that the agent can call to fetch
   any document's full content only when needed.

Designed to be configurable — works with any directory of markdown files.

Idea credit: Minksgo (https://github.com/Minksgo)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Default config ───────────────────────────────────────────────
# These can be overridden via plugin.yaml config section.

SCAN_DIR = os.path.expanduser(
    os.environ.get(
        "HERMES_CONTEXT_DIR",
        "~/.hermes/context-docs",
    )
)

INDEX_TITLE = os.environ.get(
    "HERMES_CONTEXT_INDEX_TITLE",
    "📋 Available Knowledge",
)

FILE_PATTERN = "*.md"

# ── Cache ────────────────────────────────────────────────────────

_docs: Dict[str, dict] = {}
_loaded: bool = False


# -------------------------------------------------------------------
# Document discovery
# -------------------------------------------------------------------

def _discover_docs(root: str) -> Dict[str, dict]:
    """Scan *root* for markdown files and build an index."""
    result = {}
    base = Path(root)
    if not base.exists():
        logger.info("on-demand-context: scan dir %s not found — skipping", root)
        return result

    for doc_file in base.rglob(FILE_PATTERN):
        try:
            text = doc_file.read_text(encoding="utf-8")
            lines = text.split("\n")

            # Extract title (first heading) and first few content lines
            title = ""
            summary_lines = []
            in_frontmatter = False

            for line in lines[:50]:
                if line.startswith("---"):
                    in_frontmatter = not in_frontmatter
                    continue
                if in_frontmatter:
                    continue
                if line.startswith("# "):
                    title = line.lstrip("# ").strip()
                elif line.strip() and title:
                    summary_lines.append(line.strip())

            doc_id = doc_file.stem  # filename without extension
            summary = " ".join(summary_lines[:5])

            result[doc_id] = {
                "id": doc_id,
                "path": str(doc_file),
                "title": title or doc_id,
                "summary": summary[:200],
                "content": text,
            }
        except Exception as exc:
            logger.warning(
                "on-demand-context: failed to read %s: %s", doc_file, exc
            )

    return result


def _ensure_loaded() -> None:
    """Load documents once."""
    global _docs, _loaded
    if _loaded:
        return
    _docs = _discover_docs(SCAN_DIR)
    _loaded = True
    logger.info(
        "on-demand-context: loaded %d documents from %s",
        len(_docs), SCAN_DIR,
    )


# -------------------------------------------------------------------
# Index generation
# -------------------------------------------------------------------

def _build_index() -> str:
    """Build a compact index of available documents."""
    _ensure_loaded()

    lines = []
    for doc_id, doc in sorted(_docs.items()):
        lines.append(f"  • {doc['title']}  —  {doc['summary'][:100]}")

    if not lines:
        return ""  # No docs found — silently skip

    return (
        f"{INDEX_TITLE}:\n"
        + "\n".join(lines)
        + "\n\n"
        + "Use load_context() to fetch the full content of any document above."
    )


# -------------------------------------------------------------------
# pre_llm_call hook: inject index on first turn
# -------------------------------------------------------------------


def _on_pre_llm_call(
    session_id: str = "",
    user_message: str = "",
    conversation_history: list | None = None,
    is_first_turn: bool = False,
    model: str = "",
    platform: str = "",
    sender_id: str = "",
    **kwargs: Any,
) -> Optional[Dict[str, str]]:
    """Inject the context index on the first turn only (~400 bytes)."""
    _ensure_loaded()

    if not is_first_turn:
        return None

    index = _build_index()
    if not index:
        return None

    return {"context": index}


# -------------------------------------------------------------------
# load_context() tool
# -------------------------------------------------------------------

LOAD_CONTEXT_TOOL_SPEC = {
    "name": "load_context",
    "description": (
        "Load the full content of a context document by its ID. "
        "Use this when you need details beyond the index summary."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Document ID (filename without extension, e.g. 'team-governance')",
            }
        },
        "required": ["id"],
    },
}


def load_context(id: str) -> str:
    """Return the full content of a document."""
    _ensure_loaded()
    doc = _docs.get(id)
    if not doc:
        available = ", ".join(sorted(_docs.keys())) if _docs else "(no documents loaded)"
        return f"Document '{id}' not found. Available: {available}"
    return doc["content"]


# -------------------------------------------------------------------
# Plugin entry point
# -------------------------------------------------------------------

TOOL_REGISTRY = {
    "load_context": {
        "spec": LOAD_CONTEXT_TOOL_SPEC,
        "implementation": load_context,
    },
}


def get_tool_specs() -> list[dict]:
    """Return tool specs for Hermes Plugin system."""
    return [v["spec"] for v in TOOL_REGISTRY.values()]


def execute_tool(name: str, args: dict) -> str:
    """Execute a plugin tool."""
    tool = TOOL_REGISTRY.get(name)
    if not tool:
        return f"Unknown tool: {name}"
    try:
        return tool["implementation"](**args)
    except Exception as exc:
        logger.error("on-demand-context: tool %s failed: %s", name, exc)
        return f"Tool execution failed: {exc}"
