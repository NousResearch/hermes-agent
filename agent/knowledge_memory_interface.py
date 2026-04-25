"""Knowledge base interface for Hermes memory system.

Provides read access to the knowledge/ directory indexed by knowledge/index.json.
Used by BuiltinMemoryProvider to surface knowledge context to the agent.

The knowledge base is organized under HERMES_HOME/knowledge/ with:
  - docs/plans/     — architecture and design plans
  - docs/specs/     — technical specifications
  - docs/migration/ — migration guides
  - facts/          — learnings, incident logs, optimization history
  - policies/       — operational policies
  - AGENTS.md       — agent identity and routing rules
  - SOUL.md         — core identity and values
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_KNOWLEDGE_DIR: Optional[Path] = None


def _get_knowledge_dir() -> Path:
    """Return the absolute path to the knowledge/ directory."""
    global _KNOWLEDGE_DIR
    if _KNOWLEDGE_DIR is None:
        from hermes_constants import get_hermes_home
        _KNOWLEDGE_DIR = Path(get_hermes_home()) / "knowledge"
    return _KNOWLEDGE_DIR


# ---------------------------------------------------------------------------
# Index loading (caches in module scope)
# ---------------------------------------------------------------------------

_index_cache: Optional[Dict[str, Any]] = None


def get_index() -> Dict[str, Any]:
    """Load and cache knowledge/index.json."""
    global _index_cache
    if _index_cache is not None:
        return _index_cache

    index_path = _get_knowledge_dir() / "index.json"
    result: Dict[str, Any] = {}

    if index_path.exists():
        try:
            import json
            with open(index_path, "r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception as e:
            logger.warning("Failed to load knowledge/index.json: %s", e)

    _index_cache = result
    return result


def invalidate_index() -> None:
    """Clear the index cache so the next call reloads from disk."""
    global _index_cache
    _index_cache = None


# ---------------------------------------------------------------------------
# TOML integration (delegates to skill_commands.read_index_toml)
# ---------------------------------------------------------------------------

def get_skill_index() -> Dict[str, Any]:
    """Return the skills .index.toml contents via skill_commands.

    Returns an empty dict on any error so callers always get a usable value.
    """
    try:
        from agent.skill_commands import read_index_toml

        return read_index_toml()
    except Exception as e:
        logger.debug("read_index_toml unavailable: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Knowledge entry lookup
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```[\w]*\s*$", re.MULTILINE)


def strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing markdown fence lines."""
    return _FENCE_RE.sub("", text).strip()


def list_entries(
    category: Optional[str] = None,
    file_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return matching entries from the knowledge index.

    Args:
        category: Filter by category key in structure (e.g. 'docs', 'facts').
                  If None, returns all categories.
        file_type: Filter by 'type' field (e.g. 'plan', 'spec', 'fact').

    Returns:
        List of entry dicts with keys: path, type, description.
    """
    index = get_index()
    files: List[Dict[str, Any]] = index.get("files", [])

    results: List[Dict[str, Any]] = []

    if category:
        structure: Dict[str, Any] = index.get("structure", {})
        category_paths: List[str] = structure.get(category, [])
        allowed_paths = set(_ensure_absolute(p) for p in category_paths)
        results = [f for f in files if _ensure_absolute(f.get("path", "")) in allowed_paths]
    else:
        results = list(files)

    if file_type:
        results = [f for f in results if f.get("type") == file_type]

    return results


def _ensure_absolute(path: str) -> str:
    """Normalize a knowledge index path to always start with 'knowledge/'."""
    if not path.startswith("knowledge/"):
        return "knowledge/" + path.lstrip("/")
    return path


def resolve_knowledge_path(relative_path: str) -> Optional[Path]:
    """Resolve a relative knowledge path to an absolute filesystem path.

    Args:
        relative_path: Path relative to the knowledge/ directory
                       (e.g. 'docs/plans/plan.md' or just 'SOUL.md').

    Returns:
        Absolute Path if the file exists, else None.
    """
    knowledge_dir = _get_knowledge_dir()

    # Normalize: strip any leading 'knowledge/' since we prepend it
    clean = relative_path.removeprefix("knowledge/").lstrip("/")

    full_path = knowledge_dir / clean
    if full_path.is_file():
        return full_path

    # Try without .md extension
    if not full_path.suffix and not full_path.exists():
        md_path = full_path.with_suffix(".md")
        if md_path.is_file():
            return md_path

    return None


def read_knowledge_entry(relative_path: str) -> Optional[str]:
    """Read a knowledge entry file.

    Args:
        relative_path: Path relative to knowledge/ (e.g. 'SOUL.md').

    Returns:
        File contents as string, or None if not found.
    """
    path = resolve_knowledge_path(relative_path)
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to read knowledge entry '%s': %s", relative_path, e)
        return None


# ---------------------------------------------------------------------------
# Search / query
# ---------------------------------------------------------------------------

def search_knowledge(
    query: str,
    *,
    categories: Optional[List[str]] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Search the knowledge index for entries matching a query string.

    Simple title/description substring search.  For semantic search,
    external memory providers (Honcho, Mem0) should be used.

    Args:
        query:     Lowercase string to match against entry descriptions.
        categories: Optional list of structure categories to scope the search.
        limit:     Maximum number of results to return.

    Returns:
        List of entry dicts with path, type, description, and score.
    """
    index = get_index()
    files: List[Dict[str, Any]] = index.get("files", [])
    structure: Dict[str, Any] = index.get("structure", {})

    # Build scoped pool
    if categories:
        allowed_paths: set = set()
        for cat in categories:
            for p in structure.get(cat, []):
                allowed_paths.add(_ensure_absolute(p))
        pool = [f for f in files if _ensure_absolute(f.get("path", "")) in allowed_paths]
    else:
        pool = list(files)

    query_lower = query.lower()
    scored: List[tuple[int, Dict[str, Any]]] = []

    for entry in pool:
        path = entry.get("path", "")
        desc = entry.get("description", "")
        entry_type = entry.get("type", "")

        # Score: exact type match first, then description contain, then path contain
        score = 0
        if query_lower in entry_type:
            score += 3
        if query_lower in desc:
            score += 2
        if query_lower in path.lower():
            score += 1

        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:limit]]


def get_knowledge_summary() -> str:
    """Return a human-readable summary of the knowledge base for system prompts."""
    index = get_index()
    structure: Dict[str, Any] = index.get("structure", {})
    horizon: int = index.get("horizon", 1)
    version: str = index.get("knowledge_base_version", "1.0.0")
    description: str = index.get("description", "")

    lines = [
        f"[Knowledge Base v{version} | horizon={horizon}]",
        f"{description}",
        "",
    ]

    for category, items in structure.items():
        if not items:
            continue
        if isinstance(items, list):
            count = len(items)
        else:
            count = "multiple"
        lines.append(f"  {category}: {count} entries")

    return "\n".join(lines)


def get_recent_plans(limit: int = 5) -> List[Dict[str, Any]]:
    """Return the most recent plan entries by modification time."""
    entries = list_entries(category="docs", file_type="plan")
    knowledge_dir = _get_knowledge_dir()

    def mtime(entry: Dict[str, Any]) -> float:
        p = resolve_knowledge_path(entry.get("path", ""))
        if p and p.exists():
            return p.stat().st_mtime
        return 0.0

    sorted_entries = sorted(entries, key=mtime, reverse=True)
    return sorted_entries[:limit]


# ---------------------------------------------------------------------------
# CLI / tool-facing helpers
# ---------------------------------------------------------------------------

def list_all_categories() -> List[str]:
    """Return all top-level structure categories."""
    index = get_index()
    return list(index.get("structure", {}).keys())


def get_entry_info(relative_path: str) -> Optional[Dict[str, Any]]:
    """Return the index entry dict for a path, or None if not indexed."""
    index = get_index()
    norm = _ensure_absolute(relative_path)
    for entry in index.get("files", []):
        if _ensure_absolute(entry.get("path", "")) == norm:
            return entry
    return None
