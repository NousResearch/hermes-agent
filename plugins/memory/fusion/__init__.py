"""Fusion memory provider — aggregates multiple memory sources with RRF fusion.

This provider replaces multiple individual memory providers and instead
queries them in parallel, fusing results using Reciprocal Rank Fusion (RRF).

Config (in memory.provider config):
  fusion:
    sources:
      - holographic    # SQLite FTS5 fact store
      - honcho         # Session-based user modeling
      - hindsight      # KG + entity extraction (if available)
    top_k: 10          # Max results to return
    rrf_k: 60          # RRF constant (default 60)
    enable_tools: true # Expose fusion search tool to model

Usage:
  Set memory.provider to "fusion" in config.yaml
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Tool schema
# -----------------------------------------------------------------------------

FUSION_SEARCH_SCHEMA = {
    "name": "memory_fusion_search",
    "description": (
        "Fused memory search across multiple providers (holographic, honcho, hindsight). "
        "Uses Reciprocal Rank Fusion to combine and rerank results from all memory sources. "
        "Returns the top-K most relevant memories across all providers.\n\n"
        "Use when you need comprehensive recall — it searches ALL memory providers "
        "simultaneously and fuses the results.\n\n"
        "Best for: cross-provider queries, comprehensive research, finding related facts "
        "across different memory systems."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — describe what you're looking for.",
            },
            "top_k": {
                "type": "integer",
                "description": "Max results to return (default: 8, max: 20).",
                "default": 8,
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Which sources to search (default: all available). Options: holographic, honcho, hindsight.",
            },
        },
        "required": ["query"],
    },
}


# -----------------------------------------------------------------------------
# RRF Fusion
# -----------------------------------------------------------------------------

def rrf_fuse(ranked_lists: List[List[Tuple[str, Any]]], k: int = 60) -> List[Tuple[str, float, str]]:
    """Reciprocal Rank Fusion across multiple ranked result lists.

    Args:
        ranked_lists: List of ranked lists, each as [(item_id, metadata), ...]
        k: RRF constant (default 60). Higher = more weight to lower ranks.

    Returns:
        List of (item_id, fused_score, source) tuples sorted by fused_score.
    """
    scores: Dict[str, float] = {}
    sources: Dict[str, str] = {}

    for ranked_list in ranked_lists:
        for rank, (item_id, metadata) in enumerate(ranked_list):
            if isinstance(metadata, dict):
                source = metadata.get("source", "unknown")
            else:
                source = "unknown"

            rrf_score = 1.0 / (k + rank + 1)
            if item_id not in scores:
                scores[item_id] = 0.0
                sources[item_id] = source
            scores[item_id] += rrf_score

    # Sort by fused score descending
    sorted_items = sorted(scores.items(), key=lambda x: -x[1])
    return [(item_id, score, sources[item_id]) for item_id, score in sorted_items]


# -----------------------------------------------------------------------------
# Source adapters
# -----------------------------------------------------------------------------

class HolographicSource:
    """Adapter for holographic (SQLite FTS5) memory store."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def is_available(self) -> bool:
        return self.db_path.exists()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, Dict]]:
        """Search holographic facts using FTS5."""
        if not self.is_available():
            return []

        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # FTS5 search
            cur.execute("""
                SELECT f.fact_id, f.content, f.category, f.trust_score,
                       bm25(facts_fts) as rank
                FROM facts_fts fts
                JOIN facts f ON f.fact_id = fts.rowid
                WHERE facts_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, top_k))

            results = []
            for row in cur.fetchall():
                results.append((
                    f"holo_{row['fact_id']}",
                    {
                        "source": "holographic",
                        "content": row["content"],
                        "category": row["category"],
                        "trust": row["trust_score"],
                    }
                ))

            conn.close()
            return results
        except Exception as e:
            logger.debug("Holographic search failed: %s", e)
            return []

    def get_all_facts(self, limit: int = 100) -> List[Tuple[str, Dict]]:
        """Get all facts ordered by trust score."""
        if not self.is_available():
            return []

        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            cur.execute("""
                SELECT fact_id, content, category, trust_score
                FROM facts
                ORDER BY trust_score DESC, retrieval_count DESC
                LIMIT ?
            """, (limit,))

            results = []
            for row in cur.fetchall():
                results.append((
                    f"holo_{row['fact_id']}",
                    {
                        "source": "holographic",
                        "content": row["content"],
                        "category": row["category"],
                        "trust": row["trust_score"],
                    }
                ))

            conn.close()
            return results
        except Exception as e:
            logger.debug("Holographic get_all failed: %s", e)
            return []


class HonchoSource:
    """Adapter for honcho session memory."""

    def __init__(self, hermes_home: Path):
        self.hermes_home = hermes_home
        self.honcho_dir = hermes_home / "honcho_data"

    def is_available(self) -> bool:
        return self.honcho_dir.exists() and any(self.honcho_dir.iterdir())

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, Dict]]:
        """Search honcho sessions using simple text match."""
        if not self.is_available():
            return []

        query_lower = query.lower()
        results = []
        seen_texts = set()

        try:
            for session_file in self.honcho_dir.glob("*.json"):
                try:
                    with open(session_file) as f:
                        session = json.load(f)

                    # Search in messages
                    for msg in session.get("messages", []):
                        content = msg.get("content", "")
                        if isinstance(content, str) and query_lower in content.lower():
                            # Truncate and dedupe
                            truncated = content[:500]
                            if truncated not in seen_texts:
                                seen_texts.add(truncated)
                                results.append((
                                    f"honcho_{session_file.stem}_{msg.get('id', len(results))}",
                                    {
                                        "source": "honcho",
                                        "content": truncated,
                                        "role": msg.get("role", "unknown"),
                                    }
                                ))
                                if len(results) >= top_k:
                                    return results
                except Exception:
                    continue
        except Exception as e:
            logger.debug("Honcho search failed: %s", e)

        return results


class HindsightSource:
    """Adapter for hindsight KG memory (if local mode is available)."""

    def __init__(self, config_path: Path):
        self.config_path = config_path

    def is_available(self) -> bool:
        # Check if hindsight is configured in local mode
        if not self.config_path.exists():
            return False
        try:
            with open(self.config_path) as f:
                config = json.load(f)
            return config.get("mode") == "local_embedded" and config.get("api_url")
        except Exception:
            return False

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, Dict]]:
        """Search hindsight via REST API (local mode)."""
        if not self.is_available():
            return []

        try:
            import urllib.request
            import urllib.error

            with open(self.config_path) as f:
                config = json.load(f)

            req = urllib.request.Request(
                f"{config['api_url']}/v1/search",
                data=json.dumps({"query": query, "top_k": top_k}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config.get('api_key', '')}"
                },
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                results = []
                for item in data.get("results", []):
                    results.append((
                        f"hindsight_{item.get('id', len(results))}",
                        {
                            "source": "hindsight",
                            "content": item.get("content", ""),
                            "entity": item.get("entity", ""),
                        }
                    ))
                return results
        except Exception as e:
            logger.debug("Hindsight search failed: %s", e)
            return []


# -----------------------------------------------------------------------------
# Fusion Provider
# -----------------------------------------------------------------------------

class FusionMemoryProvider(MemoryProvider):
    """Memory provider that fuses multiple memory sources using RRF."""

    name = "fusion"

    def __init__(self):
        self._sources: Dict[str, Any] = {}
        self._enabled_sources: List[str] = []
        self._top_k: int = 10
        self._rrf_k: int = 60
        self._enable_tools: bool = True
        self._hermes_home: Optional[Path] = None
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        """Check if fusion can run (at least one underlying dependency available).

        SQLite is always available (for holographic), and honcho/hindsight
        are optional. We return True if SQLite is available (always) since
        that's the primary data store.
        """
        # SQLite is always available (used by holographic source)
        import sqlite3
        try:
            sqlite3.connect(":memory:")
            return True
        except Exception:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize all configured sources."""
        self._hermes_home = Path(kwargs.get("hermes_home", Path.home() / ".hermes"))

        # Load config
        from hermes_cli.config import load_config
        try:
            config = load_config()
        except Exception:
            config = {}

        fusion_config = config.get("memory", {}).get("fusion", {})
        self._enabled_sources = fusion_config.get("sources", ["holographic", "honcho"])
        self._top_k = fusion_config.get("top_k", 10)
        self._rrf_k = fusion_config.get("rrf_k", 60)
        self._enable_tools = fusion_config.get("enable_tools", True)

        # Initialize sources
        self._sources = {
            "holographic": HolographicSource(
                self._hermes_home / "memory_store.db"
            ),
            "honcho": HonchoSource(self._hermes_home),
            "hindsight": HindsightSource(
                self._hermes_home / "hindsight" / "config.json"
            ),
        }

        # Filter to only available sources
        self._enabled_sources = [
            name for name in self._enabled_sources
            if name in self._sources and self._sources[name].is_available()
        ]

        logger.info(
            "Fusion memory initialized with sources: %s (top_k=%d, rrf_k=%d)",
            self._enabled_sources, self._top_k, self._rrf_k
        )

    def system_prompt_block(self) -> str:
        if not self._enabled_sources:
            return ""
        return (
            "## Memory: Multi-Source Fusion\n"
            f"Memory searches {len(self._enabled_sources)} sources: "
            f"{', '.join(self._enabled_sources)}. "
            "Results are fused using Reciprocal Rank Fusion for optimal relevance."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Search all sources and fuse results."""
        if not self._enabled_sources:
            return ""

        try:
            results = self._fused_search(query, top_k=self._top_k)
            if not results:
                return ""

            lines = ["<memory-context>"]
            lines.append("[System note: Memory recall from multiple sources via RRF fusion]\n")

            for item_id, score, source_data in results[:self._top_k]:
                content = source_data.get("content", "")
                source = source_data.get("source", "unknown")
                if content:
                    lines.append(f"**[{source}]** {content}")

            lines.append("</memory-context>")
            return "\n".join(lines)
        except Exception as e:
            logger.debug("Fusion prefetch failed: %s", e)
            return ""

    def _fused_search(self, query: str, top_k: int) -> List[Tuple[str, float, Dict]]:
        """Execute parallel search across all sources and fuse results."""
        ranked_lists: List[List[Tuple[str, Dict]]] = []

        def search_source(source_name: str) -> List[Tuple[str, Dict]]:
            src = self._sources.get(source_name)
            if not src or not src.is_available():
                return []
            return src.search(query, top_k=top_k)

        # Parallel search across sources
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(search_source, name): name
                for name in self._enabled_sources
            }

            for future in concurrent.futures.as_completed(futures, timeout=5.0):
                try:
                    results = future.result()
                    if results:
                        ranked_lists.append(results)
                except Exception as e:
                    logger.debug("Source search failed: %s", e)

        if not ranked_lists:
            return []

        # RRF fusion
        fused = rrf_fuse(ranked_lists, k=self._rrf_k)

        # Enrich with full content from each source
        enriched = []
        for item_id, score, source in fused[:top_k]:
            # Find the original metadata
            for ranked_list in ranked_lists:
                for rid, metadata in ranked_list:
                    if rid == item_id:
                        enriched.append((item_id, score, metadata))
                        break

        return enriched[:top_k]

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue background prefetch for next turn (no-op, we do it inline)."""
        pass

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Sync turn to underlying sources (best effort)."""
        # We don't own the data, so nothing to sync
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if self._enable_tools:
            return [FUSION_SEARCH_SCHEMA]
        return []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "memory_fusion_search":
            query = args.get("query", "")
            top_k = args.get("top_k", 8)
            sources = args.get("sources", self._enabled_sources)

            # Filter to enabled sources
            sources = [s for s in sources if s in self._enabled_sources]
            if not sources:
                sources = self._enabled_sources

            # Override sources temporarily if specified
            if sources != self._enabled_sources:
                original_sources = self._enabled_sources
                self._enabled_sources = sources
                results = self._fused_search(query, top_k=top_k)
                self._enabled_sources = original_sources
            else:
                results = self._fused_search(query, top_k=top_k)

            # Format results
            output = []
            for item_id, score, metadata in results[:top_k]:
                content = metadata.get("content", "")
                source = metadata.get("source", "unknown")
                output.append({
                    "id": item_id,
                    "source": source,
                    "score": round(score, 4),
                    "content": content,
                })

            return json.dumps({
                "query": query,
                "sources_searched": sources,
                "total_results": len(output),
                "results": output,
            })

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def shutdown(self) -> None:
        """Clean shutdown."""
        self._sources.clear()


# -----------------------------------------------------------------------------
# Plugin registration
# -----------------------------------------------------------------------------

def register(ctx):
    ctx.register_memory_provider(FusionMemoryProvider())
