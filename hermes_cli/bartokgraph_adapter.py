"""BartokGraph adapter — bridges BartokGraph's knowledge graph to the Proactive Communication Loop.

BartokGraph is a knowledge graph builder that maps concepts, projects, people, and
ideas from the user's files and conversation history into a weighted graph with typed
edges (TEACHES, BUILDS_ON, CONTRADICTS, MENTIONS, IS_ABOUT, IMPLEMENTS).

This adapter provides the thin interface the ProactiveCommunicationLoop needs:
  - Find active topics in today's conversation
  - Query the graph for dormant related nodes
  - Return BartokGraphConnection objects describing the cross-temporal links

Local model support
-------------------
BartokGraph runs entirely on-device. It detects available LLM providers in order:

  1. Explicit override: BARTOKGRAPH_API_BASE + BARTOKGRAPH_API_KEY + BARTOKGRAPH_LLM_MODEL
  2. Ollama at localhost:11434 (default model: qwen3:8b)
  3. LM Studio at localhost:1234 (auto-detected)
  4. Any OpenAI-compatible server on common local ports

Users with a local LLM pay zero API cost for graph building.

BartokGraph plugin
------------------
The full BartokGraph tool is bundled as ``plugins/bartokgraph/`` so users can
also run it standalone via ``hermes bartokgraph build <path>``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from hermes_cli.proactive_communication_loop import BartokGraphConnection, BartokGraphContext

logger = logging.getLogger(__name__)

# Common local LLM ports to auto-detect
_LOCAL_PORTS = [11434, 1234, 8080, 8000, 5000]

# Default Ollama model — lightweight, fast, free
_DEFAULT_LOCAL_MODEL = os.environ.get("BARTOKGRAPH_LLM_MODEL", "qwen3:8b")
_DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


class BartokGraphAdapter:
    """Thin adapter between BartokGraph's graph store and ProactiveCommunicationLoop.

    Constructed once per ProactiveCommunicationLoop instance. Handles:
    - Detecting whether BartokGraph data is available
    - Resolving the local model provider (Ollama, LM Studio, or explicit override)
    - Querying for cross-temporal connections given today's active topics
    """

    def __init__(self, config: Any) -> None:
        self._cfg = config
        self._model_provider = _resolve_local_model_provider()
        self._graph_data: Optional[Dict[str, Any]] = self._try_load_graph()

    @property
    def is_available(self) -> bool:
        return self._graph_data is not None

    async def get_connections(
        self,
        active_topics: List[str],
        top_k: int = 10,
        min_strength: float = 0.35,
        exclude_recent_hours: int = 24,
    ) -> Optional[BartokGraphContext]:
        """Find BartokGraph connections between today's topics and past knowledge.

        Returns None if BartokGraph data is unavailable (graceful degradation).
        """
        if not self._graph_data:
            return None

        t0 = time.monotonic()
        try:
            connections = await self._find_connections(
                active_topics, top_k, min_strength, exclude_recent_hours
            )
            return BartokGraphContext(
                connections=connections,
                provider_name=self._model_provider.get("name", "local"),
                traversal_ms=int((time.monotonic() - t0) * 1000),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("BartokGraphAdapter: traversal failed: %s", exc)
            return None

    async def _find_connections(
        self,
        active_topics: List[str],
        top_k: int,
        min_strength: float,
        exclude_recent_hours: int,
    ) -> List[BartokGraphConnection]:
        """Find dormant nodes that connect to today's active topics."""
        nodes = self._graph_data.get("nodes", [])
        cutoff_ts = time.time() - exclude_recent_hours * 3600

        # Find dormant nodes (not active in last 24h)
        dormant = [
            n for n in nodes
            if n.get("last_seen_ts", 0) < cutoff_ts
            and n.get("weight", 0) > 0.1
        ]

        connections = []
        for topic in active_topics[:5]:
            for node in dormant:
                strength = _overlap_score(topic, node.get("content", ""))
                if strength >= min_strength:
                    days_apart = max(0, int((time.time() - node.get("last_seen_ts", 0)) / 86400))
                    conn_type = _classify(topic, node, days_apart)
                    connections.append(BartokGraphConnection(
                        node_a_content=topic,
                        node_b_content=node.get("content", ""),
                        connection_type=conn_type,
                        strength=strength,
                        days_apart=days_apart,
                        explanation=_build_explanation(topic, node, conn_type),
                    ))

        # Sort by surprise value: strong + old = most interesting
        connections.sort(key=lambda c: c.strength * (1 + c.days_apart / 30), reverse=True)
        return connections[:top_k]

    def _try_load_graph(self) -> Optional[Dict[str, Any]]:
        """Try to load BartokGraph data from configured workspace."""
        try:
            workspace = os.path.expanduser(
                self._cfg.get("proactive_communication.bartokgraph.workspace", "~")
            )
            graph_path = os.path.join(workspace, ".bartokgraph", "graph.json")
            if not os.path.exists(graph_path):
                logger.debug("BartokGraphAdapter: no graph at %s", graph_path)
                return None
            with open(graph_path, encoding="utf-8") as f:
                data = json.load(f)
            node_count = len(data.get("nodes", []))
            logger.debug("BartokGraphAdapter: loaded graph with %d nodes from %s", node_count, graph_path)
            return data
        except Exception as exc:  # noqa: BLE001
            logger.debug("BartokGraphAdapter: graph load failed: %s", exc)
            return None


# ──────────────────────────────────────────────────────────────────────
# Local model provider detection
# ──────────────────────────────────────────────────────────────────────


def _resolve_local_model_provider() -> Dict[str, str]:
    """Detect available LLM provider for BartokGraph in priority order.

    Priority:
    1. Explicit env override (BARTOKGRAPH_API_BASE + key + model)
    2. Ollama at localhost:11434
    3. LM Studio at localhost:1234
    4. Any OpenAI-compatible server on common local ports
    5. Falls back to None (graph building won't use LLM, uses topology only)
    """
    # 1. Explicit override
    api_base = os.environ.get("BARTOKGRAPH_API_BASE", "")
    api_key = os.environ.get("BARTOKGRAPH_API_KEY", "")
    model = os.environ.get("BARTOKGRAPH_LLM_MODEL", _DEFAULT_LOCAL_MODEL)
    if api_base and api_key:
        logger.debug("BartokGraph: using explicit API provider at %s", api_base)
        return {"name": "api_override", "base_url": api_base, "api_key": api_key, "model": model}

    # 2. Ollama check
    try:
        import urllib.request
        with urllib.request.urlopen(f"{_DEFAULT_OLLAMA_URL}/api/tags", timeout=2) as r:
            if r.status == 200:
                logger.debug("BartokGraph: Ollama detected at %s, model: %s", _DEFAULT_OLLAMA_URL, model)
                return {"name": "ollama", "base_url": _DEFAULT_OLLAMA_URL, "model": model}
    except Exception:
        pass

    # 3. LM Studio check
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:1234/v1/models", timeout=2) as r:
            if r.status == 200:
                logger.debug("BartokGraph: LM Studio detected at localhost:1234")
                return {"name": "lmstudio", "base_url": "http://localhost:1234/v1", "model": "auto"}
    except Exception:
        pass

    # 4. Other common local ports (OpenAI-compatible /v1/models)
    try:
        import urllib.error
        import urllib.request

        for port in _LOCAL_PORTS:
            if port in (11434, 1234):
                continue
            url = f"http://127.0.0.1:{port}/v1/models"
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        logger.debug("BartokGraph: OpenAI-compatible server at port %s", port)
                        return {
                            "name": f"local_compat_{port}",
                            "base_url": f"http://127.0.0.1:{port}/v1",
                            "model": "auto",
                        }
            except (urllib.error.URLError, TimeoutError, OSError):
                continue
    except Exception:
        pass

    # 5. Topology-only fallback (no LLM needed for basic overlap detection)
    logger.debug("BartokGraph: no local LLM detected — using topology-only graph traversal")
    return {"name": "topology_only"}


# ──────────────────────────────────────────────────────────────────────
# Scoring helpers
# ──────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "was", "are", "were", "i", "you", "me", "my", "your", "it", "its",
}


def _overlap_score(a: str, b: str) -> float:
    """Word-overlap strength between two content strings.

    Simple heuristic. Replace with embedding cosine similarity for
    production-quality semantic matching.
    """
    wa = set(a.lower().split()) - _STOPWORDS
    wb = set(b.lower().split()) - _STOPWORDS
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def _classify(active_topic: str, node: Dict[str, Any], days_apart: int) -> str:
    """Classify the type of connection."""
    node_type = node.get("node_type", "topic")
    active_type = "topic"  # active topics are always extracted as topics
    if days_apart > 14:
        return "temporal_bridge"
    if active_type != node_type:
        return "cross_domain"
    return "temporal_bridge"


def _build_explanation(topic: str, node: Dict[str, Any], conn_type: str) -> str:
    """Build a human-readable explanation for the connection."""
    content = node.get("content", "")
    if conn_type == "temporal_bridge":
        days = max(0, int((time.time() - node.get("last_seen_ts", 0)) / 86400))
        return f"Same concept appeared {days} days ago: '{content}'"
    if conn_type == "cross_domain":
        return f"'{topic}' and '{content}' share structural overlap across domains"
    return f"'{topic}' connects to '{content}'"
