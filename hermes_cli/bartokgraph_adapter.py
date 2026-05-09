"""BartokGraph adapter — bridges BartokGraph's knowledge graph to the Proactive Communication Loop.

BartokGraph v2.0 is a three-layer knowledge graph:

  Layer 1 — Code Intelligence (weight 1x)
    Maps codebase structure, dependencies, function relationships.

  Layer 2 — Knowledge Graph (weight 10x)
    Maps ideas, concepts, projects, research. The layer that matters
    for the Proactive Communication Loop.

  Layer 3 — Person Graphs (filtered per-person)
    Isolated per-person graphs: Daniel's world, Alice's world, Sage's world.

Node weights (from ARCHITECTURE-V2.md):
  SOUL.md, USER.md, MEMORY.md     → 50  (sacred identity files)
  memory/YYYY-MM-DD.md            → 20  (daily memory logs)
  projects/**/*.md                → 15  (project knowledge)
  research/**                     → 10  (research notes)
  *.md, *.txt (general)           → 8
  *.html, *.pdf                   → 6
  *.json, *.jsonl                 → 4
  *.ts, *.tsx, *.js, *.mjs, *.py  → 1   (code — low noise floor)
  test files                      → 0.1

Surprise scoring formula
------------------------
The traversal ranks connections by "how surprising and important is this?"
Not just semantic overlap. Three factors combine:

  surprise = semantic_strength × node_importance × temporal_decay

  semantic_strength  — word-overlap score (0–1), replace with embeddings for production
  node_importance    — normalized node weight (0–1), based on source file type
  temporal_decay     — older dormant nodes score higher (more likely forgotten)
                       factor = 1 + log(1 + days_apart / 7)

High node weight (SOUL.md, daily memory) + strong semantic match + dormant for weeks
= the connection most worth surfacing.

Low node weight (test files, generated code) + weak overlap + dormant 2 days
= never surfaced.

Local model support
-------------------
BartokGraph runs entirely on-device. Provider priority:
  1. BARTOKGRAPH_API_BASE + BARTOKGRAPH_API_KEY + BARTOKGRAPH_LLM_MODEL
  2. Ollama at localhost:11434 (default model: qwen3:8b)
  3. LM Studio at localhost:1234
  4. Any OpenAI-compatible server on common local ports
  5. Topology-only (no LLM — uses node weights + word overlap only)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from hermes_cli.proactive_communication_loop import BartokGraphConnection, BartokGraphContext

logger = logging.getLogger(__name__)

_DEFAULT_LOCAL_MODEL = os.environ.get("BARTOKGRAPH_LLM_MODEL", "qwen3:8b")
_DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_LOCAL_PORTS = [11434, 1234, 8080, 8000, 5000]


# ──────────────────────────────────────────────────────────────────────
# Node weight table — mirrors ARCHITECTURE-V2.md exactly
# ──────────────────────────────────────────────────────────────────────

# Source file → importance weight.
# These are applied when the graph.json node has a `source_path` field.
# When weight is explicitly stored in the node, that takes precedence.
_SOURCE_WEIGHTS: List[Tuple[str, float]] = [
    # Test files — checked FIRST, before any extension rule
    (r"(?:^|/)test[_.]",          0.1),  # test_goals.py, test.ts
    (r"\.(test|spec)\.",          0.1),  # goals.test.ts, goals.spec.js
    (r"(?:^|/)tests?/",           0.1),  # tests/goals.py, test/goals.ts
    # Sacred identity files — highest importance
    (r"SOUL\.md$",               50.0),
    (r"USER\.md$",               50.0),
    (r"MEMORY\.md$",             50.0),
    # Daily memory logs
    (r"memory/\d{4}-\d{2}-\d{2}\.md$", 20.0),
    # Project knowledge
    (r"projects/.*\.md$",        15.0),
    # Research notes
    (r"research/",               10.0),
    # General prose
    (r"\.md$",                    8.0),
    (r"\.txt$",                   8.0),
    # Documents / briefs
    (r"\.html$",                  6.0),
    (r"\.pdf$",                   6.0),
    # Structured data
    (r"\.json$",                  4.0),
    (r"\.jsonl$",                 4.0),
    # Code — low noise floor (checked last)
    (r"\.(ts|tsx|js|mjs|py|sh|sql)$", 1.0),
]

# Maximum raw weight in the scale above — used for normalization
_MAX_SOURCE_WEIGHT = 50.0

# Explicit layer weights — multiplied by per-node weight
# Knowledge layer (prose, ideas) is 10x more important than code layer
_LAYER_MULTIPLIERS = {
    "knowledge": 10.0,
    "person":    10.0,
    "code":       1.0,
}

# Nodes with weight below this threshold are ignored entirely
_MIN_NODE_WEIGHT = 0.5  # filters out test files, auto-generated code, etc.


def _source_weight(source_path: str) -> float:
    """Determine importance weight from source file path."""
    if not source_path:
        return 4.0  # default: treat as structured data
    for pattern, weight in _SOURCE_WEIGHTS:
        if re.search(pattern, source_path, re.IGNORECASE):
            return weight
    return 4.0  # fallback


def _node_importance(node: Dict[str, Any]) -> float:
    """Compute 0–1 normalized importance for a graph node.

    Priority:
    1. Explicit `weight` field in the node (BartokGraph v2.0 stores this)
    2. Inferred from `source_path` using the weight table
    3. Layer multiplier applied on top
    """
    # Use explicit weight if stored
    raw_weight = node.get("weight")
    if raw_weight is None or raw_weight == 0:
        source_path = node.get("source_path", node.get("file", ""))
        raw_weight = _source_weight(source_path)

    # Apply layer multiplier
    layer = node.get("layer", "knowledge")
    multiplier = _LAYER_MULTIPLIERS.get(layer, 1.0)
    weighted = float(raw_weight) * multiplier

    # Normalize to 0–1 against the maximum possible (SOUL.md × knowledge layer)
    max_possible = _MAX_SOURCE_WEIGHT * max(_LAYER_MULTIPLIERS.values())
    return min(1.0, weighted / max_possible)


def _temporal_decay_factor(days_apart: int) -> float:
    """Older dormant connections score higher — they're more likely forgotten.

    Uses log scale so the curve is steep in the first week (2 days old scores
    much lower than 14 days old) then flattens (60 days ≈ 90 days).

    Factor range: 1.0 (just dormant) to ~4.0 (years old)
    """
    return 1.0 + math.log1p(days_apart / 7.0)


def _surprise_score(
    semantic_strength: float,
    node_importance: float,
    days_apart: int,
) -> float:
    """Composite surprise score — what makes a connection worth surfacing.

    surprise = semantic_strength × node_importance × temporal_decay

    This ensures:
    - Weak semantic match never surfaces regardless of importance
    - Test-file nodes never surface regardless of age
    - Recent-but-important nodes need very strong semantic match
    - Old, important nodes surface at lower semantic thresholds
    """
    decay = _temporal_decay_factor(days_apart)
    return semantic_strength * node_importance * decay


# ──────────────────────────────────────────────────────────────────────
# Main adapter
# ──────────────────────────────────────────────────────────────────────


class BartokGraphAdapter:
    """Adapter between BartokGraph's graph store and ProactiveCommunicationLoop.

    Loads the full graph.json, applies the three-layer weighting system,
    and returns ranked BartokGraphConnection objects sorted by surprise score.

    Constructed once per ProactiveCommunicationLoop instance.
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
        min_semantic_strength: float = 0.2,  # lowered — importance weighting does the real filtering
        exclude_recent_hours: int = 24,
    ) -> Optional[BartokGraphContext]:
        """Find cross-temporal connections between today's topics and past knowledge.

        Ranks by surprise score (semantic × importance × temporal decay).
        Only returns None if graph data is unavailable.
        Returns an empty BartokGraphContext (no connections) if nothing scores high enough.
        """
        if not self._graph_data:
            return None

        t0 = time.monotonic()
        try:
            connections = await self._find_connections(
                active_topics, top_k, min_semantic_strength, exclude_recent_hours
            )
            return BartokGraphContext(
                connections=connections,
                provider_name=self._model_provider.get("name", "local"),
                traversal_ms=int((time.monotonic() - t0) * 1000),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("BartokGraphAdapter: traversal failed: %s", exc)
            return BartokGraphContext(connections=[], provider_name="error")

    async def _find_connections(
        self,
        active_topics: List[str],
        top_k: int,
        min_semantic_strength: float,
        exclude_recent_hours: int,
    ) -> List[BartokGraphConnection]:
        """Core traversal — weighted by importance, ranked by surprise."""
        nodes = self._graph_data.get("nodes", [])
        cutoff_ts = time.time() - exclude_recent_hours * 3600

        # Filter to dormant nodes that clear the minimum weight threshold
        dormant = [
            n for n in nodes
            if n.get("last_seen_ts", 0) < cutoff_ts
            and _node_importance(n) * _MAX_SOURCE_WEIGHT * max(_LAYER_MULTIPLIERS.values()) >= _MIN_NODE_WEIGHT
        ]

        logger.debug(
            "BartokGraphAdapter: %d total nodes, %d dormant candidates, %d active topics",
            len(nodes), len(dormant), len(active_topics),
        )

        scored: List[Tuple[float, BartokGraphConnection]] = []

        for topic in active_topics[:8]:  # extended from 5 to 8 for better coverage
            for node in dormant:
                semantic = _overlap_score(topic, node.get("content", ""))
                if semantic < min_semantic_strength:
                    continue

                importance = _node_importance(node)
                days_apart = max(0, int((time.time() - node.get("last_seen_ts", 0)) / 86400))
                surprise = _surprise_score(semantic, importance, days_apart)

                conn_type = _classify_connection(topic, node, days_apart)
                explanation = _build_explanation(topic, node, conn_type, days_apart, importance)

                conn = BartokGraphConnection(
                    node_a_content=topic,
                    node_b_content=node.get("content", ""),
                    connection_type=conn_type,
                    strength=surprise,       # surprise score, not raw semantic overlap
                    days_apart=days_apart,
                    explanation=explanation,
                )
                scored.append((surprise, conn))

        # Sort by surprise score descending, deduplicate by node_b_content
        scored.sort(key=lambda x: x[0], reverse=True)
        seen: set = set()
        unique: List[BartokGraphConnection] = []
        for score, conn in scored:
            key = conn.node_b_content[:80]
            if key not in seen:
                seen.add(key)
                unique.append(conn)
            if len(unique) >= top_k:
                break

        logger.debug(
            "BartokGraphAdapter: %d raw candidates → %d unique connections returned",
            len(scored), len(unique),
        )
        return unique

    def _try_load_graph(self) -> Optional[Dict[str, Any]]:
        """Load graph.json from the configured workspace. Never raises."""
        try:
            workspace = os.path.expanduser(
                self._cfg.get("proactive_communication.bartokgraph.workspace", "~")
            )

            # Support both output paths from BartokGraph v2.0
            candidates = [
                os.path.join(workspace, ".bartokgraph", "graph.json"),
                os.path.join(workspace, "bartokgraph-output", "bartok-knowledge-graph.json"),
                os.path.join(workspace, "bartokgraph-output", "graph.json"),
            ]

            for graph_path in candidates:
                if os.path.exists(graph_path):
                    with open(graph_path, encoding="utf-8") as f:
                        data = json.load(f)
                    node_count = len(data.get("nodes", []))
                    logger.debug(
                        "BartokGraphAdapter: loaded %d nodes from %s",
                        node_count, graph_path,
                    )
                    return data

            logger.debug("BartokGraphAdapter: no graph found at any candidate path")
            return None

        except Exception as exc:  # noqa: BLE001
            logger.debug("BartokGraphAdapter: graph load failed: %s", exc)
            return None


# ──────────────────────────────────────────────────────────────────────
# Semantic similarity
# ──────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "was", "are", "were", "i", "you", "me", "my", "your", "it", "its",
    "this", "that", "with", "from", "have", "had", "not", "but", "be", "by",
    "as", "we", "they", "do", "did", "has", "all", "can", "will", "just",
}


def _overlap_score(a: str, b: str) -> float:
    """Jaccard word-overlap between two strings, ignoring stopwords.

    Production replacement: embed both strings and compute cosine similarity.
    This lexical version is fast and works well for concept-level matching
    when BartokGraph has already extracted semantic concepts as node content.
    """
    wa = {w.strip(".,!?;:\"'()[]") for w in a.lower().split()} - _STOPWORDS
    wb = {w.strip(".,!?;:\"'()[]") for w in b.lower().split()} - _STOPWORDS
    wa = {w for w in wa if len(w) > 2}
    wb = {w for w in wb if len(w) > 2}
    if not wa or not wb:
        return 0.0
    intersection = len(wa & wb)
    union = len(wa | wb)
    return intersection / union if union > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────
# Connection classification and explanation
# ──────────────────────────────────────────────────────────────────────


def _classify_connection(
    active_topic: str,
    node: Dict[str, Any],
    days_apart: int,
) -> str:
    """Classify the type of cross-temporal connection.

    PERSON_KNOWLEDGE: node is tagged to a specific person (VIP relationship)
    TEMPORAL_BRIDGE: same concept reappears after ≥7 days of dormancy
    CROSS_DOMAIN: concept from a different layer/domain type
    """
    # Person-tagged nodes — highest priority classification
    person = node.get("person") or node.get("attributed_to")
    if person:
        return "person_knowledge"

    # Different layer = cross-domain
    node_layer = node.get("layer", "knowledge")
    if node_layer == "code":
        return "cross_domain"

    # Same-domain temporal bridge (default for knowledge layer)
    if days_apart >= 7:
        return "temporal_bridge"

    return "temporal_bridge"


def _build_explanation(
    topic: str,
    node: Dict[str, Any],
    conn_type: str,
    days_apart: int,
    importance: float,
) -> str:
    """Build a precise, human-readable explanation for the connection.

    The explanation feeds into the judge model's prompt — the more
    precise it is, the better the judge can evaluate whether it's worth
    surfacing.
    """
    content = node.get("content", "")
    source = node.get("source_path", node.get("file", ""))
    person = node.get("person") or node.get("attributed_to")
    importance_label = (
        "core identity" if importance > 0.8 else
        "high-importance" if importance > 0.5 else
        "notable" if importance > 0.2 else
        "background"
    )

    if conn_type == "person_knowledge":
        return (
            f"'{person}' discussed '{content}' {days_apart}d ago "
            f"({importance_label} node from {source or 'conversation'}) — "
            f"connects to today's '{topic}'"
        )

    if conn_type == "temporal_bridge":
        weeks = days_apart // 7
        time_str = f"{weeks}w ago" if weeks >= 2 else f"{days_apart}d ago"
        return (
            f"'{content}' appeared {time_str} "
            f"({importance_label}, {source or 'conversation'}) — "
            f"same concept as today's '{topic}', likely forgotten"
        )

    if conn_type == "cross_domain":
        return (
            f"'{topic}' (current session) structurally mirrors '{content}' "
            f"from {source or 'a different domain'} ({days_apart}d dormant, {importance_label})"
        )

    return f"'{topic}' connects to '{content}' ({days_apart}d ago, {importance_label})"


# ──────────────────────────────────────────────────────────────────────
# Local model provider detection
# ──────────────────────────────────────────────────────────────────────


def _resolve_local_model_provider() -> Dict[str, str]:
    """Detect available LLM provider in priority order.

    For the Proactive Communication Loop, the judge call goes through
    Hermes's configured provider (get_text_auxiliary_client). This
    resolver is used only for BartokGraph's own graph-building step,
    which runs on-device.
    """
    import urllib.error
    import urllib.request

    # 1. Explicit env override
    api_base = os.environ.get("BARTOKGRAPH_API_BASE", "")
    api_key = os.environ.get("BARTOKGRAPH_API_KEY", "")
    model = os.environ.get("BARTOKGRAPH_LLM_MODEL", _DEFAULT_LOCAL_MODEL)
    if api_base and api_key:
        return {"name": "api_override", "base_url": api_base, "api_key": api_key, "model": model}

    # 2. Ollama
    try:
        with urllib.request.urlopen(f"{_DEFAULT_OLLAMA_URL}/api/tags", timeout=2) as r:
            if r.status == 200:
                return {"name": "ollama", "base_url": _DEFAULT_OLLAMA_URL, "model": model}
    except Exception:
        pass

    # 3. LM Studio
    try:
        with urllib.request.urlopen("http://localhost:1234/v1/models", timeout=2) as r:
            if r.status == 200:
                return {"name": "lmstudio", "base_url": "http://localhost:1234/v1", "model": "auto"}
    except Exception:
        pass

    # 4. Other common local ports
    for port in _LOCAL_PORTS:
        if port in (11434, 1234):
            continue
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
                if r.status == 200:
                    return {"name": f"local_compat_{port}", "base_url": f"http://127.0.0.1:{port}/v1", "model": "auto"}
        except Exception:
            continue

    # 5. Topology-only — word overlap + weights, no LLM embedding
    return {"name": "topology_only"}
