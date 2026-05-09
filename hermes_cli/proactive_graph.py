"""Graph-augmented context for the Proactive Loop.

This module extends ``ProactiveLoop`` with knowledge-graph traversal so that
synthesis can surface **cross-temporal and cross-domain connections** that
the user cannot see on their own.

The problem with recency-only synthesis
----------------------------------------
Looking at the last 16 hours is like using a goldfish's memory. A user who
mentioned soil carbon today and the Kenya project three weeks ago will never
connect those two threads themselves — they can't hold months of context.
The agent can.

What BartokGraph-style traversal adds
--------------------------------------
Three new message types that only a connected graph enables:

1. **Temporal bridge** — "You asked this same question 3 weeks ago. The
   answer you found then applies here."

2. **Cross-domain connection** — "Your trading regime detection work and
   your regen-ag soil work share the same underlying signal structure."

3. **Person-knowledge bridge** — "Alice mentioned X last week. You asked
   about Y today. These converge on Guruji objective #6."

Without the graph, the synthesis model sees a *transcript snippet*.
With the graph, it sees *a web of weighted, time-stamped knowledge nodes*
and can ask: "does anything from today activate a dormant thread?"

Architecture
------------
``KnowledgeGraphContext`` is a **thin adapter** over whatever memory
backend is configured (MemPalace, mem0, Holographic, Honcho, etc.) or
over a local SQLite graph if no provider is active.

It does NOT require a specific graph library or backend. The interface is
deliberately minimal so it works with every memory provider::

    nodes = await ctx.get_nodes_for_topics(topics, top_k=10)
    connections = await ctx.find_cross_connections(nodes, recent_nodes)

The synthesis prompt builder picks these up and augments the judge prompt
with a "GRAPH CONTEXT" section.

Integration point
-----------------
Import ``build_graph_augmented_prompt`` from this module in
``proactive_loop._build_synthesis_prompt`` (future wiring) to upgrade
every synthesis pass automatically when a memory provider is active.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────


@dataclass
class KnowledgeNode:
    """A single node in the knowledge graph.

    Nodes are topics, entities, questions, decisions, or insights that
    were extracted from past conversation turns. The weight reflects how
    frequently or recently this node appeared.
    """

    node_id: str
    content: str          # The concept, fact, or insight
    node_type: str        # "topic" | "entity" | "question" | "decision" | "insight"
    weight: float         # 0–1: recency + frequency combined
    first_seen_ts: int    # Unix timestamp (seconds)
    last_seen_ts: int
    session_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphConnection:
    """A detected connection between two knowledge nodes.

    Connections are either explicit (the user stated a relationship) or
    inferred (the synthesis model detected a semantic link).
    """

    node_a: KnowledgeNode
    node_b: KnowledgeNode
    connection_type: str   # "temporal_bridge" | "cross_domain" | "person_knowledge"
    strength: float        # 0–1
    explanation: str       # human-readable: "Both discuss soil carbon sequestration"
    days_apart: int        # how long since node_a and node_b were last discussed together


@dataclass
class GraphAugmentedContext:
    """All graph context gathered for one synthesis pass."""

    active_nodes: List[KnowledgeNode]           # from today's history
    related_dormant_nodes: List[KnowledgeNode]  # surfaced by graph traversal
    connections: List[GraphConnection]          # bridging active → dormant
    provider_name: str                          # which memory backend was used
    traversal_ms: int = 0                       # wall-clock time


# ──────────────────────────────────────────────────────────────────────
# Memory provider protocol (thin adapter)
# ──────────────────────────────────────────────────────────────────────


@runtime_checkable
class MemoryGraphBackend(Protocol):
    """Minimal protocol any memory provider must satisfy to enable graph augmentation.

    Third-party memory plugins (mem0, MemPalace, Holographic, etc.) can
    implement this to opt in. The adapter pattern means we never hard-code
    a dependency on any single backend.
    """

    async def search_nodes(
        self,
        query: str,
        top_k: int = 10,
        exclude_recent_hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """Return top_k nodes matching ``query``, excluding very recent ones."""
        ...

    async def get_connections(
        self,
        node_ids: List[str],
        max_hops: int = 2,
    ) -> List[Dict[str, Any]]:
        """Return connections reachable from ``node_ids`` within ``max_hops``."""
        ...


# ──────────────────────────────────────────────────────────────────────
# Context builder
# ──────────────────────────────────────────────────────────────────────


class KnowledgeGraphContext:
    """Builds graph-augmented context for the proactive synthesis pass.

    Usage::

        ctx = KnowledgeGraphContext(backend=memory_provider)
        graph_ctx = await ctx.build(
            recent_history="user asked about soil carbon...",
            active_topics=["soil carbon", "Kenya project"],
        )
        prompt = build_graph_augmented_prompt(history, graph_ctx)
    """

    def __init__(self, backend: Optional[MemoryGraphBackend] = None) -> None:
        self._backend = backend

    @property
    def is_available(self) -> bool:
        return self._backend is not None

    async def build(
        self,
        recent_history: str,
        active_topics: List[str],
        top_k: int = 10,
        connection_min_strength: float = 0.4,
    ) -> Optional[GraphAugmentedContext]:
        """Build graph context for the current synthesis window.

        Returns ``None`` if no backend is available (graceful degradation —
        synthesis continues without graph augmentation).
        """
        if not self._backend:
            return None

        import time
        t0 = time.monotonic()

        try:
            # Step 1: Extract active nodes from today's history topics
            active_nodes = await self._extract_active_nodes(active_topics)

            # Step 2: Find dormant nodes related to today's topics
            dormant_nodes = await self._find_dormant_nodes(
                active_topics, top_k=top_k
            )

            # Step 3: Detect connections between active and dormant
            connections = await self._detect_connections(
                active_nodes, dormant_nodes, min_strength=connection_min_strength
            )

            provider = getattr(self._backend, "provider_name", "unknown")
            return GraphAugmentedContext(
                active_nodes=active_nodes,
                related_dormant_nodes=dormant_nodes,
                connections=connections,
                provider_name=provider,
                traversal_ms=int((time.monotonic() - t0) * 1000),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("proactive_graph: traversal failed: %s", exc)
            return None

    async def _extract_active_nodes(self, topics: List[str]) -> List[KnowledgeNode]:
        """Convert today's detected topics into KnowledgeNode objects."""
        nodes = []
        import time
        now = int(time.time())
        for i, topic in enumerate(topics):
            nodes.append(KnowledgeNode(
                node_id=f"active_{i}",
                content=topic,
                node_type="topic",
                weight=1.0,  # active today = max weight
                first_seen_ts=now,
                last_seen_ts=now,
            ))
        return nodes

    async def _find_dormant_nodes(
        self, topics: List[str], top_k: int
    ) -> List[KnowledgeNode]:
        """Query the memory backend for dormant nodes related to today's topics."""
        if not self._backend:
            return []
        results = []
        for topic in topics[:3]:  # limit to top 3 topics to control cost
            try:
                raw = await self._backend.search_nodes(
                    query=topic,
                    top_k=top_k // len(topics[:3]),
                    exclude_recent_hours=24,  # dormant = not discussed today
                )
                for r in raw:
                    results.append(KnowledgeNode(
                        node_id=r.get("id", ""),
                        content=r.get("content", r.get("text", "")),
                        node_type=r.get("type", "topic"),
                        weight=float(r.get("weight", r.get("score", 0.5))),
                        first_seen_ts=int(r.get("created_at", 0)),
                        last_seen_ts=int(r.get("updated_at", 0)),
                        metadata=r,
                    ))
            except Exception as exc:  # noqa: BLE001
                logger.debug("proactive_graph: search failed for '%s': %s", topic, exc)
        return results

    async def _detect_connections(
        self,
        active: List[KnowledgeNode],
        dormant: List[KnowledgeNode],
        min_strength: float,
    ) -> List[GraphConnection]:
        """Find meaningful connections between active and dormant nodes.

        Uses a simple heuristic: if a dormant node's content overlaps
        significantly with an active node's content, it's a potential
        temporal bridge. Semantic similarity scoring can be added later.
        """
        connections = []
        import time
        now = int(time.time())

        for dormant_node in dormant:
            for active_node in active:
                overlap = _content_overlap_score(active_node.content, dormant_node.content)
                if overlap >= min_strength:
                    days_apart = max(0, int((now - dormant_node.last_seen_ts) / 86400))
                    conn_type = _classify_connection_type(active_node, dormant_node, days_apart)
                    connections.append(GraphConnection(
                        node_a=active_node,
                        node_b=dormant_node,
                        connection_type=conn_type,
                        strength=overlap,
                        explanation=f"'{active_node.content}' connects to '{dormant_node.content}'",
                        days_apart=days_apart,
                    ))

        # Sort by strength + novelty (older + stronger = most surprising)
        connections.sort(key=lambda c: (c.strength * (1 + c.days_apart / 30)), reverse=True)
        return connections[:5]  # top 5 most interesting


def _content_overlap_score(a: str, b: str) -> float:
    """Simple word-overlap score between two content strings.

    Returns 0–1. In production, replace with embedding similarity.
    """
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    # Remove common stopwords
    stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "is", "was"}
    words_a -= stopwords
    words_b -= stopwords
    if not words_a or not words_b:
        return 0.0
    overlap = len(words_a & words_b)
    return overlap / max(len(words_a), len(words_b))


def _classify_connection_type(
    active: KnowledgeNode,
    dormant: KnowledgeNode,
    days_apart: int,
) -> str:
    if days_apart > 14:
        return "temporal_bridge"
    if active.node_type != dormant.node_type:
        return "cross_domain"
    return "temporal_bridge"


# ──────────────────────────────────────────────────────────────────────
# Prompt augmentation
# ──────────────────────────────────────────────────────────────────────


def build_graph_augmented_prompt(
    history: str,
    already_sent: str,
    graph_ctx: Optional[GraphAugmentedContext],
) -> str:
    """Build the synthesis prompt, optionally augmented with graph context.

    When ``graph_ctx`` is None (no memory provider), falls back to the
    standard recency-only prompt — graceful degradation.
    """
    base_prompt = f"""You are reviewing a conversation history to decide whether to send the user
an unprompted message. Your job is to find something genuinely worth saying — not
to generate noise.

RECENT CONVERSATION HISTORY:
{history}

ALREADY SENT TODAY (do not repeat these):
{already_sent}
"""

    graph_section = ""
    if graph_ctx and graph_ctx.connections:
        conn_lines = []
        for conn in graph_ctx.connections[:3]:  # top 3 most interesting
            conn_lines.append(
                f"- [{conn.connection_type.upper()}] '{conn.node_a.content}' ↔ "
                f"'{conn.node_b.content}' "
                f"(strength: {conn.strength:.2f}, {conn.days_apart}d ago)"
            )
        graph_section = f"""
KNOWLEDGE GRAPH CONTEXT — CONNECTIONS FROM PAST CONVERSATIONS:
These are non-obvious links between today's topics and things discussed weeks or months ago.
If any of these represent a genuinely surprising insight the user hasn't seen, consider surfacing it.

{chr(10).join(conn_lines)}

Connection types:
- TEMPORAL_BRIDGE: same concept appeared weeks ago — the user may have forgotten
- CROSS_DOMAIN: concept from one domain connects to a different domain
- PERSON_KNOWLEDGE: connects to something a specific person mentioned
"""

    instruction = """
Review the history (and graph context if present) and identify anything meeting these criteria:
1. An unresolved question that now has an answer
2. A completed background task with a result worth sharing
3. A TEMPORAL BRIDGE — something from today echoes an older thread the user has forgotten
4. A CROSS-DOMAIN connection — "your X work and your Y work share the same underlying structure"
5. Something the user asked you to "let them know about" earlier

THE BAR IS HIGH. If you're not confident this is worth interrupting the user, set should_send=false.

If you find something worth sending, write a SHORT, NATURAL message (2-4 sentences).
Write as if continuing a conversation, NOT as a report.
Good: "Hey — just connected something. Your regime detection work and your soil carbon signal 
work are solving the same problem from different angles. Interesting."
Bad: "GRAPH ANALYSIS REPORT: cross-domain connection detected."

For graph-based messages, lead with the surprise, not the mechanism.

Respond in JSON:
{
  "should_send": true/false,
  "message": "the natural message to send, or null",
  "novelty": 0.0-1.0,
  "relevance": 0.0-1.0,
  "reasoning": "1-2 sentences on why you decided to send or not",
  "connection_type": "temporal_bridge|cross_domain|person_knowledge|none",
  "candidates": ["list of things considered"]
}"""

    return base_prompt + graph_section + instruction
