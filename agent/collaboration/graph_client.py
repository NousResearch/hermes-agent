"""Graph API client for the collaboration learning framework.

Wraps the fleet graph HTTP API (graph.xentropy.ai) for storing and querying
observations, preferences, and decisions. Uses the memory endpoints for
node storage and decisions for structured findings.

All methods are fire-and-forget safe — network errors are logged and never
propagate.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default API base URL
_DEFAULT_BASE = os.environ.get("GRAPH_API_URL", "https://graph.xentropy.ai")

# Request timeout in seconds
_TIMEOUT = 10

# Short-term memory TTL (seconds). Observations expire after this.
_SHORT_TTL = 86400 * 7  # 7 days

# Long-term memory TTL. Preferences and patterns persist longer.
_LONG_TTL = 86400 * 90  # 90 days


# ---------------------------------------------------------------------------
# Graph node models
# ---------------------------------------------------------------------------


@dataclass
class GraphMemory:
    """A node in the fleet graph, stored via the /graph/memory API."""

    content: str
    category: str               # "observation", "preference", "pattern", "session"
    tags: List[str] = field(default_factory=list)
    memory_type: str = "long_term"  # "short_term" or "long_term"
    confidence: float = 0.5
    agent_id: str = "hermes:default"
    session_id: str = ""
    ttl: int = _LONG_TTL
    memory_id: str = ""          # populated on create

    def to_create_payload(self) -> Dict:
        return {
            "content": self.content,
            "memoryType": self.memory_type,
            "category": self.category,
            "tags": self.tags,
            "confidence": self.confidence,
            "agentId": self.agent_id,
            "sessionId": self.session_id,
            "ttl": self.ttl,
        }


# ---------------------------------------------------------------------------
# Graph client
# ---------------------------------------------------------------------------


class GraphClient:
    """HTTP client for the fleet graph API.

    All methods are resilient — failures are logged, never raised.
    """

    def __init__(self, base_url: str = _DEFAULT_BASE, agent_id: str = "hermes:default"):
        self._base = base_url.rstrip("/")
        self._agent_id = agent_id
        self._available: Optional[bool] = None  # lazily probed

    @property
    def base_url(self) -> str:
        return self._base

    def is_available(self) -> bool:
        """Check if the graph API is reachable. Caches result for 60s."""
        if self._available is False:
            return False
        try:
            req = urllib.request.Request(f"{self._base}/graph/health", method="GET")
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                data = json.loads(resp.read().decode())
                self._available = data.get("status") in ("healthy", "degraded")
                return self._available
        except Exception:
            self._available = False
            logger.debug("Graph API unavailable: %s", self._base)
            return False

    # -- Memory CRUD --------------------------------------------------------

    def create_memory(self, memory: GraphMemory) -> Optional[str]:
        """Store a memory node in the graph. Returns the node ID or None on failure."""
        try:
            payload = memory.to_create_payload()
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self._base}/graph/memory",
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                result = json.loads(resp.read().decode())
                mem_id = result.get("id", "")
                logger.debug("Graph memory created: %s", mem_id)
                return mem_id
        except Exception as e:
            logger.debug("Graph create_memory failed: %s", e)
            return None

    def search_memories(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Search memory nodes by content + filters.

        Args:
            query: Full-text search query.
            category: Filter by category (e.g. "observation", "preference").
            tags: Filter by tags. Results must match ALL tags.
            limit: Max results.

        Returns:
            List of memory dicts with keys: id, content, category, tags,
            confidence, timestamp, etc.
        """
        try:
            # Build query string
            q_parts = [query]
            if category:
                q_parts.append(f"category:{category}")
            if tags:
                for tag in tags:
                    q_parts.append(f"tag:{tag}")

            q = " ".join(q_parts)
            url = f"{self._base}/graph/memory/search?q={urllib.parse.quote(q)}&limit={limit}"

            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            logger.debug("Graph search_memories failed: %s", e)
            return []

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory node by ID. Returns True on success."""
        try:
            req = urllib.request.Request(
                f"{self._base}/graph/memory/{memory_id}",
                method="DELETE",
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                return resp.status == 200
        except Exception as e:
            logger.debug("Graph delete_memory failed: %s", e)
            return False

    # -- Decisions ----------------------------------------------------------

    def create_decision(
        self,
        description: str,
        context: str = "",
        outcome: str = "pending",
        confidence: float = 0.5,
        tags: Optional[List[str]] = None,
        severity: str = "low",
    ) -> Optional[str]:
        """Record a decision in the fleet graph. Returns decision ID or None."""
        try:
            payload = {
                "description": description,
                "context": context,
                "outcome": outcome,
                "confidence": confidence,
                "tags": tags or [],
                "tenantId": "fleet",
                "severity": severity,
                "agentId": self._agent_id,
            }
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self._base}/graph/fleet/decisions",
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                result = json.loads(resp.read().decode())
                logger.debug("Graph decision created: %s", result.get("id", ""))
                return result.get("id")
        except Exception as e:
            logger.debug("Graph create_decision failed: %s", e)
            return None

    def list_decisions(
        self,
        limit: int = 10,
        tags: Optional[List[str]] = None,
    ) -> List[Dict]:
        """List recent decisions from the fleet graph."""
        try:
            url = f"{self._base}/graph/fleet/decisions?limit={limit}"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                results = json.loads(resp.read().decode())
                if tags:
                    results = [
                        r for r in results
                        if any(t in r.get("tags", []) for t in tags)
                    ]
                return results
        except Exception as e:
            logger.debug("Graph list_decisions failed: %s", e)
            return []


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_client: Optional[GraphClient] = None


def get_client() -> GraphClient:
    """Return the singleton graph client."""
    global _client
    if _client is None:
        _client = GraphClient()
    return _client


def reset_client() -> None:
    """Reset the singleton (for testing)."""
    global _client
    _client = None
