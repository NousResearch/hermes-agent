"""Agent Routing Table — the Program Control Table (PCT) equivalent.

Following the IBM CICS model, the routing table maps incoming session
keys to agent identifiers.  It is loaded from a declarative YAML file
and supports hierarchical matching: topic → dm → default.

CICS analogue
-------------
* CICS PCT  →  AgentRoutingTable
  TRANSACTION_ID → PROGRAM_NAME
  session_key   → agent_id

Route resolution order (highest priority first)
-----------------------------------------------
1. **topic** — exact match on ``platform:chat_id:topic_id``
2. **dm** — user-level match on ``platform:user_id``
3. **default_agent** — catch-all fallback

Configuration format (``~/.hermes/agent_routing.yaml``)::

    routing:
      - topic: telegram:-100XXXX:101    # coding forum topic
        agent: coding-agent
      - topic: telegram:-100XXXX:201    # support topic
        agent: cs-agent
      - dm: telegram:user123            # direct message from user123
        agent: personal-agent

    default_agent: personal-agent        # fallback when nothing matches

The file is re-read on each ``resolve()`` call so operators can change
assignments at runtime (hot reload — no gateway restart required).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hermes_state import get_hermes_home


# ── data types ────────────────────────────────────────────────────────


@dataclass
class RouteEntry:
    """One entry in the routing table."""

    kind: str           # "topic" or "dm"
    key: str            # e.g. "telegram:-100XXXX:101" or "telegram:user123"
    agent: str          # target agent id


@dataclass
class AgentRoutingTable:
    """Routable mapping from session keys to agent ids.

    Loads from ``~/.hermes/agent_routing.yaml``.  Supports hot-reload:
    the file is re-parsed on every ``resolve()`` so you can update
    assignments without restarting the gateway.
    """

    config_path: Path = field(
        default_factory=lambda: get_hermes_home() / "agent_routing.yaml"
    )
    default_agent: str = "default"

    # ── loading ──────────────────────────────────────────────────────

    def _load_routes(self) -> tuple[list[RouteEntry], str]:
        """Return (routes, default_agent) from the YAML file.

        If the file is missing or malformed, returns empty routes + the
        stored default_agent.
        """
        try:
            raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, yaml.YAMLError, OSError):
            return [], self.default_agent

        if not isinstance(raw, dict):
            return [], self.default_agent

        entries: list[RouteEntry] = []
        for item in raw.get("routing") or []:
            if not isinstance(item, dict):
                continue
            for kind in ("topic", "dm"):
                val = item.get(kind)
                if val and isinstance(val, str) and "agent" in item:
                    entries.append(
                        RouteEntry(kind=kind, key=val, agent=item["agent"])
                    )

        default = raw.get("default_agent", self.default_agent)
        return entries, str(default) if default else self.default_agent

    # ── resolution ───────────────────────────────────────────────────

    def resolve(
        self,
        platform: str,
        chat_id: str,
        topic_id: str | None = None,
    ) -> str:
        """Return the agent id that should handle this session.

        Resolution order:  topic → dm → default.
        """
        routes, default = self._load_routes()

        if topic_id:
            topic_key = f"{platform}:{chat_id}:{topic_id}"
            for r in routes:
                if r.kind == "topic" and r.key == topic_key:
                    return r.agent

        dm_key = f"{platform}:{chat_id}"
        for r in routes:
            if r.kind == "dm" and r.key == dm_key:
                return r.agent

        return default
