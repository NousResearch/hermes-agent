"""CerebroCortex memory provider for Hermes Agent.

Brain-analogous AI memory with 6 modalities, associative graph search,
ACT-R + FSRS biologically-inspired decay, and LLM-powered dream consolidation.

All data is local — SQLite + ChromaDB + igraph. No cloud required.
Designed to run on hardware as minimal as a Raspberry Pi 5.

Install: pip install cerebro-cortex
Repo:    https://github.com/buckster123/CerebroCortex
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas — the 5 core tools exposed to the agent
# ---------------------------------------------------------------------------

CC_REMEMBER_SCHEMA = {
    "name": "cc_remember",
    "description": (
        "Save information to CerebroCortex long-term memory. Auto-classifies type "
        "(episodic/semantic/procedural/affective/prospective), detects duplicates, "
        "extracts concepts, and links to related memories automatically.\n\n"
        "Use for: facts, decisions, lessons learned, user preferences, project context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The memory content to store."},
            "tags": {
                "type": "array", "items": {"type": "string"},
                "description": "Optional tags for categorization.",
            },
            "salience": {
                "type": "number",
                "description": "Importance 0-1 (auto-estimated if omitted).",
            },
        },
        "required": ["content"],
    },
}

CC_RECALL_SCHEMA = {
    "name": "cc_recall",
    "description": (
        "Search CerebroCortex memories by meaning, not just keywords. Uses vector "
        "similarity + spreading activation through the associative graph + ACT-R/FSRS "
        "decay scoring. Finds related memories even if they don't share exact terms.\n\n"
        "Use for: finding relevant context, checking what you know about a topic."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query text."},
            "top_k": {
                "type": "integer", "description": "Max results (default: 5).",
            },
        },
        "required": ["query"],
    },
}

CC_INTENTION_SCHEMA = {
    "name": "cc_todo",
    "description": (
        "Manage TODOs and reminders. Actions:\n"
        "• store — Save a TODO or reminder for future action.\n"
        "• list — List pending TODOs.\n"
        "• resolve — Mark a TODO as done (requires memory_id)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["store", "list", "resolve"],
            },
            "content": {"type": "string", "description": "TODO content (for 'store')."},
            "memory_id": {"type": "string", "description": "Memory ID (for 'resolve')."},
            "tags": {
                "type": "array", "items": {"type": "string"},
                "description": "Tags for categorization (for 'store').",
            },
        },
        "required": ["action"],
    },
}

CC_MESSAGE_SCHEMA = {
    "name": "cc_message",
    "description": (
        "Cross-agent messaging through shared memory. Actions:\n"
        "• send — Send a message to another agent (bypasses gating, always delivered).\n"
        "• inbox — Check for messages addressed to you (newest first)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["send", "inbox"],
            },
            "to": {"type": "string", "description": "Recipient agent ID or 'all' (for 'send')."},
            "content": {"type": "string", "description": "Message content (for 'send')."},
            "limit": {"type": "integer", "description": "Max messages to return (for 'inbox', default: 10)."},
        },
        "required": ["action"],
    },
}

CC_HEALTH_SCHEMA = {
    "name": "cc_health",
    "description": (
        "Get CerebroCortex system health: memory count, link count, episodes, "
        "schemas, and memory type breakdown."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------

class CerebroCortexProvider(MemoryProvider):
    """CerebroCortex memory provider — associative memory with decay and consolidation."""

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        self._cortex = None
        self._agent_id: str = "HERMES"
        self._session_id: str = ""
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "cerebrocortex"

    def is_available(self) -> bool:
        """Check if cerebro-cortex is installed."""
        try:
            import cerebro  # noqa: F401
            return True
        except ImportError:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize CerebroCortex for this session."""
        from cerebro.cortex import CerebroCortex

        self._session_id = session_id
        self._agent_id = self._config.get(
            "agent_id",
            os.environ.get("CEREBRO_AGENT_ID", "HERMES"),
        )

        # Allow custom data dir via config
        data_dir = self._config.get("data_dir")
        if data_dir:
            os.environ.setdefault("CEREBRO_DATA_DIR", data_dir)

        self._cortex = CerebroCortex()
        self._cortex.initialize()
        logger.info(
            "CerebroCortex initialized (agent=%s, session=%s)",
            self._agent_id, session_id,
        )

    def system_prompt_block(self) -> str:
        """Inject CerebroCortex status into the system prompt."""
        if not self._cortex:
            return ""
        try:
            s = self._cortex.stats(agent_id=self._agent_id)
            parts = [
                "# CerebroCortex Memory",
                f"Active. {s['nodes']} memories, {s['links']} links, {s['episodes']} episodes.",
                "Use cc_remember to store, cc_recall to search (semantic + graph activation).",
                "Use cc_todo to manage TODOs, cc_message for cross-agent messaging.",
            ]
            return "\n".join(parts)
        except Exception as e:
            logger.debug("CerebroCortex system_prompt_block failed: %s", e)
            return "# CerebroCortex Memory\nActive. Use cc_remember/cc_recall."

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant memories before each turn."""
        if not self._cortex or not query or len(query.strip()) < 3:
            return ""
        try:
            results = self._cortex.recall(
                query=query,
                top_k=5,
                agent_id=self._agent_id,
            )
            if not results:
                return ""
            lines = []
            for node, score in results:
                lines.append(f"- [{score:.2f}] {node.content[:200]}")
            return "## CerebroCortex Context\n" + "\n".join(lines)
        except Exception as e:
            logger.debug("CerebroCortex prefetch failed: %s", e)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Auto-remember significant user statements in background."""
        if not self._cortex or not user_content:
            return
        # Only auto-store substantial user messages (not short commands)
        if len(user_content.strip()) < 50:
            return

        def _bg_store():
            try:
                with self._lock:
                    self._cortex.remember(
                        content=user_content[:500],
                        agent_id=self._agent_id,
                        session_id=self._session_id,
                        tags=["auto_sync", "user_turn"],
                    )
            except Exception as e:
                logger.debug("CerebroCortex sync_turn failed: %s", e)

        threading.Thread(target=_bg_store, daemon=True).start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            CC_REMEMBER_SCHEMA,
            CC_RECALL_SCHEMA,
            CC_INTENTION_SCHEMA,
            CC_MESSAGE_SCHEMA,
            CC_HEALTH_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._cortex:
            return tool_error("CerebroCortex not initialized")
        try:
            if tool_name == "cc_remember":
                return self._handle_remember(args)
            elif tool_name == "cc_recall":
                return self._handle_recall(args)
            elif tool_name == "cc_todo":
                return self._handle_intention(args)
            elif tool_name == "cc_message":
                return self._handle_message(args)
            elif tool_name == "cc_health":
                return self._handle_health()
            return tool_error(f"Unknown tool: {tool_name}")
        except Exception as e:
            return tool_error(str(e))

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Save a session summary to CerebroCortex."""
        if not self._cortex or not messages:
            return
        try:
            # Build a brief summary from the last few messages
            user_msgs = [
                m.get("content", "")[:200]
                for m in messages[-10:]
                if m.get("role") == "user" and isinstance(m.get("content"), str)
            ]
            if not user_msgs:
                return
            summary = f"Session topics: {'; '.join(user_msgs[:5])}"
            self._cortex.session_save(
                session_summary=summary[:500],
                agent_id=self._agent_id,
            )
        except Exception as e:
            logger.debug("CerebroCortex on_session_end failed: %s", e)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to CerebroCortex."""
        if action == "add" and self._cortex and content:
            try:
                tags = ["hermes_builtin", f"target:{target}"]
                self._cortex.remember(
                    content=content,
                    agent_id=self._agent_id,
                    tags=tags,
                )
            except Exception as e:
                logger.debug("CerebroCortex memory_write mirror failed: %s", e)

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """Store delegation outcomes as episodic memories."""
        if not self._cortex:
            return
        try:
            content = f"Delegated task: {task[:200]}\nResult: {result[:300]}"
            self._cortex.remember(
                content=content,
                agent_id=self._agent_id,
                session_id=self._session_id,
                tags=["delegation", "subagent"],
            )
        except Exception as e:
            logger.debug("CerebroCortex on_delegation failed: %s", e)

    def shutdown(self) -> None:
        if self._cortex:
            self._cortex.close()
            self._cortex = None

    # -- Config ---------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "agent_id",
                "description": "Agent identifier for multi-agent setups",
                "default": "HERMES",
            },
            {
                "key": "data_dir",
                "description": "Data directory (default: ~/.cerebro-cortex/)",
                "default": "",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path) as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["cerebrocortex"] = values
            with open(config_path, "w") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception:
            pass

    # -- Tool handlers --------------------------------------------------------

    def _handle_remember(self, args: dict) -> str:
        node = self._cortex.remember(
            content=args["content"],
            tags=args.get("tags"),
            salience=args.get("salience"),
            agent_id=self._agent_id,
            session_id=self._session_id,
        )
        if node is None:
            return json.dumps({"stored": False, "reason": "gated_out (duplicate or noise)"})
        return json.dumps({
            "stored": True,
            "id": node.id,
            "type": node.metadata.memory_type.value,
            "salience": round(node.metadata.salience, 3),
            "concepts": node.metadata.concepts[:5],
            "links": node.link_count,
        })

    def _handle_recall(self, args: dict) -> str:
        results = self._cortex.recall(
            query=args["query"],
            top_k=args.get("top_k", 5),
            agent_id=self._agent_id,
        )
        return json.dumps({
            "count": len(results),
            "results": [
                {
                    "id": node.id,
                    "content": node.content,
                    "type": node.metadata.memory_type.value,
                    "score": round(score, 4),
                    "tags": node.metadata.tags,
                }
                for node, score in results
            ],
        })

    def _handle_intention(self, args: dict) -> str:
        action = args["action"]
        if action == "store":
            if "content" not in args:
                return tool_error("'content' required for store")
            node = self._cortex.store_intention(
                content=args["content"],
                agent_id=self._agent_id,
                tags=args.get("tags"),
            )
            return json.dumps({"stored": True, "id": node.id})

        elif action == "list":
            intentions = self._cortex.list_intentions(agent_id=self._agent_id)
            return json.dumps({
                "count": len(intentions),
                "items": [
                    {"id": n.id, "content": n.content, "salience": round(n.metadata.salience, 3)}
                    for n in intentions
                ],
            })

        elif action == "resolve":
            if "memory_id" not in args:
                return tool_error("'memory_id' required for resolve")
            self._cortex.resolve_intention(args["memory_id"])
            return json.dumps({"resolved": True, "id": args["memory_id"]})

        return tool_error(f"Unknown action: {action}")

    def _handle_message(self, args: dict) -> str:
        action = args["action"]
        if action == "send":
            if "to" not in args or "content" not in args:
                return tool_error("'to' and 'content' required for send")
            node = self._cortex.send_message(
                to=args["to"],
                content=args["content"],
                agent_id=self._agent_id,
                session_id=self._session_id,
            )
            return json.dumps({"sent": True, "id": node.id, "to": args["to"]})

        elif action == "inbox":
            messages = self._cortex.check_inbox(
                agent_id=self._agent_id,
                limit=args.get("limit", 10),
            )
            return json.dumps({
                "count": len(messages),
                "messages": [
                    {
                        "id": m.id,
                        "from": m.metadata.agent_id,
                        "content": m.content,
                        "created_at": m.created_at.isoformat(),
                    }
                    for m in messages
                ],
            })

        return tool_error(f"Unknown action: {action}")

    def _handle_health(self) -> str:
        s = self._cortex.stats(agent_id=self._agent_id)
        return json.dumps({
            "memories": s["nodes"],
            "links": s["links"],
            "episodes": s["episodes"],
            "types": s.get("memory_types", {}),
            "layers": s.get("layers", {}),
        })


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the CerebroCortex memory provider with Hermes."""
    config = _load_plugin_config()
    provider = CerebroCortexProvider(config=config)
    ctx.register_memory_provider(provider)


def _load_plugin_config() -> dict:
    try:
        from hermes_constants import get_hermes_home
        config_path = get_hermes_home() / "config.yaml"
        if not config_path.exists():
            return {}
        import yaml
        with open(config_path) as f:
            all_config = yaml.safe_load(f) or {}
        return all_config.get("plugins", {}).get("cerebrocortex", {}) or {}
    except Exception:
        return {}
