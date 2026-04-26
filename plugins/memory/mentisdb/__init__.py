"""MentisDB memory plugin — MemoryProvider for MentisDB semantic memory.

Connects to a MentisDB MCP server via synchronous JSON-RPC over HTTP.
Replaces markdown-file-based memory with a proper append-only semantic store.

The provider:
  - Bootstraps the MentisDB chain on session start
  - Mirrors built-in memory writes to MentisDB (via on_memory_write)
  - Injects relevant context before each turn (via prefetch)
  - Provides instructions for using MentisDB tools (via system_prompt_block)

Architecture: pure synchronous HTTP + JSON-RPC 2.0.  No asyncio, no
thread-locals, no anyio cancel-scope traps.  A background writer thread
flushes a write queue every 2 seconds.

Requires: requests package + a running MentisDB MCP server.

Config: reads the MentisDB URL + protocol_version from
  ~/.hermes/config.yaml -> mcp_servers.mentisdb
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import requests

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentry log to verify this module version is loaded
# ---------------------------------------------------------------------------
logger.info("MentisDB plugin v2 (sync HTTP/JSON-RPC) loaded from %s", __file__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get_mentisdb_config() -> Optional[Dict[str, Any]]:
    """Read mentisdb MCP server config from ~/.hermes/config.yaml."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        servers = cfg.get("mcp_servers", {})
        return servers.get("mentisdb")
    except Exception as e:
        logger.debug("Failed to read mentisdb config: %s", e)
        return None


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 client for MentisDB MCP (sync)
# ---------------------------------------------------------------------------

class MentisDbRpcClient:
    """Synchronous JSON-RPC 2.0 client for a MentisDB MCP HTTP endpoint.

    Talks to the MentisDB streamable-HTTP MCP transport directly,
    avoiding the async MCP SDK and its anyio cancel-scope issues.
    """

    def __init__(self, url: str, protocol_version: Optional[str] = None,
                 timeout: float = 30):
        self._url = url.rstrip("/")
        self._proto_ver = protocol_version
        self._session_id: Optional[str] = None
        self._req_id: int = 0
        self._sess = requests.Session()
        self._sess.headers["Content-Type"] = "application/json"
        self._sess.headers["Accept"] = "application/json, text/event-stream"
        if protocol_version:
            self._sess.headers["mcp-protocol-version"] = protocol_version
        self._sess.timeout = timeout

    def initialize(self) -> bool:
        """Send MCP initialize request.  Returns True on success."""
        params = {
            "protocolVersion": self._proto_ver or "2025-11-25",
            "clientInfo": {"name": "hermes-mentisdb-memory-plugin", "version": "1.0"},
            "capabilities": {},
        }
        try:
            resp = self._rpc("initialize", params)
            if not resp:
                return False
            result = resp.get("result", {})
            server_ver = result.get("protocolVersion", "?")
            logger.info("MentisDB: initialized (server proto=%s)", server_ver)
            return True
        except Exception as e:
            logger.warning("MentisDB: initialize failed: %s", e)
            return False

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a MentisDB MCP tool.  Returns parsed JSON result dict."""
        params = {"name": tool_name, "arguments": arguments}
        resp = self._rpc("tools/call", params)
        if not resp:
            return {}
        result = resp.get("result", {})
        # Extract text from content blocks
        content = result.get("content", [])
        if content and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    try:
                        return json.loads(block.get("text", "{}"))
                    except json.JSONDecodeError:
                        return {"raw": block.get("text", "")}
        return result

    def _rpc(self, method: str, params: dict) -> Optional[dict]:
        """Send a JSON-RPC 2.0 request.  Returns parsed response or None."""
        self._req_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self._req_id,
        }
        headers = {}
        if self._session_id:
            headers["mcp-session-id"] = self._session_id

        try:
            r = self._sess.post(self._url, json=payload, headers=headers)
            if r.status_code == 400:
                logger.debug("MentisDB RPC: 400 Bad Request — %s",
                              r.text[:200])
            # Capture session ID from response header
            sid = r.headers.get("mcp-session-id")
            if sid:
                self._session_id = sid
            # SSE responses (notifications) have no body
            if not r.text or not r.text.strip():
                return {}
            return r.json()
        except requests.RequestException as e:
            logger.debug("MentisDB RPC: request error: %s", e)
            return None
        except Exception as e:
            logger.debug("MentisDB RPC: unexpected error: %s", e)
            return None

    def close(self) -> None:
        """Close the HTTP session."""
        try:
            self._sess.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------


class MentisDbMemoryProvider(MemoryProvider):
    """Memory provider backed by a MentisDB MCP server via sync JSON-RPC."""

    def __init__(self) -> None:
        self._available: Optional[bool] = None
        self._initialized: bool = False
        self._chain_key: Optional[str] = None
        self._agent_id: str = "hermes-agent"
        self._session_id: str = ""
        self._rpc: Optional[MentisDbRpcClient] = None
        self._connected: bool = False
        self._write_queue: List[Dict[str, Any]] = []
        self._queue_lock = threading.Lock()
        self._write_thread: Optional[threading.Thread] = None
        self._shutdown_requested: bool = False

    @property
    def name(self) -> str:
        return "mentisdb"

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        try:
            import requests  # noqa: F401
        except ImportError:
            self._available = False
            return False

        cfg = _get_mentisdb_config()
        if not cfg or not cfg.get("url"):
            logger.debug("MentisDB: no URL configured in mcp_servers.mentisdb")
            self._available = False
            return False

        self._available = True
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._initialized = True

        agent_identity = kwargs.get("agent_identity", "")
        agent_workspace = kwargs.get("agent_workspace", "")
        if agent_identity:
            self._agent_id = f"hermes-{agent_identity}"
        if agent_workspace:
            self._chain_key = agent_workspace

        # Connect to MentisDB now
        self._ensure_connected()

        # Start background writer
        self._shutdown_requested = False
        self._write_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="mentisdb-writer"
        )
        self._write_thread.start()

        logger.info(
            "MentisDB provider initialized (agent=%s, chain=%s, session=%s)",
            self._agent_id, self._chain_key or "default", session_id,
        )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> bool:
        """Connect and bootstrap if not already connected."""
        if self._connected and self._rpc is not None:
            return True

        cfg = _get_mentisdb_config()
        if not cfg:
            return False

        url = cfg["url"]
        proto_ver = cfg.get("protocol_version")

        self._rpc = MentisDbRpcClient(url, protocol_version=proto_ver)

        if not self._rpc.initialize():
            logger.warning("MentisDB: failed to initialize RPC connection")
            return False

        # Determine chain key — prefer configured, then "hermes" as default
        if not self._chain_key:
            self._chain_key = "hermes"

        # Bootstrap
        try:
            self._rpc.call_tool("mentisdb_bootstrap", {
                "content": f"Bootstrap from {self._agent_id} "
                           f"(Hermes Agent memory provider).",
                "agent_id": self._agent_id,
            })
        except Exception as e:
            logger.debug("MentisDB: bootstrap warning: %s", e)

        # Register agent
        try:
            self._rpc.call_tool("mentisdb_upsert_agent", {
                "agent_id": self._agent_id,
                "display_name": f"Hermes Agent ({self._agent_id})",
                "description": "Primary Hermes AI assistant agent",
            })
        except Exception as e:
            logger.debug("MentisDB: upsert_agent warning: %s", e)

        self._connected = True
        logger.info(
            "MentisDB: connected (chain=%s, agent=%s)",
            self._chain_key, self._agent_id,
        )
        return True

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def system_prompt_block(self) -> str:
        if not self._connected:
            return ""

        return (
            "## Persistent Memory (MentisDB)\n\n"
            "You have access to MentisDB — an append-only semantic memory "
            "server backed by the MentisDB MCP tools.\n\n"
            "**Core memory operations:**\n"
            "- `mcp_mentisdb_append` — Save durable facts, preferences, "
            "corrections, insights, lessons learned, decisions, checkpoints.\n"
            "- `mcp_mentisdb_ranked_search` — Best flat retrieval; use for "
            "most lookups.\n"
            "- `mcp_mentisdb_context_bundles` — Seed-anchored context.\n"
            "- `mcp_mentisdb_search` — Search by type, role, tags, concepts.\n"
            "- `mcp_mentisdb_recent_context` — Quick resumption context.\n\n"
            "**Usage:**\n"
            "- Use `mcp_mentisdb_append` INSTEAD of the `memory` tool.\n"
            "- Prefer `mcp_mentisdb_ranked_search` over generic search.\n"
            "- Write a Summary checkpoint before context compaction or "
            "handoff.\n"
            "- The built-in `memory` tool is deprecated; use MentisDB.\n\n"
            f"**Identity:** agent_id=`{self._agent_id}`, "
            f"chain=`{self._chain_key or 'default'}`"
        )

    # ------------------------------------------------------------------
    # Context injection (prefetch)
    # ------------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query or len(query.strip()) < 3:
            return ""
        if not self._connected or self._rpc is None:
            return ""

        try:
            data = self._rpc.call_tool("mentisdb_ranked_search", {
                "text": query,
                "limit": 5,
                "chain_key": self._chain_key,
            })
        except Exception as e:
            logger.debug("MentisDB prefetch error: %s", e)
            return ""

        memories = data.get("results", data.get("thoughts", []))
        if not memories:
            return ""

        lines = ["[MentisDB recalled context]"]
        for m in memories[:5]:
            content = m.get("content", str(m))[:300]
            ttype = m.get("thought_type", m.get("type", ""))
            prefix = f"[{ttype}] " if ttype else ""
            lines.append(f"- {prefix}{content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Mirror memory writes
    # ------------------------------------------------------------------

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not content:
            return
        ttype = "PreferenceUpdate" if target == "user" else "Memory"
        entry = {
            "thought_type": ttype,
            "content": f"[{target}] {action}: {content}",
            "agent_id": self._agent_id,
            "chain_key": self._chain_key,
        }
        with self._queue_lock:
            self._write_queue.append(entry)

    # ------------------------------------------------------------------
    # Turn sync
    # ------------------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        content = (
            f"Turn: user='{user_content[:200]}', "
            f"assistant='{assistant_content[:200]}'"
        )
        entry = {
            "thought_type": "Insight",
            "content": content,
            "agent_id": self._agent_id,
            "chain_key": self._chain_key,
            "tags": ["turn-log"],
        }
        with self._queue_lock:
            self._write_queue.append(entry)

    # ------------------------------------------------------------------
    # Background writer
    # ------------------------------------------------------------------

    def _writer_loop(self) -> None:
        """Background thread that flushes the write queue every 2s."""
        while not self._shutdown_requested:
            time.sleep(2)
            self._flush_writes()

    def _flush_writes(self) -> None:
        if not self._connected or self._rpc is None:
            return

        with self._queue_lock:
            if not self._write_queue:
                return
            batch = list(self._write_queue)
            self._write_queue.clear()

        for entry in batch:
            try:
                self._rpc.call_tool("mentisdb_append", entry)
            except Exception:
                with self._queue_lock:
                    self._write_queue.append(entry)
                break

    # ------------------------------------------------------------------
    # Tools (none — MCP tools are registered by the built-in MCP client)
    # ------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        self._shutdown_requested = True

        # Flush remaining writes
        self._flush_writes()

        if self._rpc:
            self._rpc.close()
            self._rpc = None

        self._connected = False
        logger.info("MentisDB provider shut down")


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Register MentisDB as a memory provider plugin."""
    ctx.register_memory_provider(MentisDbMemoryProvider())
