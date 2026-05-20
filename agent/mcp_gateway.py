"""MCPGateway — session-scoped credential injection for MCP tool calls.

Plan 002-C: MCP Connection Pool (MCPGateway).

Architecture:
  - mcp_tool.py already manages a process pool (_servers dict + background
    event loop).  Each MCP server process is spawned once and shared by all
    sessions.  This gateway does NOT rewrite that — it adds an auth middleware
    layer on top.

  - MCPGateway is a process-wide singleton (thread-safe via double-checked
    locking).  It holds one CredentialResolver and delegates actual tool
    dispatch to mcp_tool.call_tool_with_credential().

  - Per-session credential injection happens at the CALL level, not at process
    startup.  This preserves the N-users, M-servers = M-processes invariant.

  - Access control: a session with no credential for a server raises
    MCPAccessDenied rather than failing at the tool level with an opaque error.
    This lets the agent surface a clear message to the user.

Open questions resolved (per 002-hermes-self-organization.md Q-C.1–3):
  Q-C.1: Same-process for local (current); sidecar for Fargate (future).
  Q-C.2: refresh() on first 401 before propagating.
  Q-C.3: stdio servers without per-call auth → spawn dedicated process
          (current behavior unchanged — _server_requires_auth returns False
          for all servers until user credential binding files are populated).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from agent.credential_resolver import CredentialResolver

logger = logging.getLogger(__name__)


class MCPAccessDenied(Exception):
    """Raised when a session has no valid credential for an MCP server.

    Attributes:
        session_id: The session that made the call.
        server_name: The MCP server name for which no credential was found.
        user_id: The user identity of the session.
    """

    def __init__(self, message: str, *, session_id: str = "", server_name: str = "", user_id: str = ""):
        super().__init__(message)
        self.session_id = session_id
        self.server_name = server_name
        self.user_id = user_id


class MCPGateway:
    """Process-wide singleton: one MCP process pool, per-session credential injection.

    The underlying mcp_tool.py already manages the process pool (one process
    per server, all sessions share it).  This layer adds:

    1. Per-user credential resolution via CredentialResolver
    2. Access control: no credential → MCPAccessDenied (for servers that require auth)
    3. Credential refresh on transient 401 responses (try once before failing)

    Usage::

        gw = MCPGateway.get()
        result = gw.call_tool(
            session_id="20250518_abc123",
            user_id="blake",
            server_name="gmail",
            tool_name="search_emails",
            args={"query": "from:boss"},
        )
    """

    _instance: Optional["MCPGateway"] = None
    _class_lock = threading.Lock()

    @classmethod
    def get(cls) -> "MCPGateway":
        """Return the process-wide singleton, creating it if necessary."""
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._resolver = CredentialResolver()

    def call_tool(
        self,
        session_id: str,
        user_id: str,
        server_name: str,
        tool_name: str,
        args: Dict[str, Any],
    ) -> Any:
        """Route a tool call through the MCP pool with credential injection.

        Args:
            session_id: Caller's session ID (used in error messages).
            user_id: Caller's user identity (used for credential lookup).
            server_name: MCP server name as defined in config.yaml or
                         system/mcp-servers.json.
            tool_name: Tool name within the server.
            args: Tool arguments dict.

        Returns:
            The raw result from mcp_tool.call_tool_with_credential().

        Raises:
            MCPAccessDenied: When _server_requires_auth() is True and no
                             credential could be resolved.
        """
        from tools.mcp_tool import call_tool_with_credential  # avoid circular at import

        credential = self._resolver.resolve(user_id, server_name)

        if credential is None and self._server_requires_auth(server_name):
            raise MCPAccessDenied(
                f"Session {session_id!r} (user={user_id!r}) has no credential "
                f"for MCP server {server_name!r}. "
                f"Add a ref file at "
                f"~/.hermes/users/{user_id}/credentials/{server_name}.ref",
                session_id=session_id,
                server_name=server_name,
                user_id=user_id,
            )

        try:
            return call_tool_with_credential(
                server_name=server_name,
                tool_name=tool_name,
                args=args,
                credential=credential,
            )
        except Exception as exc:
            exc_str = str(exc)
            if "401" in exc_str or "Unauthorized" in exc_str or "unauthorized" in exc_str:
                logger.info(
                    "MCPGateway: got auth error for %s/%s (session=%s), refreshing credential",
                    server_name,
                    tool_name,
                    session_id,
                )
                refreshed = self._resolver.refresh(user_id, server_name)
                # Retry exactly once with the refreshed credential
                return call_tool_with_credential(
                    server_name=server_name,
                    tool_name=tool_name,
                    args=args,
                    credential=refreshed,
                )
            raise

    def _server_requires_auth(self, server_name: str) -> bool:
        """Return True if this MCP server requires a per-user credential.

        Phase C stub: currently returns False for all servers, meaning
        calls proceed without a credential (open access to all configured
        MCP servers).  This is the safe default for local single-user
        operation.

        Future: read from system/mcp-servers.json to determine which servers
        have credential bindings. When a server has `"requires_auth": true`
        in its config, this returns True and sessions without a matching
        .ref file receive MCPAccessDenied.
        """
        return False

    def clear_session(self, user_id: str, server_name: Optional[str] = None) -> None:
        """Shred cached credentials for a user.

        Called on session close.  If server_name is None, clears all cached
        credentials for the user (brute-force clear of the resolver cache).
        """
        if server_name:
            self._resolver.refresh(user_id, server_name)
        else:
            # Shred all cache entries — clear() doesn't filter by user_id
            # but that's acceptable: values will be re-resolved on next access
            self._resolver.clear()
