"""Unified-memory plugin — context-only MemoryProvider for Paul's unified-memory service.

Wires Hermes to the unified-memory gateway running on loopback. Two
endpoints are used, both UNAUTHENTICATED loopback paths (no bearer token):

  READ:  GET  {endpoint}/api/memory/relevant?q=<query>&budget=1500
         Returns text/markdown — relevant context for the upcoming turn.
  WRITE: POST {endpoint}/api/memory/hooks/after-reply
         Fire-and-forget capture of each completed conversation turn.

The endpoint defaults to http://localhost:18790 and can be overridden via
the UNIFIED_MEMORY_ENDPOINT environment variable.

Writes carry channel="hermes" so they are distinguishable in the store from
other producers (desktop_session, uno, etc.). All HTTP is best-effort:
reads return "" on any error and writes are fire-and-forget on a background
thread — the conversation turn is never blocked or crashed by this provider.

Config via environment variable (profile-scoped via each profile's .env):
  UNIFIED_MEMORY_ENDPOINT  — gateway URL (default: http://localhost:18790)
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://localhost:18790"
_RELEVANT_PATH = "/api/memory/relevant"
_AFTER_REPLY_PATH = "/api/memory/hooks/after-reply"
_PREFETCH_BUDGET = 1500
_READ_TIMEOUT = 2.5
_WRITE_TIMEOUT = 2.5


def _get_httpx():
    """Lazy import httpx (a core hermes dependency)."""
    try:
        import httpx
        return httpx
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class UnifiedMemoryProvider(MemoryProvider):
    """Context-only memory via Paul's unified-memory loopback gateway."""

    def __init__(self):
        self._httpx = None
        self._endpoint = ""
        self._session_id = ""
        self._platform = "cli"
        self._cron_skipped = False
        self._sync_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "unified_memory"

    def is_available(self) -> bool:
        """Available whenever an endpoint is configured. No network calls.

        The endpoint always defaults to http://localhost:18790, so this is
        effectively always True once httpx (a core dependency) is importable.
        """
        import os
        return bool(os.environ.get("UNIFIED_MEMORY_ENDPOINT", _DEFAULT_ENDPOINT))

    def get_config_schema(self):
        return [
            {
                "key": "endpoint",
                "description": "unified-memory gateway URL",
                "required": False,
                "secret": False,
                "default": _DEFAULT_ENDPOINT,
                "env_var": "UNIFIED_MEMORY_ENDPOINT",
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        import os

        # Skip non-primary contexts (cron/flush system prompts would pollute
        # the user's live store). See MemoryProvider ABC docstring.
        agent_context = kwargs.get("agent_context", "")
        platform = kwargs.get("platform", "cli")
        if agent_context in {"cron", "flush"} or platform == "cron":
            logger.debug(
                "unified_memory skipped: cron/flush context (agent_context=%s, platform=%s)",
                agent_context, platform,
            )
            self._cron_skipped = True
            return

        self._endpoint = os.environ.get("UNIFIED_MEMORY_ENDPOINT", _DEFAULT_ENDPOINT).rstrip("/")
        self._session_id = session_id
        self._platform = platform

        self._httpx = _get_httpx()
        if self._httpx is None:
            logger.warning("httpx not installed — unified_memory plugin disabled")

    def _url(self, path: str) -> str:
        return f"{self._endpoint}{path}"

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """READ: fetch relevant context from unified-memory (bounded, never raises).

        Bounded synchronous GET against /api/memory/relevant. Returns the
        markdown body on a 200 with non-empty content; returns "" on any
        error, timeout, non-200, or empty body.
        """
        if self._cron_skipped or not self._httpx or not self._endpoint:
            return ""

        params: Dict[str, Any] = {"q": query, "budget": _PREFETCH_BUDGET}
        sid = session_id or self._session_id
        if sid:
            params["session_id"] = sid

        try:
            resp = self._httpx.get(
                self._url(_RELEVANT_PATH), params=params, timeout=_READ_TIMEOUT
            )
            if resp.status_code != 200:
                return ""
            body = resp.text or ""
            return body if body.strip() else ""
        except Exception as e:
            logger.debug("unified_memory prefetch failed: %s", e)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """WRITE: persist a completed turn to unified-memory (non-blocking).

        Fire-and-forget POST to /api/memory/hooks/after-reply on a daemon
        thread. Swallows all errors — never blocks or crashes the turn.
        """
        if self._cron_skipped or not self._httpx or not self._endpoint:
            return

        sid = session_id or self._session_id or "hermes"
        payload = {
            "event": {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "success": True,
                "durationMs": 0,
            },
            "hookCtx": {
                "sessionKey": sid,
                "channel": "hermes",
            },
        }

        def _sync():
            try:
                self._httpx.post(
                    self._url(_AFTER_REPLY_PATH), json=payload, timeout=_WRITE_TIMEOUT
                )
            except Exception as e:
                logger.debug("unified_memory sync_turn failed: %s", e)

        # Wait for any previous sync to finish before starting a new one.
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(
            target=_sync, daemon=True, name="unified-memory-sync"
        )
        self._sync_thread.start()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """No-op: every turn is already captured by sync_turn.

        unified-memory's after-reply hook receives each turn as it completes
        (see sync_turn). Re-POSTing the full message list here would double-
        write every turn into Paul's live store, since the hook's dedup
        contract is not guaranteed. We only flush the pending per-turn write.
        """
        if self._cron_skipped:
            return
        # Let the last per-turn write finish; don't re-send the conversation.
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Context-only provider — exposes no tools."""
        return []

    def shutdown(self) -> None:
        """Wait for the pending background write, then drop the client reference."""
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._httpx = None


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register unified-memory as a memory provider plugin."""
    ctx.register_memory_provider(UnifiedMemoryProvider())
