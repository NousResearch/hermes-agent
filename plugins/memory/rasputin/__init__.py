"""Rasputin memory provider scaffold for Hermes.

V1 scope from the planning docs:
- Rasputin is a derived retrieval and enrichment sidecar
- built-in Hermes memory plus Ryan Book/JSONL remain canonical
- no model-facing Rasputin tools in v1
- all failures must fail open so turns continue normally

This module intentionally scaffolds the provider without attempting a full
production implementation.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

from .client import RasputinClient, RasputinClientConfig
from .formatter import format_recall_block
from .mappers import build_memory_write_commit, build_turn_commit

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class RasputinProviderConfig:
    """Environment-backed provider config for the scaffold."""

    enabled: bool = True
    base_url: str = "http://127.0.0.1:7777"
    timeout_seconds: float = 8.0
    commit_timeout_seconds: float = 20.0
    source_namespace: str = "hermes"
    prefetch_limit: int = 8
    commit_importance_default: int = 60
    fail_open: bool = True

    @classmethod
    def from_env(cls) -> "RasputinProviderConfig":
        return cls(
            enabled=_env_bool("RASPUTIN_ENABLED", True),
            base_url=os.getenv("RASPUTIN_BASE_URL", cls.base_url).strip() or cls.base_url,
            timeout_seconds=_env_float("RASPUTIN_TIMEOUT_SECONDS", cls.timeout_seconds),
            commit_timeout_seconds=_env_float(
                "RASPUTIN_COMMIT_TIMEOUT_SECONDS",
                cls.commit_timeout_seconds,
            ),
            source_namespace=os.getenv("RASPUTIN_SOURCE_NAMESPACE", cls.source_namespace).strip() or cls.source_namespace,
            prefetch_limit=max(1, _env_int("RASPUTIN_PREFETCH_LIMIT", cls.prefetch_limit)),
            commit_importance_default=max(
                1,
                min(100, _env_int("RASPUTIN_COMMIT_IMPORTANCE_DEFAULT", cls.commit_importance_default)),
            ),
            fail_open=_env_bool("RASPUTIN_FAIL_OPEN", True),
        )


class RasputinMemoryProvider(MemoryProvider):
    """Derived-memory Rasputin provider scaffold.

    The implementation stays intentionally conservative: it wires up Hermes'
    provider lifecycle, performs best-effort search and commit calls, and keeps
    every path safe when Rasputin is unavailable.
    """

    def __init__(self) -> None:
        self._config = RasputinProviderConfig.from_env()
        self._client: Optional[RasputinClient] = None
        self._session_id = ""
        self._platform = "cli"
        self._agent_context = "primary"
        self._user_id = ""
        self._healthy = False
        self._prefetch_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._prefetch_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "rasputin"

    def is_available(self) -> bool:
        """Return True when the scaffold is enabled by config.

        Per the MemoryProvider contract, this does not make network calls.
        Actual server reachability is checked in initialize().
        """
        self._config = RasputinProviderConfig.from_env()
        return self._config.enabled and bool(self._config.base_url)

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize the provider for a session and run a best-effort health check."""
        self._config = RasputinProviderConfig.from_env()
        self._session_id = session_id
        self._platform = str(kwargs.get("platform") or "cli")
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        self._user_id = str(kwargs.get("user_id") or "")
        self._client = RasputinClient(
            RasputinClientConfig(
                base_url=self._config.base_url,
                timeout_seconds=self._config.timeout_seconds,
                commit_timeout_seconds=self._config.commit_timeout_seconds,
                fail_open=self._config.fail_open,
            )
        )

        # Fail-open by design: provider readiness should never break canonical memory.
        self._healthy = self._client.healthcheck()
        if self._healthy:
            logger.info("Rasputin provider initialized against %s", self._config.base_url)
        else:
            logger.warning(
                "Rasputin health check failed for %s; continuing in fail-open mode",
                self._config.base_url,
            )

    def system_prompt_block(self) -> str:
        if not self._config.enabled:
            return ""
        return (
            "# Rasputin Memory\n"
            "Derived retrieval sidecar only. Canonical memory remains Hermes built-ins "
            "plus Ryan Book/JSONL.\n"
            "V1 exposes no Rasputin tools to the model; recall and mirrored writes are "
            "best-effort and may be absent when Rasputin is unavailable."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return formatted recall, using cached results when available."""
        query = (query or "").strip()
        if not query or not self._client:
            return ""

        cache_key = session_id or self._session_id
        with self._prefetch_lock:
            cached = self._prefetch_cache.pop(cache_key, None)
        if cached is not None:
            return format_recall_block(cached, limit=self._config.prefetch_limit)

        results = self._client.search(query, limit=self._config.prefetch_limit)
        return format_recall_block(results, limit=self._config.prefetch_limit)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Start a best-effort background search for the next turn."""
        query = (query or "").strip()
        if not query or not self._client:
            return

        cache_key = session_id or self._session_id

        def _run() -> None:
            try:
                results = self._client.search(query, limit=self._config.prefetch_limit)
                with self._prefetch_lock:
                    self._prefetch_cache[cache_key] = results
            except Exception:
                logger.debug("Rasputin queue_prefetch failed", exc_info=True)

        threading.Thread(target=_run, daemon=True, name="rasputin-prefetch").start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Mirror a completed turn into Rasputin without blocking the response path."""
        if not self._client or self._agent_context != "primary":
            return
        if not (user_content or "").strip() and not (assistant_content or "").strip():
            return

        payload = build_turn_commit(
            user_content,
            assistant_content,
            session_id=session_id or self._session_id,
            platform=self._platform,
            agent_context=self._agent_context,
            user_id=self._user_id,
            namespace=self._config.source_namespace,
        )
        self._commit_best_effort(payload, label="turn")

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes while preserving canonical ownership."""
        if not self._client:
            return
        if action not in {"add", "replace"}:
            return
        if not (content or "").strip():
            return

        payload = build_memory_write_commit(
            action,
            target,
            content,
            namespace=self._config.source_namespace,
            session_id=self._session_id,
        )
        self._commit_best_effort(payload, label="memory-write")

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """V1 is context-only; no model-facing Rasputin tools are exposed."""
        return []

    def _commit_best_effort(self, payload: Dict[str, Any], *, label: str) -> None:
        """Fire a background commit and swallow failures.

        TODO:
        - add a bounded queue and flush on shutdown if reliability needs increase
        - consider retries once real traffic patterns are known
        """
        if not self._client:
            return

        def _run() -> None:
            try:
                ok = self._client.commit(payload)
                if not ok:
                    logger.debug("Rasputin %s commit skipped/failed (fail-open)", label)
            except Exception:
                logger.debug("Rasputin %s commit failed", label, exc_info=True)

        threading.Thread(target=_run, daemon=True, name=f"rasputin-{label}").start()


def register(ctx) -> None:
    """Register Rasputin as a Hermes memory provider plugin."""
    ctx.register_memory_provider(RasputinMemoryProvider())
