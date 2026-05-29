"""Semantic cache for the opt-in provider gateway.

Stores LLM response history in a local SQLite database to prevent redundant
API requests for identical conversation histories (hash-perfect matching).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from provider_gateway.usage_tracker import ProviderUsageTracker

logger = logging.getLogger(__name__)


class SemanticCache:
    """SQLite-backed cache for conversation histories."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        # Re-use the usage tracker's default db path resolver to keep profiles clean
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = ProviderUsageTracker().db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """Establish a concurrent-safe WAL connection to SQLite."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _init_schema(self) -> None:
        """Initialize the semantic cache schema idempotently."""
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_hash TEXT UNIQUE,
                    prompt_text TEXT,
                    response_text TEXT,
                    model TEXT,
                    provider TEXT,
                    created_at REAL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sc_hash ON semantic_cache(prompt_hash)"
            )
            conn.commit()
        finally:
            conn.close()

    def compute_hash(self, messages: list[dict[str, Any]]) -> str:
        """Compute a deterministic SHA-256 hash of conversation history."""
        # Sanitize messages defensively to only hash JSON-serializable keys
        clean_msgs = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            clean_msg = {
                "role": str(msg.get("role", "")),
                "content": msg.get("content") or "",
            }
            # Include tool calls if present to differentiate intents
            if "tool_calls" in msg:
                clean_msg["tool_calls"] = msg["tool_calls"]
            clean_msgs.append(clean_msg)

        serialized = json.dumps(clean_msgs, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get_cached_response(
        self,
        agent: Any,
        api_messages: list[dict[str, Any]],
    ) -> SimpleNamespace | None:
        """Look up the exact conversation history hash and return mock response on hit.

        Returns None on miss.
        """
        # Ensure gateway config is active and cache is enabled
        config = getattr(agent, "_provider_gateway_config", None)
        if config is None or not config.enabled:
            return None

        prompt_hash = self.compute_hash(api_messages)
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT response_text, model, provider 
                FROM semantic_cache 
                WHERE prompt_hash = ?
                """,
                (prompt_hash,),
            ).fetchone()
        except Exception as exc:
            logger.debug("Failed to query semantic cache: %s", exc)
            return None
        finally:
            conn.close()

        if row is None:
            return None

        response_text, model, provider = row
        logger.info(
            "Semantic Cache HIT! Replaying cached response for %s/%s",
            provider,
            model,
        )

        # Build mock response matching the native transport expectation
        mock_message = SimpleNamespace(
            role="assistant",
            content=response_text,
            tool_calls=None,
            reasoning_content=None,
        )
        mock_choice = SimpleNamespace(
            index=0,
            message=mock_message,
            finish_reason="stop",
        )
        return SimpleNamespace(
            id="cache-" + str(uuid.uuid4()),
            model=model,
            choices=[mock_choice],
            usage=None,
        )

    def set_cached_response(
        self,
        agent: Any,
        api_messages: list[dict[str, Any]],
        response_text: str,
    ) -> None:
        """Cache a successful response against the conversation history hash."""
        config = getattr(agent, "_provider_gateway_config", None)
        if config is None or not config.enabled:
            return

        # Do not cache empty or tool-call responses
        if not response_text or response_text.strip() == "":
            return

        prompt_hash = self.compute_hash(api_messages)
        model = getattr(agent, "model", "unknown")
        provider = getattr(agent, "provider", "unknown")
        prompt_text = ""
        try:
            # Reconstruct user prompt representation for human visibility in DB
            prompt_text = str(api_messages[-1].get("content", "")) if api_messages else ""
        except Exception:
            pass

        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO semantic_cache 
                (prompt_hash, prompt_text, response_text, model, provider, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (prompt_hash, prompt_text, response_text, model, provider, time.time()),
            )
            conn.commit()
            logger.debug("Cached successful response with hash: %s", prompt_hash)
        except Exception as exc:
            logger.debug("Failed to set semantic cache entry: %s", exc)
        finally:
            conn.close()
