"""Shared test fixtures and utilities for provider_gateway tests.

This module provides a standard `BaseMockAgent` class that satisfies all
attribute contracts expected by `chat_completion_helpers.py` and provider
gateway runtime hooks.  Using this base class eliminates the recurring
pattern of mock objects failing due to missing attributes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from provider_gateway.config import GatewayConfig
from provider_gateway.semantic_cache import SemanticCache


class FakeRequestClient:
    """Minimal OpenAI-compatible client for testing."""

    def __init__(self, responder) -> None:
        self._responder = responder
        self.closed = False
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        return self._responder(**kwargs)

    def close(self) -> None:
        self.closed = True


class BaseMockAgent:
    """Standard mock agent with all attributes required by provider_gateway.

    This centralised fixture prevents the recurring test failure pattern
    where mock agents are missing attributes like ``_primary_runtime``,
    ``_create_request_openai_client``, etc.
    """

    # Identity
    provider = "openrouter"
    model = "anthropic/claude-sonnet-4.6"
    base_url = "https://openrouter.ai/api/v1"
    api_key = "sk-test-key"
    api_mode = "chat_completions"
    session_id = "test-session"

    # Internal state used by chat_completion_helpers
    _interrupt_requested = False
    _base_url_lower = ""
    _base_url_hostname = ""
    _fallback_activated = False
    _fallback_index = 0

    # Primary runtime snapshot (read by try_activate_fallback)
    _primary_runtime = {
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet-4.6",
        "base_url": "https://openrouter.ai/api/v1",
    }

    # Fallback chain (read by policy.build_gateway_policy)
    _fallback_chain = [
        {
            "provider": "openrouter",
            "model": "openai/gpt-4o",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        },
    ]

    def __init__(
        self,
        response_or_exc=None,
        tracker=None,
        *,
        config: GatewayConfig | None = None,
        db_path: Path | None = None,
    ) -> None:
        self._response_or_exc = response_or_exc
        self._provider_gateway_config = config or GatewayConfig(
            enabled=True,
            track_usage=True,
            track_cost=False,
        )
        self._provider_usage_tracker = tracker

        # Isolated temp cache DB
        self._temp_dir = tempfile.TemporaryDirectory()
        _cache_path = db_path or (Path(self._temp_dir.name) / "test_cache.db")
        self._provider_semantic_cache = SemanticCache(db_path=_cache_path)

    def _create_request_openai_client(self, *, reason, api_kwargs):
        def responder(**kwargs):
            if isinstance(self._response_or_exc, Exception):
                raise self._response_or_exc
            return self._response_or_exc

        return FakeRequestClient(responder)

    def _close_request_openai_client(self, client, *, reason) -> None:
        client.close()

    def _abort_request_openai_client(self, client, *, reason) -> None:
        client.close()

    def _compute_non_stream_stale_timeout(self, api_payload) -> float:
        return 5.0

    def _touch_activity(self, message: str) -> None:
        pass


class CapturingTracker:
    """In-memory usage tracker that captures records for assertion."""

    def __init__(self, db_path=None) -> None:
        self.records: list = []
        self.db_path = db_path

    def record_usage(self, record):
        self.records.append(record)
        return len(self.records)

    def summarize_by_provider(self, *, since=None, until=None):
        return []
