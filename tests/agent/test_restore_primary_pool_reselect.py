# Copyright 2025 Nous Research (Licensed under the Apache License, Version 2.0)
"""Test that _restore_primary_runtime re-selects from the credential pool
instead of using a stale snapshot key.

Bug: when a credential pool entry is revoked/marked-exhausted during a turn,
_restore_primary_runtime restores the original (now-stale) api_key from the
construction-time snapshot. The next turn immediately hits the same error,
exhausting remaining entries and falling through to cross-provider fallback.
"""

import json
import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    PooledCredential,
)


def _make_entry(
    label: str,
    access_token: str,
    *,
    source: str = "device_code",
    priority: int = 0,
    last_status: str | None = None,
    last_status_at: float | None = None,
) -> dict:
    return {
        "id": label,
        "label": label,
        "provider": "openai-codex",
        "auth_type": AUTH_TYPE_OAUTH,
        "source": source,
        "priority": priority,
        "access_token": access_token,
        "refresh_token": f"rt-{label}",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "last_status": last_status,
        "last_status_at": last_status_at,
    }


def _build_mock_pool(entries: list[dict], *, strategy: str = "round_robin"):
    """Build a mock CredentialPool with the given entries."""
    from agent.credential_pool import CredentialPool

    pool = CredentialPool(
        provider="openai-codex",
        entries=[PooledCredential.from_dict("openai-codex", e) for e in entries],
    )
    pool._strategy = strategy
    return pool


class TestRestorePrimaryPoolReselect:
    """_restore_primary_runtime should re-select from the credential pool."""

    def _make_agent(self, pool):
        """Create a minimal AIAgent with the given credential pool."""
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        agent.model = "gpt-5.5"
        agent.provider = "openai-codex"
        agent.base_url = "https://chatgpt.com/backend-api/codex"
        agent.api_mode = "codex_responses"
        agent.api_key = "original-key-entry-1"
        agent._client_kwargs = {
            "api_key": "original-key-entry-1",
            "base_url": "https://chatgpt.com/backend-api/codex",
        }
        agent._credential_pool = pool
        agent._fallback_activated = True
        agent._fallback_index = 1
        agent._rate_limited_until = 0
        agent._use_prompt_caching = False
        agent._use_native_cache_layout = False
        agent.context_compressor = MagicMock()
        agent.context_compressor.update_model = MagicMock()

        # Snapshot the original state
        agent._primary_runtime = {
            "model": "gpt-5.5",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
            "api_key": "original-key-entry-1",
            "client_kwargs": {
                "api_key": "original-key-entry-1",
                "base_url": "https://chatgpt.com/backend-api/codex",
            },
            "use_prompt_caching": False,
            "use_native_cache_layout": False,
            "compressor_model": "gpt-5.5",
            "compressor_base_url": "https://chatgpt.com/backend-api/codex",
            "compressor_api_key": "original-key-entry-1",
            "compressor_provider": "openai-codex",
            "compressor_context_length": 128000,
            "compressor_threshold_tokens": 0.8,
        }

        # Mock client creation methods
        agent._create_openai_client = MagicMock(return_value=MagicMock())
        agent._apply_client_headers_for_base_url = MagicMock()
        agent._replace_primary_openai_client = MagicMock(return_value=True)

        return agent

    def test_restore_reselects_from_pool_after_rotation(self):
        """After pool rotation, restore should use the new entry, not the stale snapshot key."""
        entries = [
            _make_entry("entry-1", "original-key-entry-1", priority=0),
            _make_entry("entry-2", "rotated-key-entry-2", priority=1),
            _make_entry("entry-3", "fresh-key-entry-3", priority=2),
        ]
        pool = _build_mock_pool(entries)

        # Simulate: entry-1 was exhausted, pool rotated to entry-2
        exhausted = pool._entries[0]
        pool._mark_exhausted(exhausted, 401)
        pool._current_id = "entry-2"

        agent = self._make_agent(pool)
        result = agent._restore_primary_runtime()

        assert result is True
        # The agent should have the NEW key from entry-2, not the stale snapshot key
        assert agent.api_key == "rotated-key-entry-2"
        assert agent._client_kwargs["api_key"] == "rotated-key-entry-2"

    def test_restore_uses_freshest_available_entry(self):
        """When multiple entries are available, restore should select the pool's best pick."""
        entries = [
            _make_entry("entry-1", "key-1", priority=0,
                         last_status="exhausted", last_status_at=time.time() + 3600),
            _make_entry("entry-2", "key-2", priority=1),
            _make_entry("entry-3", "key-3", priority=2),
        ]
        pool = _build_mock_pool(entries)

        agent = self._make_agent(pool)
        result = agent._restore_primary_runtime()

        assert result is True
        # entry-1 is exhausted, so pool should select entry-2
        assert agent.api_key == "key-2"
        assert agent._client_kwargs["api_key"] == "key-2"

    def test_restore_without_pool_uses_snapshot(self):
        """When no pool exists, restore should use the snapshot key (existing behavior)."""
        agent = self._make_agent(pool=None)
        result = agent._restore_primary_runtime()

        assert result is True
        assert agent.api_key == "original-key-entry-1"

    def test_restore_with_empty_pool_uses_snapshot(self):
        """When pool exists but has no available entries, use snapshot key."""
        entries = [
            _make_entry("entry-1", "key-1", priority=0,
                         last_status="exhausted", last_status_at=time.time() + 3600),
        ]
        pool = _build_mock_pool(entries)

        agent = self._make_agent(pool)
        result = agent._restore_primary_runtime()

        assert result is True
        # Pool has no available entries, so fall back to snapshot key
        assert agent.api_key == "original-key-entry-1"

    def test_restore_rebuilds_client_after_reselect(self):
        """After re-selecting from pool, client should be rebuilt with new key."""
        entries = [
            _make_entry("entry-1", "key-1", priority=0),
        ]
        pool = _build_mock_pool(entries)

        agent = self._make_agent(pool)
        result = agent._restore_primary_runtime()

        assert result is True
        # _swap_credential rebuilds the live OpenAI client with the fresh key.
        agent._replace_primary_openai_client.assert_called_once_with(
            reason="credential_rotation",
        )

    def test_restore_skips_reselect_if_entry_has_no_key(self):
        """If pool entry has an empty access token, fall back to snapshot key."""
        entries = [
            _make_entry("entry-1", "", priority=0),
        ]
        pool = _build_mock_pool(entries)

        agent = self._make_agent(pool)
        result = agent._restore_primary_runtime()

        assert result is True
        # Entry has no key, so use snapshot
        assert agent.api_key == "original-key-entry-1"

    def test_restore_updates_base_url_from_pool_entry(self):
        """If pool entry has a different base_url, restore should update it."""
        entries = [
            {
                **_make_entry("entry-1", "key-1", priority=0),
                "base_url": "https://custom-endpoint.example.com/v1",
            },
        ]
        pool = _build_mock_pool(entries)

        agent = self._make_agent(pool)
        result = agent._restore_primary_runtime()

        assert result is True
        assert "custom-endpoint.example.com" in agent.base_url

    def test_restore_skips_pool_reselect_when_provider_mismatch(self):
        """When the pool belongs to a fallback provider, skip re-selection.

        _try_activate_fallback swaps _credential_pool to the fallback
        provider's pool.  If restore_primary_runtime re-selects from that
        pool, _swap_credential overwrites the just-restored primary base_url
        with the fallback provider's URL, causing 404s (#56885).
        """
        from agent.credential_pool import CredentialPool

        # Build a deepseek pool (different provider from the primary openai-codex)
        deepseek_entries = [
            PooledCredential.from_dict("deepseek", {
                "id": "ds-1",
                "label": "ds-1",
                "auth_type": AUTH_TYPE_OAUTH,
                "source": "device_code",
                "priority": 0,
                "access_token": "deepseek-key",
                "base_url": "https://api.deepseek.com/v1",
            }),
        ]
        pool = CredentialPool(provider="deepseek", entries=deepseek_entries)

        agent = self._make_agent(pool)
        # Verify the agent's primary is openai-codex
        assert agent.provider == "openai-codex"
        assert agent._credential_pool.provider == "deepseek"

        result = agent._restore_primary_runtime()

        assert result is True
        # base_url must stay as the primary's, NOT deepseek's
        assert agent.base_url == "https://chatgpt.com/backend-api/codex"
        assert agent.api_key == "original-key-entry-1"
        # _swap_credential should NOT have been called (no client rebuild)
        agent._replace_primary_openai_client.assert_not_called()
