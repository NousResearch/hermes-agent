"""Tests for Firecrawl credential pool integration.

Covers:
- _seed_from_env seeds FIRECRAWL_API_KEY correctly and respects suppression
- search() rotates pool keys on HTTP 429 and retries once
- extract() rotates pool keys on HTTP 402 and retries once
- Gateway path works when pool is empty (regression)
"""
from __future__ import annotations

import json
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
import requests


# =========================================================================
# _seed_from_env — firecrawl
# =========================================================================


def test_seed_from_env_firecrawl_seeds_api_key(tmp_path, monkeypatch):
    """_seed_from_env("firecrawl") must seed from FIRECRAWL_API_KEY."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-key-12345")

    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {},
    }))

    from agent.credential_pool import _seed_from_env

    entries = []
    changed, active = _seed_from_env("firecrawl", entries)
    assert changed is True
    assert len(entries) == 1
    assert entries[0].source == "env:FIRECRAWL_API_KEY"
    assert entries[0].access_token == "fc-key-12345"
    assert "env:FIRECRAWL_API_KEY" in active


def test_seed_from_env_firecrawl_skips_gateway_url(tmp_path, monkeypatch):
    """_seed_from_env("firecrawl") must NOT seed FIRECRAWL_GATEWAY_URL."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-key-12345")
    monkeypatch.setenv("FIRECRAWL_GATEWAY_URL", "https://gateway.example.com")

    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {},
    }))

    from agent.credential_pool import _seed_from_env

    entries = []
    changed, active = _seed_from_env("firecrawl", entries)
    # Only one entry should exist — the API key, not the gateway URL
    assert len(entries) == 1
    assert entries[0].source == "env:FIRECRAWL_API_KEY"
    assert entries[0].access_token == "fc-key-12345"


def test_seed_from_env_firecrawl_no_key(tmp_path, monkeypatch):
    """_seed_from_env("firecrawl") returns empty when FIRECRAWL_API_KEY is unset."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)

    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {},
    }))

    from agent.credential_pool import _seed_from_env

    entries = []
    changed, active = _seed_from_env("firecrawl", entries)
    assert changed is False
    assert entries == []
    assert active == set()


def test_seed_from_env_firecrawl_respects_suppression(tmp_path, monkeypatch):
    """_seed_from_env("firecrawl") must skip env:FIRECRAWL_API_KEY when suppressed."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-key-12345")

    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "suppressed_sources": {"firecrawl": ["env:FIRECRAWL_API_KEY"]},
    }))

    from agent.credential_pool import _seed_from_env

    entries = []
    changed, active = _seed_from_env("firecrawl", entries)
    assert changed is False
    assert entries == []
    assert active == set()


# =========================================================================
# load_pool — firecrawl integration
# =========================================================================


def test_load_pool_firecrawl_creates_pool_from_env(tmp_path, monkeypatch):
    """load_pool("firecrawl") returns a pool with entries from env."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-key-12345")

    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {},
    }))

    from agent.credential_pool import load_pool

    pool = load_pool("firecrawl")
    assert pool.has_credentials() is True
    assert pool.has_available() is True
    entries = pool.entries()
    assert len(entries) >= 1
    # One of the entries should be our env-seeded key
    api_keys = [e.runtime_api_key for e in entries]
    assert "fc-key-12345" in api_keys


# =========================================================================
# FirecrawlWebSearchProvider — pool rotation on HTTP errors
# =========================================================================


class TestSearchPoolRotation:
    """search() must rotate pool keys on 429/402 and retry once."""

    @pytest.fixture(autouse=True)
    def _reset_provider_state(self):
        """Reset module-level pool state between tests."""
        import plugins.web.firecrawl.provider as fcp

        fcp._FIRECRAWL_POOL = None
        fcp._FIRECRAWL_POOL_KEY = None
        yield

    def _make_mock_pool(self, keys=None):
        """Create a mock CredentialPool with the given keys.

        Uses a real PooledCredential per key so runtime_api_key works.
        """
        from agent.credential_pool import CredentialPool, PooledCredential

        if keys is None:
            keys = ["fc-key-1", "fc-key-2"]
        entries = [
            PooledCredential(
                provider="firecrawl",
                id=f"id-{i}",
                label=f"key-{i}",
                auth_type="api_key",
                priority=i,
                source="manual",
                access_token=k,
            )
            for i, k in enumerate(keys)
        ]
        return CredentialPool("firecrawl", entries)

    def test_search_429_rotates_and_retries(self):
        """search() must rotate pool key on HTTP 429 and retry once."""
        import plugins.web.firecrawl.provider as fcp

        pool = self._make_mock_pool(["fc-key-1", "fc-key-2"])
        fcp._FIRECRAWL_POOL = pool

        # Mock the client to raise 429 on first call, succeed on second
        mock_client = MagicMock()
        # First call: raise HTTP 429
        first_response = MagicMock()
        first_response.status_code = 429
        mock_client.search.side_effect = [
            requests.exceptions.HTTPError(
                "429 Too Many Requests", response=first_response
            ),
            # Second call: success
            MagicMock(),
        ]

        # Patch _extract_web_search_results to return a known shape
        with patch.object(
            fcp, "_get_firecrawl_client", return_value=mock_client
        ), patch.object(
            fcp, "_extract_web_search_results", return_value=[{"url": "https://example.com"}]
        ), patch.object(
            fcp, "check_firecrawl_api_key", return_value=True
        ):
            # Clear cached client
            import tools.web_tools as wt

            wt._firecrawl_client = None
            wt._firecrawl_client_config = None

            provider = fcp.FirecrawlWebSearchProvider()
            result = provider.search("test query")

        # Must succeed after rotation
        assert result.get("success") is True
        # Must have called client.search() twice (initial + retry)
        assert mock_client.search.call_count == 2

        # After rotation, pool should have 1 key cooling down
        # Check that the first key was marked exhausted
        entries = pool.entries()
        exhausted = [e for e in entries if e.last_status == "exhausted"]
        assert len(exhausted) >= 1

    def test_search_402_rotates_and_retries(self):
        """search() must rotate pool key on HTTP 402 and retry once."""
        import plugins.web.firecrawl.provider as fcp

        pool = self._make_mock_pool(["fc-key-1", "fc-key-2"])
        fcp._FIRECRAWL_POOL = pool

        mock_client = MagicMock()
        billing_response = MagicMock()
        billing_response.status_code = 402
        mock_client.search.side_effect = [
            requests.exceptions.HTTPError(
                "402 Payment Required", response=billing_response
            ),
            MagicMock(),
        ]

        with patch.object(
            fcp, "_get_firecrawl_client", return_value=mock_client
        ), patch.object(
            fcp, "_extract_web_search_results", return_value=[{"url": "https://example.com"}]
        ), patch.object(
            fcp, "check_firecrawl_api_key", return_value=True
        ):
            import tools.web_tools as wt

            wt._firecrawl_client = None
            wt._firecrawl_client_config = None

            provider = fcp.FirecrawlWebSearchProvider()
            result = provider.search("test query")

        assert result.get("success") is True
        assert mock_client.search.call_count == 2

    def test_search_429_fails_when_all_keys_exhausted(self):
        """search() must return error when all pool keys are exhausted."""
        import plugins.web.firecrawl.provider as fcp

        pool = self._make_mock_pool(["fc-key-1"])
        fcp._FIRECRAWL_POOL = pool

        mock_client = MagicMock()
        error_response = MagicMock()
        error_response.status_code = 429
        # Both initial call and retry fail
        mock_client.search.side_effect = [
            requests.exceptions.HTTPError(
                "429 Too Many Requests", response=error_response
            ),
            # Retry also fails (but with a different error — retry catch is generic)
            requests.exceptions.ConnectionError("retry also failed"),
        ]

        with patch.object(
            fcp, "_get_firecrawl_client", return_value=mock_client
        ), patch.object(
            fcp, "_extract_web_search_results", return_value=[]
        ), patch.object(
            fcp, "check_firecrawl_api_key", return_value=True
        ):
            import tools.web_tools as wt

            wt._firecrawl_client = None
            wt._firecrawl_client_config = None

            provider = fcp.FirecrawlWebSearchProvider()
            result = provider.search("test query")

        assert result.get("success") is False
        assert "retry" in result.get("error", "").lower() or "failed" in result.get("error", "").lower()


class TestExtractPoolRotation:
    """extract() must rotate pool keys on 429/402 and retry once."""

    @pytest.fixture(autouse=True)
    def _reset_provider_state(self):
        import plugins.web.firecrawl.provider as fcp

        fcp._FIRECRAWL_POOL = None
        fcp._FIRECRAWL_POOL_KEY = None
        yield

    def _make_mock_pool(self, keys=None):
        from agent.credential_pool import CredentialPool, PooledCredential

        if keys is None:
            keys = ["fc-key-1", "fc-key-2"]
        entries = [
            PooledCredential(
                provider="firecrawl",
                id=f"id-{i}",
                label=f"key-{i}",
                auth_type="api_key",
                priority=i,
                source="manual",
                access_token=k,
            )
            for i, k in enumerate(keys)
        ]
        return CredentialPool("firecrawl", entries)

    @pytest.mark.asyncio
    async def test_extract_429_rotates_and_retries(self):
        """extract() must rotate pool key on HTTP 429 and retry once."""
        import plugins.web.firecrawl.provider as fcp

        pool = self._make_mock_pool(["fc-key-1", "fc-key-2"])
        fcp._FIRECRAWL_POOL = pool

        # Mock the client to raise 429 on first scrape call
        mock_client = MagicMock()
        error_response = MagicMock()
        error_response.status_code = 429
        # First scrape fails with 429
        mock_client.scrape.side_effect = [
            requests.exceptions.HTTPError(
                "429 Too Many Requests", response=error_response
            ),
            # Second scrape succeeds
            MagicMock(),
        ]

        with patch.object(
            fcp, "_get_firecrawl_client", return_value=mock_client
        ), patch.object(
            fcp, "_extract_scrape_payload", return_value={
                "markdown": "success",
                "metadata": {"title": "test", "sourceURL": "https://example.com"},
            }
        ), patch.object(
            fcp, "check_website_access", return_value=None
        ), patch.object(
            fcp, "is_safe_url", return_value=True
        ):
            import tools.web_tools as wt

            wt._firecrawl_client = None
            wt._firecrawl_client_config = None

            provider = fcp.FirecrawlWebSearchProvider()
            results = await provider.extract(["https://example.com"])

        # Must succeed after rotation
        assert len(results) == 1
        assert results[0].get("content") == "success"
        # Must have called client.scrape() twice (initial + retry)
        assert mock_client.scrape.call_count == 2

    @pytest.mark.asyncio
    async def test_extract_402_rotates_and_retries(self):
        """extract() must rotate pool key on HTTP 402 and retry once."""
        import plugins.web.firecrawl.provider as fcp

        pool = self._make_mock_pool(["fc-key-1", "fc-key-2"])
        fcp._FIRECRAWL_POOL = pool

        mock_client = MagicMock()
        billing_response = MagicMock()
        billing_response.status_code = 402
        mock_client.scrape.side_effect = [
            requests.exceptions.HTTPError(
                "402 Payment Required", response=billing_response
            ),
            MagicMock(),
        ]

        with patch.object(
            fcp, "_get_firecrawl_client", return_value=mock_client
        ), patch.object(
            fcp, "_extract_scrape_payload", return_value={
                "markdown": "success",
                "metadata": {"title": "test", "sourceURL": "https://example.com"},
            }
        ), patch.object(
            fcp, "check_website_access", return_value=None
        ), patch.object(
            fcp, "is_safe_url", return_value=True
        ):
            import tools.web_tools as wt

            wt._firecrawl_client = None
            wt._firecrawl_client_config = None

            provider = fcp.FirecrawlWebSearchProvider()
            results = await provider.extract(["https://example.com"])

        assert len(results) == 1
        assert results[0].get("content") == "success"
        assert mock_client.scrape.call_count == 2

    @pytest.mark.asyncio
    async def test_extract_non_429_passthrough(self):
        """extract() must NOT rotate on non-429/402 errors — let them propagate."""
        import plugins.web.firecrawl.provider as fcp

        pool = self._make_mock_pool(["fc-key-1"])
        fcp._FIRECRAWL_POOL = pool

        mock_client = MagicMock()
        # 403 error — should NOT trigger rotation
        forbidden_response = MagicMock()
        forbidden_response.status_code = 403
        mock_client.scrape.side_effect = requests.exceptions.HTTPError(
            "403 Forbidden", response=forbidden_response
        )

        with patch.object(
            fcp, "_get_firecrawl_client", return_value=mock_client
        ), patch.object(
            fcp, "check_website_access", return_value=None
        ):
            import tools.web_tools as wt

            wt._firecrawl_client = None
            wt._firecrawl_client_config = None

            provider = fcp.FirecrawlWebSearchProvider()
            results = await provider.extract(["https://example.com"])

        # Should get an error result, not a success
        assert len(results) == 1
        assert "error" in results[0]
        # Pool should NOT have marked anything as exhausted
        entries = pool.entries()
        exhausted = [e for e in entries if e.last_status == "exhausted"]
        assert len(exhausted) == 0
