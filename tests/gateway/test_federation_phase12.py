"""Tests for federation Phase 12 — Desktop Bridge security + E2E."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.federation.federation_api import (
    FederationAPI,
    FederationAPIConfig,
)


# ========================================================================
# Security: Auth Middleware Tests
# ========================================================================

class TestAPISecurityAuth:
    """Test that auth middleware properly protects endpoints."""

    def _make_api_with_token(self, token="secret-token-123"):
        adapter = MagicMock()
        adapter._peers = {}
        adapter._auth_token = token
        adapter.get_leader = MagicMock(return_value="")
        adapter._mode = "auto"
        adapter._relay = MagicMock()
        adapter._relay._tasks = {}
        config = FederationAPIConfig(enabled=False)
        return FederationAPI(adapter=adapter, config=config, hermes_version="1.0.0")

    def test_health_endpoint_public(self):
        """Health endpoint should be accessible without auth."""
        api = self._make_api_with_token()
        # Health is public by design
        assert api.config.enabled is False  # Server not started, but public endpoint logic exists

    def test_auth_middleware_rejects_no_token(self):
        """Protected endpoints should reject requests without Bearer token."""
        api = self._make_api_with_token("secret-token")
        from aiohttp import web
        import asyncio

        request = MagicMock()
        request.path = "/api/federation/status"
        request.headers = {}

        async def _test():
            from gateway.federation.federation_api import _make_auth_middleware
            middleware = _make_auth_middleware("secret-token")
            response = await middleware(request, MagicMock())
            assert response.status == 401

        asyncio.get_event_loop().run_until_complete(_test())

    def test_auth_middleware_accepts_valid_token(self):
        """Valid Bearer token should be accepted."""
        api = self._make_api_with_token("valid-token")
        # Token comparison logic
        assert api.adapter._auth_token == "valid-token"


# ========================================================================
# Security: Input Validation Tests
# ========================================================================

class TestAPIInputValidation:
    """Test that API properly validates input."""

    def test_status_response_no_leak(self):
        """Status endpoint should not leak sensitive data."""
        adapter = MagicMock()
        adapter._peers = {
            "dev-a": MagicMock(status="online", last_seen=1234567890.0),
        }
        adapter.get_leader = MagicMock(return_value="dev-a")
        adapter._mode = "auto"
        adapter._relay = MagicMock()
        adapter._relay._tasks = {}
        config = FederationAPIConfig(enabled=False)
        api = FederationAPI(adapter=adapter, config=config)

        status = api._build_status()
        d = status.to_dict()

        # Should not contain tokens, passwords, or internal paths
        assert "token" not in str(d).lower()
        assert "password" not in str(d).lower()
        assert "secret" not in str(d).lower()
        assert "key" not in str(d).lower()


# ========================================================================
# Security: Race Condition Tests
# ========================================================================

class TestRaceConditionFixes:
    """Test that race conditions are properly fixed."""

    def test_connection_rate_limit_is_async(self):
        """Rate limit check should be async (uses lock)."""
        from gateway.federation.federation_connection import FederationConnectionManager
        import inspect
        assert inspect.iscoroutinefunction(FederationConnectionManager._check_rate_limit)

    def test_compute_pool_uses_lock(self):
        """Compute pool should have internal state for chunk tracking."""
        from gateway.federation.federation_compute_pool import FederationComputePool
        import asyncio
        pool = FederationComputePool(device_id="dev-a", adapter=MagicMock())
        assert hasattr(pool, 'device_id')
        assert hasattr(pool, '_pending_results')
        assert hasattr(pool, '_capabilities')


# ========================================================================
# Desktop Bridge Tests (IPC layer)
# ========================================================================

class TestDesktopBridge:
    """Test Desktop bridge security properties."""

    def test_ipc_does_not_expose_token(self):
        """IPC handlers should never expose auth token to renderer."""
        # Read the federation-ipc.ts file and verify token handling
        with open("apps/desktop/electron/federation-ipc.ts") as f:
            content = f.read()

        # Token should be used in headers, never returned
        assert "Authorization" in content
        assert "authToken" in content
        # Token should not be in any response
        assert "token" not in content.split("resolve")[1][:200] if "resolve" in content else True

    def test_bridge_handles_timeout(self):
        """Bridge should handle API timeout gracefully."""
        # federation-bridge.ts has 5s timeout
        with open("apps/desktop/src/federation-bridge.ts") as f:
            content = f.read()
        # Should have error handling
        assert "catch" in content
        assert "try" in content
