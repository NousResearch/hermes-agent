"""Tests for correlation ID tracking in gateway requests."""
import asyncio

import pytest


class TestCorrelationIds:
    """Gateway requests should be tagged with correlation IDs."""

    def test_correlation_id_generated_on_request(self):
        """Each API request should generate a unique correlation ID."""
        from gateway.platforms.api_server import _get_correlation_id, _set_correlation_id

        # Before setting, should be None
        assert _get_correlation_id() is None

        # After setting
        _set_correlation_id("test-123")
        assert _get_correlation_id() == "test-123"

    def test_correlation_id_isolated_between_contexts(self):
        """Correlation IDs should not leak between concurrent requests."""
        from gateway.platforms.api_server import _get_correlation_id, _set_correlation_id

        results = {}

        async def request_a():
            _set_correlation_id("req-a-123")
            await asyncio.sleep(0.01)
            results["a"] = _get_correlation_id()

        async def request_b():
            _set_correlation_id("req-b-456")
            await asyncio.sleep(0.01)
            results["b"] = _get_correlation_id()

        async def main():
            await asyncio.gather(request_a(), request_b())

        asyncio.run(main())

        assert results["a"] == "req-a-123"
        assert results["b"] == "req-b-456"
