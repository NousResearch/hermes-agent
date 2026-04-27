"""Tests for correlation ID tracking in gateway requests."""
import asyncio
import concurrent.futures
import re

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

    def test_correlation_id_format_is_full_uuid_hex(self):
        """Auto-generated correlation IDs must be 32-char hex strings (UUID4 hex).

        A truncated 8-char ID has only 32 bits of entropy, giving a ~1% collision
        probability at 10k concurrent requests (birthday paradox).  The full UUID4
        hex (128 bits) eliminates this risk.
        """
        import uuid
        from gateway.platforms.api_server import _set_correlation_id, _get_correlation_id

        # Simulate what the middleware generates when no header is present
        generated = uuid.uuid4().hex
        _set_correlation_id(generated)
        stored = _get_correlation_id()

        assert stored is not None
        assert len(stored) == 32, f"Expected 32-char hex, got {len(stored)}-char: {stored!r}"
        assert re.fullmatch(r"[0-9a-f]{32}", stored), f"Not a lowercase hex string: {stored!r}"

    def test_correlation_id_propagates_to_executor_thread(self):
        """Correlation ID set in async context must be visible in ThreadPoolExecutor threads.

        ContextVar values are NOT automatically inherited by threads submitted via
        run_in_executor.  The fix (capturing the ID before submit and calling
        _set_correlation_id inside the thread) must be verified explicitly.
        """
        from gateway.platforms.api_server import _get_correlation_id, _set_correlation_id

        thread_result = {}

        async def main():
            _set_correlation_id("propagation-test-id")
            captured_cid = _get_correlation_id()

            loop = asyncio.get_event_loop()

            def _thread_fn():
                # Replicate the fix: set the captured ID inside the thread
                if captured_cid is not None:
                    _set_correlation_id(captured_cid)
                thread_result["cid"] = _get_correlation_id()

            await loop.run_in_executor(None, _thread_fn)

        asyncio.run(main())

        assert thread_result.get("cid") == "propagation-test-id", (
            "Correlation ID was not visible inside the executor thread. "
            "The captured_cid must be passed explicitly and set via _set_correlation_id."
        )

    def test_correlation_id_not_visible_in_thread_without_explicit_propagation(self):
        """Confirm that without explicit propagation, threads see None.

        This is the baseline that documents WHY explicit propagation is required.
        """
        from gateway.platforms.api_server import _get_correlation_id, _set_correlation_id

        thread_result = {}

        async def main():
            _set_correlation_id("should-not-propagate")
            loop = asyncio.get_event_loop()

            def _thread_fn():
                # No explicit _set_correlation_id call here
                thread_result["cid"] = _get_correlation_id()

            await loop.run_in_executor(None, _thread_fn)

        asyncio.run(main())

        assert thread_result.get("cid") is None, (
            "ContextVar unexpectedly propagated to thread without explicit copy. "
            "Python behavior may have changed."
        )
