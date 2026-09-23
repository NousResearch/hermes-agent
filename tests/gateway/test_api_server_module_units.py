"""Module-level unit tests for the api_server adapter's own helpers and stores.

Byte-verbatim shard of ``tests/gateway/test_api_server.py`` (2K-law fracture,
#79951): the requirements gate, the outward error redactor, and the two
in-memory stores need neither an aiohttp app nor an adapter fixture, so they
live here while the endpoint suites stay in the original module.

Part of #79951, #78647
"""

import asyncio
from unittest.mock import patch

import pytest

from gateway.platforms.api_server import (
    ResponseStore,
    _IdempotencyCache,
    _redact_api_error_text,
    check_api_server_requirements,
)


# ---------------------------------------------------------------------------
# check_api_server_requirements
# ---------------------------------------------------------------------------


class TestCheckRequirements:

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", False)
    def test_returns_false_without_aiohttp(self):
        assert check_api_server_requirements() is False


# ---------------------------------------------------------------------------
# _redact_api_error_text — guards every outward error site (envelopes, SSE
# error events, cron-endpoint 500 bodies) that routes raw exception text to
# authenticated HTTP clients. #37733
# ---------------------------------------------------------------------------


class TestRedactApiErrorText:
    def test_masks_secret_value_but_preserves_structure(self):
        secret = "sk-api-server-leak-1234567890"
        out = _redact_api_error_text(Exception(f"auth failed OPENAI_API_KEY={secret}"))
        assert secret not in out
        assert "OPENAI_API_KEY=" in out

    def test_redacts_regardless_of_global_redaction_setting(self):
        # force=True must mask even when global redaction is disabled.
        secret = "sk-forced-redaction-0987654321"
        with patch("agent.redact._REDACT_ENABLED", False):
            out = _redact_api_error_text(Exception(f"boom AWS_SECRET_ACCESS_KEY={secret}"))
        assert secret not in out

    def test_limit_truncates_after_redaction(self):
        assert len(_redact_api_error_text("x" * 500, limit=50)) == 50


# ---------------------------------------------------------------------------
# ResponseStore
# ---------------------------------------------------------------------------


class TestResponseStore:
    def test_put_and_get(self):
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "hello"})
        assert store.get("resp_1") == {"output": "hello"}

    def test_get_missing_returns_none(self):
        store = ResponseStore(max_size=10)
        assert store.get("resp_missing") is None

    def test_lru_eviction(self):
        store = ResponseStore(max_size=3)
        store.put("resp_1", {"output": "one"})
        store.put("resp_2", {"output": "two"})
        store.put("resp_3", {"output": "three"})
        # Adding a 4th should evict resp_1
        store.put("resp_4", {"output": "four"})
        assert store.get("resp_1") is None
        assert store.get("resp_2") is not None
        assert len(store) == 3


    def test_delete_clears_conversation_mapping(self):
        """Deleting a response also removes conversation mappings that reference it."""
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "hello"})
        store.set_conversation("chat-a", "resp_1")
        assert store.get_conversation("chat-a") == "resp_1"
        store.delete("resp_1")
        assert store.get_conversation("chat-a") is None


# ---------------------------------------------------------------------------
# _IdempotencyCache
# ---------------------------------------------------------------------------


class TestIdempotencyCache:
    @pytest.mark.asyncio
    async def test_concurrent_same_key_and_fingerprint_runs_once(self):
        cache = _IdempotencyCache()
        gate = asyncio.Event()
        started = asyncio.Event()
        calls = 0

        async def compute():
            nonlocal calls
            calls += 1
            started.set()
            await gate.wait()
            return ("response", {"total_tokens": 1})

        first = asyncio.create_task(cache.get_or_set("idem-key", "fp-1", compute))
        second = asyncio.create_task(cache.get_or_set("idem-key", "fp-1", compute))

        await started.wait()
        assert calls == 1

        gate.set()
        first_result, second_result = await asyncio.gather(first, second)

        assert first_result == second_result == ("response", {"total_tokens": 1})
