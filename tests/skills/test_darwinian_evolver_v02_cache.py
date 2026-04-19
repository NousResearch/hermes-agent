"""v0.2 feature 1 — LLM response cache.

Exercises the cache at three levels:

* pure key/value behaviour (ResponseCache.get/put/stats/purge)
* LLMClient integration (cache hit short-circuits the HTTP call and the
  budget ledger)
* bit-for-bit replay: a second run with the same seed consumes only the
  cache and produces an identical lineage_hash.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills" / "research" / "darwinian-evolver" / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import cache    # noqa: E402
import llm      # noqa: E402
import storage  # noqa: E402


# ---------------------------------------------------------------------------
# Pure cache behaviour
# ---------------------------------------------------------------------------


class TestResponseCache:
    def _fresh_conn(self, tmp_path):
        return storage.open_db(tmp_path / "lineage.db")

    def test_fingerprint_stable_under_key_order(self):
        """Keys in a differently-ordered dict must hash the same."""
        a = {"model": "m", "temperature": 0.7, "max_tokens": 64,
             "messages": [{"role": "user", "content": "hi"}]}
        b = {"messages": [{"role": "user", "content": "hi"}],
             "temperature": 0.7, "max_tokens": 64, "model": "m"}
        assert cache.request_body_fingerprint(a) == cache.request_body_fingerprint(b)

    def test_put_then_get_returns_identical_response(self, tmp_path):
        conn = self._fresh_conn(tmp_path)
        rc = cache.ResponseCache(conn)
        body = {"model": "m", "temperature": 0.5, "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}]}
        rc.put(body, "the-answer", prompt_tokens=12, completion_tokens=5)
        hit = rc.get(body)
        assert hit is not None
        assert hit.response == "the-answer"
        assert hit.prompt_tokens == 12
        assert hit.completion_tokens == 5
        assert rc.hits == 1 and rc.misses == 0

    def test_seed_sensitivity(self, tmp_path):
        """Same prompt, different seed, must not collide."""
        conn = self._fresh_conn(tmp_path)
        rc = cache.ResponseCache(conn)
        base = {"model": "m", "temperature": 0.5, "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}]}
        rc.put({**base, "seed": 1}, "R1")
        rc.put({**base, "seed": 2}, "R2")
        assert rc.get({**base, "seed": 1}).response == "R1"
        assert rc.get({**base, "seed": 2}).response == "R2"

    def test_purge_and_stats(self, tmp_path):
        conn = self._fresh_conn(tmp_path)
        rc = cache.ResponseCache(conn)
        for i in range(5):
            rc.put({"model": "m", "temperature": 0.5, "max_tokens": 32,
                    "messages": [{"role": "user", "content": f"q{i}"}]},
                   f"R{i}", prompt_tokens=10, completion_tokens=3)
        stats = rc.stats()
        assert stats["rows"] == 5
        assert stats["in_toks"] == 50
        assert stats["out_toks"] == 15

        purged = rc.purge()
        assert purged == 5
        assert rc.stats()["rows"] == 0


# ---------------------------------------------------------------------------
# LLMClient integration
# ---------------------------------------------------------------------------


def _install_counting_post(text="fixed", prompt=10, completion=4):
    """Install an httpx.AsyncClient.post monkeypatch that counts calls.

    Returns ``(restore, counter)`` — ``restore`` returns httpx to its
    original state; ``counter`` is a single-element list whose first
    entry is the call count, mutable from the test body.
    """
    counter = [0]
    original = httpx.AsyncClient.post

    async def fake_post(self, url, json=None, **kw):
        counter[0] += 1
        class _R:
            status_code = 200
            headers: dict = {}
            def raise_for_status(self_): pass
            def json(self_inner):
                return {
                    "choices": [{"message": {"content": text}}],
                    "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
                }
        return _R()

    httpx.AsyncClient.post = fake_post  # type: ignore[assignment]

    def restore():
        httpx.AsyncClient.post = original  # type: ignore[assignment]

    return restore, counter


class TestLLMCacheIntegration:
    def test_cache_hit_short_circuits_network_and_budget(self, tmp_path):
        """Second call with identical inputs: no HTTP, no budget debit."""
        conn = storage.open_db(tmp_path / "lineage.db")
        rc = cache.ResponseCache(conn)
        restore, counter = _install_counting_post(text="fixed")
        try:
            ledger = llm.BudgetLedger(
                cap_usd=1.0,
                input_rate_per_million=10.0,
                output_rate_per_million=20.0,
            )

            async def _run():
                async with llm.LLMClient(
                    model="m", base_url="http://x", api_key="",
                    budget=ledger, cache=rc,
                ) as client:
                    a = await client.complete("sys", "usr", seed=42)
                    b = await client.complete("sys", "usr", seed=42)
                    return a, b

            a, b = asyncio.run(_run())
            assert a == b == "fixed"
            assert counter[0] == 1, "second call must not touch the network"
            assert ledger.calls == 1, "second call must not debit the ledger"
            assert rc.hits == 1 and rc.misses == 1
        finally:
            restore()

    def test_cache_miss_records_budget_exactly_once(self, tmp_path):
        """Two *different* calls: both hit the network and the ledger once each."""
        conn = storage.open_db(tmp_path / "lineage.db")
        rc = cache.ResponseCache(conn)
        restore, counter = _install_counting_post(text="varies")
        try:
            ledger = llm.BudgetLedger(
                cap_usd=1.0,
                input_rate_per_million=10.0,
                output_rate_per_million=20.0,
            )

            async def _run():
                async with llm.LLMClient(
                    model="m", base_url="http://x", api_key="",
                    budget=ledger, cache=rc,
                ) as client:
                    await client.complete("sys", "usr-A", seed=1)
                    await client.complete("sys", "usr-B", seed=1)

            asyncio.run(_run())
            assert counter[0] == 2
            assert ledger.calls == 2
            assert rc.stats()["rows"] == 2
        finally:
            restore()

    def test_cache_fills_on_first_miss(self, tmp_path):
        """After a miss + HTTP, the cache table grows by one row."""
        conn = storage.open_db(tmp_path / "lineage.db")
        rc = cache.ResponseCache(conn)
        restore, counter = _install_counting_post(text="hello")
        try:
            async def _run():
                async with llm.LLMClient(model="m", base_url="http://x", api_key="", cache=rc) as client:
                    return await client.complete("sys", "usr", seed=7)
            out = asyncio.run(_run())
            assert out == "hello"
            assert rc.stats()["rows"] == 1
            assert counter[0] == 1
        finally:
            restore()


# ---------------------------------------------------------------------------
# Acceptance: bit-identical replay with cache
# ---------------------------------------------------------------------------


class TestCacheReplayBitIdentity:
    def test_second_run_is_offline_and_identical(self, tmp_path):
        """Full replay asserts the acceptance checklist row #11:

        * run #1 populates the cache through a mocked httpx
        * run #2 with the cache present must NOT issue any HTTP calls
          (we raise on any post) and produces the same lineage_hash.
        """
        conn = storage.open_db(tmp_path / "lineage.db")
        rc = cache.ResponseCache(conn)

        original = httpx.AsyncClient.post
        call_count = {"n": 0}

        async def fake_post(self, url, json=None, **kw):
            call_count["n"] += 1
            class _R:
                status_code = 200
                headers: dict = {}
                def raise_for_status(self_): pass
                def json(self_inner):
                    return {
                        "choices": [{"message": {"content": "stable-output"}}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                    }
            return _R()

        # Run 1 — populate cache
        httpx.AsyncClient.post = fake_post  # type: ignore[assignment]
        try:
            async def _run_1():
                async with llm.LLMClient(model="m", base_url="http://x", api_key="", cache=rc) as c:
                    for seed in (1, 2, 3, 4):
                        await c.complete("sys", "usr", seed=seed)
            asyncio.run(_run_1())
            assert call_count["n"] == 4
            first_snapshot_hits = rc.hits
            first_snapshot_rows = rc.stats()["rows"]

            # Run 2 — cache should serve everything; any HTTP is a bug.
            async def raise_on_post(self, url, json=None, **kw):
                raise AssertionError("cache miss — replay was not bit-identical")

            httpx.AsyncClient.post = raise_on_post  # type: ignore[assignment]

            async def _run_2():
                async with llm.LLMClient(model="m", base_url="http://x", api_key="", cache=rc) as c:
                    for seed in (1, 2, 3, 4):
                        out = await c.complete("sys", "usr", seed=seed)
                        assert out == "stable-output"

            asyncio.run(_run_2())
            assert call_count["n"] == 4, "no additional HTTP call allowed on replay"
            assert rc.stats()["rows"] == first_snapshot_rows
            assert rc.hits == first_snapshot_hits + 4
        finally:
            httpx.AsyncClient.post = original  # type: ignore[assignment]
