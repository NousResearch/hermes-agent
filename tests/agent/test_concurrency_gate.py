"""Comprehensive tests for the per-model concurrency gate in agent.auxiliary_client.

The concurrency gate is entirely config-driven — limits come from config.yaml's
``concurrency_limits`` section with no hardcoded defaults.

Tests cover:
  - _load_concurrency_limits: config loading, validation, error handling, caching
  - _get_model_semaphore: lookup, caching, case normalisation, edge inputs
  - reset_concurrency_state: full cleanup
  - _async_acquire_semaphore: async/threaded interop
  - Serialisation correctness: blocking, release ordering, multi-slot models
  - Cross-model isolation: different models don't interfere
  - Thread safety: concurrent access to the semaphore registry
  - Integration with call_llm: acquire/release on success, error, max_tokens retry,
    payment fallback
  - Integration with async_call_llm: same, plus async-specific edge cases
  - Edge cases and stress tests
"""

import asyncio
import concurrent.futures
import threading
import time
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agent.auxiliary_client import (
    _get_model_semaphore,
    _async_acquire_semaphore,
    _load_concurrency_limits,
    reset_concurrency_state,
    call_llm,
    async_call_llm,
)


# ── Fixtures ────────────────────────────────────────────────────────────


# Sample config used by most tests — mimics a z.ai setup
_SAMPLE_CONFIG = {
    "concurrency_limits": {
        "glm-5-turbo": 1,
        "glm-5v-turbo": 1,
        "glm-5.1": 1,
        "glm-5": 2,
        "glm-4.5": 10,
        "glm-4-plus": 20,
    }
}


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset concurrency gate state before and after each test."""
    reset_concurrency_state()
    yield
    reset_concurrency_state()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip provider env vars so each test starts clean."""
    for key in (
        "OPENROUTER_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY",
        "OPENAI_MODEL", "LLM_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def sample_config():
    """Patch config to return the sample z.ai concurrency limits."""
    with patch("hermes_cli.config.load_config", return_value=_SAMPLE_CONFIG):
        yield _SAMPLE_CONFIG


def _mock_call_llm_patches(model, provider="zai"):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "ok"
    mock_client.chat.completions.create.return_value = mock_response
    resolve_patch = patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=(provider, model, None, None, None),
    )
    client_patch = patch(
        "agent.auxiliary_client._get_cached_client",
        return_value=(mock_client, model),
    )
    return resolve_patch, client_patch, mock_client, mock_response


def _mock_async_call_llm_patches(model, provider="zai"):
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "ok"
    mock_client.chat.completions.create.return_value = mock_response
    resolve_patch = patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=(provider, model, None, None, None),
    )
    client_patch = patch(
        "agent.auxiliary_client._get_cached_client",
        return_value=(mock_client, model),
    )
    return resolve_patch, client_patch, mock_client, mock_response


# ═══════════════════════════════════════════════════════════════════════
# Unit tests: _load_concurrency_limits
# ═══════════════════════════════════════════════════════════════════════


class TestLoadConcurrencyLimits:

    def test_loads_from_config(self):
        cfg = {"concurrency_limits": {"model-a": 3, "model-b": 1}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            limits = _load_concurrency_limits()
        assert limits == {"model-a": 3, "model-b": 1}

    def test_empty_config_returns_empty(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            assert _load_concurrency_limits() == {}

    def test_none_config_returns_empty(self):
        with patch("hermes_cli.config.load_config", return_value=None):
            assert _load_concurrency_limits() == {}

    def test_missing_section_returns_empty(self):
        with patch("hermes_cli.config.load_config", return_value={"model": "foo"}):
            assert _load_concurrency_limits() == {}

    def test_ignores_zero(self):
        cfg = {"concurrency_limits": {"m": 0}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert "m" not in _load_concurrency_limits()

    def test_ignores_negative(self):
        cfg = {"concurrency_limits": {"m": -5}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert "m" not in _load_concurrency_limits()

    def test_ignores_string(self):
        cfg = {"concurrency_limits": {"m": "ten"}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert "m" not in _load_concurrency_limits()

    def test_ignores_float(self):
        cfg = {"concurrency_limits": {"m": 2.5}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert "m" not in _load_concurrency_limits()

    def test_ignores_none_value(self):
        cfg = {"concurrency_limits": {"m": None}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert "m" not in _load_concurrency_limits()

    def test_valid_kept_invalid_skipped(self):
        cfg = {"concurrency_limits": {
            "good": 5, "bad-zero": 0, "bad-neg": -1,
            "bad-str": "x", "also-good": 1,
        }}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            limits = _load_concurrency_limits()
        assert limits == {"good": 5, "also-good": 1}

    def test_case_normalisation(self):
        cfg = {"concurrency_limits": {"MY-Model": 4}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert _load_concurrency_limits()["my-model"] == 4

    def test_whitespace_stripping(self):
        cfg = {"concurrency_limits": {"  spaced  ": 2}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert _load_concurrency_limits()["spaced"] == 2

    def test_config_exception_returns_empty(self):
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("oops")):
            assert _load_concurrency_limits() == {}

    def test_import_error_returns_empty(self):
        with patch("hermes_cli.config.load_config", side_effect=ImportError("missing")):
            assert _load_concurrency_limits() == {}

    def test_non_dict_section_returns_empty(self):
        cfg = {"concurrency_limits": "not-a-dict"}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert _load_concurrency_limits() == {}

    def test_list_section_returns_empty(self):
        cfg = {"concurrency_limits": [1, 2, 3]}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert _load_concurrency_limits() == {}

    def test_results_cached(self):
        call_count = {"n": 0}
        def counting_load():
            call_count["n"] += 1
            return {"concurrency_limits": {"m": 1}}

        with patch("hermes_cli.config.load_config", side_effect=counting_load):
            _load_concurrency_limits()
            _load_concurrency_limits()
            _load_concurrency_limits()
        assert call_count["n"] == 1

    def test_cache_cleared_by_reset(self):
        cfg1 = {"concurrency_limits": {"m": 3}}
        cfg2 = {"concurrency_limits": {"m": 8}}
        with patch("hermes_cli.config.load_config", return_value=cfg1):
            assert _load_concurrency_limits()["m"] == 3
        reset_concurrency_state()
        with patch("hermes_cli.config.load_config", return_value=cfg2):
            assert _load_concurrency_limits()["m"] == 8


# ═══════════════════════════════════════════════════════════════════════
# Unit tests: _get_model_semaphore
# ═══════════════════════════════════════════════════════════════════════


class TestGetModelSemaphore:

    def test_returns_semaphore_for_configured_model(self, sample_config):
        sem = _get_model_semaphore("glm-5-turbo")
        assert sem is not None
        assert isinstance(sem, threading.Semaphore)
        assert sem._value == 1

    def test_returns_none_for_unconfigured_model(self, sample_config):
        assert _get_model_semaphore("gpt-4o") is None

    def test_returns_none_when_no_config(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            assert _get_model_semaphore("glm-5-turbo") is None

    def test_returns_none_for_none(self, sample_config):
        assert _get_model_semaphore(None) is None

    def test_returns_none_for_empty_string(self, sample_config):
        assert _get_model_semaphore("") is None

    def test_returns_none_for_whitespace(self, sample_config):
        assert _get_model_semaphore("   ") is None

    def test_case_insensitive(self, sample_config):
        assert _get_model_semaphore("GLM-5-TURBO") is _get_model_semaphore("glm-5-turbo")

    def test_strips_whitespace(self, sample_config):
        assert _get_model_semaphore("  glm-5-turbo  ") is _get_model_semaphore("glm-5-turbo")

    def test_singleton(self, sample_config):
        refs = [_get_model_semaphore("glm-5-turbo") for _ in range(100)]
        assert all(r is refs[0] for r in refs)

    def test_different_models_different_semaphores(self, sample_config):
        s1 = _get_model_semaphore("glm-5-turbo")
        s2 = _get_model_semaphore("glm-5")
        s3 = _get_model_semaphore("glm-4-plus")
        assert len({id(s1), id(s2), id(s3)}) == 3

    @pytest.mark.parametrize("model,expected", [
        ("glm-5-turbo", 1), ("glm-5v-turbo", 1), ("glm-5.1", 1),
        ("glm-5", 2), ("glm-4.5", 10), ("glm-4-plus", 20),
    ])
    def test_value_matches_config(self, sample_config, model, expected):
        assert _get_model_semaphore(model)._value == expected

    def test_custom_model(self):
        cfg = {"concurrency_limits": {"my-local-llm": 4}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert _get_model_semaphore("my-local-llm")._value == 4

    def test_openrouter_style_name(self):
        cfg = {"concurrency_limits": {"google/gemini-3-flash": 5}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert _get_model_semaphore("google/gemini-3-flash")._value == 5

    def test_colon_style_name(self):
        cfg = {"concurrency_limits": {"anthropic:claude-3-opus": 2}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert _get_model_semaphore("anthropic:claude-3-opus")._value == 2


# ═══════════════════════════════════════════════════════════════════════
# Unit tests: reset_concurrency_state
# ═══════════════════════════════════════════════════════════════════════


class TestResetConcurrencyState:

    def test_clears_semaphores(self, sample_config):
        sem1 = _get_model_semaphore("glm-5-turbo")
        reset_concurrency_state()
        with patch("hermes_cli.config.load_config", return_value=_SAMPLE_CONFIG):
            assert _get_model_semaphore("glm-5-turbo") is not sem1

    def test_clears_limits_cache(self):
        with patch("hermes_cli.config.load_config", return_value={"concurrency_limits": {"m": 3}}):
            assert _get_model_semaphore("m")._value == 3
        reset_concurrency_state()
        with patch("hermes_cli.config.load_config", return_value={"concurrency_limits": {"m": 8}}):
            assert _get_model_semaphore("m")._value == 8

    def test_idempotent(self):
        reset_concurrency_state()
        reset_concurrency_state()


# ═══════════════════════════════════════════════════════════════════════
# Serialisation tests
# ═══════════════════════════════════════════════════════════════════════


class TestSerialization:

    def test_concurrency_1_blocks(self, sample_config):
        sem = _get_model_semaphore("glm-5-turbo")
        sem.acquire()
        blocked, acquired = threading.Event(), threading.Event()

        def _second():
            blocked.set()
            sem.acquire()
            acquired.set()
            sem.release()

        t = threading.Thread(target=_second, daemon=True)
        t.start()
        blocked.wait(timeout=1)
        time.sleep(0.05)
        assert not acquired.is_set()
        sem.release()
        acquired.wait(timeout=1)
        assert acquired.is_set()

    def test_concurrency_2_allows_two(self, sample_config):
        sem = _get_model_semaphore("glm-5")
        sem.acquire(); sem.acquire()
        assert not sem.acquire(blocking=False)
        sem.release(); sem.release()

    def test_concurrency_10_allows_batch(self, sample_config):
        sem = _get_model_semaphore("glm-4.5")
        for _ in range(10):
            assert sem.acquire(blocking=False)
        assert not sem.acquire(blocking=False)
        for _ in range(10):
            sem.release()

    def test_all_waiters_served(self, sample_config):
        sem = _get_model_semaphore("glm-5-turbo")
        sem.acquire()
        order = []
        barriers = [threading.Event() for _ in range(3)]

        def _waiter(idx):
            barriers[idx].set()
            sem.acquire()
            order.append(idx)
            sem.release()

        threads = []
        for i in range(3):
            t = threading.Thread(target=_waiter, args=(i,), daemon=True)
            t.start()
            barriers[i].wait(timeout=1)
            time.sleep(0.02)
            threads.append(t)

        for _ in range(3):
            sem.release()
            time.sleep(0.05)
        for t in threads:
            t.join(timeout=2)
        assert set(order) == {0, 1, 2}

    def test_mutex_behaviour(self, sample_config):
        sem = _get_model_semaphore("glm-5-turbo")
        sem.acquire()
        assert not sem.acquire(blocking=False)
        sem.release()
        assert sem.acquire(blocking=False)
        sem.release()


class TestCrossModelIsolation:

    def test_different_models_independent(self, sample_config):
        s1 = _get_model_semaphore("glm-5-turbo")
        s2 = _get_model_semaphore("glm-4.5")
        s1.acquire()
        assert s2.acquire(blocking=False)
        s1.release(); s2.release()

    def test_ungated_model_returns_none(self, sample_config):
        assert _get_model_semaphore("gpt-4o") is None


# ═══════════════════════════════════════════════════════════════════════
# Thread safety
# ═══════════════════════════════════════════════════════════════════════


class TestThreadSafety:

    def test_concurrent_same_model(self, sample_config):
        results = [None] * 50
        def _get(idx): results[idx] = _get_model_semaphore("glm-5-turbo")
        threads = [threading.Thread(target=_get, args=(i,)) for i in range(50)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=5)
        assert all(r is results[0] for r in results)

    def test_concurrent_different_models(self, sample_config):
        models = list(_SAMPLE_CONFIG["concurrency_limits"].keys())
        results = {}
        lock = threading.Lock()

        def _get(m):
            sem = _get_model_semaphore(m)
            with lock: results[m] = sem

        threads = [threading.Thread(target=_get, args=(m,)) for m in models]
        for t in threads: t.start()
        for t in threads: t.join(timeout=5)
        for m in models:
            assert results[m]._value == _SAMPLE_CONFIG["concurrency_limits"][m]


# ═══════════════════════════════════════════════════════════════════════
# Async tests
# ═══════════════════════════════════════════════════════════════════════


class TestAsync:

    @pytest.mark.asyncio
    async def test_acquire_and_release(self, sample_config):
        sem = _get_model_semaphore("glm-5-turbo")
        await _async_acquire_semaphore(sem)
        assert sem._value == 0
        sem.release()
        assert sem._value == 1

    @pytest.mark.asyncio
    async def test_blocks_until_released(self, sample_config):
        sem = _get_model_semaphore("glm-5-turbo")
        sem.acquire()
        acquired = asyncio.Event()

        async def _waiter():
            await _async_acquire_semaphore(sem)
            acquired.set()
            sem.release()

        task = asyncio.create_task(_waiter())
        await asyncio.sleep(0.05)
        assert not acquired.is_set()
        sem.release()
        await asyncio.wait_for(acquired.wait(), timeout=2)
        await task

    @pytest.mark.asyncio
    async def test_multiple_waiters(self, sample_config):
        sem = _get_model_semaphore("glm-5-turbo")
        order = []

        async def _w(idx):
            await _async_acquire_semaphore(sem)
            order.append(idx)
            await asyncio.sleep(0.01)
            sem.release()

        await asyncio.gather(*[asyncio.create_task(_w(i)) for i in range(5)])
        assert set(order) == {0, 1, 2, 3, 4}

    @pytest.mark.asyncio
    async def test_mixed_sync_async(self, sample_config):
        sem = _get_model_semaphore("glm-5-turbo")
        sem.acquire()
        acquired = asyncio.Event()

        async def _try():
            await _async_acquire_semaphore(sem)
            acquired.set()
            sem.release()

        task = asyncio.create_task(_try())
        await asyncio.sleep(0.05)
        assert not acquired.is_set()
        sem.release()
        await asyncio.wait_for(acquired.wait(), timeout=2)
        await task


# ═══════════════════════════════════════════════════════════════════════
# Integration: call_llm
# ═══════════════════════════════════════════════════════════════════════


class TestCallLlmGate:

    def test_success(self, sample_config):
        rp, cp, _, resp = _mock_call_llm_patches("glm-5-turbo")
        with rp, cp:
            assert call_llm(task="compression", messages=[{"role": "user", "content": "t"}]) is resp
        assert _get_model_semaphore("glm-5-turbo")._value == 1

    def test_error_releases(self, sample_config):
        rp, cp, mc, _ = _mock_call_llm_patches("glm-5-turbo")
        mc.chat.completions.create.side_effect = RuntimeError("boom")
        with rp, cp:
            with pytest.raises(RuntimeError): call_llm(task="compression", messages=[{"role": "user", "content": "t"}])
        assert _get_model_semaphore("glm-5-turbo")._value == 1

    def test_max_tokens_retry_success(self, sample_config):
        rp, cp, mc, resp = _mock_call_llm_patches("glm-5-turbo")
        mc.chat.completions.create.side_effect = [Exception("unsupported_parameter: max_tokens"), resp]
        with rp, cp:
            assert call_llm(task="compression", messages=[{"role": "user", "content": "t"}]) is resp
        assert _get_model_semaphore("glm-5-turbo")._value == 1

    def test_max_tokens_retry_failure(self, sample_config):
        rp, cp, mc, _ = _mock_call_llm_patches("glm-5-turbo")
        mc.chat.completions.create.side_effect = [Exception("unsupported_parameter: max_tokens"), ValueError("no")]
        with rp, cp:
            with pytest.raises(ValueError): call_llm(task="compression", messages=[{"role": "user", "content": "t"}])
        assert _get_model_semaphore("glm-5-turbo")._value == 1

    def test_payment_fallback(self, sample_config):
        rp, cp, mc, _ = _mock_call_llm_patches("glm-5-turbo", provider="auto")
        err = Exception("402"); err.status_code = 402
        mc.chat.completions.create.side_effect = err
        fb = MagicMock(); fb_r = MagicMock(); fb.chat.completions.create.return_value = fb_r
        with rp, cp, patch("agent.auxiliary_client._try_payment_fallback", return_value=(fb, "fb", "nous")):
            assert call_llm(task="compression", messages=[{"role": "user", "content": "t"}]) is fb_r
        assert _get_model_semaphore("glm-5-turbo")._value == 1

    def test_ungated_model(self, sample_config):
        rp, cp, _, resp = _mock_call_llm_patches("gpt-4o", provider="openai")
        with rp, cp:
            assert call_llm(task="compression", messages=[{"role": "user", "content": "t"}]) is resp
        assert _get_model_semaphore("gpt-4o") is None

    def test_concurrent_serialised(self, sample_config):
        order = []
        def slow(**kw):
            order.append("s"); time.sleep(0.1); order.append("e")
            r = MagicMock(); r.choices = [MagicMock()]; return r

        def _do():
            rp, cp, mc, _ = _mock_call_llm_patches("glm-5-turbo")
            mc.chat.completions.create.side_effect = slow
            with rp, cp: call_llm(task="compression", messages=[{"role": "user", "content": "t"}])

        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda _: _do(), range(2)))
        assert order == ["s", "e", "s", "e"]


# ═══════════════════════════════════════════════════════════════════════
# Integration: async_call_llm
# ═══════════════════════════════════════════════════════════════════════


class TestAsyncCallLlmGate:

    @pytest.mark.asyncio
    async def test_success(self, sample_config):
        rp, cp, _, resp = _mock_async_call_llm_patches("glm-5-turbo")
        with rp, cp:
            assert await async_call_llm(task="compression", messages=[{"role": "user", "content": "t"}]) is resp
        assert _get_model_semaphore("glm-5-turbo")._value == 1

    @pytest.mark.asyncio
    async def test_error_releases(self, sample_config):
        rp, cp, mc, _ = _mock_async_call_llm_patches("glm-5-turbo")
        mc.chat.completions.create.side_effect = RuntimeError("boom")
        with rp, cp:
            with pytest.raises(RuntimeError): await async_call_llm(task="compression", messages=[{"role": "user", "content": "t"}])
        assert _get_model_semaphore("glm-5-turbo")._value == 1

    @pytest.mark.asyncio
    async def test_max_tokens_retry(self, sample_config):
        rp, cp, mc, resp = _mock_async_call_llm_patches("glm-5-turbo")
        mc.chat.completions.create.side_effect = [Exception("unsupported_parameter: max_tokens"), resp]
        with rp, cp:
            assert await async_call_llm(task="compression", messages=[{"role": "user", "content": "t"}]) is resp
        assert _get_model_semaphore("glm-5-turbo")._value == 1

    @pytest.mark.asyncio
    async def test_payment_fallback(self, sample_config):
        rp, cp, mc, _ = _mock_async_call_llm_patches("glm-5-turbo", provider="auto")
        err = Exception("402"); err.status_code = 402
        mc.chat.completions.create.side_effect = err
        fb = AsyncMock(); fb_r = MagicMock(); fb.chat.completions.create.return_value = fb_r
        with rp, cp, \
             patch("agent.auxiliary_client._try_payment_fallback", return_value=(fb, "fb", "nous")), \
             patch("agent.auxiliary_client._to_async_client", return_value=(fb, "fb")):
            assert await async_call_llm(task="compression", messages=[{"role": "user", "content": "t"}]) is fb_r
        assert _get_model_semaphore("glm-5-turbo")._value == 1

    @pytest.mark.asyncio
    async def test_ungated_model(self, sample_config):
        rp, cp, _, resp = _mock_async_call_llm_patches("gpt-4o", provider="openai")
        with rp, cp:
            assert await async_call_llm(task="compression", messages=[{"role": "user", "content": "t"}]) is resp
        assert _get_model_semaphore("gpt-4o") is None

    @pytest.mark.asyncio
    async def test_concurrent_serialised(self, sample_config):
        events = []
        async def slow(**kw):
            events.append("s"); await asyncio.sleep(0.1); events.append("e")
            r = MagicMock(); r.choices = [MagicMock()]; return r

        async def _do():
            rp, cp, mc, _ = _mock_async_call_llm_patches("glm-5-turbo")
            mc.chat.completions.create = slow
            with rp, cp: await async_call_llm(task="compression", messages=[{"role": "user", "content": "t"}])

        await asyncio.gather(_do(), _do())
        assert events == ["s", "e", "s", "e"]


# ═══════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:

    def test_1000_cycles(self, sample_config):
        sem = _get_model_semaphore("glm-5-turbo")
        for _ in range(1000):
            sem.acquire(); sem.release()
        assert sem._value == 1

    def test_very_high_limit(self):
        with patch("hermes_cli.config.load_config", return_value={"concurrency_limits": {"bulk": 10000}}):
            assert _get_model_semaphore("bulk")._value == 10000

    def test_multiple_providers(self):
        cfg = {"concurrency_limits": {
            "glm-5-turbo": 1, "claude-3-opus": 5, "gpt-4o": 3, "gemini-2-flash": 10,
        }}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            sems = {m: _get_model_semaphore(m) for m in cfg["concurrency_limits"]}
        for m, sem in sems.items():
            assert sem._value == cfg["concurrency_limits"][m]
        ids = [id(s) for s in sems.values()]
        assert len(set(ids)) == len(ids)  # all distinct
