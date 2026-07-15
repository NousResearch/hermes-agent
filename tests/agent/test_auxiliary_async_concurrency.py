"""Tests for the async concurrency semaphore in auxiliary_client.

When context compression fires, it can spawn many parallel ``async_call_llm``
calls (one per summarisation chunk). Without a cap, these exhaust rate-limited
providers — e.g. ZAI's 7-concurrent-slot limit produced 43 instant 429s that
cascaded through every fallback provider and crashed the gateway.

The fix wraps ``async_call_llm`` in an ``asyncio.Semaphore`` whose limit is
read from ``compression.processing.max_concurrent_requests`` in config.yaml
(default 3).

These tests exercise the real ``async_call_llm`` production path with a mocked
LLM client and assert the concurrency invariant: at most ``max_concurrent``
calls are in-flight at any time, regardless of how many are dispatched
concurrently.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agent.auxiliary_client import (
    async_call_llm,
    _get_async_semaphore,
    _reset_async_semaphore,
    _get_async_concurrency_limit,
    _DEFAULT_ASYNC_CONCURRENCY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_response():
    return {"ok": True}


def _make_async_client():
    """A mock async client that records concurrency when create() is called."""
    client = MagicMock()
    client.base_url = "https://api.openai.com/v1"
    client.chat.completions.create = AsyncMock(return_value=_ok_response())
    return client


def _patches(client):
    """Common mocks: provider resolution, cached client, response validation."""
    return (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("openai", "gpt-4o", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client",
              return_value=(client, "gpt-4o")),
        patch("agent.auxiliary_client._validate_llm_response",
              side_effect=lambda resp, _task: resp),
        patch("agent.auxiliary_client._get_task_extra_body",
              return_value={}),
        patch("agent.auxiliary_client._effective_aux_timeout",
              return_value=60.0),
    )


@pytest.fixture(autouse=True)
def _reset_sem_between_tests():
    """Ensure each test gets a fresh semaphore."""
    _reset_async_semaphore()
    yield
    _reset_async_semaphore()


# ---------------------------------------------------------------------------
# Config reading
# ---------------------------------------------------------------------------

class TestConcurrencyLimitConfig:
    """The limit is read from config with a sensible default."""

    def test_default_when_config_missing(self):
        """When config.yaml has no compression.processing section, the
        default (3) is returned."""
        with patch("hermes_cli.config.load_config", return_value={}):
            _reset_async_semaphore()
            limit = _get_async_concurrency_limit()
        assert limit == _DEFAULT_ASYNC_CONCURRENCY

    def test_reads_config_value(self):
        """A configured value is honoured."""
        fake_cfg = {
            "compression": {
                "processing": {"max_concurrent_requests": 5},
            },
        }
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            _reset_async_semaphore()
            limit = _get_async_concurrency_limit()
        assert limit == 5

    def test_clamps_to_minimum_1(self):
        """A value below 1 is clamped to 1 (no deadlock)."""
        fake_cfg = {
            "compression": {
                "processing": {"max_concurrent_requests": 0},
            },
        }
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            _reset_async_semaphore()
            limit = _get_async_concurrency_limit()
        assert limit == 1

    def test_falls_back_on_config_error(self):
        """A config-read failure returns the default, not an exception."""
        with patch("hermes_cli.config.load_config", side_effect=Exception("boom")):
            _reset_async_semaphore()
            limit = _get_async_concurrency_limit()
        assert limit == _DEFAULT_ASYNC_CONCURRENCY


# ---------------------------------------------------------------------------
# Concurrency invariant
# ---------------------------------------------------------------------------

class TestAsyncConcurrencySemaphore:
    """At most ``max_concurrent`` calls are in-flight simultaneously."""

    @pytest.mark.asyncio
    async def test_default_limit_caps_concurrency(self):
        """With the default limit (3), dispatching 10 concurrent calls
        never exceeds 3 in-flight at once."""
        client = _make_async_client()
        p1, p2, p3, p4, p5 = _patches(client)

        in_flight = 0
        max_observed = 0

        original_create = client.chat.completions.create

        async def tracking_create(**kwargs):
            nonlocal in_flight, max_observed
            in_flight += 1
            max_observed = max(max_observed, in_flight)
            await asyncio.sleep(0.02)  # hold the slot briefly
            in_flight -= 1
            return await original_create(**kwargs)

        client.chat.completions.create = tracking_create

        with p1, p2, p3, p4, p5:
            # Use default config (limit = 3)
            await asyncio.gather(*[
                async_call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": f"chunk {i}"}],
                )
                for i in range(10)
            ])

        assert max_observed <= _DEFAULT_ASYNC_CONCURRENCY, (
            f"observed {max_observed} concurrent calls but limit is "
            f"{_DEFAULT_ASYNC_CONCURRENCY}"
        )
        assert max_observed == _DEFAULT_ASYNC_CONCURRENCY, (
            "expected the semaphore to actually be contended (all 3 slots used), "
            f"but only {max_observed} were observed"
        )

    @pytest.mark.asyncio
    async def test_limit_1_serialises_all_calls(self):
        """With limit=1, calls are fully serialised (max in-flight = 1)."""
        fake_cfg = {
            "compression": {
                "processing": {"max_concurrent_requests": 1},
            },
        }
        client = _make_async_client()
        p1, p2, p3, p4, p5 = _patches(client)

        in_flight = 0
        max_observed = 0

        original_create = client.chat.completions.create

        async def tracking_create(**kwargs):
            nonlocal in_flight, max_observed
            in_flight += 1
            max_observed = max(max_observed, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return await original_create(**kwargs)

        client.chat.completions.create = tracking_create

        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            _reset_async_semaphore()
            with p1, p2, p3, p4, p5:
                await asyncio.gather(*[
                    async_call_llm(
                        task="compression",
                        messages=[{"role": "user", "content": f"chunk {i}"}],
                    )
                    for i in range(6)
                ])

        assert max_observed == 1, (
            f"limit=1 should serialise all calls, but {max_observed} "
            "were in-flight simultaneously"
        )

    @pytest.mark.asyncio
    async def test_all_calls_complete_successfully(self):
        """The semaphore limits concurrency but does not lose calls —
        all dispatched calls return a result."""
        client = _make_async_client()
        p1, p2, p3, p4, p5 = _patches(client)

        with p1, p2, p3, p4, p5:
            results = await asyncio.gather(*[
                async_call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": f"chunk {i}"}],
                )
                for i in range(8)
            ])

        assert len(results) == 8
        assert all(r == _ok_response() for r in results)

    @pytest.mark.asyncio
    async def test_semaphore_is_reused_across_calls(self):
        """The semaphore is module-level and lazily initialised once —
        subsequent calls reuse the same instance."""
        _reset_async_semaphore()
        sem1 = _get_async_semaphore()
        sem2 = _get_async_semaphore()
        assert sem1 is sem2, "semaphore should be cached after first init"

    @pytest.mark.asyncio
    async def test_reset_picks_up_config_change(self):
        """After _reset_async_semaphore(), the next call re-reads config."""
        # Initialise with default
        _reset_async_semaphore()
        sem_before = _get_async_semaphore()

        # Change config and reset
        fake_cfg = {
            "compression": {
                "processing": {"max_concurrent_requests": 7},
            },
        }
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            _reset_async_semaphore()
            sem_after = _get_async_semaphore()

        assert sem_before is not sem_after, (
            "after reset, a new semaphore should be created from updated config"
        )
