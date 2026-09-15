"""Tests for the process-wide per-provider LLM concurrency gate (#109889).

``providers.<id>.max_in_flight`` caps concurrent provider requests for the whole
process (main loop + auxiliary/relay clients, sync and async), so several agents
sharing a one-request-per-key provider queue locally instead of colliding into 429s.
"""

import asyncio
import contextlib
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import chat_completion_helpers as h
from agent import llm_concurrency
from agent.auxiliary_client import async_call_llm, call_llm
from agent.llm_concurrency import (
    acquire_provider_permit,
    async_acquire_provider_permit,
    provider_max_in_flight,
    provider_request_slot,
    release_permit_when_stream_ends,
    reset_provider_gates,
)

PROVIDER = "acme"


@pytest.fixture(autouse=True)
def _clean_gates():
    reset_provider_gates()
    yield
    reset_provider_gates()


def _config(limit=None, *, provider: str = PROVIDER, present: bool = True):
    """Patch the config loader with ``providers.<provider>.max_in_flight: limit``."""
    providers = {provider: ({"max_in_flight": limit} if present else {})} if present else {}
    return patch("hermes_cli.config.load_config_readonly", return_value={"providers": providers})


class _Counter:
    """Tracks concurrent holders of a fake transport, worst case included."""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self._lock = threading.Lock()

    def __enter__(self):
        with self._lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        return self

    def __exit__(self, *exc):
        with self._lock:
            self.active -= 1
        return False


# ── configuration ───────────────────────────────────────────────────────────

class TestConfig:
    def test_configured_limit(self):
        with _config(1):
            assert provider_max_in_flight(PROVIDER) == 1
        with _config(4):
            assert provider_max_in_flight(PROVIDER) == 4

    def test_unset_or_unknown_provider_is_unlimited(self):
        with _config(None):
            assert provider_max_in_flight(PROVIDER) is None
        with patch("hermes_cli.config.load_config_readonly", return_value={"providers": {}}):
            assert provider_max_in_flight(PROVIDER) is None
            assert provider_max_in_flight("") is None
            assert provider_max_in_flight(None) is None

    def test_malformed_values_are_unlimited(self):
        for bad in (0, -3, "nope", "1.5"):
            with _config(bad):
                assert provider_max_in_flight(PROVIDER) is None, bad

    def test_string_number_is_accepted(self):
        with _config("2"):
            assert provider_max_in_flight(PROVIDER) == 2

    def test_broken_config_never_raises(self):
        with patch("hermes_cli.config.load_config_readonly", side_effect=RuntimeError("boom")):
            assert provider_max_in_flight(PROVIDER) is None


# ── the gate itself ─────────────────────────────────────────────────────────

class TestGateSync:
    def test_one_permits_serializes(self):
        with _config(1):
            order, first_holds = [], threading.Event()
            entered = threading.Semaphore(0)

            def worker(name):
                with provider_request_slot(PROVIDER):
                    order.append(name)
                    entered.release()
                    if name == "a":
                        first_holds.wait(2)

            a = threading.Thread(target=worker, args=("a",))
            b = threading.Thread(target=worker, args=("b",))
            a.start()
            entered.acquire()
            b.start()
            time.sleep(0.1)
            assert order == ["a"], "second request must wait while the first holds the permit"
            first_holds.set()
            a.join(2)
            b.join(2)
            assert order == ["a", "b"]

    def test_two_permits_run_parallel(self):
        with _config(2):
            gate = threading.Barrier(3, timeout=5)

            def worker():
                with provider_request_slot(PROVIDER):
                    gate.wait()

            threads = [threading.Thread(target=worker) for _ in range(2)]
            for t in threads:
                t.start()
            # A third party only clears the barrier if BOTH workers held a permit at once.
            gate.wait()
            for t in threads:
                t.join(5)
            assert not any(t.is_alive() for t in threads)

    def test_unconfigured_provider_does_not_block(self):
        with _config(None):
            first = acquire_provider_permit(PROVIDER)
            assert first is None  # no gate, no bookkeeping
            with provider_request_slot(PROVIDER):
                pass

    def test_nested_slot_is_reentrant(self):
        with _config(1):
            with provider_request_slot(PROVIDER):
                with provider_request_slot(PROVIDER):
                    assert llm_concurrency._gate_for(PROVIDER)._in_flight == 1

    def test_released_on_exception(self):
        with _config(1):
            with pytest.raises(RuntimeError):
                with provider_request_slot(PROVIDER):
                    raise RuntimeError("boom")
            permit = acquire_provider_permit(PROVIDER)
            assert permit is not None
            permit.release()

    def test_double_release_is_rejected(self):
        with _config(1):
            permit = acquire_provider_permit(PROVIDER)
            permit.release()
            with pytest.raises(ValueError):
                permit.release()

    def test_limit_change_applies_without_rebuilding_the_gate(self):
        with _config(1):
            gate = llm_concurrency._gate_for(PROVIDER)
            with provider_request_slot(PROVIDER):
                with _config(3):
                    assert llm_concurrency._gate_for(PROVIDER) is gate
                    assert gate.limit == 3


class TestGateAsync:
    @pytest.mark.asyncio
    async def test_one_permit_serializes_tasks(self):
        with _config(1):
            order = []

            async def worker(name):
                async with llm_concurrency.async_provider_request_slot(PROVIDER):
                    order.append(name)
                    await asyncio.sleep(0.05)

            await asyncio.gather(*(worker(n) for n in "abc"))
            assert order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_shared_budget_with_sync_callers(self):
        """A sync holder must keep an async waiter out — one budget, not two."""
        with _config(1):
            held = threading.Event()
            holder_started = threading.Event()

            def holder():
                with provider_request_slot(PROVIDER):
                    holder_started.set()
                    held.wait(2)

            thread = threading.Thread(target=holder)
            thread.start()
            assert holder_started.wait(2)
            waiter = asyncio.create_task(
                llm_concurrency.async_provider_request_slot(PROVIDER).__aenter__())
            await asyncio.sleep(0.05)
            assert not waiter.done(), "async caller must queue behind the sync holder"
            held.set()
            thread.join(2)
            await asyncio.wait_for(waiter, 2)

    @pytest.mark.asyncio
    async def test_cancel_while_queued_strands_no_permit(self):
        with _config(1):
            holder_ready = asyncio.Event()
            release_holder = asyncio.Event()

            async def holder():
                async with llm_concurrency.async_provider_request_slot(PROVIDER):
                    holder_ready.set()
                    await release_holder.wait()

            holder_task = asyncio.create_task(holder())
            await asyncio.wait_for(holder_ready.wait(), 2)
            waiter = asyncio.create_task(async_acquire_provider_permit(PROVIDER))
            await asyncio.sleep(0.05)
            assert not waiter.done(), "the second caller must queue behind the permit"
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert llm_concurrency._gate_for(PROVIDER)._in_flight == 1
            release_holder.set()
            await asyncio.wait_for(holder_task, 2)
            # The permit the cancelled waiter never took is free again.
            permit = await asyncio.wait_for(async_acquire_provider_permit(PROVIDER), 2)
            assert permit is not None
            permit.release()

    @pytest.mark.asyncio
    async def test_nested_slot_is_reentrant(self):
        with _config(1):
            async with llm_concurrency.async_provider_request_slot(PROVIDER):
                async with llm_concurrency.async_provider_request_slot(PROVIDER):
                    assert llm_concurrency._gate_for(PROVIDER)._in_flight == 1

    @pytest.mark.asyncio
    async def test_released_on_exception(self):
        with _config(1):
            with pytest.raises(RuntimeError):
                async with llm_concurrency.async_provider_request_slot(PROVIDER):
                    raise RuntimeError("boom")
            permit = await asyncio.wait_for(async_acquire_provider_permit(PROVIDER), 2)
            permit.release()

    @pytest.mark.asyncio
    async def test_unconfigured_provider_is_unlimited(self):
        with _config(None):
            async with llm_concurrency.async_provider_request_slot(PROVIDER):
                async with llm_concurrency.async_provider_request_slot(PROVIDER):
                    pass


# ── auxiliary path (fake transport) ─────────────────────────────────────────

def _aux_client(counter: _Counter, *, stream_chunks=None):
    client = MagicMock()
    client.base_url = "https://acme.test/v1"

    def fake_create(**kwargs):
        with counter:
            if stream_chunks is not None:
                return iter(stream_chunks)
            time.sleep(0.05)
        return MagicMock()

    client.chat.completions.create.side_effect = fake_create
    return client


@contextlib.contextmanager
def _aux_patches(client):
    with (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=(PROVIDER, "test-model", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(client, "test-model")),
        patch("agent.auxiliary_client._validate_llm_response",
              side_effect=lambda resp, _task, **_kwargs: resp),
        patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={}),
    ):
        yield


def _run_threads(n, target):
    threads = [threading.Thread(target=target) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert not any(t.is_alive() for t in threads), "a caller hung behind the gate"


class TestAuxiliaryCalls:
    def test_single_permit_serializes_aux_requests(self):
        counter = _Counter()
        client = _aux_client(counter)
        with _config(1), _aux_patches(client):
            _run_threads(4, lambda: call_llm(task="title_generation",
                                             messages=[{"role": "user", "content": "hi"}]))
        assert counter.calls == 4
        assert counter.max_active == 1

    def test_limit_above_one_allows_parallel_aux_requests(self):
        counter = _Counter()
        client = _aux_client(counter)
        with _config(3), _aux_patches(client):
            _run_threads(3, lambda: call_llm(task="title_generation",
                                             messages=[{"role": "user", "content": "hi"}]))
        assert counter.max_active == 3, "N>1 must still run concurrently"

    def test_unconfigured_provider_stays_unbounded(self):
        counter = _Counter()
        client = _aux_client(counter)
        with _config(None), _aux_patches(client):
            _run_threads(3, lambda: call_llm(task="title_generation",
                                             messages=[{"role": "user", "content": "hi"}]))
        assert counter.max_active == 3

    def test_stream_holds_permit_until_consumed(self):
        counter = _Counter()
        client = _aux_client(counter, stream_chunks=["chunk"])
        second_started = threading.Event()

        def second():
            second_started.set()
            call_llm(task="compression", messages=[{"role": "user", "content": "second"}])

        with _config(1), _aux_patches(client):
            stream = call_llm(task="compression", messages=[{"role": "user", "content": "first"}],
                              stream=True)
            thread = threading.Thread(target=second)
            thread.start()
            assert second_started.wait(1)
            time.sleep(0.05)
            assert counter.calls == 1, "streaming request must keep holding the permit"
            assert list(stream) == ["chunk"]
            thread.join(2)

        assert counter.calls == 2

    def test_stream_shim_returning_a_response_is_passed_through(self):
        """A client that answers stream=True with a completed response must not be wrapped."""
        completed = SimpleNamespace(choices=[SimpleNamespace()])
        client = MagicMock()
        client.base_url = "https://acme.test/v1"
        client.chat.completions.create.return_value = completed

        with _config(1), _aux_patches(client):
            result = call_llm(task="moa_aggregator", provider=PROVIDER,
                              messages=[{"role": "user", "content": "q"}], stream=True)
            assert result is completed
            # Permit was released inline, so the next call is not blocked by a phantom stream.
            call_llm(task="moa_aggregator", provider=PROVIDER,
                     messages=[{"role": "user", "content": "q"}])
        assert client.chat.completions.create.call_count == 2

    def test_permit_released_when_aux_call_raises(self):
        counter = _Counter()
        client = MagicMock()
        client.base_url = "https://acme.test/v1"
        client.chat.completions.create.side_effect = RuntimeError("boom")
        with _config(1), _aux_patches(client):
            for _ in range(3):
                with pytest.raises(RuntimeError, match="boom"):
                    call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}])


class TestAuxiliaryAsyncCalls:
    @pytest.mark.asyncio
    async def test_single_permit_serializes_async_aux_requests(self):
        active = 0
        max_active = 0

        async def fake_create(**kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            try:
                await asyncio.sleep(0.05)
            finally:
                active -= 1
            return MagicMock()

        client = MagicMock()
        client.base_url = "https://acme.test/v1"
        client.chat.completions.create = AsyncMock(side_effect=fake_create)

        with _config(1), _aux_patches(client):
            await asyncio.gather(*(
                async_call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
                for _ in range(4)))

        assert client.chat.completions.create.await_count == 4
        assert max_active == 1

    @pytest.mark.asyncio
    async def test_cancelled_waiter_leaves_gate_usable(self):
        gate_open = asyncio.Event()
        started = asyncio.Event()

        async def fake_create(**kwargs):
            started.set()
            await gate_open.wait()
            return MagicMock()

        client = MagicMock()
        client.base_url = "https://acme.test/v1"
        client.chat.completions.create = AsyncMock(side_effect=fake_create)

        with _config(1), _aux_patches(client):
            holder = asyncio.create_task(
                async_call_llm(task="compression", messages=[{"role": "user", "content": "a"}]))
            await asyncio.wait_for(started.wait(), 2)
            waiter = asyncio.create_task(
                async_call_llm(task="compression", messages=[{"role": "user", "content": "b"}]))
            await asyncio.sleep(0.05)
            waiters_before = len(llm_concurrency._gate_for(PROVIDER)._async_waiters)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert len(llm_concurrency._gate_for(PROVIDER)._async_waiters) == waiters_before - 1
            gate_open.set()
            await asyncio.wait_for(holder, 2)
            # Both permits are back: a fresh call must go straight through.
            await asyncio.wait_for(
                async_call_llm(task="compression", messages=[{"role": "user", "content": "c"}]),
                2)

        assert client.chat.completions.create.await_count == 2  # the cancelled one never sent a request


# ── main agent path (fake transport) ────────────────────────────────────────

def _direct_agent(counter: _Counter, *, fail: bool = False):
    """A MagicMock agent shaped like the cron/subagent inline path (see tests/cron)."""
    agent = MagicMock()
    agent.platform = "cron"
    agent.provider = PROVIDER
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    agent._touch_activity = MagicMock()
    agent._close_request_openai_client = MagicMock()

    def fake_create(**kwargs):
        with counter:
            if fail:
                raise RuntimeError("provider down")
            time.sleep(0.05)
        return SimpleNamespace(id="completion")

    def make_client(*_args, **_kwargs):
        client = MagicMock()
        client.chat.completions.create.side_effect = fake_create
        return client

    agent._create_request_openai_client.side_effect = make_client
    return agent


class TestMainPathDispatch:
    def test_direct_api_call_serializes_across_threads(self):
        counter = _Counter()
        agent = _direct_agent(counter)
        with _config(1):
            _run_threads(4, lambda: h.direct_api_call(agent, {"model": "m", "messages": []}))
        assert counter.calls == 4
        assert counter.max_active == 1

    def test_direct_api_call_is_ungated_when_unconfigured(self):
        counter = _Counter()
        agent = _direct_agent(counter)
        with _config(None):
            _run_threads(3, lambda: h.direct_api_call(agent, {"model": "m", "messages": []}))
        assert counter.max_active == 3

    def test_direct_api_call_releases_permit_on_error(self):
        counter = _Counter()
        agent = _direct_agent(counter, fail=True)
        with _config(1):
            for _ in range(3):
                with pytest.raises(RuntimeError, match="provider down"):
                    h.direct_api_call(agent, {"model": "m", "messages": []})
            assert llm_concurrency._gate_for(PROVIDER)._in_flight == 0

    def test_interruptible_api_call_holds_permit_over_the_worker(self):
        seen = {}

        class _FakeRequest:
            def __init__(self, agent, api_kwargs):
                pass

            def run(self):
                seen["in_flight"] = llm_concurrency._gate_for(PROVIDER)._in_flight
                return "response"

        agent = MagicMock()
        agent.platform = "cli"
        agent.provider = PROVIDER
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        agent._consecutive_stale_streams = 0

        with _config(1), patch("agent.chat_completion_nonstream._NonStreamRequest", _FakeRequest):
            gate = llm_concurrency._gate_for(PROVIDER)
            assert h.interruptible_api_call(agent, {}) == "response"
            assert gate._in_flight == 0

        assert seen["in_flight"] == 1, "the permit must be held while the request runs"

    def test_streaming_call_holds_permit_for_its_whole_duration(self):
        seen = {}

        class _FakeStreamCall:
            def __init__(self, agent, api_kwargs, on_first_delta):
                pass

            def run(self):
                seen["in_flight"] = llm_concurrency._gate_for(PROVIDER)._in_flight
                return "streamed"

        agent = MagicMock()
        agent.provider = PROVIDER
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        agent._consecutive_stale_streams = 0

        with _config(1), patch.object(h, "_StreamingCall", _FakeStreamCall):
            gate = llm_concurrency._gate_for(PROVIDER)
            assert h.interruptible_streaming_api_call(agent, {}, on_first_delta=None) == "streamed"
            assert gate._in_flight == 0

        assert seen["in_flight"] == 1, "a stream keeps its permit for the whole call"

    def test_stream_wrapper_releases_when_consumed(self):
        with _config(1):
            permit = acquire_provider_permit(PROVIDER)
            assert permit is not None
            stream = release_permit_when_stream_ends(iter(["chunk"]), permit)
            assert list(stream) == ["chunk"]
            assert llm_concurrency._gate_for(PROVIDER)._in_flight == 0

    def test_stream_wrapper_releases_on_early_close(self):
        def breaking_stream():
            yield "chunk"
            raise KeyboardInterrupt

        with _config(1):
            permit = acquire_provider_permit(PROVIDER)
            stream = release_permit_when_stream_ends(breaking_stream(), permit)
            assert next(stream) == "chunk"
            with pytest.raises(KeyboardInterrupt):
                next(stream)
            assert llm_concurrency._gate_for(PROVIDER)._in_flight == 0
