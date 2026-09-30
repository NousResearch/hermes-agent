"""Per-provider request admission invariants (#109889, #31802)."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from agent import llm_concurrency, relay_llm
from agent.rate_limit_tracker import RateLimitBucket, RateLimitState
from hermes_cli import config as config_module


@pytest.fixture
def providers(monkeypatch):
    def configure(entries: dict) -> None:
        monkeypatch.setattr(config_module, "load_config_readonly", lambda: {"providers": entries})

    llm_concurrency._reset_provider_limiters()
    yield configure
    llm_concurrency._reset_provider_limiters()


def _spawn(fn, *args, **kwargs) -> Future:
    """Run ``fn`` on a daemon thread, so a regression that deadlocks fails on a timeout, not a hang."""
    future: Future = Future()

    def run() -> None:
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)

    threading.Thread(target=run, daemon=True).start()
    return future


class _Physical:
    """Counts provider requests that are open at the same time."""

    def __init__(self) -> None:
        self.active = self.max_active = 0
        self._lock = threading.Lock()

    def enter(self) -> None:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def exit(self) -> None:
        with self._lock:
            self.active -= 1


class _ProviderStream:
    def __init__(self, physical: _Physical, *, fail_after: int | None) -> None:
        physical.enter()
        self._physical, self._fail_after, self._sent, self._open = physical, fail_after, 0, True

    def __iter__(self):
        return self

    def __next__(self):
        if self._fail_after is not None and self._sent >= self._fail_after:
            raise ConnectionError("provider dropped the stream")
        if self._sent >= 2:
            raise StopIteration
        self._sent += 1
        return f"chunk-{self._sent}"

    def close(self) -> None:
        if self._open:
            self._open = False
            self._physical.exit()


@pytest.mark.parametrize("stream_end", ["exhausted", "aborted", "failed"])
def test_max_in_flight_bounds_requests_and_streams_without_leaking(providers, stream_end):
    providers({"openrouter": {"max_in_flight": 1}})
    physical = _Physical()
    queued_entered = threading.Event()

    held = relay_llm.stream(
        {}, lambda _request: _ProviderStream(physical, fail_after=1 if stream_end == "failed" else None),
        name="openrouter", model_name="primary", session_id="", finalizer=dict,
    )

    def queued_call(_request):
        physical.enter()
        try:
            queued_entered.set()
            # A same-provider call nested inside a request is that request: it must not self-deadlock.
            return relay_llm.execute({}, lambda _r: "nested", name="openrouter", model_name="m", session_id="")
        finally:
            physical.exit()

    class InterruptedCaller:
        _interrupt_requested = False
        dispatched = False

        def call(self, _request):
            self.dispatched = True

    interrupted = InterruptedCaller()

    queued = _spawn(
        relay_llm.execute, {}, queued_call, name="openrouter", model_name="aux", session_id="",
        metadata={"call_role": "auxiliary:title_generation"},
    )
    waiter = _spawn(
        relay_llm.execute, {}, interrupted.call, name="openrouter", model_name="m", session_id="")
    # Other providers are not queued behind this budget.
    assert relay_llm.execute({}, lambda _r: "other", name="anthropic", model_name="m", session_id="") == "other"
    assert not queued_entered.is_set()

    interrupted._interrupt_requested = True
    with pytest.raises(InterruptedError):
        waiter.result(timeout=5)
    assert not interrupted.dispatched
    assert not queued_entered.is_set()

    assert next(held) == "chunk-1"
    if stream_end == "exhausted":
        assert list(held) == ["chunk-2"]
    elif stream_end == "aborted":
        held.close()
    else:
        with pytest.raises(ConnectionError):
            next(held)

    assert queued.result(timeout=5) == "nested"

    # Nothing leaked: a fresh request is admitted straight away.
    fresh = _spawn(relay_llm.execute, {}, lambda _r: "fresh", name="openrouter", model_name="m", session_id="")
    assert fresh.result(timeout=5) == "fresh"
    assert physical.max_active == 1
    assert physical.active == 0


def test_requests_per_minute_paces_starts_fairly_and_honors_rate_limit_headers(providers, monkeypatch):
    providers({"openrouter": {"requests_per_minute": 60}})
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(llm_concurrency, "time", SimpleNamespace(monotonic=lambda: clock.now))
    started: list[str] = []

    async def request(name: str, role: str) -> None:
        permit = await llm_concurrency.acquire_provider_slot_async("openrouter", role=role)
        started.append(name)
        permit.release()

    async def settle() -> None:
        await asyncio.sleep(0.2)

    async def scenario() -> None:
        await request("main-0", "main")
        tasks = [asyncio.create_task(request(name, role)) for name, role in (
            ("aux-1", "auxiliary"), ("aux-2", "auxiliary"), ("main-1", "main"))]
        await settle()
        assert started == ["main-0"]  # 60 rpm: one start per second

        for step, expected in ((1.0, "aux-1"), (2.0, "main-1"), (3.0, "aux-2")):
            clock.now = 1000.0 + step - 0.01
            await settle()
            assert started[-1] != expected
            clock.now = 1000.0 + step
            await settle()
            # Queued auxiliary work cannot starve the main loop (or the reverse).
            assert started[-1] == expected
        await asyncio.gather(*tasks)

        # The provider reports its request window exhausted for another 30 s.
        state = RateLimitState(
            requests_min=RateLimitBucket(limit=60, remaining=0, reset_seconds=30.0, captured_at=llm_real_time()),
            captured_at=llm_real_time(), provider="openrouter",
        )
        llm_concurrency.note_rate_limit_state("openrouter", state)
        late = asyncio.create_task(request("after-reset", "main"))
        clock.now += 20.0
        await settle()
        assert started[-1] == "aux-2"
        clock.now += 11.0
        await asyncio.wait_for(late, timeout=5)
        assert started[-1] == "after-reset"

    asyncio.run(scenario())


def llm_real_time() -> float:
    import time

    return time.time()
