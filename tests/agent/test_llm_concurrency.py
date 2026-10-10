"""Per-provider request admission invariants (#109889, #31802)."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

import run_agent
from agent import chat_completion_helpers, llm_concurrency, relay_llm
from agent.chat_completion_helpers import _context_thread_target
from agent.rate_limit_tracker import RateLimitBucket, RateLimitState
from hermes_cli import config as config_module
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


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


def _nested(callback):
    return relay_llm.execute({}, lambda _r: callback(), name="openrouter", model_name="m", session_id="")


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize("stream_end", ["exhausted", "aborted", "failed"])
def test_max_in_flight_bounds_requests_and_streams_without_leaking(providers, stream_end, lazy, tmp_path):
    providers({"openrouter": {"max_in_flight": 1}})
    physical = _Physical()
    queued_entered = threading.Event()

    def open_stream(_request):
        return _ProviderStream(physical, fail_after=1 if stream_end == "failed" else None)

    def lazy_shim(request):
        # Opens and closes its one provider request through nested same-provider calls,
        # on first pull and on teardown: both are that stream's request.
        stream = _nested(lambda: open_stream(request))
        try:
            yield from stream
        finally:
            _nested(stream.close)

    held = relay_llm.stream(
        {}, lazy_shim if lazy else open_stream,
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
    # Nor is another profile multiplexed into this process (A -> B -> A: A's budget is still held).
    def in_profile_b():
        token = set_hermes_home_override(tmp_path / "profile-b")
        try:
            return relay_llm.execute({}, lambda _r: "profile-b", name="openrouter", model_name="m", session_id="")
        finally:
            reset_hermes_home_override(token)

    assert _spawn(in_profile_b).result(timeout=5) == "profile-b"
    assert not queued_entered.is_set()

    interrupted._interrupt_requested = True
    with pytest.raises(InterruptedError):
        waiter.result(timeout=5)
    assert not interrupted.dispatched
    assert not queued_entered.is_set()

    assert _spawn(next, held).result(timeout=5) == "chunk-1"
    if stream_end == "exhausted":
        assert _spawn(list, held).result(timeout=5) == ["chunk-2"]
    elif stream_end == "aborted":
        _spawn(held.close).result(timeout=5)
    else:
        with pytest.raises(ConnectionError):
            _spawn(next, held).result(timeout=5)

    assert queued.result(timeout=5) == "nested"

    # Nothing leaked: a fresh request is admitted straight away.
    fresh = _spawn(relay_llm.execute, {}, lambda _r: "fresh", name="openrouter", model_name="m", session_id="")
    assert fresh.result(timeout=5) == "fresh"
    assert physical.max_active == 1
    assert physical.active == 0


def _request_window(remaining: int, reset_seconds: float) -> RateLimitState:
    bucket = RateLimitBucket(limit=60, remaining=remaining, reset_seconds=reset_seconds, captured_at=time.time())
    return RateLimitState(requests_min=bucket, captured_at=time.time(), provider="openrouter")


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
        llm_concurrency.note_rate_limit_state("openrouter", _request_window(0, 30.0))
        late = asyncio.create_task(request("after-reset", "main"))
        clock.now += 20.0
        await settle()
        assert started[-1] == "aux-2"
        clock.now += 11.0
        await asyncio.wait_for(late, timeout=5)
        assert started[-1] == "after-reset"

        # Rate inputs that change between two starts govern the very next start, in both directions:
        # a positive header window (30 left of 60 s: one per 2 s), then live 60 -> 1 -> 60 rpm edits.
        async def starts_after(name: str, delay: float) -> None:
            begin = clock.now
            task = asyncio.create_task(request(name, "main"))
            clock.now = begin + delay - 0.01
            await settle()
            assert started[-1] != name
            clock.now = begin + delay
            await asyncio.wait_for(task, timeout=5)
            assert started[-1] == name

        llm_concurrency.note_rate_limit_state("openrouter", _request_window(30, 60.0))
        await starts_after("header-paced", 2.0)
        providers({"openrouter": {"requests_per_minute": 1}})
        clock.now += 0.1
        await starts_after("tightened", 59.9)
        providers({"openrouter": {"requests_per_minute": 60}})
        clock.now += 0.1
        await starts_after("loosened", 0.9)

        # Unsetting the pacing key (auto or numeric -> concurrency cap only) drops an exhausted
        # window's hold too: the next request is admitted with nothing in flight and no clock advance.
        for paced in ("auto", 60):
            providers({"openrouter": {"requests_per_minute": paced, "max_in_flight": 1}})
            llm_concurrency.note_rate_limit_state("openrouter", _request_window(0, 60.0))
            providers({"openrouter": {"max_in_flight": 1}})
            await asyncio.wait_for(request(f"cap-only-after-{paced}", "main"), timeout=5)
        providers({"openrouter": {"requests_per_minute": 60}})

    asyncio.run(scenario())

    # A streaming retry is a new physical request: admitted and paced afresh, not re-entry of
    # the logical call's admission (the attempts run on a context-copied worker, as in production).
    llm_concurrency._reset_provider_limiters()
    clock.now = 5000.0
    opened: list[float] = []

    def provider_stream(_request):
        opened.append(clock.now)
        return _ProviderStream(_Physical(), fail_after=0 if len(opened) == 1 else None)

    def attempts() -> list:
        for attempt in range(2):
            if attempt:
                llm_concurrency.readmit_prepaid("openrouter")
            try:
                return list(relay_llm.stream(
                    {}, provider_stream, name="openrouter", model_name="m", session_id="", finalizer=dict))
            except ConnectionError:
                pass
        return []

    def logical_call() -> list:
        with llm_concurrency.prepaid_provider_slot("openrouter"):
            return _spawn(_context_thread_target(attempts)).result(timeout=5)

    call = _spawn(logical_call)
    time.sleep(0.3)
    clock.now = 5000.99
    time.sleep(0.3)
    assert opened == [5000.0]
    clock.now = 5001.0
    assert call.result(timeout=5) == ["chunk-1", "chunk-2"]
    assert opened == [5000.0, 5001.0]

    # Queueing for that admission is not provider silence: the real stall monitor beside the real
    # retry path neither warns, kills nor strikes while the retry waits past the stale threshold,
    # and still kills an admitted attempt that goes silent.
    monkeypatch.setattr(llm_concurrency, "time", time)
    llm_concurrency._reset_provider_limiters()
    providers({"openrouter": {"max_in_flight": 1}})
    agent = run_agent.AIAgent(
        api_key="k", base_url="https://openrouter.ai/api/v1", model="m", provider="openrouter",
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[], max_iterations=1,
    )
    streaming = chat_completion_helpers._StreamingCall(agent, {"model": "m", "messages": []}, None)
    streaming._stream_stale_timeout = 0.5
    streaming._call_done = threading.Event()
    stale_kills: list[float] = []
    streaming._kill_stale_stream = stale_kills.append
    monitor = threading.Thread(target=streaming._monitor_loop, daemon=True)
    monitor.start()
    try:
        with llm_concurrency.prepaid_provider_slot("openrouter"):
            first_attempt = llm_concurrency.acquire_provider_slot("openrouter")
            threading.Timer(1.5, first_attempt.release).start()
            streaming._readmit_retry()  # queued ~1.5 s against a 0.5 s stale threshold
            assert stale_kills == []
            time.sleep(1.2)  # admitted, and the provider never answers
            assert stale_kills
    finally:
        streaming._call_done.set()
        monitor.join(timeout=5)
