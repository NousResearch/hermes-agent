"""Behavior tests for process-wide model request admission."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import pytest

from agent.model_admission import (
    ModelAdmissionCancelled,
    ModelAdmissionRegistry,
    ModelAdmissionSettings,
    ModelAdmissionTimeout,
    normalize_model_target,
)


class _Clock:
    def __init__(self, *, monotonic: float = 100.0, wall: float = 1_800_000_000.0):
        self.monotonic_value = monotonic
        self.wall_value = wall

    def monotonic(self) -> float:
        return self.monotonic_value

    def wall(self) -> float:
        return self.wall_value

    def advance(self, seconds: float) -> None:
        self.monotonic_value += seconds
        self.wall_value += seconds


class _RateLimitError(Exception):
    status_code = 429

    def __init__(self, retry_after=None):
        super().__init__("rate limited")
        headers = {} if retry_after is None else {"Retry-After": retry_after}
        self.response = SimpleNamespace(status_code=429, headers=headers)


def _settings(**overrides) -> ModelAdmissionSettings:
    values = {
        "enabled": True,
        "max_in_flight": 8,
        "per_target": 4,
        "min_per_target": 1,
        "queue_timeout_seconds": 2.0,
        "additive_successes": 2,
        "retry_after_max_seconds": 600.0,
        "idle_state_ttl_seconds": 3600.0,
        "max_target_states": 256,
        "wait_poll_seconds": 0.01,
    }
    values.update(overrides)
    return ModelAdmissionSettings(**values)


def _target_snapshot(registry: ModelAdmissionRegistry, model: str) -> dict:
    return next(
        target for target in registry.snapshot()["targets"] if target["model"] == model
    )


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def test_normalize_model_target_removes_credentials_and_preserves_route_path():
    target = normalize_model_target(
        " OpenAI ",
        "HTTPS://user:super-secret@API.Example.COM:443/v1/deployments/GPT/?token=leak#part",
        " GPT-5 ",
    )

    assert target.provider == "openai"
    assert target.base_url == "https://api.example.com/v1/deployments/GPT"
    assert target.model == "gpt-5"
    assert "super-secret" not in repr(target)
    assert "token" not in repr(target)


def test_concurrent_admission_never_exceeds_global_or_per_target_limits():
    registry = ModelAdmissionRegistry(_settings(max_in_flight=8, per_target=3))
    lock = threading.Lock()
    global_active = 0
    target_active: dict[str, int] = {}
    observed_global = 0
    observed_target: dict[str, int] = {}

    def worker(index: int) -> None:
        nonlocal global_active, observed_global
        model = f"model-{index % 4}"
        with registry.acquire("provider", "https://api.example.com/v1", model):
            with lock:
                global_active += 1
                target_active[model] = target_active.get(model, 0) + 1
                observed_global = max(observed_global, global_active)
                observed_target[model] = max(
                    observed_target.get(model, 0), target_active[model]
                )
            time.sleep(0.005)
            with lock:
                global_active -= 1
                target_active[model] -= 1

    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = [pool.submit(worker, index) for index in range(64)]
        for future in futures:
            future.result(timeout=10)

    assert observed_global <= 8
    assert observed_global >= 4
    assert all(value <= 3 for value in observed_target.values())
    assert registry.snapshot()["global"]["in_flight"] == 0


def test_ready_waiters_are_fifo_within_the_same_target():
    registry = ModelAdmissionRegistry(_settings(max_in_flight=1, per_target=1))
    owner = registry.acquire("p", "https://one.example/v1", "m")
    admitted: list[int] = []
    threads: list[threading.Thread] = []

    def wait_for_turn(index: int) -> None:
        with registry.acquire("p", "https://one.example/v1", "m"):
            admitted.append(index)

    for index in range(4):
        thread = threading.Thread(target=wait_for_turn, args=(index,), daemon=True)
        thread.start()
        threads.append(thread)
        assert _wait_until(
            lambda expected=index + 1: registry.snapshot()["global"]["queued"]
            == expected
        )

    owner.succeed()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()

    assert admitted == [0, 1, 2, 3]


def test_retry_after_target_does_not_head_of_line_block_another_target():
    registry = ModelAdmissionRegistry(
        _settings(max_in_flight=1, per_target=1, wait_poll_seconds=0.01)
    )
    limiter = registry.acquire("p", "https://api.example/v1", "blocked")
    limiter.fail(_RateLimitError("60"))

    cancel = threading.Event()
    blocked_error: list[BaseException] = []

    def wait_on_blocked_target() -> None:
        try:
            registry.acquire(
                "p",
                "https://api.example/v1",
                "blocked",
                cancel_check=cancel.is_set,
            )
        except BaseException as exc:
            blocked_error.append(exc)

    waiter = threading.Thread(target=wait_on_blocked_target, daemon=True)
    waiter.start()
    assert _wait_until(lambda: registry.snapshot()["global"]["queued"] == 1)

    healthy = registry.acquire("p", "https://api.example/v1", "healthy", timeout=0.2)
    healthy.succeed()
    cancel.set()
    waiter.join(timeout=2)

    assert not waiter.is_alive()
    assert len(blocked_error) == 1
    assert isinstance(blocked_error[0], ModelAdmissionCancelled)


def test_timeout_and_cancellation_remove_waiters_without_leaking_capacity():
    registry = ModelAdmissionRegistry(_settings(max_in_flight=1, per_target=1))
    owner = registry.acquire("p", "https://api.example/v1", "m")

    with pytest.raises(ModelAdmissionTimeout, match="timed out"):
        registry.acquire("p", "https://api.example/v1", "m", timeout=0.02)
    assert registry.snapshot()["global"]["queued"] == 0

    cancel = threading.Event()
    seen: list[BaseException] = []

    def cancelled_waiter() -> None:
        try:
            registry.acquire(
                "p",
                "https://api.example/v1",
                "m",
                cancel_check=cancel.is_set,
            )
        except BaseException as exc:
            seen.append(exc)

    thread = threading.Thread(target=cancelled_waiter, daemon=True)
    thread.start()
    assert _wait_until(lambda: registry.snapshot()["global"]["queued"] == 1)
    cancel.set()
    thread.join(timeout=2)
    owner.release()

    assert len(seen) == 1
    assert isinstance(seen[0], ModelAdmissionCancelled)
    assert registry.snapshot()["global"] == {
        "in_flight": 0,
        "max_in_flight": 1,
        "queued": 0,
    }


def test_async_task_cancellation_removes_waiter_and_future_acquire_works():
    async def scenario() -> None:
        registry = ModelAdmissionRegistry(_settings(max_in_flight=1, per_target=1))
        owner = registry.acquire("p", "https://api.example/v1", "m")
        task = asyncio.create_task(
            registry.acquire_async("p", "https://api.example/v1", "m")
        )
        for _ in range(100):
            if registry.snapshot()["global"]["queued"] == 1:
                break
            await asyncio.sleep(0.005)
        assert registry.snapshot()["global"]["queued"] == 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert registry.snapshot()["global"]["queued"] == 0

        owner.succeed()
        permit = await registry.acquire_async(
            "p", "https://api.example/v1", "m", timeout=0.2
        )
        permit.succeed()
        assert registry.snapshot()["global"]["in_flight"] == 0

    asyncio.run(scenario())


def test_same_epoch_rate_limits_reduce_once_then_additive_success_recovers():
    clock = _Clock()
    registry = ModelAdmissionRegistry(
        _settings(per_target=4, additive_successes=2),
        monotonic=clock.monotonic,
        wall_clock=clock.wall,
    )
    permits = [registry.acquire("p", "https://api.example/v1", "m") for _ in range(4)]

    permits[0].fail(_RateLimitError("10"))
    permits[1].fail(_RateLimitError("20"))
    permits[2].succeed()
    permits[3].succeed()

    state = _target_snapshot(registry, "m")
    assert state["limit"] == 2
    assert state["congestion_epoch"] == 1
    assert state["successes_toward_increase"] == 0
    assert state["blocked_for_seconds"] == pytest.approx(20.0)

    clock.advance(20)
    for _ in range(2):
        registry.acquire("p", "https://api.example/v1", "m").succeed()

    state = _target_snapshot(registry, "m")
    assert state["limit"] == 3
    assert state["successes_toward_increase"] == 0


def test_new_epoch_rate_limit_can_reduce_again():
    clock = _Clock()
    registry = ModelAdmissionRegistry(
        _settings(per_target=4),
        monotonic=clock.monotonic,
        wall_clock=clock.wall,
    )

    registry.acquire("p", "https://api.example/v1", "m").fail(_RateLimitError("1"))
    clock.advance(1)
    registry.acquire("p", "https://api.example/v1", "m").fail(_RateLimitError("1"))

    state = _target_snapshot(registry, "m")
    assert state["limit"] == 1
    assert state["congestion_epoch"] == 2


def test_same_epoch_rate_limits_do_not_multiply_fallback_cooldown():
    clock = _Clock()
    registry = ModelAdmissionRegistry(
        _settings(per_target=4),
        monotonic=clock.monotonic,
        wall_clock=clock.wall,
    )
    permits = [registry.acquire("p", "https://api.example/v1", "m") for _ in range(4)]

    for permit in permits:
        permit.fail(_RateLimitError())

    state = _target_snapshot(registry, "m")
    assert state["limit"] == 2
    assert state["congestion_epoch"] == 1
    assert state["blocked_for_seconds"] == pytest.approx(1.0)


def test_retry_after_http_date_is_parsed_and_capped():
    clock = _Clock()
    registry = ModelAdmissionRegistry(
        _settings(per_target=4, retry_after_max_seconds=90),
        monotonic=clock.monotonic,
        wall_clock=clock.wall,
    )
    retry_at = format_datetime(
        datetime.fromtimestamp(clock.wall() + 120, tz=timezone.utc), usegmt=True
    )

    registry.acquire("p", "https://api.example/v1", "dated").fail(
        _RateLimitError(retry_at)
    )

    state = _target_snapshot(registry, "dated")
    assert state["blocked_for_seconds"] == pytest.approx(90.0)


def test_blocked_state_survives_zero_ttl_and_target_capacity_cleanup():
    clock = _Clock()
    registry = ModelAdmissionRegistry(
        _settings(
            max_in_flight=1,
            per_target=1,
            idle_state_ttl_seconds=0,
            max_target_states=1,
        ),
        monotonic=clock.monotonic,
        wall_clock=clock.wall,
    )
    registry.acquire("p", "https://api.example/v1", "blocked").fail(
        _RateLimitError("60")
    )

    assert _target_snapshot(registry, "blocked")["blocked_for_seconds"] == 60
    with pytest.raises(ModelAdmissionTimeout):
        registry.acquire("p", "https://api.example/v1", "blocked", timeout=0)

    registry.acquire("p", "https://api.example/v1", "healthy").succeed()
    assert {target["model"] for target in registry.snapshot()["targets"]} == {"blocked"}

    clock.advance(60)
    assert len(registry.snapshot()["targets"]) <= 1


def test_sync_stream_holds_permit_until_exhaustion_and_release_is_idempotent():
    registry = ModelAdmissionRegistry(_settings(max_in_flight=1, per_target=1))
    permit = registry.acquire("p", "https://api.example/v1", "stream")
    stream = permit.wrap_stream(iter(["a", "b"]))

    assert registry.snapshot()["global"]["in_flight"] == 1
    assert next(stream) == "a"
    assert registry.snapshot()["global"]["in_flight"] == 1
    assert next(stream) == "b"
    with pytest.raises(StopIteration):
        next(stream)

    permit.succeed()
    permit.release()
    assert registry.snapshot()["global"]["in_flight"] == 0


def test_permit_context_transfers_lifetime_to_sync_stream():
    registry = ModelAdmissionRegistry(_settings(max_in_flight=1, per_target=1))

    with registry.acquire("p", "https://api.example/v1", "stream") as permit:
        stream = permit.wrap_stream(iter(["a", "b"]))

    assert registry.snapshot()["global"]["in_flight"] == 1
    assert list(stream) == ["a", "b"]
    assert registry.snapshot()["global"]["in_flight"] == 0


def test_sync_stream_context_manager_can_defer_iterator_creation_until_entry():
    registry = ModelAdmissionRegistry(_settings(max_in_flight=1, per_target=1))

    class StreamManager:
        def __enter__(self):
            return iter(["a", "b"])

        def __exit__(self, exc_type, exc, traceback):
            return False

    stream = registry.acquire(
        "p", "https://api.example/v1", "managed-stream"
    ).wrap_stream(StreamManager())

    with stream as entered:
        assert list(entered) == ["a", "b"]
    assert registry.snapshot()["global"]["in_flight"] == 0


def test_sync_stream_context_exit_rate_limit_overrides_exhaustion_success():
    registry = ModelAdmissionRegistry(_settings(per_target=4))

    class StreamManager:
        def __enter__(self):
            return iter(["a"])

        def __exit__(self, exc_type, exc, traceback):
            raise _RateLimitError("30")

    stream = registry.acquire(
        "p", "https://api.example/v1", "managed-stream-exit"
    ).wrap_stream(StreamManager())

    with pytest.raises(_RateLimitError):
        with stream as entered:
            assert list(entered) == ["a"]
            entered.close()

    state = _target_snapshot(registry, "managed-stream-exit")
    assert state["limit"] == 2
    assert state["rate_limit_events"] == 1
    assert state["blocked_for_seconds"] > 0


def test_stream_iteration_rate_limit_updates_aimd_and_releases_permit():
    registry = ModelAdmissionRegistry(_settings(per_target=4))

    def failing_stream():
        yield "first"
        raise _RateLimitError("15")

    stream = registry.acquire(
        "p", "https://api.example/v1", "stream-error"
    ).wrap_stream(failing_stream())

    assert next(stream) == "first"
    with pytest.raises(_RateLimitError):
        next(stream)

    state = _target_snapshot(registry, "stream-error")
    assert registry.snapshot()["global"]["in_flight"] == 0
    assert state["limit"] == 2
    assert state["blocked_for_seconds"] > 0


def test_async_stream_holds_permit_until_exhaustion():
    async def scenario() -> None:
        registry = ModelAdmissionRegistry(_settings(max_in_flight=1, per_target=1))

        async def chunks():
            yield "a"
            yield "b"

        permit = await registry.acquire_async(
            "p", "https://api.example/v1", "async-stream"
        )
        stream = permit.wrap_async_stream(chunks())

        assert registry.snapshot()["global"]["in_flight"] == 1
        assert [item async for item in stream] == ["a", "b"]
        assert registry.snapshot()["global"]["in_flight"] == 0

    asyncio.run(scenario())


def test_async_permit_context_transfers_lifetime_to_stream():
    async def scenario() -> None:
        registry = ModelAdmissionRegistry(_settings(max_in_flight=1, per_target=1))

        async def chunks():
            yield "a"
            yield "b"

        async with await registry.acquire_async(
            "p", "https://api.example/v1", "async-stream"
        ) as permit:
            stream = permit.wrap_async_stream(chunks())

        assert registry.snapshot()["global"]["in_flight"] == 1
        assert [item async for item in stream] == ["a", "b"]
        assert registry.snapshot()["global"]["in_flight"] == 0

    asyncio.run(scenario())


def test_async_stream_context_manager_can_defer_iterator_creation_until_entry():
    async def scenario() -> None:
        registry = ModelAdmissionRegistry(_settings(max_in_flight=1, per_target=1))

        async def chunks():
            yield "a"

        class StreamManager:
            async def __aenter__(self):
                return chunks()

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        permit = await registry.acquire_async(
            "p", "https://api.example/v1", "managed-async-stream"
        )
        stream = permit.wrap_async_stream(StreamManager())
        async with stream as entered:
            assert [item async for item in entered] == ["a"]
        assert registry.snapshot()["global"]["in_flight"] == 0

    asyncio.run(scenario())


def test_async_stream_context_exit_rate_limit_overrides_exhaustion_success():
    async def scenario() -> None:
        registry = ModelAdmissionRegistry(_settings(per_target=4))

        async def chunks():
            yield "a"

        class StreamManager:
            async def __aenter__(self):
                return chunks()

            async def __aexit__(self, exc_type, exc, traceback):
                raise _RateLimitError("30")

        permit = await registry.acquire_async(
            "p", "https://api.example/v1", "managed-async-stream-exit"
        )
        stream = permit.wrap_async_stream(StreamManager())
        with pytest.raises(_RateLimitError):
            async with stream as entered:
                assert [item async for item in entered] == ["a"]
                await entered.aclose()

        state = _target_snapshot(registry, "managed-async-stream-exit")
        assert state["limit"] == 2
        assert state["rate_limit_events"] == 1
        assert state["blocked_for_seconds"] > 0

    asyncio.run(scenario())


def test_async_waiter_notifications_are_bounded_by_queue_size():
    async def scenario() -> None:
        registry = ModelAdmissionRegistry(
            _settings(max_in_flight=1, per_target=1, wait_poll_seconds=0.05)
        )
        owner = registry.acquire("p", "https://api.example/v1", "owner")
        loop = asyncio.get_running_loop()
        original_call = loop.call_soon_threadsafe
        wake_calls = 0

        def counted_call(callback, *args, context=None):
            nonlocal wake_calls
            wake_calls += 1
            return original_call(callback, *args, context=context)

        setattr(loop, "call_soon_threadsafe", counted_call)
        tasks: list[asyncio.Task] = []
        try:

            async def worker(index: int) -> None:
                permit = await registry.acquire_async(
                    "p", "https://api.example/v1", f"model-{index % 4}"
                )
                permit.succeed()

            waiter_count = 32
            tasks = [
                asyncio.create_task(worker(index)) for index in range(waiter_count)
            ]
            for _ in range(100):
                if registry.snapshot()["global"]["queued"] == waiter_count:
                    break
                await asyncio.sleep(0.001)
            assert registry.snapshot()["global"]["queued"] == waiter_count

            owner.succeed()
            await asyncio.gather(*tasks)
            assert wake_calls <= waiter_count * 3
        finally:
            setattr(loop, "call_soon_threadsafe", original_call)
            owner.release()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(scenario())


def test_reported_closed_async_loop_waiter_does_not_block_sync_admission():
    async def scenario() -> None:
        registry = ModelAdmissionRegistry(
            _settings(max_in_flight=1, per_target=1, wait_poll_seconds=0.01)
        )
        owner = registry.acquire("p", "https://api.example/v1", "owner")
        waiting = asyncio.create_task(
            registry.acquire_async("p", "https://api.example/v1", "orphan")
        )
        for _ in range(100):
            if registry.snapshot()["global"]["queued"] == 1:
                break
            await asyncio.sleep(0.001)

        loop = asyncio.get_running_loop()
        original_is_closed = loop.is_closed
        setattr(loop, "is_closed", lambda: True)
        failure = None
        healthy = None
        try:
            owner.succeed()
            healthy = registry.acquire(
                "p", "https://api.example/v1", "healthy", timeout=0.05
            )
        except BaseException as error:
            failure = error
        finally:
            setattr(loop, "is_closed", original_is_closed)
            if healthy is not None:
                healthy.succeed()
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
        if failure is not None:
            raise failure
        assert registry.snapshot()["global"]["queued"] == 0

    asyncio.run(scenario())


def test_expired_async_waiter_is_pruned_before_sync_admission():
    async def scenario() -> None:
        clock = _Clock()
        registry = ModelAdmissionRegistry(
            _settings(max_in_flight=1, per_target=1),
            monotonic=clock.monotonic,
            wall_clock=clock.wall,
        )
        owner = registry.acquire("p", "https://api.example/v1", "owner")
        expired = asyncio.create_task(
            registry.acquire_async("p", "https://api.example/v1", "expired", timeout=5)
        )
        for _ in range(100):
            if registry.snapshot()["global"]["queued"] == 1:
                break
            await asyncio.sleep(0.001)

        clock.advance(5)
        healthy = None
        try:
            owner.succeed()
            healthy = registry.acquire(
                "p", "https://api.example/v1", "healthy", timeout=0
            )
        finally:
            if healthy is not None:
                healthy.succeed()
        with pytest.raises(ModelAdmissionTimeout):
            await expired
        assert registry.snapshot()["global"]["queued"] == 0

    asyncio.run(scenario())


def test_bounded_idle_cleanup_and_snapshot_never_expose_url_secrets():
    registry = ModelAdmissionRegistry(
        _settings(max_target_states=2, idle_state_ttl_seconds=3600)
    )
    for index in range(3):
        registry.acquire(
            "provider",
            f"https://user:password-{index}@api{index}.example/v1?api_key=secret-{index}",
            f"model-{index}",
        ).succeed()

    snapshot = registry.snapshot()
    rendered = repr(snapshot)

    assert len(snapshot["targets"]) <= 2
    assert "password" not in rendered
    assert "api_key" not in rendered
    assert "secret" not in rendered
    assert all(target["in_flight"] == 0 for target in snapshot["targets"])


def test_snapshot_hashes_route_path_instead_of_exposing_it():
    registry = ModelAdmissionRegistry(_settings())
    registry.acquire(
        "provider",
        "https://api.example/v1/tenant-secret-value/deployments/model?key=query-secret",
        "model",
    ).succeed()

    target = _target_snapshot(registry, "model")
    rendered = repr(target)
    assert target["base_url"] == "https://api.example"
    assert target["route_id"]
    assert "tenant-secret-value" not in rendered
    assert "query-secret" not in rendered


def test_disabled_registry_returns_noop_permits_without_tracking_state():
    registry = ModelAdmissionRegistry(_settings(enabled=False))

    with registry.acquire(
        "p", "https://user:secret@api.example/v1?token=secret", "m"
    ) as permit:
        assert permit.is_noop

    assert registry.snapshot() == {
        "enabled": False,
        "global": {"in_flight": 0, "max_in_flight": 8, "queued": 0},
        "targets": [],
    }
