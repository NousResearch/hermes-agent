"""Tests for the process-local Z.AI concurrency gate."""

import importlib
import threading
import time

import pytest

from agent import zai_concurrency


def _reset(max_concurrent=2, timeout=0.0):
    zai_concurrency._reset_for_tests(max_concurrent, timeout)


@pytest.fixture(autouse=True)
def _stable_gate():
    _reset()
    yield
    _reset()


class TestDetection:
    @pytest.mark.parametrize(
        ("provider", "base_url"),
        [
            ("zai", ""),
            ("ZAI", ""),
            ("zhipu", ""),
            ("glm", ""),
            ("z-ai", ""),
            ("Z.AI", ""),
            ("custom", "https://api.z.ai/api/coding/paas/v4"),
            ("custom", "https://open.bigmodel.cn/api/paas/v4"),
        ],
    )
    def test_zai_provider_and_host_markers(self, provider, base_url):
        assert zai_concurrency.is_zai_request(
            provider=provider,
            model="glm-5.2",
            base_url=base_url,
        )

    @pytest.mark.parametrize(
        ("provider", "model", "base_url"),
        [
            ("openai", "gpt-5", "https://api.openai.com/v1"),
            ("anthropic", "claude-opus-4", "https://api.anthropic.com"),
            ("custom", "glm-5.2", "http://localhost:8080/v1"),
            (None, None, None),
        ],
    )
    def test_non_zai_routes_are_not_gated(self, provider, model, base_url):
        assert not zai_concurrency.is_zai_request(
            provider=provider,
            model=model,
            base_url=base_url,
        )


class TestLifecycle:
    def test_non_zai_and_disabled_gate_pass_through(self):
        with zai_concurrency.acquire_zai_slot(
            provider="openai",
            model="gpt-5",
            base_url="https://api.openai.com/v1",
        ):
            pass

        _reset(0)
        with zai_concurrency.acquire_zai_slot(
            provider="zai",
            model="glm-5.2",
            base_url="https://api.z.ai/api/coding/paas/v4",
        ):
            pass

    def test_slot_is_acquired_on_enter_not_handle_creation(self):
        _reset(1)
        sem = zai_concurrency._gate._semaphore()
        handle = zai_concurrency.acquire_zai_slot(
            provider="zai",
            model="glm-5.2",
            base_url="https://api.z.ai/api/coding/paas/v4",
        )

        assert sem.acquire(blocking=False)
        sem.release()

        with handle:
            assert not sem.acquire(blocking=False)
        assert sem.acquire(blocking=False)
        sem.release()

    def test_slot_is_released_after_exception(self):
        _reset(1)
        sem = zai_concurrency._gate._semaphore()

        with pytest.raises(RuntimeError, match="boom"):
            with zai_concurrency.acquire_zai_slot(
                provider="zai",
                model="glm-5.2",
                base_url="https://api.z.ai/api/coding/paas/v4",
            ):
                raise RuntimeError("boom")

        assert sem.acquire(blocking=False)
        sem.release()

    def test_handle_cannot_be_reentered(self):
        handle = zai_concurrency.acquire_zai_slot(
            provider="zai",
            model="glm-5.2",
            base_url="https://api.z.ai/api/coding/paas/v4",
        )
        with handle:
            pass
        with pytest.raises(RuntimeError, match="cannot be re-entered"):
            with handle:
                pass


class TestStrictBound:
    def test_parallel_calls_never_exceed_cap(self):
        _reset(2)
        observed_peak = 0
        in_flight = 0
        lock = threading.Lock()

        def _worker():
            nonlocal observed_peak, in_flight
            with zai_concurrency.acquire_zai_slot(
                provider="zai",
                model="glm-5.2",
                base_url="https://api.z.ai/api/coding/paas/v4",
            ):
                with lock:
                    in_flight += 1
                    observed_peak = max(observed_peak, in_flight)
                time.sleep(0.03)
                with lock:
                    in_flight -= 1

        threads = [threading.Thread(target=_worker) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
            assert not thread.is_alive()

        assert observed_peak == 2

    def test_timeout_raises_instead_of_proceeding_uncapped(self):
        _reset(1, timeout=0.03)
        sem = zai_concurrency._gate._semaphore()
        assert sem.acquire()

        with pytest.raises(zai_concurrency.ZaiConcurrencyTimeout):
            with zai_concurrency.acquire_zai_slot(
                provider="zai",
                model="glm-5.2",
                base_url="https://api.z.ai/api/coding/paas/v4",
            ):
                pytest.fail("a timed-out caller must never enter uncapped")

        assert not sem.acquire(blocking=False)
        sem.release()

    def test_interrupt_stops_wait_without_releasing_foreign_slot(self):
        _reset(1)
        sem = zai_concurrency._gate._semaphore()
        assert sem.acquire()

        with pytest.raises(InterruptedError):
            with zai_concurrency.acquire_zai_slot(
                provider="zai",
                model="glm-5.2",
                base_url="https://api.z.ai/api/coding/paas/v4",
                interrupt_check=lambda: True,
            ):
                pytest.fail("an interrupted caller must not enter")

        assert not sem.acquire(blocking=False)
        sem.release()

    def test_interrupt_racing_with_acquire_returns_newly_granted_slot(self):
        interrupted = False

        class _RaceSemaphore:
            releases = 0

            def acquire(self, timeout):
                nonlocal interrupted
                interrupted = True
                return True

            def release(self):
                self.releases += 1

        sem = _RaceSemaphore()
        handle = zai_concurrency._SlotHandle(
            sem,
            timeout=0.0,
            interrupt_check=lambda: interrupted,
        )

        with pytest.raises(InterruptedError):
            with handle:
                pytest.fail("an interrupt that races with acquire must win")

        assert sem.releases == 1


class TestConfiguration:
    def test_defaults_are_strict_cap_two_with_unbounded_wait(self, monkeypatch):
        monkeypatch.delenv("HERMES_ZAI_MAX_CONCURRENT", raising=False)
        monkeypatch.delenv("HERMES_ZAI_ACQUIRE_TIMEOUT_S", raising=False)
        reloaded = importlib.reload(zai_concurrency)
        try:
            assert reloaded.configured_max_concurrent() == 2
            assert reloaded._ZAI_ACQUIRE_TIMEOUT == 0.0
        finally:
            reloaded._reset_for_tests(2, 0.0)

    def test_environment_overrides_cap_and_fractional_timeout(self, monkeypatch):
        monkeypatch.setenv("HERMES_ZAI_MAX_CONCURRENT", "7")
        monkeypatch.setenv("HERMES_ZAI_ACQUIRE_TIMEOUT_S", "0.5")
        reloaded = importlib.reload(zai_concurrency)
        try:
            assert reloaded.configured_max_concurrent() == 7
            assert reloaded._ZAI_ACQUIRE_TIMEOUT == pytest.approx(0.5)
        finally:
            monkeypatch.delenv("HERMES_ZAI_MAX_CONCURRENT", raising=False)
            monkeypatch.delenv("HERMES_ZAI_ACQUIRE_TIMEOUT_S", raising=False)
            importlib.reload(zai_concurrency)._reset_for_tests(2, 0.0)
