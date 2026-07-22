"""Tests for the process-local Z.AI concurrency gate."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from agent import zai_concurrency
from run_agent import AIAgent


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
            ("custom", "api.z.ai/api/coding/paas/v4"),
            ("custom", "https://edge.api.z.ai/v1"),
            ("custom", "api.z.ai/v1?next=http://elsewhere.example"),
            ("custom", "https:api.z.ai/api/coding/paas/v4"),
            (" zai ", ""),
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
            ("custom", "glm-5.2", "https://api.z.ai.evil.example/v1"),
            ("custom", "glm-5.2", "https://example.com/proxy/api.z.ai"),
            ("custom", "glm-5.2", "evil.example/v1?u=http://api.z.ai"),
            ("custom", "glm-5.2", "https:evil.example/v1"),
            (None, None, None),
        ],
    )
    def test_non_zai_routes_are_not_gated(self, provider, model, base_url):
        assert not zai_concurrency.is_zai_request(
            provider=provider,
            model=model,
            base_url=base_url,
        )

    @pytest.mark.parametrize(
        "provider",
        ["zai", "ZAI", "glm", "zhipu", "z-ai", "z.ai"],
    )
    def test_zai_provider_with_overridden_non_zai_host_is_not_gated(self, provider):
        # A ``zai``/``glm`` provider whose GLM_BASE_URL is overridden to a
        # local or third-party endpoint must not be throttled: the resolved
        # host is authoritative and does not point at Z.AI.
        assert not zai_concurrency.is_zai_request(
            provider=provider,
            model="glm-5.2",
            base_url="http://localhost:8080/v1",
        )
        assert not zai_concurrency.is_zai_request(
            provider=provider,
            model="glm-5.2",
            base_url="https://glm.internal.corp/v1",
        )

    @pytest.mark.parametrize(
        "base_url",
        ["", None, "   "],
    )
    def test_zai_provider_without_a_resolvable_host_falls_back_to_name(self, base_url):
        # No parseable host → the built-in Z.AI provider name is the fallback
        # signal, so the gate still applies.
        assert zai_concurrency.is_zai_request(
            provider="zai",
            model="glm-5.2",
            base_url=base_url,
        )

    @pytest.mark.parametrize(
        "base_url",
        [
            "//api.z.ai/api/coding/paas/v4",
            "//open.bigmodel.cn/api/paas/v4",
        ],
    )
    def test_protocol_relative_zai_host_is_gated(self, base_url):
        assert zai_concurrency.is_zai_request(
            provider="custom",
            model="glm-5.2",
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


def test_direct_agent_request_path_holds_zai_slot(monkeypatch):
    _reset(1)
    sem = zai_concurrency._gate._semaphore()
    observed = []

    agent = AIAgent(
        api_key="test-key",
        base_url="https://api.z.ai/api/coding/paas/v4",
        model="glm-5.2",
        provider="zai",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=[],
        max_iterations=1,
    )

    def _fake_api_call(_api_kwargs, **_kwargs):
        acquired = sem.acquire(blocking=False)
        if acquired:
            sem.release()
        observed.append(acquired)
        message = type("_Message", (), {"content": "done", "tool_calls": []})()
        choice = type(
            "_Choice",
            (),
            {"message": message, "finish_reason": "stop"},
        )()
        return type(
            "_Response",
            (),
            {"choices": [choice], "usage": None, "model": "glm-5.2"},
        )()

    monkeypatch.setattr(agent, "_interruptible_api_call", _fake_api_call)
    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", _fake_api_call)
    monkeypatch.setattr(agent, "_persist_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent, "_save_trajectory", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent, "_cleanup_task_resources", lambda *args, **kwargs: None)

    result = agent.run_conversation("hello")

    assert result["final_response"] == "done"
    assert observed == [False]


class TestConfiguration:
    @staticmethod
    def _subprocess_env(home: Path, **overrides: str) -> dict[str, str]:
        env = os.environ.copy()
        env["HERMES_HOME"] = str(home)
        repo_root = str(Path(__file__).resolve().parents[2])
        existing_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            repo_root if not existing_path else repo_root + os.pathsep + existing_path
        )
        env.update(overrides)
        return env

    @classmethod
    def _set_profile_config(cls, home: Path, max_concurrent: str, timeout: str) -> None:
        script = (
            "from hermes_cli.config import set_config_value;"
            f"set_config_value('providers.zai.max_concurrent', {max_concurrent!r});"
            "set_config_value("
            f"'providers.zai.acquire_timeout_seconds', {timeout!r})"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            env=cls._subprocess_env(home),
            check=True,
            capture_output=True,
            text=True,
        )

    @classmethod
    def _read_profile_limits(cls, home: Path, **env_overrides: str) -> tuple[int, float]:
        script = (
            "from agent.zai_concurrency import "
            "configured_acquire_timeout, configured_max_concurrent;"
            "print(f'{configured_max_concurrent()}|{configured_acquire_timeout()}')"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            env=cls._subprocess_env(home, **env_overrides),
            check=True,
            capture_output=True,
            text=True,
        )
        raw_max, raw_timeout = result.stdout.strip().split("|")
        return int(raw_max), float(raw_timeout)

    def test_defaults_are_strict_cap_two_with_unbounded_wait(self, tmp_path):
        assert self._read_profile_limits(tmp_path / "default") == (2, 0.0)

    def test_config_set_is_profile_scoped_and_accepts_fractional_timeout(self, tmp_path):
        profile_one = tmp_path / "profile-one"
        profile_two = tmp_path / "profile-two"
        self._set_profile_config(profile_one, "1", "0.25")
        self._set_profile_config(profile_two, "4", "1.5")

        assert self._read_profile_limits(profile_one) == (1, 0.25)
        assert self._read_profile_limits(profile_two) == (4, 1.5)

    def test_zero_disables_and_legacy_environment_controls_are_ignored(self, tmp_path):
        profile = tmp_path / "disabled"
        self._set_profile_config(profile, "0", "0")

        assert self._read_profile_limits(
            profile,
            HERMES_ZAI_MAX_CONCURRENT="9",
            HERMES_ZAI_ACQUIRE_TIMEOUT_S="12.5",
        ) == (0, 0.0)

    def test_invalid_config_values_fall_back_to_safe_defaults(self, tmp_path):
        profile = tmp_path / "invalid"
        self._set_profile_config(profile, "-1", "not-a-number")

        assert self._read_profile_limits(profile) == (2, 0.0)

    @pytest.mark.parametrize(
        ("max_concurrent", "timeout"),
        [
            ("1.5", "0"),
            ("2", "inf"),
            ("2", "nan"),
        ],
    )
    def test_non_integral_or_non_finite_config_falls_back_safely(
        self,
        tmp_path,
        max_concurrent,
        timeout,
    ):
        profile = tmp_path / f"invalid-{max_concurrent}-{timeout}"
        self._set_profile_config(profile, max_concurrent, timeout)

        assert self._read_profile_limits(profile) == (2, 0.0)
