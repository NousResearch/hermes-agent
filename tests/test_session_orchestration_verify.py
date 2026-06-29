"""
Unit tests for session_orchestration/adapters/verify.py.

Coverage
--------
1. A healthy adapter (binary found, all declared capability flags present in
   --help) → returned in the available dict.
2. A deliberately-wrong declared capability (adapter claims has_hooks=True,
   but the probed --help contains no hook flag) → that adapter is disabled,
   others remain available.
3. A missing-binary adapter (binary not on PATH) → adapter unavailable,
   no crash, others unaffected.
4. An adapter with no registered probe spec → accepted with a warning (no
   crash).
5. An adapter whose capabilities() raises → unavailable, no crash.
6. Multiple adapters: one fails, others remain available.

All tests use an injected ``FakeProbeRunner`` so no real binaries are
invoked and the test suite is hermetic.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from session_orchestration.adapters.base import AgentAdapter
from session_orchestration.adapters.verify import (
    AdapterProbeSpec,
    _ADAPTER_PROBE_SPECS,
    _check_capabilities,
    verify_adapters,
)
from session_orchestration.types import Capabilities, SessionHandle, SessionLifecycle


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class FakeProbeRunner:
    """Injectable test double for ProbeRunner.

    ``known_binaries``: set of binary names considered on PATH.
    ``help_outputs``:   mapping binary → simulated --help text.
    """

    def __init__(
        self,
        known_binaries: set[str] | None = None,
        help_outputs: dict[str, str] | None = None,
    ) -> None:
        self._known = known_binaries or set()
        self._help = help_outputs or {}

    def which(self, binary: str) -> str | None:
        return f"/usr/local/bin/{binary}" if binary in self._known else None

    def help_text(self, binary: str) -> str:
        return self._help.get(binary, "")


def _minimal_adapter(caps: Capabilities) -> AgentAdapter:
    """Create a throwaway AgentAdapter stub that returns ``caps``."""

    class _StubAdapter(AgentAdapter):
        def capabilities(self) -> Capabilities:  # type: ignore[override]
            return caps

        def launch(self, workdir: str, prompt: str) -> SessionHandle:
            raise NotImplementedError

        def drive(self, handle: SessionHandle, message: str) -> None:
            raise NotImplementedError

        def detect(self, handle: SessionHandle) -> SessionLifecycle:
            raise NotImplementedError

        def resume(self, handle: SessionHandle, prompt: str) -> None:
            raise NotImplementedError

        def terminate(self, handle: SessionHandle) -> None:
            raise NotImplementedError

    return _StubAdapter()


# A probe spec that covers the two test-adapter class names below.
_TEST_SPECS: dict[str, AdapterProbeSpec] = {
    "HealthyAdapter": AdapterProbeSpec(
        binary="healthybin",
        has_hooks_flag="--hook",
        supports_print_mode_flag="--print",
    ),
    "WrongCapAdapter": AdapterProbeSpec(
        binary="wrongbin",
        has_hooks_flag="--hook",
        supports_print_mode_flag="--print",
    ),
    "MissingBinAdapter": AdapterProbeSpec(
        binary="missingbin",
        has_hooks_flag="--hook",
    ),
}


def _healthy_runner() -> FakeProbeRunner:
    return FakeProbeRunner(
        known_binaries={"healthybin", "wrongbin"},
        help_outputs={
            "healthybin": "Usage: healthybin\n  --hook  lifecycle hook\n  --print  print mode\n",
            "wrongbin": "Usage: wrongbin\n  no hook flag here\n  --print  print mode\n",
        },
    )


# ---------------------------------------------------------------------------
# Named adapter stubs for use with _TEST_SPECS
# ---------------------------------------------------------------------------


class HealthyAdapter(AgentAdapter):
    """Declares has_hooks=True + supports_print_mode=True; probe confirms both."""

    def capabilities(self) -> Capabilities:
        return Capabilities(has_hooks=True, supports_print_mode=True)

    def launch(self, workdir: str, prompt: str) -> SessionHandle:
        raise NotImplementedError

    def drive(self, handle: SessionHandle, message: str) -> None:
        raise NotImplementedError

    def detect(self, handle: SessionHandle) -> SessionLifecycle:
        raise NotImplementedError

    def resume(self, handle: SessionHandle, prompt: str) -> None:
        raise NotImplementedError

    def terminate(self, handle: SessionHandle) -> None:
        raise NotImplementedError


class WrongCapAdapter(AgentAdapter):
    """Declares has_hooks=True but its binary --help has no '--hook' flag."""

    def capabilities(self) -> Capabilities:
        return Capabilities(has_hooks=True, supports_print_mode=True)

    def launch(self, workdir: str, prompt: str) -> SessionHandle:
        raise NotImplementedError

    def drive(self, handle: SessionHandle, message: str) -> None:
        raise NotImplementedError

    def detect(self, handle: SessionHandle) -> SessionLifecycle:
        raise NotImplementedError

    def resume(self, handle: SessionHandle, prompt: str) -> None:
        raise NotImplementedError

    def terminate(self, handle: SessionHandle) -> None:
        raise NotImplementedError


class MissingBinAdapter(AgentAdapter):
    """Declares has_hooks=True; binary is not on PATH at all."""

    def capabilities(self) -> Capabilities:
        return Capabilities(has_hooks=True)

    def launch(self, workdir: str, prompt: str) -> SessionHandle:
        raise NotImplementedError

    def drive(self, handle: SessionHandle, message: str) -> None:
        raise NotImplementedError

    def detect(self, handle: SessionHandle) -> SessionLifecycle:
        raise NotImplementedError

    def resume(self, handle: SessionHandle, prompt: str) -> None:
        raise NotImplementedError

    def terminate(self, handle: SessionHandle) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCheckCapabilities:
    """Unit tests for the pure _check_capabilities helper."""

    def test_all_ok_returns_empty(self) -> None:
        caps = Capabilities(has_hooks=True, supports_print_mode=True)
        spec = AdapterProbeSpec(
            binary="bin",
            has_hooks_flag="--hook",
            supports_print_mode_flag="--print",
        )
        runner = FakeProbeRunner(
            known_binaries={"bin"},
            help_outputs={"bin": "--hook  hook flag\n--print  print flag\n"},
        )
        mismatches = _check_capabilities(caps, spec, runner)
        assert mismatches == []

    def test_missing_binary_returns_mismatch(self) -> None:
        caps = Capabilities(has_hooks=True)
        spec = AdapterProbeSpec(binary="ghostbin", has_hooks_flag="--hook")
        runner = FakeProbeRunner(known_binaries=set())
        mismatches = _check_capabilities(caps, spec, runner)
        assert len(mismatches) == 1
        assert "not found on PATH" in mismatches[0]

    def test_missing_hook_flag_in_help(self) -> None:
        caps = Capabilities(has_hooks=True)
        spec = AdapterProbeSpec(binary="mybin", has_hooks_flag="--hook")
        runner = FakeProbeRunner(
            known_binaries={"mybin"},
            help_outputs={"mybin": "no hook flag here"},
        )
        mismatches = _check_capabilities(caps, spec, runner)
        assert len(mismatches) == 1
        assert "has_hooks=True" in mismatches[0]
        assert "--hook" in mismatches[0]

    def test_skip_probe_when_flag_is_none(self) -> None:
        """A None flag means 'skip this probe'; no mismatch even if help is empty."""
        caps = Capabilities(rpc_mode=True)
        spec = AdapterProbeSpec(binary="mybin", rpc_mode_flag=None)
        runner = FakeProbeRunner(
            known_binaries={"mybin"},
            help_outputs={"mybin": ""},
        )
        mismatches = _check_capabilities(caps, spec, runner)
        assert mismatches == []

    def test_false_declared_cap_not_probed(self) -> None:
        """If declared has_hooks=False, there is no mismatch even if the flag is absent."""
        caps = Capabilities(has_hooks=False)
        spec = AdapterProbeSpec(binary="mybin", has_hooks_flag="--hook")
        runner = FakeProbeRunner(
            known_binaries={"mybin"},
            help_outputs={"mybin": "nothing useful"},
        )
        mismatches = _check_capabilities(caps, spec, runner)
        assert mismatches == []

    def test_multiple_mismatches_all_reported(self) -> None:
        caps = Capabilities(has_hooks=True, supports_print_mode=True, json_mode=True)
        spec = AdapterProbeSpec(
            binary="bin",
            has_hooks_flag="--hook",
            supports_print_mode_flag="--print",
            json_mode_flag="--json",
        )
        runner = FakeProbeRunner(
            known_binaries={"bin"},
            help_outputs={"bin": "no relevant flags at all"},
        )
        mismatches = _check_capabilities(caps, spec, runner)
        assert len(mismatches) == 3


class TestVerifyAdapters:
    """Integration tests for verify_adapters — the function the watcher calls."""

    def test_healthy_adapter_is_available(self) -> None:
        adapters: dict[str, AgentAdapter] = {"healthy": HealthyAdapter()}
        result = verify_adapters(adapters, probe_runner=_healthy_runner(), probe_specs=_TEST_SPECS)
        assert "healthy" in result
        assert result["healthy"] is adapters["healthy"]

    def test_wrong_capability_disables_adapter(self) -> None:
        """Adapter claiming has_hooks=True whose --help has no '--hook' → disabled."""
        adapters: dict[str, AgentAdapter] = {"wrong": WrongCapAdapter()}
        result = verify_adapters(adapters, probe_runner=_healthy_runner(), probe_specs=_TEST_SPECS)
        assert "wrong" not in result

    def test_missing_binary_disables_adapter(self) -> None:
        """Adapter whose binary is not on PATH → unavailable, no crash."""
        runner = FakeProbeRunner(known_binaries=set())  # no binary available
        adapters: dict[str, AgentAdapter] = {"missing": MissingBinAdapter()}
        result = verify_adapters(adapters, probe_runner=runner, probe_specs=_TEST_SPECS)
        assert "missing" not in result

    def test_only_failing_adapter_is_disabled(self) -> None:
        """When one adapter fails and another passes, only the healthy one is returned."""
        adapters: dict[str, AgentAdapter] = {
            "healthy": HealthyAdapter(),
            "wrong": WrongCapAdapter(),
        }
        result = verify_adapters(adapters, probe_runner=_healthy_runner(), probe_specs=_TEST_SPECS)
        assert "healthy" in result
        assert "wrong" not in result

    def test_missing_binary_does_not_crash_other_adapters(self) -> None:
        """A missing-binary adapter must not prevent healthy adapters from being verified."""
        runner = FakeProbeRunner(
            known_binaries={"healthybin"},
            help_outputs={"healthybin": "--hook  h\n--print  p\n"},
        )
        adapters: dict[str, AgentAdapter] = {
            "healthy": HealthyAdapter(),
            "missing": MissingBinAdapter(),
        }
        result = verify_adapters(adapters, probe_runner=runner, probe_specs=_TEST_SPECS)
        assert "healthy" in result
        assert "missing" not in result

    def test_no_probe_spec_adapter_accepted_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Adapter with no registered spec → accepted (cannot verify) + warning logged."""

        class UnknownAdapter(AgentAdapter):
            def capabilities(self) -> Capabilities:
                return Capabilities()

            def launch(self, workdir: str, prompt: str) -> SessionHandle:
                raise NotImplementedError

            def drive(self, handle: SessionHandle, message: str) -> None:
                raise NotImplementedError

            def detect(self, handle: SessionHandle) -> SessionLifecycle:
                raise NotImplementedError

            def resume(self, handle: SessionHandle, prompt: str) -> None:
                raise NotImplementedError

            def terminate(self, handle: SessionHandle) -> None:
                raise NotImplementedError

        adapters: dict[str, AgentAdapter] = {"unknown": UnknownAdapter()}
        with caplog.at_level(logging.WARNING, logger="session_orchestration.adapters.verify"):
            result = verify_adapters(adapters, probe_runner=FakeProbeRunner(), probe_specs={})
        assert "unknown" in result
        assert any("no probe spec" in r.message for r in caplog.records)

    def test_capabilities_raises_disables_adapter(self, caplog: pytest.LogCaptureFixture) -> None:
        """If capabilities() raises, the adapter is marked unavailable, no crash."""

        class BrokenAdapter(AgentAdapter):
            def capabilities(self) -> Capabilities:
                raise RuntimeError("broken!")

            def launch(self, workdir: str, prompt: str) -> SessionHandle:
                raise NotImplementedError

            def drive(self, handle: SessionHandle, message: str) -> None:
                raise NotImplementedError

            def detect(self, handle: SessionHandle) -> SessionLifecycle:
                raise NotImplementedError

            def resume(self, handle: SessionHandle, prompt: str) -> None:
                raise NotImplementedError

            def terminate(self, handle: SessionHandle) -> None:
                raise NotImplementedError

        specs = {"BrokenAdapter": AdapterProbeSpec(binary="bin")}
        adapters: dict[str, AgentAdapter] = {"broken": BrokenAdapter()}
        with caplog.at_level(logging.ERROR, logger="session_orchestration.adapters.verify"):
            result = verify_adapters(
                adapters,
                probe_runner=FakeProbeRunner(known_binaries={"bin"}),
                probe_specs=specs,
            )
        assert "broken" not in result
        assert any("capabilities()" in r.message for r in caplog.records)

    def test_empty_adapters_returns_empty(self) -> None:
        result = verify_adapters({}, probe_runner=FakeProbeRunner())
        assert result == {}

    def test_error_is_logged_not_raised_on_mismatch(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Capability mismatch must log an error, never raise."""
        adapters: dict[str, AgentAdapter] = {"wrong": WrongCapAdapter()}
        with caplog.at_level(logging.ERROR, logger="session_orchestration.adapters.verify"):
            result = verify_adapters(
                adapters, probe_runner=_healthy_runner(), probe_specs=_TEST_SPECS
            )
        assert "wrong" not in result
        error_msgs = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("capability mismatch" in m for m in error_msgs)
        assert any("disabled" in m for m in error_msgs)
