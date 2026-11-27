"""Contracts for the aarch64 SVE/SIGILL guard (``hermes_cli/_openssl_arm_guard.py``).

The guard exists because some aarch64 hosts advertise SVE/SME in ``AT_HWCAP`` while faulting on
the instructions OpenSSL's load-time cpuprobe executes, which kills the process before it can
report anything. It must therefore (a) act on real evidence only, and (b) never disable hardware
crypto on a guess.
"""

import subprocess
import sys

import pytest

from hermes_cli import _openssl_arm_guard as guard


@pytest.fixture
def arm_linux(monkeypatch):
    """Pretend we are on the platform the guard is for; tests opt out individually."""
    monkeypatch.setattr(guard, "_is_aarch64_linux", lambda: True)
    monkeypatch.delenv(guard.OPENSSL_ARMCAP_ENV, raising=False)


class _FakeCompleted:
    def __init__(self, returncode):
        self.returncode = returncode


class TestActsOnlyOnEvidence:
    def test_disables_the_probe_when_the_child_dies_on_sigill(self, arm_linux, monkeypatch):
        monkeypatch.setattr(guard, "openssl_probe_sigills", lambda: True)

        reason = guard.ensure_openssl_armcap_safe()

        assert reason and "SVE" in reason
        assert guard.os.environ[guard.OPENSSL_ARMCAP_ENV] == guard.SAFE_ARMCAP

    def test_does_nothing_when_the_child_imports_cleanly(self, arm_linux, monkeypatch):
        monkeypatch.setattr(guard, "openssl_probe_sigills", lambda: False)

        assert guard.ensure_openssl_armcap_safe() is None
        assert guard.OPENSSL_ARMCAP_ENV not in guard.os.environ

    def test_a_failing_import_is_not_evidence(self, arm_linux, monkeypatch):
        """An ordinary non-zero exit (no ``cryptography`` installed) must not trip the guard."""
        monkeypatch.setattr(
            guard.subprocess, "run",
            lambda *a, **k: _FakeCompleted(1),
        )

        assert guard.openssl_probe_sigills() is False
        assert guard.ensure_openssl_armcap_safe() is None

    def test_a_sigill_killed_child_is_evidence(self, arm_linux, monkeypatch):
        monkeypatch.setattr(
            guard.subprocess, "run",
            lambda *a, **k: _FakeCompleted(-guard.signal.SIGILL),
        )

        assert guard.openssl_probe_sigills() is True


class TestRespectsAnExistingChoice:
    def test_an_operator_set_mask_is_left_alone(self, arm_linux, monkeypatch):
        monkeypatch.setenv(guard.OPENSSL_ARMCAP_ENV, "0x3d")
        probed: list[int] = []
        monkeypatch.setattr(
            guard, "openssl_probe_sigills", lambda: probed.append(1) or True)

        assert guard.ensure_openssl_armcap_safe() is None
        # Never spend a subprocess on a decision that is already made.
        assert probed == []
        assert guard.os.environ[guard.OPENSSL_ARMCAP_ENV] == "0x3d"

    def test_second_call_is_a_no_op(self, arm_linux, monkeypatch):
        monkeypatch.setattr(guard, "openssl_probe_sigills", lambda: True)

        assert guard.ensure_openssl_armcap_safe() is not None
        assert guard.ensure_openssl_armcap_safe() is None


class TestOtherPlatformsAreUntouched:
    def test_a_non_aarch64_platform_never_probes(self, monkeypatch):
        monkeypatch.setattr(guard, "_is_aarch64_linux", lambda: False)
        monkeypatch.delenv(guard.OPENSSL_ARMCAP_ENV, raising=False)
        probed: list[int] = []
        monkeypatch.setattr(
            guard, "openssl_probe_sigills", lambda: probed.append(1) or True)

        assert guard.ensure_openssl_armcap_safe() is None
        assert probed == []

    @pytest.mark.parametrize("machine", ["x86_64", "AMD64", "riscv64", ""])
    def test_machine_detection_rejects_non_aarch64(self, monkeypatch, machine):
        monkeypatch.setattr(guard.platform, "machine", lambda: machine)
        monkeypatch.setattr(guard.sys, "platform", "linux")

        assert guard._is_aarch64_linux() is False

    @pytest.mark.parametrize("machine", ["aarch64", "arm64", "AARCH64"])
    def test_machine_detection_accepts_aarch64_spellings(self, monkeypatch, machine):
        monkeypatch.setattr(guard.platform, "machine", lambda: machine)
        monkeypatch.setattr(guard.sys, "platform", "linux")

        assert guard._is_aarch64_linux() is True

    @pytest.mark.parametrize("os_name", ["darwin", "win32"])
    def test_only_linux_is_probed(self, monkeypatch, os_name):
        monkeypatch.setattr(guard.platform, "machine", lambda: "arm64")
        monkeypatch.setattr(guard.sys, "platform", os_name)

        assert guard._is_aarch64_linux() is False


class TestTheProbeItself:
    def test_a_hung_child_is_not_evidence(self, arm_linux, monkeypatch):
        def _timeout(*_a, **_k):
            raise subprocess.TimeoutExpired(cmd="python", timeout=1)

        monkeypatch.setattr(guard.subprocess, "run", _timeout)

        assert guard.openssl_probe_sigills() is False

    def test_a_missing_interpreter_is_not_evidence(self, arm_linux, monkeypatch):
        def _boom(*_a, **_k):
            raise OSError("no such executable")

        monkeypatch.setattr(guard.subprocess, "run", _boom)

        assert guard.openssl_probe_sigills() is False

    def test_the_child_probes_with_the_variable_stripped(self, arm_linux, monkeypatch):
        """The child must reproduce a fresh, unguarded process — not inherit our answer."""
        monkeypatch.setenv(guard.OPENSSL_ARMCAP_ENV, "0")
        seen: dict = {}

        def _capture(argv, **kwargs):
            seen["env"] = kwargs.get("env")
            seen["argv"] = argv
            return _FakeCompleted(0)

        monkeypatch.setattr(guard.subprocess, "run", _capture)
        guard.openssl_probe_sigills()

        assert guard.OPENSSL_ARMCAP_ENV not in seen["env"]
        assert seen["argv"][0] == sys.executable
        assert "cryptography" in seen["argv"][-1]
