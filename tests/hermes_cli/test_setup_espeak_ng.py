"""Tests for espeak-ng installation subprocess handling in setup.py.

These tests verify the regression contract for PR #64802:
- timeout=300 and stdin=subprocess.DEVNULL are passed to every platform
  branch's subprocess.run call.
- subprocess.TimeoutExpired is caught alongside CalledProcessError /
  FileNotFoundError and causes _install_neutts_deps to return False.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_setup_helpers(monkeypatch):
    """Stub out helpers that would otherwise prompt or print during tests."""
    from hermes_cli import setup as setup_mod

    monkeypatch.setattr(setup_mod, "_check_espeak_ng", lambda: False)
    monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda *a, **kw: True)
    monkeypatch.setattr(setup_mod, "print_success", lambda *a: None)
    monkeypatch.setattr(setup_mod, "print_warning", lambda *a: None)
    monkeypatch.setattr(setup_mod, "print_info", lambda *a: None)
    monkeypatch.setattr(setup_mod, "print_error", lambda *a: None)


# ---------------------------------------------------------------------------
# Platform-branch subprocess.kwargs tests
# ---------------------------------------------------------------------------


def _install_for_platform(platform):
    """Patch sys.platform and subprocess.run globally, capturing all calls,
    then call _install_neutts_deps."""
    captured = []

    def _fake_run(*args, **kwargs):
        captured.append({"args": list(args), "kwargs": dict(kwargs)})
        return SimpleNamespace(returncode=0)

    with (
        patch("subprocess.run", _fake_run),
        patch("sys.platform", platform),
    ):
        from hermes_cli.setup import _install_neutts_deps
        result = _install_neutts_deps()

    return result, captured


def _find_espeak_call(captured):
    """Find the subprocess.run call that installs espeak-ng."""
    for call in captured:
        cmd = call["args"][0] if call["args"] else []
        if "espeak-ng" in str(cmd):
            return call
    return None


class TestEspeakNgSubprocessKwargs:
    """Verify each platform branch calls subprocess.run with timeout=300
    and stdin=subprocess.DEVNULL."""

    def test_darwin_branch_has_timeout_and_devnull(self):
        """macOS (darwin) brew install gets timeout=300 and stdin=DEVNULL."""
        result, captured = _install_for_platform("darwin")
        assert result is True
        espeak_call = _find_espeak_call(captured)
        assert espeak_call is not None
        assert espeak_call["kwargs"]["timeout"] == 300
        assert espeak_call["kwargs"]["stdin"] is subprocess.DEVNULL
        assert espeak_call["args"][0] == ["brew", "install", "espeak-ng"]

    def test_win32_branch_has_timeout_and_devnull(self):
        """Windows (win32) choco install gets timeout=300 and stdin=DEVNULL."""
        result, captured = _install_for_platform("win32")
        assert result is True
        espeak_call = _find_espeak_call(captured)
        assert espeak_call is not None
        assert espeak_call["kwargs"]["timeout"] == 300
        assert espeak_call["kwargs"]["stdin"] is subprocess.DEVNULL
        assert espeak_call["args"][0] == ["choco", "install", "espeak-ng", "-y"]

    def test_linux_branch_has_timeout_and_devnull(self):
        """Linux (generic) apt install gets timeout=300 and stdin=DEVNULL."""
        result, captured = _install_for_platform("linux")
        assert result is True
        espeak_call = _find_espeak_call(captured)
        assert espeak_call is not None
        assert espeak_call["kwargs"]["timeout"] == 300
        assert espeak_call["kwargs"]["stdin"] is subprocess.DEVNULL
        assert espeak_call["args"][0] == ["sudo", "apt", "install", "-y", "espeak-ng"]


# ---------------------------------------------------------------------------
# TimeoutExpired handling
# ---------------------------------------------------------------------------


class TestEspeakNgTimeoutExpired:
    """subprocess.TimeoutExpired must be caught and _install_neutts_deps
    returns False without raising."""

    def _timeout_run(self, *args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=300)

    def test_timeout_returns_false_no_raise(self):
        """TimeoutExpired is caught; _install_neutts_deps returns False."""
        with (
            patch("subprocess.run", self._timeout_run),
            patch("sys.platform", "darwin"),
        ):
            from hermes_cli.setup import _install_neutts_deps
            result = _install_neutts_deps()
        assert result is False

    def test_timeout_warned(self):
        """A warning is printed when timeout occurs."""
        warnings = []
        from hermes_cli import setup as setup_mod
        orig_warn = setup_mod.print_warning
        setup_mod.print_warning = lambda *a: warnings.append(a)
        try:
            with (
                patch("subprocess.run", self._timeout_run),
                patch("sys.platform", "linux"),
            ):
                from hermes_cli.setup import _install_neutts_deps
                result = _install_neutts_deps()
            assert result is False
            assert any("Could not install espeak-ng" in str(w) for w in warnings)
        finally:
            setup_mod.print_warning = orig_warn


# ---------------------------------------------------------------------------
# CalledProcessError handling (existing behaviour, sanity check)
# ---------------------------------------------------------------------------


class TestEspeakNgCalledProcessError:
    """Non-zero exit code from the package manager is caught gracefully."""

    def _failing_run(self, *args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    def test_called_process_error_returns_false(self):
        with (
            patch("subprocess.run", self._failing_run),
            patch("sys.platform", "darwin"),
        ):
            from hermes_cli.setup import _install_neutts_deps
            result = _install_neutts_deps()
        assert result is False
