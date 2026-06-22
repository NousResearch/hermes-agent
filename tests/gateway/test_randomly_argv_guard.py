"""Unit tests for the pytest argv-collision guard in conftest.

The guard reduces sys.argv to just the program name so that a gateway test
importing hermes_cli.main (whose import-time _apply_profile_override scans argv
for `-p <profile>`) does not mis-parse pytest's `-p <plugin>` flag as a Hermes
profile and sys.exit(1). See conftest._sanitize_pytest_randomly_argv.
"""
import sys

import pytest

from tests.gateway.conftest import (
    _restore_sanitized_argv,
    _sanitize_pytest_randomly_argv,
)


class _Cfg:
    pass


def _reset_capture(monkeypatch):
    import tests.gateway.conftest as cf
    monkeypatch.setattr(cf, "_ORIG_ARGV", None, raising=False)


@pytest.mark.parametrize(
    "argv",
    [
        ["pytest", "tests/gateway/", "-p", "randomly", "--randomly-seed=1"],
        ["pytest", "tests/gateway/", "-p", "no:randomly"],
        ["pytest", "-pno:cacheprovider", "tests/"],
        ["pytest", "--randomly-seed", "42", "tests/"],
    ],
)
def test_reduces_argv_to_progname(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", list(argv))
    _reset_capture(monkeypatch)

    _sanitize_pytest_randomly_argv(_Cfg())

    # everything after argv0 is gone — no `-p`, no plugin name, no seed token
    assert sys.argv == [argv[0]], sys.argv


def test_capture_once_under_reentrancy(monkeypatch):
    """A 2nd call (xdist controller+worker) must NOT re-capture the stripped argv,
    else restore would write the already-reduced argv back as the 'original'."""
    monkeypatch.setattr(sys, "argv", ["pytest", "-p", "randomly", "tests/"])
    _reset_capture(monkeypatch)

    _sanitize_pytest_randomly_argv(_Cfg())  # 1st: captures + strips
    assert sys.argv == ["pytest"]
    _sanitize_pytest_randomly_argv(_Cfg())  # 2nd: no-op (already captured)
    assert sys.argv == ["pytest"]

    _restore_sanitized_argv()  # restores the TRUE original
    assert sys.argv == ["pytest", "-p", "randomly", "tests/"], sys.argv


def test_restore_returns_original(monkeypatch):
    orig = ["pytest", "tests/gateway/", "-p", "randomly", "--randomly-seed=1"]
    monkeypatch.setattr(sys, "argv", list(orig))
    _reset_capture(monkeypatch)

    _sanitize_pytest_randomly_argv(_Cfg())
    assert sys.argv == ["pytest"]
    _restore_sanitized_argv()
    assert sys.argv == orig, sys.argv


def test_hermes_cli_main_no_systemexit_after_sanitize(monkeypatch):
    """End-to-end point: with argv reduced, _apply_profile_override does not
    SystemExit on a bogus 'randomly' profile."""
    monkeypatch.setattr(
        sys, "argv", ["pytest", "tests/gateway/", "-p", "randomly", "--randomly-seed=1"]
    )
    _reset_capture(monkeypatch)

    _sanitize_pytest_randomly_argv(_Cfg())
    assert "randomly" not in sys.argv

    from hermes_cli.main import _apply_profile_override

    try:
        _apply_profile_override()  # must not raise SystemExit
    finally:
        _restore_sanitized_argv()  # always restore, even if the call raised
