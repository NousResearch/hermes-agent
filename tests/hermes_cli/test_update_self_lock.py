"""Regression coverage for the updater self-lock preflight (#83569).

``_detect_venv_python_processes`` excludes the calling process and its
ancestors by design — a CLI ``hermes update`` IS the venv python.  Before
this guard, an updater that had already imported a native venv extension
(e.g. ``cryptography.hazmat.bindings._rust``, mapped the moment
``hermes_cli.main`` resolved external secret sources) sailed through every
preflight and then died mid-sync with ``os error 5`` when ``uv`` tried to
replace the mapped ``.pyd``, stranding the venv half-updated:

- ``cryptography-48.0.1.dist-info`` left without ``RECORD`` (uninstall
  finished, reinstall could not replace ``_rust.pyd``)
- ``.update-incomplete`` marker written but no actionable guidance
- every retry (git path + ZIP fallback) failing identically

The self-lock preflight refuses the update BEFORE touching the checkout,
writes the update-incomplete marker so the next fresh launch completes the
install, and exits 2 like the other preflight refusals.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.main as cli_main


# ---------------------------------------------------------------------------
# _detect_self_loaded_native_modules
# ---------------------------------------------------------------------------


@patch.object(cli_main, "_is_windows", return_value=False)
def test_self_lock_detection_is_noop_off_windows(_winp):
    with patch.dict(sys.modules, {"cryptography.hazmat.bindings._rust": MagicMock()}):
        assert cli_main._detect_self_loaded_native_modules() == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_self_lock_detection_flags_loaded_rust_module(_winp):
    with patch.dict(sys.modules, {"cryptography.hazmat.bindings._rust": MagicMock()}):
        assert cli_main._detect_self_loaded_native_modules() == [
            "cryptography (_rust.pyd)"
        ]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_self_lock_detection_clean_when_rust_not_loaded(_winp):
    # The lazy-import startup path (#73381, #83569) must NOT trip the guard:
    # no cryptography module in sys.modules → no lock → update proceeds.
    sys.modules.pop("cryptography.hazmat.bindings._rust", None)
    assert cli_main._detect_self_loaded_native_modules() == []


# ---------------------------------------------------------------------------
# Preflight wiring inside _cmd_update_impl
# ---------------------------------------------------------------------------


def _update_args(**overrides):
    defaults = dict(
        gateway=False,
        check=False,
        no_backup=True,
        backup=False,
        yes=True,
        branch=None,
        force=False,
        force_venv=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _run_update_until_sync(args, *, self_locked: list[str]):
    """Drive _cmd_update_impl to just past the self-lock preflight.

    Everything before the preflight is stubbed. The first statement AFTER it
    is ``git_dir = PROJECT_ROOT / ".git"`` — a PROJECT_ROOT sentinel whose
    ``__truediv__`` raises marks 'preflight passed'.
    """

    class _PastPreflight(Exception):
        pass

    class _RootSentinel:
        def __truediv__(self, _other):
            raise _PastPreflight

    marker_writes = []

    with patch.object(cli_main, "_is_windows", return_value=True), patch.object(
        cli_main, "_venv_scripts_dir", return_value=None
    ), patch.object(cli_main, "_run_pre_update_backup"), patch.object(
        cli_main, "_pause_windows_gateways_for_update", return_value=None
    ), patch.object(
        cli_main, "_resume_windows_gateways_after_update"
    ), patch.object(
        cli_main, "_detect_venv_python_processes", return_value=[]
    ), patch.object(
        cli_main, "_detect_self_loaded_native_modules", return_value=self_locked
    ), patch.object(
        cli_main,
        "_write_update_incomplete_marker",
        side_effect=lambda: marker_writes.append("written"),
    ), patch.object(
        cli_main, "PROJECT_ROOT", _RootSentinel()
    ):
        try:
            cli_main._cmd_update_impl(args, gateway_mode=False)
        except _PastPreflight:
            return "past_preflight", marker_writes
        except SystemExit as exc:
            return f"exit_{exc.code}", marker_writes
    return "returned", marker_writes


def test_self_lock_preflight_refuses_and_defers(capsys):
    result, markers = _run_update_until_sync(
        _update_args(), self_locked=["cryptography (_rust.pyd)"]
    )
    assert result == "exit_2"
    # Deferral contract: marker dropped so the next fresh launch completes
    # the install before anything imports the native extension.
    assert markers == ["written"]
    out = capsys.readouterr().out
    assert "cryptography (_rust.pyd)" in out
    assert "deferred" in out


def test_self_lock_preflight_not_bypassed_by_force_venv(capsys):
    """--force-venv escapes EXTERNAL holders; it cannot unmap an image from
    the running process, so the self-lock refusal must stand."""
    result, markers = _run_update_until_sync(
        _update_args(force=True, force_venv=True),
        self_locked=["cryptography (_rust.pyd)"],
    )
    assert result == "exit_2"
    assert markers == ["written"]


def test_self_lock_preflight_passes_when_nothing_loaded():
    result, markers = _run_update_until_sync(_update_args(), self_locked=[])
    assert result == "past_preflight"
    assert markers == []
