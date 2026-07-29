"""Legacy pythonw launcher normalization + post-update launcher refresh.

Covers the two halves of the "legacy pythonw gateways survive updates
forever" gap:

1. ``gateway_windows._resolve_detached_python`` — normalizes a legacy
   ``pythonw.exe`` interpreter (pre-aa2ae36c3f launchers / argv snapshots)
   to the sibling console ``python.exe`` so respawns and regenerated
   launchers use the hidden-console design (#54220/#56747) and don't die
   with ``RuntimeError: sys.stderr is None`` (#71671).
2. ``hermes_cli.main._refresh_windows_gateway_launchers`` — ``hermes
   update`` regenerates the installed Scheduled Task / Startup launcher
   scripts instead of leaving install-time artifacts stale forever.

Windows-specific paths are exercised via ``_is_windows`` patching so they
run on any host (same approach as test_update_venv_health).
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import hermes_cli.gateway_windows as gateway_windows
import hermes_cli.main as cli_main


# ---------------------------------------------------------------------------
# _resolve_detached_python: legacy pythonw normalization
# ---------------------------------------------------------------------------


def _make_venv(tmp_path: Path, *, with_console_python: bool) -> tuple[Path, Path]:
    scripts = tmp_path / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    pythonw = scripts / "pythonw.exe"
    pythonw.write_text("", encoding="utf-8")
    python = scripts / "python.exe"
    if with_console_python:
        python.write_text("", encoding="utf-8")
    return pythonw, python


def test_resolve_detached_python_swaps_legacy_pythonw_for_console_sibling(tmp_path):
    pythonw, python = _make_venv(tmp_path, with_console_python=True)

    exe, venv_dir, extra = gateway_windows._resolve_detached_python(str(pythonw))

    assert exe == str(python)
    assert venv_dir == tmp_path / "venv"
    assert extra == []


def test_resolve_detached_python_keeps_pythonw_when_no_console_sibling(tmp_path):
    """A failed respawn is worse than a console-less gateway — never swap to
    an interpreter that doesn't exist."""
    pythonw, _python = _make_venv(tmp_path, with_console_python=False)

    exe, venv_dir, extra = gateway_windows._resolve_detached_python(str(pythonw))

    assert exe == str(pythonw)
    assert venv_dir == tmp_path / "venv"
    assert extra == []


def test_resolve_detached_python_leaves_console_python_untouched(tmp_path):
    _pythonw, python = _make_venv(tmp_path, with_console_python=True)

    exe, venv_dir, _extra = gateway_windows._resolve_detached_python(str(python))

    assert exe == str(python)
    assert venv_dir == tmp_path / "venv"


def test_restart_spec_normalizes_legacy_pythonw_argv(tmp_path):
    """A pre-rework Scheduled Task argv snapshot (leading pythonw.exe) must be
    respawned through the console python + hidden-console launch, with every
    argument after the interpreter preserved verbatim."""
    pythonw, python = _make_venv(tmp_path, with_console_python=True)

    # Pre-import so the function's lazy imports resolve from sys.modules
    # instead of re-importing under the win32 platform patch (see the
    # TestWindowlessGatewayRestartSpec comment in
    # tests/tools/test_windows_native_support.py).
    import hermes_cli.config  # noqa: F401
    import hermes_cli.gateway  # noqa: F401

    argv = [str(pythonw), "-m", "hermes_cli.main", "gateway", "run"]
    with mock.patch.object(gateway_windows.sys, "platform", "win32"), mock.patch.object(
        gateway_windows, "_stable_gateway_working_dir", return_value=str(tmp_path)
    ), mock.patch("hermes_cli.config.get_hermes_home", return_value=str(tmp_path)):
        new_argv, cwd, env = gateway_windows.windowless_gateway_restart_spec(list(argv))

    assert new_argv[0] == str(python)
    assert new_argv[1:] == argv[1:]
    assert cwd == str(tmp_path)
    assert env["VIRTUAL_ENV"] == str(tmp_path / "venv")


# ---------------------------------------------------------------------------
# _refresh_windows_gateway_launchers: hermes update regenerates launchers
# ---------------------------------------------------------------------------


def test_refresh_is_noop_off_windows():
    with mock.patch.object(cli_main, "_is_windows", return_value=False), mock.patch.object(
        gateway_windows, "is_installed"
    ) as is_installed:
        cli_main._refresh_windows_gateway_launchers()
    is_installed.assert_not_called()


@mock.patch.object(cli_main, "_is_windows", return_value=True)
def test_refresh_rewrites_launchers_when_installed(_winp):
    with mock.patch.object(gateway_windows, "is_installed", return_value=True), mock.patch.object(
        gateway_windows, "_write_task_script"
    ) as write_script:
        cli_main._refresh_windows_gateway_launchers()
    write_script.assert_called_once_with()


@mock.patch.object(cli_main, "_is_windows", return_value=True)
def test_refresh_skips_when_no_autostart_installed(_winp):
    with mock.patch.object(gateway_windows, "is_installed", return_value=False), mock.patch.object(
        gateway_windows, "_write_task_script"
    ) as write_script:
        cli_main._refresh_windows_gateway_launchers()
    write_script.assert_not_called()


@mock.patch.object(cli_main, "_is_windows", return_value=True)
def test_refresh_failure_never_raises(_winp):
    with mock.patch.object(gateway_windows, "is_installed", return_value=True), mock.patch.object(
        gateway_windows, "_write_task_script", side_effect=OSError("locked")
    ):
        cli_main._refresh_windows_gateway_launchers()  # must not raise


@mock.patch.object(cli_main, "_is_windows", return_value=True)
def test_resume_after_update_refreshes_launchers_first(_winp):
    """The resume path regenerates launchers before any respawn, including the
    cold-start-only shape (no gateway was running, autostart installed)."""
    calls: list[str] = []
    with mock.patch.object(
        cli_main,
        "_refresh_windows_gateway_launchers",
        side_effect=lambda: calls.append("refresh"),
    ), mock.patch.object(
        cli_main,
        "_cold_start_windows_gateway_after_update",
        side_effect=lambda: calls.append("cold_start"),
    ):
        cli_main._resume_windows_gateways_after_update(
            {
                "resume_needed": True,
                "profiles": {},
                "unmapped_pids": [],
                "unmapped": [],
                "cold_start_if_installed": True,
            }
        )

    assert calls == ["refresh", "cold_start"]


def test_resume_after_update_noop_token_skips_refresh():
    with mock.patch.object(cli_main, "_refresh_windows_gateway_launchers") as refresh:
        cli_main._resume_windows_gateways_after_update(None)
        cli_main._resume_windows_gateways_after_update({"resume_needed": False})
    refresh.assert_not_called()
