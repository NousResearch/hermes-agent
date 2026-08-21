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

``_resolve_detached_python`` is a pure path helper and runs on any host.
``windowless_gateway_restart_spec`` returns its argv unchanged off Windows,
so the test that exercises the rewrite is ``windows_only`` rather than run
against a faked ``sys.platform``.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

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




@pytest.mark.windows_only
def test_restart_spec_normalizes_legacy_pythonw_argv(tmp_path):
    """A pre-rework Scheduled Task argv snapshot (leading pythonw.exe) must be
    respawned through the console python + hidden-console launch, with every
    argument after the interpreter preserved verbatim.

    ``windows_only``: ``windowless_gateway_restart_spec`` returns the argv
    untouched off Windows, so the fake was the only thing making the rewrite
    (and its ``Scripts/``-layout venv derivation) run at all.
    """
    pythonw, python = _make_venv(tmp_path, with_console_python=True)

    argv = [str(pythonw), "-m", "hermes_cli.main", "gateway", "run"]
    with mock.patch.object(
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









# ---------------------------------------------------------------------------
# _refresh_windows_gateway_launchers: hermes update regenerates launchers
# ---------------------------------------------------------------------------


@pytest.mark.windows_only
def test_refresh_retargets_legacy_cmd_task(monkeypatch, capsys):
    """A pre-#45599 Scheduled Task whose action launches the console .cmd
    wrapper must be recreated (delete+create) with the hidden VBS launcher
    during the post-update refresh — a launcher-file rewrite alone cannot
    retarget the task's action."""
    import hermes_cli.update_cmd as update_cmd

    script_path = Path(r"C:\Users\me\AppData\Local\hermes\gateway-service\Hermes_Gateway.cmd")
    called_with = {}

    def fake_install(task_name, path):
        called_with["task_name"] = task_name
        called_with["path"] = path
        return (True, "Created Scheduled Task 'Hermes_Gateway'")

    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: True)
    monkeypatch.setattr(gateway_windows, "_write_task_script", lambda: script_path)
    monkeypatch.setattr(gateway_windows, "task_launcher_is_current", lambda *a, **k: False)
    monkeypatch.setattr(gateway_windows, "_install_scheduled_task", fake_install)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway")
    monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda: script_path)

    update_cmd._refresh_windows_gateway_launchers()

    assert called_with == {"task_name": "Hermes_Gateway", "path": script_path}
    out = capsys.readouterr().out
    assert "Retargeted legacy Scheduled Task to the hidden VBS launcher" in out


@pytest.mark.windows_only
def test_refresh_leaves_current_task_alone(monkeypatch, capsys):
    """A task that already launches the hidden VBS must not be recreated."""
    import hermes_cli.update_cmd as update_cmd

    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: True)
    monkeypatch.setattr(gateway_windows, "_write_task_script", lambda: Path("unused.cmd"))
    monkeypatch.setattr(gateway_windows, "task_launcher_is_current", lambda *a, **k: True)

    update_cmd._refresh_windows_gateway_launchers()

    out = capsys.readouterr().out
    assert "Retargeted legacy" not in out
    assert "Refreshed Windows gateway launcher scripts" in out


@pytest.mark.windows_only
def test_refresh_warns_when_legacy_retarget_blocked(monkeypatch, capsys):
    """Access-denied on the recreate must surface an actionable warning, not
    fail the update (best-effort contract)."""
    import hermes_cli.update_cmd as update_cmd

    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: True)
    monkeypatch.setattr(gateway_windows, "_write_task_script", lambda: Path("unused.cmd"))
    monkeypatch.setattr(gateway_windows, "task_launcher_is_current", lambda *a, **k: False)
    monkeypatch.setattr(
        gateway_windows,
        "_install_scheduled_task",
        lambda *a, **k: (False, "schtasks /Create failed (code 5): access is denied"),
    )

    update_cmd._refresh_windows_gateway_launchers()

    out = capsys.readouterr().out
    assert "needs admin to fix" in out
    assert "hermes gateway install" in out
