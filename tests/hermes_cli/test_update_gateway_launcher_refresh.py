"""Legacy pythonw launcher normalization + post-update launcher refresh.

Covers the two halves of the "legacy pythonw gateways survive updates
forever" gap:

1. ``gateway_windows._resolve_detached_python`` — normalizes a legacy
   ``pythonw.exe`` interpreter (pre-aa2ae36c3f launchers / argv snapshots)
   to the sibling console ``python.exe`` so respawns and regenerated
   launchers use the hidden-console design (#54220/#56747) and don't die
   with ``RuntimeError: sys.stderr is None`` (#71671).
2. ``cli_main._refresh_windows_gateway_launchers`` — ``hermes
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
from hermes_cli import update_cmd


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


def test_refresh_windows_gateway_launchers_retargets_js_when_no_vbscript_engine(monkeypatch, tmp_path):
    """When vbscript.dll is gone, refresh must re-register the task at the .js launcher."""
    import hermes_cli.main as cli_main
    from hermes_cli.update_cmd_windows import _refresh_windows_gateway_launchers

    xml_seen = {}
    script_path = tmp_path / "Hermes_Gateway.cmd"
    script_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    monkeypatch.setattr(gateway_windows, "_vbscript_engine_available", lambda: False, raising=False)
    monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: True)
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: True)
    monkeypatch.setattr(gateway_windows, "is_startup_entry_installed", lambda: False)
    monkeypatch.setattr(gateway_windows, "_write_task_script", lambda: script_path)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway")
    monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda: script_path)
    monkeypatch.setattr(gateway_windows, "_resolve_task_user", lambda: None)

    def fake_schtasks(args):
        if args[0] == "/Delete":
            return (0, "SUCCESS", "")
        if args[0] == "/Create":
            xml_path = Path(args[args.index("/XML") + 1])
            xml_seen["text"] = xml_path.read_text(encoding="utf-16")
            return (0, "SUCCESS", "")
        return (0, "", "")

    monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)

    _refresh_windows_gateway_launchers()

    xml = xml_seen["text"]
    assert "<Command>wscript.exe</Command>" in xml
    assert "//B //Nologo" in xml
    assert "Hermes_Gateway.js" in xml
    assert "Hermes_Gateway.vbs" not in xml
    assert "cmd.exe" not in xml.lower()
    assert "powershell.exe" not in xml.lower()
    assert "<Command>pythonw.exe</Command>" not in xml








