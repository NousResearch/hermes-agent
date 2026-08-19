"""The Windows console-shim update self-lock (#88838, #89599, #86093).

``venv\\Scripts\\hermes.exe`` is a launcher that runs the interpreter with the
shim itself as its script, keeping the file open without FILE_SHARE_DELETE for
the whole command. An update started that way must therefore replace a file it
is holding, which Windows refuses — so ``hermes update`` re-runs itself under
``venv\\Scripts\\python.exe`` before touching anything.

``_is_windows`` is patched so these paths are exercised on any host.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from hermes_cli import main as cli_main

SHIM_NAMES = ["hermes.exe", "hermes-agent.exe", "hermes-acp.exe", "hermes-gateway.exe"]


@pytest.fixture
def venv(tmp_path, monkeypatch):
    """A Windows-shaped project venv with a python.exe, wired into main."""
    scripts = tmp_path / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").write_bytes(b"")
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    monkeypatch.setattr(cli_main, "_venv_scripts_dir", lambda: scripts)
    monkeypatch.setattr(sys, "argv", ["hermes", "update"])
    monkeypatch.delenv(cli_main._UPDATE_REEXEC_ENV, raising=False)
    _fake_psutil(monkeypatch, [])
    return scripts


def _fake_psutil(monkeypatch, ancestor_exes: list[str]):
    """Stand in for psutil with a fixed self+ancestor executable chain."""

    class _Proc:
        def __init__(self, exe=None):
            self._exe = exe

        def exe(self):
            if self._exe is None:
                raise OSError("exe unavailable")
            return self._exe

        def parents(self):
            return [_Proc(exe) for exe in ancestor_exes]

    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(Process=_Proc))


def _capture_popen(monkeypatch, raises: Exception | None = None):
    calls = []

    def fake_popen(cmd, env=None, **kwargs):
        if raises is not None:
            raise raises
        calls.append((list(cmd), dict(env or {})))
        return object()

    monkeypatch.setattr(cli_main.subprocess, "Popen", fake_popen)
    return calls


# ---------------------------------------------------------------------------
# Shim detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shim_name", SHIM_NAMES)
def test_detects_shim_as_argv0(venv, monkeypatch, shim_name):
    monkeypatch.setattr(sys, "argv", [str(venv / shim_name), "update"])
    assert cli_main._windows_shim_in_process_chain() == venv / shim_name


def test_detects_shim_from_zipapp_main_py(venv, monkeypatch):
    """runpy/zipapp launches put ``<shim>\\__main__.py`` in argv[0]."""
    monkeypatch.setattr(sys, "argv", [str(venv / "hermes.exe" / "__main__.py")])
    assert cli_main._windows_shim_in_process_chain() == venv / "hermes.exe"


def test_detects_shim_from_main_module_spec_origin(venv, monkeypatch):
    fake_main = types.SimpleNamespace(
        __file__=None,
        __spec__=types.SimpleNamespace(origin=str(venv / "hermes.exe")),
    )
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    assert cli_main._windows_shim_in_process_chain() == venv / "hermes.exe"


def test_detects_shim_in_ancestor_chain(venv, monkeypatch):
    """The launcher is usually a separate parent process, not argv[0]."""
    _fake_psutil(monkeypatch, [str(venv / "hermes.exe")])
    assert cli_main._windows_shim_in_process_chain() == venv / "hermes.exe"


def test_ignores_hermes_exe_outside_the_project_venv(venv, monkeypatch, tmp_path):
    """A shim from some other install must never trigger a re-exec."""
    other = tmp_path / "other" / "Scripts"
    other.mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", [str(other / "hermes.exe"), "update"])
    _fake_psutil(monkeypatch, [str(other / "hermes.exe")])
    assert cli_main._windows_shim_in_process_chain() is None


def test_no_shim_off_windows(venv, monkeypatch):
    monkeypatch.setattr(cli_main, "_is_windows", lambda: False)
    monkeypatch.setattr(sys, "argv", [str(venv / "hermes.exe"), "update"])
    assert cli_main._windows_shim_in_process_chain() is None


def test_no_shim_without_a_venv(venv, monkeypatch):
    monkeypatch.setattr(cli_main, "_venv_scripts_dir", lambda: None)
    monkeypatch.setattr(sys, "argv", [str(venv / "hermes.exe"), "update"])
    assert cli_main._windows_shim_in_process_chain() is None


# ---------------------------------------------------------------------------
# Re-exec hand-off
# ---------------------------------------------------------------------------


def test_reexec_runs_same_args_under_venv_python(venv, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(venv / "hermes.exe"), "update", "--yes"])
    calls = _capture_popen(monkeypatch)

    assert cli_main._reexec_update_off_windows_shim() is True
    cmd, env = calls[0]
    assert cmd == [
        str(venv / "python.exe"), "-m", "hermes_cli.main", "update", "--yes",
    ]
    assert env[cli_main._UPDATE_REEXEC_ENV] == "1"
    assert "under the venv Python" in capsys.readouterr().out


def test_reexec_does_not_recurse(venv, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(venv / "hermes.exe"), "update"])
    monkeypatch.setenv(cli_main._UPDATE_REEXEC_ENV, "1")
    calls = _capture_popen(monkeypatch)

    assert cli_main._reexec_update_off_windows_shim() is False
    assert calls == []


def test_reexec_skipped_when_not_launched_from_a_shim(venv, monkeypatch):
    calls = _capture_popen(monkeypatch)
    assert cli_main._reexec_update_off_windows_shim() is False
    assert calls == []


def test_reexec_falls_through_when_venv_python_is_missing(venv, monkeypatch, capsys):
    (venv / "python.exe").unlink()
    monkeypatch.setattr(sys, "argv", [str(venv / "hermes.exe"), "update"])

    assert cli_main._reexec_update_off_windows_shim() is False
    assert "-m hermes_cli.main update" in capsys.readouterr().out


def test_reexec_falls_through_when_spawn_fails(venv, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(venv / "hermes.exe"), "update"])
    _capture_popen(monkeypatch, raises=OSError("no exec"))

    assert cli_main._reexec_update_off_windows_shim() is False
    assert "-m hermes_cli.main update" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Reboot-deferred renames
# ---------------------------------------------------------------------------


def test_reboot_deferred_rename_fallback_is_gone():
    """MOVEFILE_DELAY_UNTIL_REBOOT needed elevation and freed nothing."""
    assert not hasattr(cli_main, "_schedule_replace_on_reboot")


def test_pending_rename_filter_drops_only_our_shim_pairs():
    shims = [Path(r"C:\hermes\venv\Scripts\hermes.exe")]
    entries = [
        r"\??\C:\other\thing.dll", r"!\??\C:\other\thing.dll.bak",
        r"\??\C:\hermes\venv\Scripts\hermes.exe",
        r"!\??\C:\hermes\venv\Scripts\hermes.exe.old.1755624735000",
    ]
    kept, removed = cli_main._filter_pending_shim_renames(entries, shims)
    assert removed == 1
    assert kept == entries[:2]


def test_pending_rename_filter_keeps_a_shim_pair_with_a_foreign_target():
    shims = [Path(r"C:\hermes\venv\Scripts\hermes.exe")]
    entries = [
        r"\??\C:\hermes\venv\Scripts\hermes.exe", r"!\??\C:\somewhere\else.exe",
    ]
    kept, removed = cli_main._filter_pending_shim_renames(entries, shims)
    assert removed == 0
    assert kept == entries


def test_pending_rename_filter_preserves_a_trailing_delete_entry():
    """A bare source with an empty target is a scheduled delete, not a pair."""
    entries = [r"\??\C:\other\thing.dll", "", r"\??\C:\other\orphan.dll"]
    kept, removed = cli_main._filter_pending_shim_renames(entries, [])
    assert removed == 0
    assert kept == entries


# ---------------------------------------------------------------------------
# venv layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("venv_name", ["venv", ".venv"])
def test_venv_scripts_dir_finds_both_layouts(tmp_path, monkeypatch, venv_name):
    """uv writes .venv; our installers write venv. Both must resolve (#79542)."""
    scripts = tmp_path / venv_name / "Scripts"
    scripts.mkdir(parents=True)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    assert cli_main._venv_scripts_dir() == scripts
