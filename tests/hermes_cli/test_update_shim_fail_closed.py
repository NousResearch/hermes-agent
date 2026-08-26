"""Fail-closed shim quarantine (#87331): a contended venv is never mutated.

The bug class: on Windows, when `hermes.exe`/sibling shims could not be
renamed aside (another process holds them without FILE_SHARE_DELETE), the
updater printed a warning and ran the installer anyway — which then died
partway on the same locks and stranded the venv between versions (3x field
reports on one machine, #87331).

Contract pinned here:
- `_run_quarantined_install(strict_quarantine=True)` raises
  ShimQuarantineError BEFORE running any install command, and rolls back the
  renames that did succeed.
- Non-strict callers keep the old warn-and-try behavior.
- The recovery installer's `_run_install_cmd` is strict unconditionally.
- The update boundary turns the error into a refusal (exit 2) + marker,
  never a ZIP fallback.
"""

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

import hermes_cli.main as cli_main
import hermes_cli._install_repair as ir
import hermes_cli.update_cmd as update_cmd


def _make_shims(scripts_dir: Path, names=("hermes", "hermes-gateway")) -> list[Path]:
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shims = []
    for name in names:
        p = scripts_dir / f"{name}.exe"
        p.write_bytes(b"MZ fake")
        shims.append(p)
    return shims


@pytest.fixture()
def windows(monkeypatch):
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    monkeypatch.setattr(ir, "_is_windows", lambda: True)


# ---------------------------------------------------------------------------
# main.py: _run_quarantined_install strict mode
# ---------------------------------------------------------------------------

def test_strict_quarantine_refuses_before_install(windows, tmp_path, monkeypatch):
    scripts = tmp_path / "venv" / "Scripts"
    shims = _make_shims(scripts)
    # hermes.exe cannot be renamed; hermes-gateway.exe can
    real_rename = Path.rename

    def deny_hermes(self, target):
        if self.name == "hermes.exe":
            raise PermissionError(13, "held open")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", deny_hermes)
    monkeypatch.setattr(cli_main, "_hermes_exe_shims", lambda d: shims)

    install_ran = []
    monkeypatch.setattr(
        cli_main, "_run_install_with_heartbeat",
        lambda cmd, env=None: install_ran.append(cmd),
    )

    with pytest.raises(cli_main.ShimQuarantineError) as exc_info:
        cli_main._run_quarantined_install(
            ["uv", "pip", "install", "-e", "."],
            scripts_dir=scripts,
            strict_quarantine=True,
        )

    # The installer NEVER ran — that is the whole fix.
    assert install_ran == []
    assert "hermes.exe" in exc_info.value.failed_shims
    # The successful rename (hermes-gateway.exe) was rolled back.
    assert (scripts / "hermes-gateway.exe").exists()
    assert not list(scripts.glob("hermes-gateway.exe.old.*"))


def test_non_strict_keeps_warn_and_try(windows, tmp_path, monkeypatch):
    scripts = tmp_path / "venv" / "Scripts"
    shims = _make_shims(scripts, names=("hermes",))
    monkeypatch.setattr(
        Path, "rename",
        mock.Mock(side_effect=PermissionError(13, "held open")),
    )
    monkeypatch.setattr(cli_main, "_hermes_exe_shims", lambda d: shims)

    install_ran = []
    monkeypatch.setattr(
        cli_main, "_run_install_with_heartbeat",
        lambda cmd, env=None: install_ran.append(cmd),
    )

    # Default (non-strict): installer still runs — old behavior for repair
    # paths whose venv is already mutated.
    cli_main._run_quarantined_install(["fake"], scripts_dir=scripts)
    assert install_ran == [["fake"]]


def test_strict_all_renames_ok_runs_install(windows, tmp_path, monkeypatch):
    scripts = tmp_path / "venv" / "Scripts"
    shims = _make_shims(scripts)
    monkeypatch.setattr(cli_main, "_hermes_exe_shims", lambda d: shims)

    install_ran = []
    monkeypatch.setattr(
        cli_main, "_run_install_with_heartbeat",
        lambda cmd, env=None: install_ran.append(cmd),
    )
    cli_main._run_quarantined_install(
        ["fake"], scripts_dir=scripts, strict_quarantine=True
    )
    assert install_ran == [["fake"]]


def test_update_sync_installs_are_strict(windows, tmp_path, monkeypatch):
    """_install_python_dependencies_with_optional_fallback must pass
    strict_quarantine=True — the #87331 site."""
    seen = {}

    def spy(cmd, *, env=None, scripts_dir=None, strict_quarantine=False):
        seen["strict"] = strict_quarantine

    monkeypatch.setattr(cli_main, "_run_quarantined_install", spy)
    monkeypatch.setattr(cli_main, "_venv_scripts_dir", lambda: tmp_path)
    monkeypatch.setattr(
        cli_main, "_verify_console_scripts_installed",
        lambda prefix, env=None: None,
    )
    monkeypatch.setattr(
        cli_main, "_verify_core_dependencies_installed",
        lambda prefix, env=None, group="all": None,
    )
    cli_main._install_python_dependencies_with_optional_fallback(["uv", "pip"])
    assert seen["strict"] is True


# ---------------------------------------------------------------------------
# _install_repair.py: recovery installer is strict unconditionally
# ---------------------------------------------------------------------------

def test_recovery_install_cmd_fail_closed(windows, tmp_path, monkeypatch):
    root = tmp_path
    scripts = root / "venv" / "Scripts"
    _make_shims(scripts, names=("hermes",))

    monkeypatch.setattr(ir, "_venv_scripts_dir", lambda r: scripts)
    monkeypatch.setattr(
        Path.__module__ and os, "rename",
        mock.Mock(side_effect=PermissionError(13, "held open")),
    )

    run_calls = []
    monkeypatch.setattr(
        ir.subprocess, "run", lambda *a, **k: run_calls.append(a)
    )

    with pytest.raises(ir.ShimQuarantineError):
        ir._run_install_cmd(["fake"], env=None, root=root)
    assert run_calls == []  # contended venv never mutated


def test_recovery_install_cmd_ok_when_uncontended(windows, tmp_path, monkeypatch):
    root = tmp_path
    scripts = root / "venv" / "Scripts"
    _make_shims(scripts, names=("hermes",))
    monkeypatch.setattr(ir, "_venv_scripts_dir", lambda r: scripts)

    run_calls = []
    monkeypatch.setattr(
        ir.subprocess, "run",
        lambda *a, **k: run_calls.append(a) or mock.Mock(returncode=0),
    )
    ir._run_install_cmd(["fake"], env=None, root=root)
    assert len(run_calls) == 1


# ---------------------------------------------------------------------------
# update_cmd.py: boundary refusal — marker + exit 2, no ZIP fallback
# ---------------------------------------------------------------------------

def test_refusal_writes_marker_and_exits_2(monkeypatch, capsys):
    wrote = []
    monkeypatch.setattr(
        update_cmd, "_write_update_incomplete_marker", lambda: wrote.append(1)
    )
    exc = cli_main.ShimQuarantineError(["hermes.exe"])
    with pytest.raises(SystemExit) as exit_info:
        update_cmd._refuse_update_for_contended_shims(exc)
    assert exit_info.value.code == 2
    assert wrote == [1]
    out = capsys.readouterr().out
    assert "hermes.exe" in out
    assert "deferred" in out


def test_shim_error_type_resolves_real_class():
    assert update_cmd._shim_quarantine_error_type() is cli_main.ShimQuarantineError


def test_shim_error_is_not_a_zip_fallback_trigger():
    exc = cli_main.ShimQuarantineError(["hermes.exe"])
    assert update_cmd._should_zip_fallback_on_update_error(exc) is False
