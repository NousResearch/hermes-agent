"""Regression tests for the quarantine no-op restore gap (#75584).

On Windows, ``_run_quarantined_install`` / ``_run_install_cmd`` rename live
``hermes*.exe`` shims aside (``hermes.exe.old.<ms>``) before invoking the
installer so uv/pip can write fresh replacements. When the install SUCCEEDS
but never rewrites entry points (uv audits an already-satisfied editable
install as a no-op), the old code only restored the shims on FAILURE — the
quarantined shims stayed renamed aside and ``hermes`` vanished from PATH
after a green install.

These tests exercise both wrapper sites with a fake installer and assert the
shims come back on every path:
  - success + installer rewrote shims  → fresh shims kept, .old garbage left
  - success + installer wrote nothing  → original shims renamed back (the bug)
  - failure                            → original shims renamed back (as before)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import _install_repair as ir
from hermes_cli import main as cli_main


def _make_scripts_dir(tmp_path: Path) -> Path:
    scripts = tmp_path / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    for name in ("hermes", "hermes-agent", "hermes-acp", "hermes-gateway"):
        (scripts / f"{name}.exe").write_bytes(b"MZ-old-" + name.encode())
    return scripts


def _shim_names(scripts: Path) -> set[str]:
    return {p.name for p in scripts.iterdir()}


# ---------------------------------------------------------------------------
# hermes_cli.main._run_quarantined_install
# ---------------------------------------------------------------------------


def test_main_noop_success_restores_shims(tmp_path):
    """A successful install that writes no entry points must restore shims."""
    scripts = _make_scripts_dir(tmp_path)

    with patch.object(cli_main, "_is_windows", lambda: True), patch.object(
        cli_main, "_run_install_with_heartbeat", lambda cmd, env=None: None
    ):
        cli_main._run_quarantined_install(["fake"], scripts_dir=scripts)

    names = _shim_names(scripts)
    assert "hermes.exe" in names, "hermes.exe must be restored after a no-op install"
    assert "hermes-acp.exe" in names
    assert "hermes-gateway.exe" in names
    assert (scripts / "hermes.exe").read_bytes() == b"MZ-old-hermes"


def test_main_rewriting_success_keeps_fresh_shims(tmp_path):
    """When the installer writes fresh shims, restore must NOT clobber them."""
    scripts = _make_scripts_dir(tmp_path)

    def fake_install(cmd, env=None):
        for name in ("hermes", "hermes-agent", "hermes-acp", "hermes-gateway"):
            (scripts / f"{name}.exe").write_bytes(b"MZ-new-" + name.encode())

    with patch.object(cli_main, "_is_windows", lambda: True), patch.object(
        cli_main, "_run_install_with_heartbeat", fake_install
    ):
        cli_main._run_quarantined_install(["fake"], scripts_dir=scripts)

    assert (scripts / "hermes.exe").read_bytes() == b"MZ-new-hermes"


def test_main_failure_restores_shims_and_reraises(tmp_path):
    scripts = _make_scripts_dir(tmp_path)

    def boom(cmd, env=None):
        raise RuntimeError("install died")

    with patch.object(cli_main, "_is_windows", lambda: True), patch.object(
        cli_main, "_run_install_with_heartbeat", boom
    ):
        with pytest.raises(RuntimeError, match="install died"):
            cli_main._run_quarantined_install(["fake"], scripts_dir=scripts)

    assert (scripts / "hermes.exe").read_bytes() == b"MZ-old-hermes"


# ---------------------------------------------------------------------------
# hermes_cli._install_repair._run_install_cmd (the deferred-recovery path)
# ---------------------------------------------------------------------------


def _patch_repair_windows(scripts: Path):
    """Force the repair module down the Windows quarantine path."""
    return (
        patch.object(ir, "_is_windows", lambda: True),
        patch.object(ir, "_venv_scripts_dir", lambda root: scripts),
    )


def test_repair_noop_success_restores_shims(tmp_path):
    """The early-recovery install path (the #75584 report) must restore too."""
    scripts = _make_scripts_dir(tmp_path)
    win, vdir = _patch_repair_windows(scripts)

    with win, vdir, patch.object(ir.subprocess, "run", lambda *a, **k: None):
        ir._run_install_cmd(["fake"], env=None, root=tmp_path)

    names = _shim_names(scripts)
    assert "hermes.exe" in names, "hermes.exe must be restored after a no-op recovery install"
    assert (scripts / "hermes.exe").read_bytes() == b"MZ-old-hermes"


def test_repair_rewriting_success_keeps_fresh_shims(tmp_path):
    scripts = _make_scripts_dir(tmp_path)
    win, vdir = _patch_repair_windows(scripts)

    def fake_run(cmd, cwd=None, check=None, env=None):
        (scripts / "hermes.exe").write_bytes(b"MZ-new-hermes")

    with win, vdir, patch.object(ir.subprocess, "run", fake_run):
        ir._run_install_cmd(["fake"], env=None, root=tmp_path)

    assert (scripts / "hermes.exe").read_bytes() == b"MZ-new-hermes"


def test_repair_failure_restores_shims_and_reraises(tmp_path):
    scripts = _make_scripts_dir(tmp_path)
    win, vdir = _patch_repair_windows(scripts)

    def fake_run(cmd, cwd=None, check=None, env=None):
        raise ir.subprocess.CalledProcessError(1, cmd)

    with win, vdir, patch.object(ir.subprocess, "run", fake_run):
        with pytest.raises(ir.subprocess.CalledProcessError):
            ir._run_install_cmd(["fake"], env=None, root=tmp_path)

    assert (scripts / "hermes.exe").read_bytes() == b"MZ-old-hermes"
