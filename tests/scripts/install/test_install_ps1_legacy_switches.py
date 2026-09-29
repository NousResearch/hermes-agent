"""Legacy install.ps1 switches from the pre-rework surface must bind (#125350).

The staged-installer rework (92686159d1) replaced install.ps1's param()
surface with the stage protocol. ``-SkipSetup`` was restored after a wrapper
hit it; ``-NoVenv``, ``-ForceCommit``, ``-Tag``, ``-Ensure`` and
``-PostInstall`` were not, so any wrapper written against the old block dies
at parameter binding with ``NamedParameterNotFound`` before the script can
emit a single line of explanation — indistinguishable from a defect.

The contract these tests pin:

* every legacy switch BINDS (no NamedParameterNotFound);
* an accepted no-op (``-NoVenv``, ``-ForceCommit``) says so on stderr and
  keeps stdout clean for the machine contracts (-ShowResolvedPaths JSON and
  the -Stage/-Json frame stream);
* a removed capability (``-Tag``, ``-Ensure``, ``-PostInstall``) stops with
  exit 2 and a message that names the replacement — never a binder error.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALLER = REPO_ROOT / "scripts" / "install.ps1"


def _run(tmp_path, *flags):
    powershell = shutil.which("powershell")
    assert powershell
    return subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(INSTALLER), *flags,
         "-HermesHome", str(tmp_path / "home"), "-InstallDir", str(tmp_path / "install")],
        capture_output=True, text=True, timeout=120,
    )


def _last_json_line(stdout: str) -> dict:
    lines = [line for line in stdout.strip().splitlines() if line.strip()]
    assert lines, "expected at least one stdout line"
    return json.loads(lines[-1])


def test_novenv_binds_as_a_stated_no_op(tmp_path):
    result = _run(tmp_path, "-NoVenv", "-ShowResolvedPaths")
    assert result.returncode == 0, result.stderr + result.stdout
    # The note goes to stderr; stdout stays machine-readable.
    assert "-NoVenv" in result.stderr
    report = _last_json_line(result.stdout)
    assert report.get("hermes_home")


def test_forcecommit_binds_as_a_stated_no_op(tmp_path):
    result = _run(tmp_path, "-ForceCommit", "-ShowResolvedPaths")
    assert result.returncode == 0, result.stderr + result.stdout
    assert "-ForceCommit" in result.stderr
    report = _last_json_line(result.stdout)
    assert report.get("hermes_home")


def test_legacy_no_ops_keep_stage_frames_parseable(tmp_path):
    """A wrapper on the old surface can also drive stages; the frame stream
    is single-line JSON on stdout and a stray note would break its parsing."""
    result = _run(tmp_path, "-NoVenv", "-ForceCommit", "-Manifest")
    assert result.returncode == 0, result.stderr + result.stdout
    manifest = json.loads(result.stdout.strip().splitlines()[-1])
    assert manifest["protocol_version"] == 1
    assert manifest["stages"]


def test_tag_stops_with_guidance_instead_of_a_binder_error(tmp_path):
    result = _run(tmp_path, "-Tag", "v2026.9.24")
    assert result.returncode == 2, result.stderr + result.stdout
    assert "-Commit" in result.stderr and "-Branch" in result.stderr
    assert "NamedParameterNotFound" not in (result.stderr + result.stdout)


def test_ensure_stops_with_guidance_instead_of_a_binder_error(tmp_path):
    result = _run(tmp_path, "-Ensure", "node,browser")
    assert result.returncode == 2, result.stderr + result.stdout
    assert "pm install" in result.stderr
    assert "NamedParameterNotFound" not in (result.stderr + result.stdout)


def test_postinstall_stops_with_guidance_instead_of_a_binder_error(tmp_path):
    result = _run(tmp_path, "-PostInstall")
    assert result.returncode == 2, result.stderr + result.stdout
    assert "pm install" in result.stderr
    assert "NamedParameterNotFound" not in (result.stderr + result.stdout)
