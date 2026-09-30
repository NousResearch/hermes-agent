"""Diagnose deliberately removed install.ps1 switches safely (#125350).

The pre-MSIX removal (47f4ab3a17) reduced install.ps1 to the stage protocol.
``-SkipSetup`` was restored after a wrapper hit it; older wrappers using
``-NoVenv``, ``-ForceCommit``, ``-Tag``, ``-Ensure`` or ``-PostInstall`` need
clear diagnostics rather than an unexplained ``NamedParameterNotFound``.
Removed capabilities stay removed.

The contract these tests pin:

* every legacy switch BINDS (no NamedParameterNotFound);
* an accepted no-op (``-NoVenv``, ``-ForceCommit``) says so on stderr and
  keeps stdout clean for the machine contracts (-ShowResolvedPaths JSON and
  the -Stage/-Json frame stream);
* a removed capability (``-Tag``, ``-Ensure``, ``-PostInstall``) returns code
  2 and replacement guidance without terminating a scriptblock caller.
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


def _ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def _run_scriptblock(tmp_path, *flags):
    powershell = shutil.which("powershell")
    assert powershell
    args = " ".join(flag if flag.startswith("-") else _ps_quote(flag) for flag in flags)
    command = (
        "$global:LASTEXITCODE = 99; "
        f"& ([ScriptBlock]::Create([IO.File]::ReadAllText({_ps_quote(INSTALLER)}))) "
        f"{args} -HermesHome {_ps_quote(tmp_path / 'home')} "
        f"-InstallDir {_ps_quote(tmp_path / 'install')}; "
        'Write-Output "session alive: $LASTEXITCODE"'
    )
    return subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-Command", command], capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize("flags,replacement", [
    (("-Tag", "v2026.9.24"), "-Commit"),
    (("-Ensure", "node,browser"), "pm install"),
    (("-PostInstall",), "pm install"),
])
def test_removed_modes_return_to_scriptblock_caller_with_one_failure_frame(tmp_path, flags, replacement):
    result = _run_scriptblock(tmp_path, *flags, "-Stage", "repository", "-Json")
    assert result.returncode == 0, result.stderr + result.stdout
    assert "session alive: 2" in result.stdout, result.stderr + result.stdout
    assert replacement in result.stderr
    frames = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(frames) == 1, result.stdout
    assert frames[0]["ok"] is False and frames[0]["stage"] == "repository"
    assert frames[0]["skipped"] is False and replacement in frames[0]["reason"]
    assert not (tmp_path / "home").exists()
    assert not (tmp_path / "install").exists()


@pytest.mark.parametrize("flags", [
    ("-Tag", "v2026.9.24"), ("-Ensure", "node,browser"), ("-PostInstall",),
])
def test_removed_modes_have_a_single_failure_frame_when_run_as_a_file(tmp_path, flags):
    result = _run(tmp_path, *flags, "-Stage", "repository", "-Json")
    assert result.returncode == 2, result.stderr + result.stdout
    frames = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(frames) == 1, result.stdout
    assert frames[0]["ok"] is False and frames[0]["stage"] == "repository"
    assert not (tmp_path / "home").exists()
    assert not (tmp_path / "install").exists()


@pytest.mark.parametrize("flag", ["-ProtocolVersion", "-Manifest", "-ShowResolvedPaths"])
def test_read_only_scriptblock_entries_return_to_the_caller_and_reset_exit_code(tmp_path, flag):
    result = _run_scriptblock(tmp_path, flag)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "session alive: 0" in result.stdout, result.stderr + result.stdout
    lines = [line for line in result.stdout.splitlines() if line and not line.startswith("session alive:")]
    assert len(lines) == 1, result.stdout
    if flag == "-ProtocolVersion":
        assert lines[0] == "1"
    elif flag == "-Manifest":
        assert json.loads(lines[0])["protocol_version"] == 1
    else:
        assert json.loads(lines[0])["hermes_home"]
    assert not (tmp_path / "home").exists()
    assert not (tmp_path / "install").exists()


@pytest.mark.parametrize("stage,flags,expected_code,ok,skipped", [
    ("unknown-stage", (), 2, False, False),
    ("setup", ("-NonInteractive",), 0, True, True),
    ("gateway", ("-SkipSetup",), 0, True, True),
    ("repository", (), 1, False, False),
    ("config", (), 0, True, False),
])
def test_stage_scriptblock_entries_preserve_the_session_and_report_outcome(
    tmp_path, stage, flags, expected_code, ok, skipped,
):
    if stage == "repository":
        install = tmp_path / "install"
        install.mkdir()
        (install / "user-file").write_text("preserve me", encoding="utf-8")
    result = _run_scriptblock(tmp_path, "-Stage", stage, "-Json", *flags)
    assert result.returncode == 0, result.stderr + result.stdout
    assert f"session alive: {expected_code}" in result.stdout, result.stderr + result.stdout
    frames = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(frames) == 1, result.stdout
    assert frames[0]["ok"] is ok and frames[0]["stage"] == stage
    assert frames[0]["skipped"] is skipped
    if stage == "repository":
        assert "exists and is not a Hermes git checkout" in frames[0]["reason"]
        assert (tmp_path / "install" / "user-file").read_text(encoding="utf-8-sig") == "preserve me"
