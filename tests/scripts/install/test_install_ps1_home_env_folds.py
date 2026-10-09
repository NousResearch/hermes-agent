"""install.ps1 must fold a whitespace-only Hermes home to the platform default.

A ``   `` HERMES_HOME (or ``-HermesHome``) is "unset" for pm —
``get_hermes_home`` strips it before deciding — and both shell bootstraps
already treat it that way (``hermes_root_of`` trims before expanding).
install.ps1 passed the trimmed empty string into ``GetFullPath``, which
throws, so the installer died in its prologue with a raw .NET error instead
of reporting and installing at the default home.
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


def _show_resolved_home(extra_args: list[str]) -> Path:
    powershell = shutil.which("powershell")
    assert powershell, "install.ps1's contract test runs Windows PowerShell"
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(INSTALLER), "-SkipSetup", "-NonInteractive", "-ShowResolvedPaths",
         *extra_args],
        capture_output=True, text=True, errors="replace", timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(result.stdout)
    return Path(report["hermes_home"])


def _platform_default_home(tmp_path: Path, suffix: str = "") -> Path:
    # install.ps1's default is %LOCALAPPDATA%\hermes plus the data-dir suffix
    # literal (Get-HermesDefaultHome, mirroring _get_platform_default_hermes_home):
    # the tests pin LOCALAPPDATA to tmp_path so the expected default is hermetic,
    # and resolve() on both sides survives an 8.3 alias in either spelling.
    return (tmp_path / f"hermes{suffix}").resolve()


def test_whitespace_only_env_home_folds_to_the_platform_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """'   ' in the ENV behaves exactly like an unset variable."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", "   ")
    assert _show_resolved_home([]).resolve() == _platform_default_home(tmp_path)


def test_whitespace_only_param_home_folds_to_the_platform_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Same for an explicitly passed -HermesHome: trim -> empty -> default."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    assert (
        _show_resolved_home(["-HermesHome", "   "]).resolve()
        == _platform_default_home(tmp_path)
    )


def test_default_home_carries_the_literal_data_dir_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An unset home must resolve to the SAME default pm computes.

    ``_get_platform_default_hermes_home`` appends ``HERMES_DATA_DIR_SUFFIX``
    literally; the installer exporting the suffix-less default as HERMES_HOME
    pins the process to one home while every fresh shell — which never
    inherits that export — computes the suffix'd default and misses the
    installed layout entirely."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HERMES_DATA_DIR_SUFFIX", "-ci")
    assert _show_resolved_home([]).resolve() == _platform_default_home(tmp_path, "-ci")
