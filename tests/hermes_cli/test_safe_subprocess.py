"""Tests for hermes_cli.safe_subprocess — Windows PATH-resolved subprocess wrapper.

Verifies the wrapper handles all the Windows quirks:
1. User-level PATH additions (registry-only) resolve correctly
2. .CMD/.BAT files are auto-resolved via shell=True
3. .PS1 files (scoop) are auto-invoked via powershell
4. The merged PATH includes system + user + env
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from hermes_cli.safe_subprocess import run, _build_merged_path, _resolve


# --- _build_merged_path ---

def test_build_merged_path_includes_git_usr_bin():
    """The merged PATH should include the Git usr/bin we added to the registry."""
    merged = _build_merged_path()
    assert r"C:\Program Files\Git\usr\bin" in merged


def test_build_merged_path_includes_cargo_toolchain():
    """The merged PATH should include the rust toolchain bin."""
    merged = _build_merged_path()
    assert r".rustup\toolchains" in merged


def test_build_merged_path_includes_nssm():
    merged = _build_merged_path()
    assert r"C:\Tools\nssm" in merged


def test_build_merged_path_deduplicates():
    """If the same path appears in user, system, and env, it appears once."""
    merged = _build_merged_path()
    entries = merged.split(";")
    lowered = [e.lower() for e in entries]
    assert len(lowered) == len(set(lowered))


# --- _resolve ---

def test_resolve_finds_ls():
    """ls (in Git usr/bin) should resolve to its full exe path."""
    merged = _build_merged_path()
    found = _resolve("ls", merged)
    assert found is not None
    assert found.lower().endswith("ls.exe")


def test_resolve_finds_cargo():
    """cargo (in toolchain bin) should resolve."""
    merged = _build_merged_path()
    found = _resolve("cargo", merged)
    assert found is not None
    assert found.lower().endswith("cargo.exe")


def test_resolve_finds_scoop_cmd():
    """scoop resolves to .CMD (uppercase)."""
    merged = _build_merged_path()
    found = _resolve("scoop", merged)
    assert found is not None
    assert found.lower().endswith(".cmd")


def test_resolve_finds_winget():
    """winget is in WindowsApps."""
    merged = _build_merged_path()
    found = _resolve("winget", merged)
    assert found is not None
    assert "WindowsApps" in found


def test_resolve_returns_none_for_unknown():
    """Unknown commands return None."""
    merged = _build_merged_path()
    assert _resolve("definitely-not-a-real-command-xyz", merged) is None


def test_resolve_finds_code_insiders_cmd():
    """code-insiders is .CMD (uppercase)."""
    merged = _build_merged_path()
    found = _resolve("code-insiders", merged)
    assert found is not None
    assert found.lower().endswith(".cmd")


# --- run ---

def test_run_ls_version():
    """ls --version should return a string with "ls" and "GNU"."""
    r = run(["ls", "--version"], timeout=10)
    assert r.returncode == 0
    assert "GNU" in r.stdout or "ls" in r.stdout


def test_run_scoop_via_ps1():
    """scoop is .ps1 under the hood. run() should handle it via powershell."""
    r = run(["scoop", "--version"], timeout=30)
    # Scoop prints version info to stdout
    assert r.returncode == 0
    assert "Scoop" in r.stdout or "version" in r.stdout.lower()


def test_run_code_insiders_via_cmd():
    """code-insiders is .CMD. run() should handle it via shell=True."""
    r = run(["code-insiders", "--version"], timeout=30)
    assert r.returncode == 0
    assert "insider" in r.stdout.lower() or "Visual Studio Code" in r.stdout


def test_run_hermes():
    """hermes is .EXE in venv Scripts."""
    r = run(["hermes", "--version"], timeout=10)
    assert r.returncode == 0
    assert "Hermes" in r.stdout or "hermes" in r.stdout.lower()


def test_run_with_full_path():
    """Calling with an absolute path should still work."""
    r = run([r"C:\Program Files\Git\usr\bin\ls.exe", "--version"], timeout=10)
    assert r.returncode == 0


def test_run_with_invalid_command_raises():
    """Unknown command raises FileNotFoundError."""
    import pytest
    try:
        run(["definitely-not-a-real-command-xyz"], timeout=5)
        assert False, "should have raised"
    except FileNotFoundError:
        pass


def test_run_empty_raises():
    """Empty cmd raises ValueError."""
    import pytest
    try:
        run([], timeout=5)
        assert False, "should have raised"
    except ValueError:
        pass


def test_run_returns_completed_process():
    """The return value should be a subprocess.CompletedProcess with stdout/stderr/returncode."""
    r = run(["ls", "--version"], timeout=10)
    assert hasattr(r, "stdout")
    assert hasattr(r, "stderr")
    assert hasattr(r, "returncode")
    assert isinstance(r.returncode, int)
