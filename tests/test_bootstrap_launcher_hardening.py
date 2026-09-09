"""Codex audit Batch 1 (field-breaking risks), RUN-2026-09-07-010.

Fix 1 - scripts/bootstrap-north-forge.ps1: PowerShell 5.1 with
  ``$ErrorActionPreference = 'Stop'`` treats a native process that merely writes
  to stderr as a terminating error. Every native call now goes through
  ``Invoke-Native`` / ``Get-NativeText`` (EAP relaxed locally, judged by
  ``$LASTEXITCODE``), and the run is wrapped so ``$ErrorActionPreference``,
  ``HERMES_HOME`` and ``UV_CACHE_DIR`` are restored on every exit path.

Fix 2 - north-forge.cmd: renaming the checkout folder derived a new data-folder
  name and silently started fresh - existing conversations / config / credentials
  in the old sibling folder were not found, not migrated, not flagged. The
  launcher now detects exactly one sibling ``*-data`` folder and makes the
  operator choose (reuse / start fresh) - never silent.

Fix 3 - north-forge.cmd: ``mkdir "%DATA%"`` was unchecked; on a read-only / full /
  disconnected drive HERMES_HOME pointed at a folder that was never created. The
  launcher now verifies the folder exists and stops cleanly if not.

Layers, per test_nf_preflight_readiness.py:
  * ``test_*_source`` - transport-independent, runs everywhere.
  * ``_WINDOWS_ONLY`` - drive the real .ps1 / .cmd and assert behaviour.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap-north-forge.ps1"
LAUNCHER = REPO_ROOT / "north-forge.cmd"
_WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="drives a .ps1 / .cmd")


# --------------------------------------------------------------------------- Fix 1 (source)


def test_bootstrap_defines_relaxed_native_helpers_source():
    src = BOOTSTRAP.read_text(encoding="utf-8")
    assert "function Invoke-Native" in src
    assert "function Get-NativeText" in src
    # each helper relaxes then restores EAP
    for fn in ("Invoke-Native", "Get-NativeText"):
        body = src.split(f"function {fn}", 1)[1].split("\nfunction ", 1)[0]
        assert "$ErrorActionPreference = 'Continue'" in body
        assert "finally { $ErrorActionPreference = $prev }" in body


def test_bootstrap_native_calls_go_through_a_helper_source():
    """No bare `& uv` / `& $pyExe` / `& $sysPy` outside Invoke-Native / Get-NativeText."""
    src = BOOTSTRAP.read_text(encoding="utf-8")
    offenders = []
    for m in re.finditer(r"&\s+(uv|\$pyExe|\$sysPy\.Source|\$hermes)\b.*", src):
        line = m.group(0)
        # allowed: the line is the body of a helper call
        before = src[: m.start()].rsplit("\n", 1)[-1]
        if "Invoke-Native" in before or "Get-NativeText" in before:
            continue
        offenders.append(line.strip())
    assert not offenders, f"native calls not wrapped: {offenders}"


def test_bootstrap_restores_env_on_every_path_source():
    src = BOOTSTRAP.read_text(encoding="utf-8")
    assert "finally {" in src
    for token in ("$ErrorActionPreference = $script:__origEAP",
                  "$env:HERMES_HOME = $script:__origHermesHome",
                  "$env:UV_CACHE_DIR = $script:__origUvCache"):
        assert token in src, token
    # body runs as a function that reports via $script:__nfExit; exactly one top-level `exit`
    assert "function Invoke-NfBootstrap" in src
    assert re.search(r"\nexit \$script:__nfExit\s*$", src)
    assert re.search(r"^\s*exit \b", src, re.M) is None or src.count("\nexit ") == 1, \
        "use `$script:__nfExit = N; return` inside the function; `exit` exactly once at the end"


def test_bootstrap_verify_step_uses_exit_code_not_stderr_text_source():
    src = BOOTSTRAP.read_text(encoding="utf-8")
    seg = src.split("# --- verify", 1)[1]
    assert "$importCode = $LASTEXITCODE" in seg
    assert "if ($importCode -ne 0)" in seg
    # the merged text is only displayed, never the pass/fail signal
    assert "import ok" in seg


# --------------------------------------------------------------------------- Fix 2 / 3 (source)


def test_launcher_detects_prior_data_folder_source():
    src = LAUNCHER.read_text(encoding="utf-8")
    assert ":find_prior_data" in src and 'for /d %%D in ("%PARENT%\\*-data")' in src
    choice_line = next(ln for ln in src.splitlines() if ln.strip().startswith("choice /c RN"))
    # explicit choice, no automatic default / timeout
    assert " /t " not in choice_line and " /d " not in choice_line
    assert "Reusing" in src and "Starting fresh" in src


def test_launcher_checks_data_dir_creation_source():
    src = LAUNCHER.read_text(encoding="utf-8")
    seg = src.split("Codex audit Fix 3", 1)[1]
    assert 'mkdir "%DATA%" 2>nul' in seg
    assert 'if not exist "%DATA%\\" (' in seg
    assert "will not start without it" in seg
    assert "exit /b 1" in seg
    # skin dir gets the same guard
    assert 'mkdir "%DATA%\\skins" 2>nul' in src


# --------------------------------------------------------------------------- Fix 1 (behaviour)

# Each probe runs at SCRIPT scope, not inside a helper function. PowerShell 5.1
# does not propagate a native command's exit code out of `& $scriptblockParam`
# when the scriptblock arrived as a *function parameter* — so a `Check $name
# $block` helper would read $LASTEXITCODE == 0 no matter what the child did (a
# bug in the harness, not in Invoke-Native). Keeping every probe inline sidesteps
# that entirely; `try {} catch {}` at script scope preserves $LASTEXITCODE fine.
_PS_HELPER_HARNESS = r"""
$ErrorActionPreference = 'Stop'
__HELPERS__
$results = @()
function Record($name, $code, $want, $eapOk, $threw) {
    $ok = ($code -eq $want) -and $eapOk -and (-not $threw)
    $script:results += ("{0} code={1} want={2} eapRestored={3} threw={4} -> {5}" -f `
        $name, $code, $want, $eapOk, $threw, $(if ($ok) { 'PASS' } else { 'FAIL' }))
}

# 1. success + informational stderr — must not throw under EAP=Stop, code 0, EAP restored
$eap = $ErrorActionPreference; $LASTEXITCODE = 0; $threw = $false
try { Invoke-Native { & cmd /c 'echo informational 1>&2 & exit 0' } } catch { $threw = $true }
Record 'info_stderr' $LASTEXITCODE 0 ($ErrorActionPreference -eq $eap) $threw

# 2. success + warning-shaped stderr
$eap = $ErrorActionPreference; $LASTEXITCODE = 0; $threw = $false
try { Invoke-Native { & cmd /c 'echo WARNING: heads up 1>&2 & exit 0' } } catch { $threw = $true }
Record 'warn_stderr' $LASTEXITCODE 0 ($ErrorActionPreference -eq $eap) $threw

# 3. non-zero exit is preserved (the whole point — bootstrap gates on $LASTEXITCODE)
$eap = $ErrorActionPreference; $LASTEXITCODE = 0; $threw = $false
try { Invoke-Native { & cmd /c 'echo boom 1>&2 & exit 3' } } catch { $threw = $true }
Record 'nonzero' $LASTEXITCODE 3 ($ErrorActionPreference -eq $eap) $threw

# 4. Get-NativeText returns the merged text AND leaves $LASTEXITCODE correct
$eap = $ErrorActionPreference; $LASTEXITCODE = 0; $threw = $false
try { $t = Get-NativeText { & cmd /c 'echo captured & echo err 1>&2 & exit 5' } } catch { $threw = $true }
Record 'get_native_text' $LASTEXITCODE 5 (($ErrorActionPreference -eq $eap) -and ($t -match 'captured')) $threw

$results -join "`n"
"""


def _extract_helpers() -> str:
    src = BOOTSTRAP.read_text(encoding="utf-8")
    out = []
    for fn in ("Invoke-Native", "Get-NativeText"):
        chunk = "function " + fn + src.split("function " + fn, 1)[1]
        chunk = chunk.split("\nfunction ", 1)[0]
        out.append(chunk)
    return "\n".join(out)


@_WINDOWS_ONLY
def test_invoke_native_survives_stderr_and_preserves_exit_code(tmp_path):
    harness = _PS_HELPER_HARNESS.replace("__HELPERS__", _extract_helpers())
    script = tmp_path / "h.ps1"
    script.write_text(harness, encoding="utf-8")
    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert "info_stderr" in out and "warn_stderr" in out, out
    assert "FAIL" not in out, out
    for line in out.splitlines():
        if "=" in line and ("code=" in line):
            assert line.strip().endswith("PASS"), line


@_WINDOWS_ONLY
def test_bootstrap_restores_env_after_early_return(tmp_path):
    """Real early-return path: HERMES_HOME / UV_CACHE_DIR / EAP restored, exit 0."""
    venv = REPO_ROOT.parent / (REPO_ROOT.name + "-venv")
    if not (venv / "Scripts" / "hermes.exe").exists():
        pytest.skip("no sibling bootstrapped venv to exercise the early-return path")
    data = tmp_path / "d"
    probe = tmp_path / "probe.ps1"
    probe.write_text(textwrap.dedent(f"""
        $env:HERMES_HOME = 'SENTINEL_HH'
        $env:UV_CACHE_DIR = 'SENTINEL_UV'
        $eap = $ErrorActionPreference
        & '{BOOTSTRAP}' -RepoRoot '{REPO_ROOT}' -VenvDir '{venv}' -DataDir '{data}'
        $rc = $LASTEXITCODE
        "rc=$rc"
        "hh=$($env:HERMES_HOME -eq 'SENTINEL_HH')"
        "uv=$($env:UV_CACHE_DIR -eq 'SENTINEL_UV')"
        "eap=$($ErrorActionPreference -eq $eap)"
    """), encoding="utf-8")
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(probe)],
                       capture_output=True, text=True)
    out = r.stdout
    assert "rc=0" in out, r.stdout + r.stderr
    assert "hh=True" in out and "uv=True" in out and "eap=True" in out, out


# --------------------------------------------------------------------------- Fix 2 / 3 (behaviour)


def _fake_drive(tmp_path, *, checkout="myco", prior=("oldco-data",), data_as_file=False):
    root = tmp_path / "drive"
    co = root / checkout
    (co / "scripts").mkdir(parents=True)
    (co / "skins").mkdir(parents=True)
    (co / "north-forge.cmd").write_text(LAUNCHER.read_text(encoding="utf-8"), encoding="utf-8")
    (co / "skins" / "north-forge.yaml").write_text("name: nf\n", encoding="utf-8")
    stub = root / f"{checkout}-venv" / "Scripts"
    stub.mkdir(parents=True)
    # a python.exe that always exits 0 so nf_tier verify never trips the tamper branch
    (stub / "python.exe.bat").write_text("@exit /b 0\n", encoding="utf-8")
    for name in ("python.exe", "hermes.exe"):
        (stub / name).write_bytes(b"")  # presence-only; guards run before these are used
    for p in prior:
        (root / p).mkdir()
        (root / p / "config.yaml").write_text("real: data\n", encoding="utf-8")
    if data_as_file:
        (root / f"{checkout}-data").write_text("not a dir\n", encoding="utf-8")
    return root, co


@_WINDOWS_ONLY
def test_launcher_prompts_on_checkout_rename_reuse(tmp_path):
    root, co = _fake_drive(tmp_path, prior=("oldco-data",))
    r = subprocess.run(["cmd", "/c", str(co / "north-forge.cmd"), "--version"],
                       input="R\n", capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert "a data folder from a previous checkout name was found" in out
    assert "oldco-data" in out and "Reusing" in out
    assert not (root / "myco-data").exists(), "reuse must NOT create a new empty data dir"
    assert (root / "oldco-data" / "config.yaml").read_text().strip() == "real: data"


@_WINDOWS_ONLY
def test_launcher_prompts_on_checkout_rename_fresh(tmp_path):
    root, co = _fake_drive(tmp_path, prior=("oldco-data",))
    r = subprocess.run(["cmd", "/c", str(co / "north-forge.cmd"), "--version"],
                       input="N\n", capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert "Renaming the checkout folder changed the expected data-folder name" in out
    assert "Starting fresh" in out
    assert (root / "myco-data").is_dir()
    assert (root / "oldco-data" / "config.yaml").exists(), "old folder left untouched"


@_WINDOWS_ONLY
def test_launcher_no_prompt_when_no_prior_data(tmp_path):
    root, co = _fake_drive(tmp_path, prior=())
    r = subprocess.run(["cmd", "/c", str(co / "north-forge.cmd"), "--version"],
                       input="", capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert "previous checkout name" not in out
    assert (root / "myco-data").is_dir()


@_WINDOWS_ONLY
def test_launcher_no_prompt_when_two_priors_ambiguous(tmp_path):
    # >1 candidate: don't guess, don't prompt a bogus single choice - just proceed fresh
    root, co = _fake_drive(tmp_path, prior=("oldco-data", "olderco-data"))
    r = subprocess.run(["cmd", "/c", str(co / "north-forge.cmd"), "--version"],
                       input="", capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert "a data folder from a previous checkout name was found" not in out
    assert (root / "myco-data").is_dir()


@_WINDOWS_ONLY
def test_launcher_stops_cleanly_when_data_dir_uncreatable(tmp_path):
    root, co = _fake_drive(tmp_path, prior=(), data_as_file=True)
    r = subprocess.run(["cmd", "/c", str(co / "north-forge.cmd"), "--version"],
                       input="", capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert r.returncode == 1
    assert "Could not create or open the drive-local data folder" in out
    assert "No data has been lost" in out
    assert "usage:" not in out.lower(), "hermes must not have been launched"
