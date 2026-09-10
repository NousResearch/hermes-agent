"""Regression: the launcher must PROBE the venv, not trust that files exist.

`ERR-2026-09-07-006` (reclassified CRITICAL after a real first-handoff failure):
`north-forge.cmd` and `bootstrap-north-forge.ps1` both treated
"``hermes.exe`` exists" as proof the environment was ready. A Python venv built
on one machine is not portable to another - copy the drive to a new computer, or
just rename / re-letter the checkout, and ``hermes.exe`` + ``.nf-bootstrapped``
are still there while the interpreter fails or imports stale code.

The fix (``scripts/lib/nf-readiness.ps1`` + ``scripts/nf-preflight.ps1``, called
by ``north-forge.cmd`` before every launch) replaces the existence check with a
real launch-time probe of four things:

  1. python_exec          - the venv's ``python.exe`` actually executes
  2. import_hermes_cli    - ``import hermes_cli`` works in it
  3. module_in_checkout   - that ``hermes_cli`` resolves under THIS checkout
  4. marker_repo_matches  - ``.nf-bootstrapped``'s ``repo=`` is THIS checkout

Any failure => silently rebuild the venv (never the data folder) and write one
line to a launcher-level log first, so a broken interpreter can't stop the log
line from existing.

Two layers, same split as ``test_bootstrap_north_forge_path_safety.py``:
  * ``test_*_source`` - transport-independent, runs on every platform (Linux CI
    cannot execute PowerShell).
  * ``test_*`` under ``_WINDOWS_ONLY`` - actually run the probe / the launcher's
    preflight and confirm a foreign-machine venv is rebuilt, not launched.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LIB_PS1 = REPO_ROOT / "scripts" / "lib" / "nf-readiness.ps1"
PREFLIGHT_PS1 = REPO_ROOT / "scripts" / "nf-preflight.ps1"
BOOTSTRAP_PS1 = REPO_ROOT / "scripts" / "bootstrap-north-forge.ps1"
LAUNCHER_CMD = REPO_ROOT / "north-forge.cmd"

FOUR_CHECKS = ("python_exec", "import_hermes_cli", "module_in_checkout", "marker_repo_matches")


# ==========================================================================
# source-level - portable
# ==========================================================================

def test_readiness_lib_defines_the_four_named_checks_source() -> None:
    src = LIB_PS1.read_text(encoding="utf-8")
    assert "function Test-NfVenvReady" in src
    for name in FOUR_CHECKS:
        assert name in src, f"readiness lib must name the '{name}' check"
    # it reports, it does not act: no exit / no rebuild from the library itself
    assert "\nexit " not in src and not src.strip().endswith("exit")
    assert "Remove-Item" not in src, "the probe library must never delete anything"


def test_readiness_probe_is_isolated_source() -> None:
    """The probe must run python with -I and from a cwd OUTSIDE the checkout, so
    a stray ``hermes_cli/`` in the working dir can't fake check 3 and an ambient
    ``PYTHONPATH`` can't rescue a broken venv."""
    src = LIB_PS1.read_text(encoding="utf-8")
    assert "-I -c" in src, "probe must isolate the interpreter (-I)"
    assert "GetTempPath" in src, "probe must not run with cwd inside the checkout"
    assert "OrdinalIgnoreCase" in src, "path comparison must be ordinal/case-insensitive"


def test_preflight_sources_lib_rebuilds_venv_never_data_source() -> None:
    src = PREFLIGHT_PS1.read_text(encoding="utf-8")
    assert "lib\\nf-readiness.ps1" in src, "preflight must dot-source the shared probe"
    assert "Test-NfVenvReady" in src
    # rebuild path calls the bootstrap with -Force ...
    assert "-Force" in src and "bootstrap-north-forge.ps1" in src
    # ... and preflight itself never removes the venv or the data folder
    # (bootstrap owns the venv wipe; data is never wiped at all). The only
    # deletion here is single-file launcher-log rotation.
    for rm in ("Remove-Item -LiteralPath $VenvDir", "Remove-Item -LiteralPath $DataDir",
               "Remove-Item -Recurse"):
        assert rm not in src, f"preflight must not run `{rm}`"
    # no interactive gate on the self-heal
    assert "Read-Host" not in src and "-Confirm" not in src


def test_preflight_writes_one_launcher_log_line_defensively_source() -> None:
    src = PREFLIGHT_PS1.read_text(encoding="utf-8")
    assert "finally" in src, "the log line must be written in a finally block (always)"
    assert "AppendAllText" in src, "one appended line per launch"
    # the log write must not itself depend on the readiness lib having loaded
    fn = src.split("function Write-NfLauncherLine", 1)
    assert len(fn) == 2, "expected a self-contained Write-NfLauncherLine helper"
    body = fn[1].split("\n}", 1)[0]
    assert "try {" in body and "catch" in body, "log write must be wrapped in try/catch"
    for field in ("host=", "drive=", "repo=", "venv=", "checks:", "action=", "result="):
        assert field in src, f"launcher log line must carry the '{field}' field"


def test_launcher_cmd_calls_preflight_before_launch_source() -> None:
    src = LAUNCHER_CMD.read_text(encoding="utf-8")
    assert "nf-preflight.ps1" in src, "north-forge.cmd must run the readiness preflight"
    # the bare existence check is now only the fallback for a missing preflight script
    pre_idx = src.index("nf-preflight.ps1")
    launch_idx = src.rindex("hermes.exe")
    assert pre_idx < launch_idx, "preflight must run before the agent is launched"
    exist_idx = src.index('exist "%VENV%\\Scripts\\hermes.exe"')
    assert exist_idx > pre_idx, "hermes.exe existence check must be downstream of preflight (fallback only)"
    assert "LAUNCHLOG" in src


def test_bootstrap_decides_create_rebuild_refuse_source() -> None:
    """bootstrap dot-sources BOTH the readiness probe and the ownership classifier,
    keeps them as separate decisions, and drives a create / rebuild / refuse
    outcome from a state table - it never derives 'delete' from a failed probe
    alone, and it emits the stable venv_state=/ownership=/action=/reason= line."""
    src = BOOTSTRAP_PS1.read_text(encoding="utf-8")
    assert "lib\\nf-readiness.ps1" in src, "bootstrap must dot-source the readiness probe"
    assert "lib\\nf-venv-state.ps1" in src, "bootstrap must dot-source the ownership classifier"
    assert "Test-NfVenvReady" in src and "Get-NfVenvState" in src
    # the old naive early-return (marker + hermes.exe exist => done) must be gone
    assert "-and -not $Force) {\n    New-Item -ItemType Directory -Path $DataDir -Force | Out-Null\n    Write-Host \"Already bootstrapped (pass -Force" not in src
    for token in ("venv_state=", "ownership=", "action=", "reason="):
        assert token in src, f"bootstrap must emit the '{token}' diagnostic field"
    # every one of the seven states is named in the decision
    for state in ("Absent", "EmptyDirectory", "OwnedNorthForgeVenv", "RecognizablePythonVenv",
                  "UnknownDirectory", "UnsafePath"):
        assert state in src, f"decision must handle the '{state}' state"
    # refuse is explicit, and -Force is documented as NOT a deletion override
    assert "'refuse'" in src and "does not override" in src.lower()
    # a re-check of ownership sits immediately before the Remove-Item
    rm = src.index("Remove-Item -LiteralPath $VenvDir -Recurse -Force")
    window = src[max(0, rm - 700):rm]
    assert "Get-NfVenvState" in window, "ownership must be re-checked immediately before the delete"


# ==========================================================================
# behavioural - Windows only (PowerShell launcher path)
# ==========================================================================

_WINDOWS_ONLY = pytest.mark.skipif(
    os.name != "nt", reason="the North Forge launcher / preflight run under Windows PowerShell only"
)
_HAVE_PWSH = shutil.which("powershell") is not None


def _toolchain_env() -> dict:
    """A minimal-but-valid Windows env (works under the canonical runner's
    ``env -i``) PLUS python / uv on PATH and uv's cache vars - the probe and the
    stub bootstrap both spawn ``python``."""
    from tests._windows_env import minimal_windows_subprocess_env

    env = minimal_windows_subprocess_env()
    extra = [str(Path(sys.executable).parent), str(Path(sys.executable).parent / "Scripts")]
    uv = shutil.which("uv")
    if uv:
        extra.append(str(Path(uv).parent))
    env["PATH"] = os.pathsep.join(extra + [env["PATH"]])
    for key in ("LOCALAPPDATA", "APPDATA", "USERPROFILE", "UV_CACHE_DIR"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def _pwsh(*args: str, **kw) -> subprocess.CompletedProcess:
    kw.setdefault("env", _toolchain_env())
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", *args],
        capture_output=True, text=True, timeout=180, **kw,
    )


def _make_fake_checkout(root: Path) -> Path:
    """A throwaway checkout with just what the probe / preflight read: a
    pyproject, an importable ``hermes_cli`` package, and copies of the real
    readiness lib + preflight."""
    checkout = root / "north-forge-agent"
    (checkout / "scripts" / "lib").mkdir(parents=True)
    (checkout / "hermes_cli").mkdir()
    (checkout / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    (checkout / "hermes_cli" / "__init__.py").write_text('"""stub"""\n', encoding="utf-8")
    shutil.copy(LIB_PS1, checkout / "scripts" / "lib" / "nf-readiness.ps1")
    shutil.copy(PREFLIGHT_PS1, checkout / "scripts" / "nf-preflight.ps1")
    return checkout


_STUB_BOOTSTRAP = textwrap.dedent(r"""
    param([switch]$Force, [string]$RepoRoot, [string]$VenvDir, [string]$DataDir)
    Add-Content -LiteralPath $env:NF_SENTINEL -Value "called Force=$Force VenvDir=$VenvDir DataDir=$DataDir"
    if ($Force -and (Test-Path -LiteralPath $VenvDir)) { Remove-Item -LiteralPath $VenvDir -Recurse -Force }
    & (Get-Command python).Source -m venv $VenvDir | Out-Null
    Set-Content -LiteralPath (Join-Path $VenvDir 'Lib\site-packages\nf_stub.pth') -Value $RepoRoot
    New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $VenvDir 'Scripts\hermes.exe') -Value ''
    [System.IO.File]::WriteAllText((Join-Path $VenvDir '.nf-bootstrapped'), "repo=$RepoRoot`nbootstrapped=rebuilt`n")
    exit 0
""").strip()


def _probe(repo: Path, venv: Path) -> dict:
    """Run Test-NfVenvReady and return {'ready': bool, 'summary': str}."""
    script = (
        f". '{LIB_PS1}'; "
        f"$p = Test-NfVenvReady -RepoRoot '{repo}' -VenvDir '{venv}'; "
        "Write-Output ('READY=' + $p.Ready); Write-Output ('SUMMARY=' + $p.Summary)"
    )
    r = _pwsh("-Command", script)
    assert r.returncode == 0, f"probe crashed:\n{r.stdout}\n{r.stderr}"
    out = r.stdout
    ready = "READY=True" in out
    summary = next((ln.split("=", 1)[1].strip() for ln in out.splitlines() if ln.startswith("SUMMARY=")), "")
    return {"ready": ready, "summary": summary}


@_WINDOWS_ONLY
@pytest.mark.skipif(not _HAVE_PWSH, reason="powershell not on PATH")
def test_foreign_marker_alone_makes_the_probe_not_ready(tmp_path: Path) -> None:
    """THE regression case: a working venv whose ONLY defect is the marker's
    recorded repo path pointing somewhere other than the current checkout. Checks
    1-3 pass; check 4 alone must flip Ready to False."""
    checkout = _make_fake_checkout(tmp_path)
    venv = tmp_path / "north-forge-agent-venv"
    r = _pwsh("-Command", f"& (Get-Command python).Source -m venv '{venv}'")
    assert r.returncode == 0, f"could not build test venv:\n{r.stderr}"
    (venv / "Lib" / "site-packages" / "nf_stub.pth").write_text(str(checkout), encoding="utf-8")
    (venv / "Scripts" / "hermes.exe").write_text("", encoding="utf-8")

    # correct marker -> ready
    (venv / ".nf-bootstrapped").write_text(f"repo={checkout}\nbootstrapped=now\n", encoding="utf-8")
    healthy = _probe(checkout, venv)
    assert healthy["ready"], f"baseline venv should be ready: {healthy['summary']}"

    # marker now points at 'another machine' -> NOT ready, and only check 4 failed
    (venv / ".nf-bootstrapped").write_text(
        "repo=D:\\north-forge-agent-on-another-pc\nbootstrapped=old\n", encoding="utf-8"
    )
    foreign = _probe(checkout, venv)
    assert not foreign["ready"], "a foreign marker path must make the venv not-ready"
    assert "marker_repo_matches=FAIL" in foreign["summary"]
    assert "python_exec=PASS" in foreign["summary"]
    assert "import_hermes_cli=PASS" in foreign["summary"]
    assert "module_in_checkout=PASS" in foreign["summary"]


@_WINDOWS_ONLY
@pytest.mark.skipif(not _HAVE_PWSH, reason="powershell not on PATH")
def test_preflight_rebuilds_foreign_venv_instead_of_launching(tmp_path: Path) -> None:
    """The launcher's preflight, given a foreign-machine venv (full venv layout,
    python's base 'home' gone, marker -> other path), must REBUILD the venv (call
    bootstrap with -Force) rather than hand a broken environment to the agent -
    and must leave exactly one launcher-log line describing what it did."""
    checkout = _make_fake_checkout(tmp_path)
    (checkout / "scripts" / "bootstrap-north-forge.ps1").write_text(_STUB_BOOTSTRAP, encoding="utf-8")

    venv = tmp_path / "north-forge-agent-venv"
    data = tmp_path / "north-forge-agent-data"
    log = tmp_path / "north-forge-agent-launcher.log"
    sentinel = tmp_path / "sentinel.txt"
    (venv / "Scripts").mkdir(parents=True)
    data.mkdir()
    (data / "keep.txt").write_text("user data - must survive", encoding="utf-8")

    # a full venv copied from another PC: python.exe present, but its base is gone
    shutil.copy(Path(sys.executable), venv / "Scripts" / "python.exe")
    (venv / "pyvenv.cfg").write_text(
        "home = C:\\this\\base\\is\\gone\ninclude-system-site-packages = false\nversion = 3.11.9\n",
        encoding="utf-8",
    )
    (venv / "Scripts" / "hermes.exe").write_text("", encoding="utf-8")
    (venv / ".nf-bootstrapped").write_text(
        "repo=D:\\north-forge-agent-on-another-pc\nbootstrapped=old\n", encoding="utf-8"
    )

    env = dict(_toolchain_env(), NF_SENTINEL=str(sentinel))
    r = _pwsh(
        "-File", str(checkout / "scripts" / "nf-preflight.ps1"),
        "-RepoRoot", str(checkout), "-VenvDir", str(venv),
        "-DataDir", str(data), "-LogFile", str(log),
        env=env,
    )

    assert sentinel.exists(), (
        f"preflight did NOT rebuild - it would have launched a broken venv.\n{r.stdout}\n{r.stderr}"
    )
    assert "Force=True" in sentinel.read_text(encoding="utf-8"), "a present-but-broken venv must be rebuilt with -Force"
    assert r.returncode == 0, f"rebuild should have fixed it (rc={r.returncode}):\n{r.stdout}\n{r.stderr}"

    assert log.exists(), "the launcher log line must exist"
    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1, f"exactly one line per launch, got {len(lines)}: {lines}"
    line = lines[0]
    assert "marker_repo_matches=FAIL" in line, "log must record the pre-rebuild failure"
    # preflight no longer classifies or labels the venv - it just requests a
    # repair from bootstrap (the sole cleanup authority) and records the outcome.
    assert "action=bootstrap-repair" in line
    assert "result=repair-ok" in line
    for field in ("host=", "drive=", f"repo={checkout}", "venv="):
        assert field in line, f"log line missing {field!r}: {line}"

    assert (data / "keep.txt").read_text(encoding="utf-8") == "user data - must survive", (
        "the data folder must be untouched by a venv rebuild"
    )


@_WINDOWS_ONLY
@pytest.mark.skipif(not _HAVE_PWSH, reason="powershell not on PATH")
def test_preflight_norebuild_detects_but_leaves_repair_alone(tmp_path: Path) -> None:
    """``-NoRebuild`` must still probe and still log, but never invoke bootstrap."""
    checkout = _make_fake_checkout(tmp_path)
    (checkout / "scripts" / "bootstrap-north-forge.ps1").write_text(_STUB_BOOTSTRAP, encoding="utf-8")
    venv = tmp_path / "north-forge-agent-venv"
    data = tmp_path / "north-forge-agent-data"
    log = tmp_path / "nf.log"
    sentinel = tmp_path / "sentinel.txt"
    (venv / "Scripts").mkdir(parents=True)
    (venv / "Scripts" / "hermes.exe").write_text("", encoding="utf-8")
    (venv / ".nf-bootstrapped").write_text("repo=D:\\somewhere-else\nbootstrapped=old\n", encoding="utf-8")

    env = dict(_toolchain_env(), NF_SENTINEL=str(sentinel))
    r = _pwsh(
        "-File", str(checkout / "scripts" / "nf-preflight.ps1"),
        "-RepoRoot", str(checkout), "-VenvDir", str(venv),
        "-DataDir", str(data), "-LogFile", str(log), "-NoRebuild",
        env=env,
    )
    assert r.returncode == 2, f"expected exit 2 (not-ready, no repair): {r.returncode}\n{r.stdout}\n{r.stderr}"
    assert not sentinel.exists(), "-NoRebuild must not invoke bootstrap"
    text = log.read_text(encoding="utf-8")
    assert "action=none" in text and "result=not-ready-norebuild" in text


@_WINDOWS_ONLY
@pytest.mark.skipif(not _HAVE_PWSH, reason="powershell not on PATH")
def test_real_sibling_venv_if_present_is_ready() -> None:
    """Sanity: the developer's own sibling venv (if bootstrapped) passes the
    probe - the checks are not so strict they reject a healthy environment."""
    venv = REPO_ROOT.parent / (REPO_ROOT.name + "-venv")
    if not (venv / "Scripts" / "python.exe").exists() or not (venv / ".nf-bootstrapped").exists():
        pytest.skip("no bootstrapped sibling venv next to this checkout")
    result = _probe(REPO_ROOT, venv)
    assert result["ready"], f"healthy sibling venv rejected by the probe: {result['summary']}"
