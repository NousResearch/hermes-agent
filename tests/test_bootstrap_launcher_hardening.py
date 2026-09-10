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

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests._windows_env import minimal_windows_subprocess_env

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap-north-forge.ps1"
LAUNCHER = REPO_ROOT / "north-forge.cmd"
_WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="drives a .ps1 / .cmd")


# --------------------------------------------------------------------------- Fix 1 (source)


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


@pytest.fixture
def run_native_helpers_probe(tmp_path):
    """Run a probe ``.ps1`` that defines ONLY the two bootstrap native helpers
    (lifted verbatim) plus *body*, under a minimal but VALID Windows environment.

    The probe gets an explicit clean env rather than inheriting the test runner's:
    the canonical runner executes under Git-Bash ``env -i``, which leaves
    ``PATHEXT`` / ``ComSpec`` / the env block in a state where a PowerShell
    subprocess there cannot spawn ANY child process (silent exit 0, no output).
    A behavioural test of native-command handling must not be hostage to that.
    Its own ``tmp_path`` probe — no sibling ``*-venv`` / bootstrapped checkout
    assumed."""

    def _run(body: str) -> subprocess.CompletedProcess:
        probe = tmp_path / "native_probe.ps1"
        probe.write_text(
            "$ErrorActionPreference = 'Stop'\n"
            + _extract_helpers()
            + "\n"
            + textwrap.dedent(body)
            + "\n",
            encoding="utf-8",
        )
        return subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(probe)],
            capture_output=True, text=True, env=minimal_windows_subprocess_env(),
        )

    return _run


@pytest.mark.windows_only
def test_native_helpers_preserve_streams_exit_code_and_restore_eap(run_native_helpers_probe):
    """Behavioural (replaces the old source-grep): a child that writes BOTH stdout
    and stderr and exits non-zero keeps all three intact through the helpers, and
    ``$ErrorActionPreference`` is back to its prior value afterward."""
    r = run_native_helpers_probe(
        r"""
        $before = $ErrorActionPreference

        # Invoke-Native: streams pass straight through untouched; the verdict is
        # $LASTEXITCODE on the very next line.
        $LASTEXITCODE = 0
        Invoke-Native { & cmd /c 'echo out-line & echo err-line 1>&2 & exit 7' }
        "INVOKE_CODE=$LASTEXITCODE"
        "INVOKE_EAP_RESTORED=$($ErrorActionPreference -eq $before)"

        # Get-NativeText: merged stdout+stderr is RETURNED; exit code still readable.
        $LASTEXITCODE = 0
        $merged = Get-NativeText { & cmd /c 'echo cap-out & echo cap-err 1>&2 & exit 4' }
        "TEXT_CODE=$LASTEXITCODE"
        "TEXT_EAP_RESTORED=$($ErrorActionPreference -eq $before)"
        "TEXT_HAS_OUT=$($merged -match 'cap-out')"
        "TEXT_HAS_ERR=$($merged -match 'cap-err')"
        """
    )
    combined = r.stdout + r.stderr
    assert r.returncode == 0, combined

    # Invoke-Native: stdout stayed on stdout, stderr stayed on stderr (no redirect,
    # no merge), the non-zero exit survived, and EAP was restored.
    assert "out-line" in r.stdout, combined
    assert "err-line" in r.stderr, combined
    assert "INVOKE_CODE=7" in r.stdout, combined
    assert "INVOKE_EAP_RESTORED=True" in r.stdout, combined

    # Get-NativeText: both streams captured into the return value, exit code kept,
    # EAP restored.
    assert "TEXT_CODE=4" in r.stdout, combined
    assert "TEXT_HAS_OUT=True" in r.stdout, combined
    assert "TEXT_HAS_ERR=True" in r.stdout, combined
    assert "TEXT_EAP_RESTORED=True" in r.stdout, combined


@_WINDOWS_ONLY
def test_invoke_native_survives_stderr_and_preserves_exit_code(tmp_path):
    harness = _PS_HELPER_HARNESS.replace("__HELPERS__", _extract_helpers())
    script = tmp_path / "h.ps1"
    script.write_text(harness, encoding="utf-8")
    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        capture_output=True, text=True, env=minimal_windows_subprocess_env())
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


# ---------------------------------------------------------------- R1: direct bootstrap
# The GUI Setup app calls bootstrap-north-forge.ps1 DIRECTLY - only -RepoRoot,
# no -Force, no preflight (CHG-2026-09-08-001). bootstrap must be self-sufficient:
# its venv ownership/cleanup safety cannot depend on nf-preflight.ps1 running first.

import hashlib as _hashlib   # noqa: E402
import os as _os             # noqa: E402
import shutil as _shutil     # noqa: E402


def _toolchain_env() -> dict:
    """minimal_windows_subprocess_env() + python/uv on PATH + uv's cache vars, so
    a case that really builds a venv works under scripts/run_tests.sh's env -i."""
    env = minimal_windows_subprocess_env()
    extra = [str(Path(sys.executable).parent), str(Path(sys.executable).parent / "Scripts")]
    uv = _shutil.which("uv")
    if uv:
        extra.append(str(Path(uv).parent))
    env["PATH"] = _os.pathsep.join(extra + [env["PATH"]])
    for key in ("LOCALAPPDATA", "APPDATA", "USERPROFILE", "UV_CACHE_DIR"):
        if _os.environ.get(key):
            env[key] = _os.environ[key]
    return env


def _r1_direct_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "north-forge-agent"
    (repo / "scripts" / "lib").mkdir(parents=True)
    (repo / "hermes_cli").mkdir()
    (repo / "pyproject.toml").write_text(
        "[build-system]\nrequires=['setuptools>=61']\nbuild-backend='setuptools.build_meta'\n"
        "[project]\nname='nf-fake'\nversion='0.0.0'\n"
        "[project.scripts]\nhermes='hermes_cli:main'\n"
        "[tool.setuptools]\npackages=['hermes_cli']\n",
        encoding="utf-8",
    )
    (repo / "hermes_cli" / "__init__.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    for name in ("nf-readiness.ps1", "nf-venv-state.ps1", "nf-toolchain.ps1"):
        (repo / "scripts" / "lib" / name).write_bytes((REPO_ROOT / "scripts" / "lib" / name).read_bytes())
    (repo / "scripts" / "bootstrap-north-forge.ps1").write_bytes(BOOTSTRAP.read_bytes())
    (repo / "scripts" / "make-drive-root-shortcut.ps1").write_text("param([string]$RepoRoot)\n", encoding="utf-8")
    return repo


def _sentinel_hash(data: Path) -> str:
    return _hashlib.sha256((data / "SENTINEL.bin").read_bytes()).hexdigest()


@_WINDOWS_ONLY
def test_direct_bootstrap_refuses_unknown_venv_dir_without_preflight(tmp_path):
    """GUI path: `bootstrap-north-forge.ps1 -RepoRoot <root>` with no -Force and
    no preflight. If the derived sibling <leaf>-venv is an occupied directory
    bootstrap can't prove it owns, it must REFUSE - delete nothing, and never
    touch the sibling data folder."""
    repo = _r1_direct_repo(tmp_path)
    venv = tmp_path / "north-forge-agent-venv"
    data = tmp_path / "north-forge-agent-data"
    venv.mkdir()
    (venv / "not-ours.txt").write_text("someone else's directory", encoding="utf-8")
    data.mkdir()
    (data / "SENTINEL.bin").write_bytes(bytes(range(256)) * 5)
    before_venv = sorted(p.name for p in venv.iterdir())
    before_hash = _sentinel_hash(data)

    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(repo / "scripts" / "bootstrap-north-forge.ps1"), "-RepoRoot", str(repo)],
        capture_output=True, text=True, timeout=180, env=minimal_windows_subprocess_env(),
    )
    combined = r.stdout + r.stderr
    assert r.returncode != 0, combined
    assert "action=refuse" in combined and "venv_state=UnknownDirectory" in combined
    assert "nothing was deleted" in combined.lower()
    assert sorted(p.name for p in venv.iterdir()) == before_venv, "refused venv dir was modified"
    assert _sentinel_hash(data) == before_hash, "data folder touched during a refused direct bootstrap"


@_WINDOWS_ONLY
def test_direct_bootstrap_recovers_interrupted_venv_without_preflight(tmp_path):
    """GUI path again: an interrupted venv (valid pyvenv.cfg, Scripts\\, no
    python.exe) handed straight to bootstrap with no -Force and no preflight is
    rebuilt (not silently populated in place), and the data sentinel survives."""
    repo = _r1_direct_repo(tmp_path)
    venv = tmp_path / "north-forge-agent-venv"
    data = tmp_path / "north-forge-agent-data"
    (venv / "Scripts").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(
        "home = C:\\Python311\ninclude-system-site-packages = false\nversion = 3.11.9\n", encoding="utf-8"
    )
    data.mkdir()
    (data / "SENTINEL.bin").write_bytes(bytes(range(256)) * 5)
    before_hash = _sentinel_hash(data)

    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(repo / "scripts" / "bootstrap-north-forge.ps1"), "-RepoRoot", str(repo)],
        capture_output=True, text=True, timeout=300, env=_toolchain_env(),
    )
    combined = r.stdout + r.stderr
    assert r.returncode == 0, combined
    assert "venv_state=RecognizablePythonVenv" in combined and "action=rebuild" in combined
    assert (venv / "Scripts" / "python.exe").exists(), combined
    assert _sentinel_hash(data) == before_hash
    assert not (data / "skins").exists(), "a rebuild via the direct path re-seeded the data folder"
