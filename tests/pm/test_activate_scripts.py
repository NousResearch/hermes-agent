"""Run the real activation scripts with setup replaced at its process boundary.

The isolated checkout uses the real PM environment reader and fake installed
artifacts. Setup records each sync and publishes a selected environment; no test
sources the working checkout or runs its installer against the developer's home.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pm.store import current_target

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVATE = REPO_ROOT / "activate"
ACTIVATE_PS1 = REPO_ROOT / "activate.ps1"
SETUP_HERMES_SH = REPO_ROOT / "setup-hermes.sh"
SETUP_HERMES_PS1 = REPO_ROOT / "setup-hermes.ps1"
CANARY = "HERMES_PM_ACTIVATE_CANARY"


def _posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def _bash() -> str:
    """A bash CreateProcess can start. conftest blanks SystemRoot/ComSpec
    and the hermetic runner's PATH may resolve `bash` to the MSIX payload
    copy under ``C:\\Program Files\\WindowsApps\\...`` (WinError 5 outside
    its package context) or miss entirely — prefer a conventional install
    (same pattern as the bootstrap version-stamp tests' _git_exe)."""
    found = shutil.which("bash")
    if found and "windowsapps" not in str(found).lower():
        return found
    if sys.platform == "win32":
        pf = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        for rel in (("Git", "bin", "bash.exe"), ("Git", "usr", "bin", "bash.exe")):
            cand = pf.joinpath(*rel)
            if cand.exists():
                return str(cand)
    return found or "bash"


def _child_env() -> dict:
    env = os.environ.copy()
    if sys.platform == "win32":
        env.setdefault("SystemRoot", r"C:\Windows")
        env.setdefault("ComSpec", r"C:\Windows\system32\cmd.exe")
        # PowerShell 5.1 silently fails to launch children through `&`
        # without PATHEXT (empty output, exit 0) — the hermetic runner
        # drops it, so every powershell-invoking child env needs it back.
        env.setdefault(
            "PATHEXT",
            ".COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC",
        )
    return env


def _spawnable_python() -> Path:
    """An interpreter this process can actually CreateProcess. The hermetic
    runner's venv python can be an emulated x64 binary on an arm64 host
    (WinError 5 on every spawn); the native pm-store python under
    WindowsApps works when invoked by full path. Prefer whichever python
    can run a trivial child; last resort is sys.executable."""
    candidates: list[Path] = []
    for env_name in ("HERMES_TEST_PYTHON",):
        val = os.environ.get(env_name)
        if val:
            candidates.append(Path(val))
    exe = Path(sys.executable)
    if "windowsapps" in str(exe).lower():
        candidates.append(exe)  # native packaged python — spawnable by path
    else:
        candidates.insert(0, exe)
    candidates.append(exe)
    for cand in candidates:
        if _can_spawn(cand):
            return cand
    return exe


def _can_spawn(python: Path) -> bool:
    try:
        r = subprocess.run(
            [str(python), "-c", "print(1)"],
            capture_output=True,
            text=True,
            timeout=30,
            env=_child_env(),
        )
        return r.returncode == 0
    except Exception:
        return False


def _fake_store(tmp_path: Path) -> tuple[Path, Path]:
    """A store laid out like pm's: facts.json marking the python package
    installed (identity matches pm/lock.json) with a canary env export, and
    an entry whose interpreter is a wrapper that runs the real python."""
    lock = json.loads(
        (REPO_ROOT / "pm" / "lock.json").read_text(encoding="utf-8-sig")
    )["packages"]
    target = current_target()
    python_pkg = lock["python"]
    sha = python_pkg["artifacts"][target]["sha256"]

    store = tmp_path / "store"
    entry = store / f"python-{python_pkg['version']}-{target}"
    entry.mkdir(parents=True)

    # The store interpreter only needs to run `python -m pm.cli env` —
    # delegate to a real, spawnable interpreter via a #!/bin/sh wrapper (a
    # copied CPython would miss its DLLs/stdlib; the wrapper is the honest
    # minimal fake). sys.executable can be an emulated x64 python on an
    # arm64 host that cannot CreateProcess children at all — resolve a
    # spawnable interpreter instead (see _spawnable_python). The wrapper
    # works even named python.exe because activate execs it through
    # bash/MSYS, which honors #!-scripts regardless of suffix.
    interpreter = entry / ("python.exe" if sys.platform.startswith("win") else "bin/python3")
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    real = _spawnable_python()
    wrapper = "#!/bin/sh\nexec '%s' \"$@\"\n" % _posix(real)
    interpreter.write_text(wrapper, encoding="utf-8")
    interpreter.chmod(0o755)

    (store / "facts.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "packages": {
                    "python": {
                        "entry": entry.name,
                        "version": python_pkg["version"],
                        "env": {CANARY: "env-ok"},
                        "target": target,
                        "artifacts": [sha],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return store, entry


def _isolated_checkout(tmp_path: Path) -> Path:
    root = tmp_path / "checkout with spaces"
    root.mkdir()
    shutil.copytree(REPO_ROOT / "pm", root / "pm", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "hermes_cli").mkdir()
    for relative in ("activate", "activate.ps1", "hermes_constants.py", "hermes_cli/__init__.py",
                     "hermes_cli/runtime_paths.py", "hermes_cli/runtime_state.py"):
        shutil.copy2(REPO_ROOT / relative, root / relative)
    # Environment-only tests do not exercise provisioning; the runtime tests
    # replace these stubs with a publisher that records and applies each sync.
    (root / "setup-hermes.sh").write_text('test "$#" = 1 && test "$1" = --runtime-only\n', encoding="utf-8")
    (root / "setup-hermes.ps1").write_text(
        "param([switch]$RuntimeOnly)\nif (-not $RuntimeOnly) { exit 2 }\n", encoding="utf-8",
    )
    return root


def _bash_env(store: Path) -> dict:
    env = _child_env()
    home = store.parent / "home"
    home.mkdir(exist_ok=True)
    env.update(HOME=_posix(home), USERPROFILE=str(home), HERMES_HOME=_posix(home / "hermes"))
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "BASH_ENV", "__HERMES_ACTIVATED"):
        env.pop(key, None)
    env["HERMES_RUNTIME_DIR"] = _posix(store)
    # Keep the real env out of the composed pm output so the canary export
    # is the only thing activate adds beyond the ambient environment.
    env.pop(CANARY, None)
    return env


def test_bash_scripts_pass_syntax_check():
    for script in (ACTIVATE, SETUP_HERMES_SH):
        result = subprocess.run(
            [_bash(), "-n", _posix(script)], capture_output=True, text=True, env=_child_env()
        )
        assert result.returncode == 0, f"{script.name}: {result.stderr}"


def test_source_activate_exports_the_pm_env(tmp_path: Path):
    root = _isolated_checkout(tmp_path)
    store, _ = _fake_store(tmp_path)
    script = (
        f'source "{_posix(root / "activate")}" && '
        f'test -n "$__HERMES_ACTIVATED" && '
        f'printf "%s" "${CANARY}"'
    )
    result = subprocess.run(
        [_bash(), "-c", script],
        capture_output=True,
        text=True,
        cwd=_posix(tmp_path),
        env=_bash_env(store),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "env-ok"


def test_deactivate_restores_the_prior_shell(tmp_path: Path):
    root = _isolated_checkout(tmp_path)
    store, _ = _fake_store(tmp_path)
    script = (
        f'source "{_posix(root / "activate")}" && deactivate && '
        f'test -z "${{{CANARY}+set}}" && '
        f'test -z "${{__HERMES_ACTIVATED+set}}" && '
        f"! declare -F deactivate >/dev/null && "
        f'echo restored'
    )
    result = subprocess.run(
        [_bash(), "-c", script],
        capture_output=True,
        text=True,
        cwd=_posix(tmp_path),
        env=_bash_env(store),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "restored"


def test_activate_fails_cleanly_without_a_store(tmp_path: Path):
    env = _bash_env(tmp_path / "empty-store")
    isolated = _isolated_checkout(tmp_path) / "activate"
    script = (
        f'source "{_posix(isolated)}" 2>/dev/null; '
        f'test $? -ne 0 && echo refused'
    )
    result = subprocess.run(
        [_bash(), "-c", script],
        capture_output=True,
        text=True,
        cwd=_posix(tmp_path),
        env=env,
    )
    # Without any provisioned python the source must refuse — never
    # silently no-op with a half-activated shell.
    assert "refused" in result.stdout


def _powershell() -> str | None:
    for name in ("pwsh", "powershell"):
        found = shutil.which(name)
        if found and "windowsapps" not in str(found).lower():
            return found
    # Prefer the conventional System32 host over an MSIX-packaged one
    # (same WinError-5-outside-package-context class as _bash()); also the
    # fallback when the hermetic runner's PATH misses both names.
    if sys.platform == "win32":
        cand = (
            Path(os.environ.get("SystemRoot", r"C:\Windows"))
            / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        )
        if cand.exists():
            return str(cand)
    return None


def test_powershell_scripts_parse():
    """Parse-check the PowerShell entry points; skip gracefully when no
    PowerShell host is available."""
    ps = _powershell()
    if ps is None:
        pytest.skip("no PowerShell host available")
    for script in (ACTIVATE_PS1, SETUP_HERMES_PS1):
        result = subprocess.run(
            [
                ps,
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                f"$errs = $null; $null = [System.Management.Automation.Language.Parser]::ParseFile("
                f"'{script}', [ref]$null, [ref]$errs); "
                f"if ($errs.Count) {{ $errs | ForEach-Object {{ $_.Message }}; exit 1 }}",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{script.name}: {result.stdout}{result.stderr}"


@pytest.mark.platforms("windows")
def test_powershell_activate_exports_and_deactivates(tmp_path: Path):
    # Native venv redirectors resolve the base DLLs/stdlib without copying CPython.
    root = _isolated_checkout(tmp_path)
    env = _bash_env(tmp_path / "store")
    subprocess.run(
        [str(_spawnable_python()), "-m", "venv", "--without-pip", str(root / ".venv")],
        check=True, capture_output=True, env=env, timeout=60,
    )
    ps = _powershell()
    assert ps, "native Windows test requires PowerShell"
    result = subprocess.run(
        [ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command",
         f"$ErrorActionPreference='Stop'; $env:PYTHONPATH='caller-original'; "
         f". '{root / 'activate.ps1'}'; "
         "Write-Output ('active=' + $env:PYTHONPATH); deactivate; "
         "Write-Output ('after=' + $env:PYTHONPATH)"],
        capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"active={root}{os.pathsep}" in result.stdout
    assert "after=caller-original" in result.stdout
