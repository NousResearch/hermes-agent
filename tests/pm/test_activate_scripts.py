"""The venv-style activate scripts (./activate, ./activate.ps1).

`source ./activate` must put the composed pm env into the CURRENT shell
without ever invoking uv: it runs the pm store's pinned python (fallback:
the repo venv) to emit `python -m pm.cli env`, then exports the result,
with a venv-activate-style `deactivate` that restores the prior state.
These are real input->output checks against a fake store (same layout the
pm store uses: facts.json + a python-<version>-<target> entry), following
tests/pm conventions for faking the store.
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
    interpreter = entry / "bin" / (
        "python.exe" if sys.platform.startswith("win") else "python"
    )
    interpreter.parent.mkdir(parents=True)
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


def _bash_env(store: Path) -> dict:
    env = os.environ.copy()
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


def test_activate_never_invokes_uv():
    """The fast path must run the provisioned python directly — setup is
    the only place uv bootstrap logic lives."""
    source = ACTIVATE.read_text(encoding="utf-8")
    assert "uv run" not in source
    assert "ensure_pinned_uv" not in source


def test_source_activate_exports_the_pm_env(tmp_path: Path):
    store, _ = _fake_store(tmp_path)
    script = (
        f'source "{_posix(ACTIVATE)}" && '
        f'test -n "$__HERMES_ACTIVATED" && '
        f'printf "%s" "${CANARY}"'
    )
    result = subprocess.run(
        [_bash(), "-c", script],
        capture_output=True,
        text=True,
        cwd=_posix(REPO_ROOT),
        env=_bash_env(store),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "env-ok"


def test_deactivate_restores_the_prior_shell(tmp_path: Path):
    store, _ = _fake_store(tmp_path)
    script = (
        f'source "{_posix(ACTIVATE)}" && deactivate && '
        f'test -z "${{{CANARY}+set}}" && '
        f'test -z "${{__HERMES_ACTIVATED+set}}" && '
        f"! declare -F deactivate >/dev/null && "
        f'echo restored'
    )
    result = subprocess.run(
        [_bash(), "-c", script],
        capture_output=True,
        text=True,
        cwd=_posix(REPO_ROOT),
        env=_bash_env(store),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "restored"


def test_activate_fails_cleanly_without_a_store(tmp_path: Path):
    env = os.environ.copy()
    env["HERMES_RUNTIME_DIR"] = _posix(tmp_path / "empty-store")
    env.pop(CANARY, None)
    script = (
        f'source "{_posix(ACTIVATE)}" 2>/dev/null; '
        f'test $? -ne 0 && echo refused'
    )
    result = subprocess.run(
        [_bash(), "-c", script],
        capture_output=True,
        text=True,
        cwd=_posix(REPO_ROOT),
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


def test_powershell_activate_exports_and_deactivates(tmp_path: Path):
    ps = _powershell()
    if ps is None:
        pytest.skip("no PowerShell host available")
    store, _ = _fake_store(tmp_path)

    # The store interpreter is a /bin/sh wrapper — PowerShell needs a real
    # python.exe, so stage the running interpreter (+ its DLLs/zips) into
    # the fake entry; if that does not yield a runnable python, skip.
    entry = store / json.loads(store.joinpath("facts.json").read_text())["packages"][
        "python"
    ]["entry"]
    exe = Path(sys.executable)
    shim = entry / "bin" / "python.exe"
    for src in exe.parent.glob("*.dll"):
        shutil.copy2(src, entry / "bin" / src.name)
    for base in (exe.parent, Path(sys.base_prefix)):
        for zipname in ("python311.zip", "python312.zip"):
            if (base / zipname).exists():
                shutil.copy2(base / zipname, entry / "bin" / zipname)
    if (exe.parent / "Lib").is_dir():
        shutil.copytree(exe.parent / "Lib", entry / "bin" / "Lib", dirs_exist_ok=True)
    shutil.copy2(exe, shim)

    probe = subprocess.run(
        [str(shim), "-c", "print(1)"], capture_output=True, text=True, env=_child_env()
    )
    if probe.returncode != 0:
        pytest.skip("copied interpreter is not runnable on this host")

    result = subprocess.run(
        [
            ps,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            f"$env:HERMES_RUNTIME_DIR = '{store}'; "
            f". '{ACTIVATE_PS1}'; "
            f"$active = $env:{CANARY}; "
            f"deactivate; "
            f"$after = $env:{CANARY}; "
            f"Write-Output ('active=' + $active + ' after=' + $after)",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_child_env(),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "active=env-ok" in result.stdout
    assert "after=" in result.stdout and "after=env-ok" not in result.stdout
