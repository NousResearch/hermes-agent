"""Fixtures shared by the shell-activation tests.

An isolated checkout holds the real activation scripts, the real PM environment
reader and fake installed artifacts. Setup is replaced at its process boundary:
the stub records each sync and publishes a selected environment. Nothing here
sources the working checkout or runs its installer against the developer's home.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from pm.store import current_target

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVATE = REPO_ROOT / "activate"
ACTIVATE_FISH = REPO_ROOT / "activate.fish"
ACTIVATE_PS1 = REPO_ROOT / "activate.ps1"
SETUP_HERMES_SH = REPO_ROOT / "setup-hermes.sh"
SETUP_HERMES_PS1 = REPO_ROOT / "setup-hermes.ps1"
CANARY = "HERMES_PM_ACTIVATE_CANARY"


def posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def bash() -> str:
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


def child_env() -> dict:
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


def spawnable_python() -> Path:
    """An interpreter this process can actually CreateProcess. The hermetic
    runner's venv python can be an emulated x64 binary on an arm64 host
    (WinError 5 on every spawn), or a macOS venv shim that stops working
    after activation sanitizes its parent environment. Prefer the venv's
    base interpreter, then whichever python can run a trivial child; last
    resort is sys.executable."""
    candidates: list[Path] = []
    for env_name in ("HERMES_TEST_PYTHON",):
        val = os.environ.get(env_name)
        if val:
            candidates.append(Path(val))
    exe = Path(sys.executable)
    if "windowsapps" in str(exe).lower():
        candidates.append(exe)  # native packaged python — spawnable by path
    else:
        base = Path(getattr(sys, "_base_executable", "") or exe)
        if base != exe:
            candidates.append(base)
        candidates.append(exe)
    for cand in candidates:
        if can_spawn(cand):
            return cand
    return exe


def can_spawn(python: Path) -> bool:
    try:
        r = subprocess.run(
            [str(python), "-c", "print(1)"],
            capture_output=True,
            text=True,
            timeout=30,
            env=child_env(),
        )
        return r.returncode == 0
    except Exception:
        return False


def fake_store(tmp_path: Path) -> tuple[Path, Path]:
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
    # spawnable interpreter instead (see spawnable_python). The wrapper
    # works even named python.exe because activate execs it through
    # bash/MSYS, which honors #!-scripts regardless of suffix.
    interpreter = entry / ("python.exe" if sys.platform.startswith("win") else "bin/python3")
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    real = spawnable_python()
    wrapper = "#!/bin/sh\nexec '%s' \"$@\"\n" % posix(real)
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


def isolated_checkout(tmp_path: Path) -> Path:
    root = tmp_path / "checkout with spaces"
    root.mkdir()
    shutil.copytree(REPO_ROOT / "pm", root / "pm", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "hermes_cli").mkdir()
    (root / "scripts").mkdir()
    for relative in ("activate", "activate.fish", "activate.ps1", "scripts/_activation.sh",
                     "hermes_constants.py", "hermes_cli/__init__.py",
                     "pm/environments.py", "hermes_cli/runtime_state.py"):
        shutil.copy2(REPO_ROOT / relative, root / relative)
    # Environment-only tests do not exercise provisioning; the runtime tests
    # replace these stubs with a publisher that records and applies each sync.
    (root / "setup-hermes.sh").write_text(
        'test "$#" = 2 && test "$1" = --runtime-only && case "$2" in --test-environment*) ;; *) exit 2 ;; esac\n',
        encoding="utf-8",
    )
    (root / "setup-hermes.ps1").write_text(
        "param([switch]$RuntimeOnly)\nif (-not $RuntimeOnly) { exit 2 }\n", encoding="utf-8",
    )
    return root


def bash_env(store: Path) -> dict:
    env = child_env()
    home = store.parent / "home"
    home.mkdir(exist_ok=True)
    env.update(HOME=posix(home), USERPROFILE=str(home), HERMES_HOME=posix(home / "hermes"))
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "BASH_ENV", "__HERMES_ACTIVATED"):
        env.pop(key, None)
    env["HERMES_RUNTIME_DIR"] = posix(store)
    # Keep the real env out of the composed pm output so the canary export
    # is the only thing activate adds beyond the ambient environment.
    env.pop(CANARY, None)
    return env


def powershell() -> str | None:
    for name in ("pwsh", "powershell"):
        found = shutil.which(name)
        if found and "windowsapps" not in str(found).lower():
            return found
    # Prefer the conventional System32 host over an MSIX-packaged one
    # (same WinError-5-outside-package-context class as bash()); also the
    # fallback when the hermetic runner's PATH misses both names.
    if sys.platform == "win32":
        cand = (
            Path(os.environ.get("SystemRoot", r"C:\Windows"))
            / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        )
        if cand.exists():
            return str(cand)
    return None


def sync_checkout(tmp_path: Path):
    root = isolated_checkout(tmp_path)
    env = bash_env(tmp_path / "store")
    python = spawnable_python()
    # The setup seam publishes a bootstrap and a selected generation. PM's
    # freshness algorithms have their own tests; here a warm setup call is a
    # no-op and activation must still call it, rather than cache its own answer.
    (root / "sync.py").write_text(textwrap.dedent('''\
        import json, os, pathlib, shutil, sys
        from pm.environments import runtime_facts_path, site_packages
        root = pathlib.Path(__file__).parent
        record = {"argv": sys.argv[1:], "python_env": {
            key: os.environ.get(key) for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")}}
        with (root / "calls.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\\n")
        assert all(value is None for value in record["python_env"].values()), record
        assert sys.argv[1:] == ["runtime-only", "--trust-recorded", "--test-environment"], record
        print("setup progress")
        if (root / "fail").exists():
            sys.exit(42)
        bootstrap = root / ".venv"
        if not bootstrap.exists():
            shutil.copytree(root / "prepared-bootstrap", bootstrap)
        # Windows PowerShell's Set-Content writes a BOM.
        generation = (root / "input").read_text(encoding="utf-8-sig").strip()
        facts = runtime_facts_path(root)
        selected = facts.parent / "environments" / generation / "venv"
        if not selected.exists():
            site_packages(selected).mkdir(parents=True)
            (selected / "pyvenv.cfg").write_text("home = fixture", encoding="utf-8")
            facts.write_text(json.dumps({"packages": {"venv": {"environment": str(selected)}}}), encoding="utf-8")
            with (root / "builds").open("a", encoding="utf-8") as stream:
                stream.write(generation + "\\n")
    '''), encoding="utf-8")
    (root / "input").write_text("first", encoding="utf-8")
    if os.name == "nt":
        subprocess.run(
            [str(python), "-m", "venv", "--without-pip", str(root / "prepared-bootstrap")],
            check=True, capture_output=True, env=env, timeout=60,
        )
    else:
        binary = root / "prepared-bootstrap" / "bin" / "python"
        binary.parent.mkdir(parents=True)
        binary.write_text(f"#!/bin/sh\nexec {shlex.quote(str(python))} \"$@\"\n", encoding="utf-8")
        binary.chmod(0o755)
    # Activation's contract with setup is the runtime-only switch; setup
    # itself maps that to PM's --trust-recorded install.
    (root / "setup-hermes.sh").write_text(
        'test "$#" = 2 && test "$1" = --runtime-only && test "$2" = --test-environment || exit 2\n'
        f'cd {shlex.quote(str(root))} || exit 3\n'
        f'exec {shlex.quote(str(python))} sync.py runtime-only --trust-recorded --test-environment\n', encoding="utf-8",
    )
    (root / "setup-hermes.ps1").write_text(
        "param([switch]$RuntimeOnly)\n"
        "if (-not $RuntimeOnly) { exit 2 }\n"
        "$ErrorActionPreference = 'Stop'\n"
        "$record = @{pid=$PID; executable=(Get-Process -Id $PID).Path; "
        "argv=[Environment]::GetCommandLineArgs()}\n"
        "$record | ConvertTo-Json -Compress | Add-Content -LiteralPath \"$PSScriptRoot\\ps-calls.jsonl\"\n"
        "Set-Location -LiteralPath $PSScriptRoot\n"
        f"& '{python}' sync.py runtime-only --trust-recorded --test-environment\nexit $LASTEXITCODE\n", encoding="utf-8",
    )
    return root, env
