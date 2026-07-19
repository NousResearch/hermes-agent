"""Behavioral regression tests for the Windows baseline-import gate.

The test runs the real ``install.ps1 -Stage dependencies`` path against a
minimal temporary project. Local stand-in modules avoid network dependency
resolution, while ``dotenv.py`` fails a controlled number of imports across
separate Python processes. This exercises PowerShell 5.1 process behavior,
not the installer's source text.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows PowerShell 5.1 installer regression",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def _command_path(name: str) -> Path:
    resolved = shutil.which(name)
    if not resolved:
        pytest.skip(f"required command not available: {name}")
    return Path(resolved).resolve()


def _prepare_minimal_install(
    tmp_path: Path, *, standins_in_managed_site_packages: bool = True
) -> tuple[Path, Path, Path]:
    uv = _command_path("uv.exe")
    powershell = _command_path("powershell.exe")

    hermes_home = tmp_path / "hermes-home"
    install_dir = tmp_path / "hermes-agent"
    managed_uv = hermes_home / "bin" / "uv.exe"
    managed_uv.parent.mkdir(parents=True)
    shutil.copy2(uv, managed_uv)

    install_dir.mkdir()
    (install_dir / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "installer-probe-test"
            version = "0.0.0"
            requires-python = ">=3.11"
            dependencies = []

            [project.optional-dependencies]
            all = []
            web = []

            [tool.uv]
            package = false
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            str(managed_uv),
            "venv",
            str(install_dir / "venv"),
            "--python",
            sys.executable,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [str(managed_uv), "lock", "--project", str(install_dir)],
        check=True,
        capture_output=True,
        text=True,
    )

    managed_python = install_dir / "venv" / "Scripts" / "python.exe"
    site_packages = Path(
        subprocess.run(
            [
                str(managed_python),
                "-I",
                "-c",
                "import site; print(next(p for p in site.getsitepackages() if p.lower().endswith('site-packages')))",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )

    # Normal fixtures live in the managed venv. One negative test deliberately
    # puts them in the checkout to prove isolated mode rejects that shadow path.
    module_root = site_packages if standins_in_managed_site_packages else install_dir
    for module in ("openai", "rich", "prompt_toolkit", "fastapi", "uvicorn"):
        (module_root / f"{module}.py").write_text("# test stand-in\n", encoding="utf-8")

    # Fail a configurable number of imports across separate Python processes.
    # The installer must tolerate a short-lived import failure, then continue
    # when the exact same interpreter succeeds on a later probe.
    (module_root / "dotenv.py").write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path
            import sys

            state = Path(os.environ["HERMES_TEST_PROBE_STATE"])
            count = int(state.read_text(encoding="utf-8")) if state.exists() else 0
            count += 1
            state.write_text(str(count), encoding="utf-8")
            failures = int(os.environ.get("HERMES_TEST_PROBE_FAILURES", "0"))
            if count <= failures:
                raise RuntimeError(f"synthetic baseline import failure {count}")
            if os.environ.get("HERMES_TEST_PROBE_STDERR") == "1":
                print("synthetic benign stderr", file=sys.stderr)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    web_server = install_dir / "hermes_cli" / "web_server.py"
    web_server.parent.mkdir()
    web_server.write_text("# syntax probe fixture\n", encoding="utf-8")

    return hermes_home, install_dir, powershell


def _run_dependency_stage(
    tmp_path: Path,
    *,
    failures: int,
    emit_stderr: bool = False,
    standins_in_managed_site_packages: bool = True,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    workspace = tmp_path / "installer workspace"
    workspace.mkdir()
    hermes_home, install_dir, powershell = _prepare_minimal_install(
        workspace,
        standins_in_managed_site_packages=standins_in_managed_site_packages,
    )
    state = workspace / "probe-count.txt"
    env = os.environ.copy()
    env["HERMES_TEST_PROBE_STATE"] = str(state)
    env["HERMES_TEST_PROBE_FAILURES"] = str(failures)
    env["HERMES_TEST_PROBE_STDERR"] = "1" if emit_stderr else "0"
    if not standins_in_managed_site_packages:
        env["PYTHONPATH"] = str(install_dir)

    result = subprocess.run(
        [
            str(powershell),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(INSTALL_PS1),
            "-Stage",
            "dependencies",
            "-NonInteractive",
            "-HermesHome",
            str(hermes_home),
            "-InstallDir",
            str(install_dir),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    managed_python = install_dir / "venv" / "Scripts" / "python.exe"
    return result, state, managed_python


def _stage_frame(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    for line in reversed(result.stdout.splitlines()):
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(frame, dict) and "ok" in frame:
            return frame
    pytest.fail(
        f"dependency stage emitted no JSON frame:\n{result.stdout}{result.stderr}"
    )


def test_dependency_stage_accepts_benign_probe_stderr(tmp_path: Path) -> None:
    result, state, managed_python = _run_dependency_stage(
        tmp_path, failures=0, emit_stderr=True
    )
    output = result.stdout + result.stderr
    frame = _stage_frame(result)
    site_packages = managed_python.parent.parent / "Lib" / "site-packages"

    assert result.returncode == 0, output
    assert frame["ok"] is True
    assert state.read_text(encoding="utf-8") == "1"
    assert "Baseline import probe failed" not in output
    assert "module_origins=" in output
    assert str(site_packages).lower() in output.lower()
    assert "Baseline imports verified in venv" in output


def test_dependency_stage_retries_transient_baseline_import_failure(
    tmp_path: Path,
) -> None:
    result, state, _ = _run_dependency_stage(tmp_path, failures=2)
    output = result.stdout + result.stderr
    frame = _stage_frame(result)

    assert result.returncode == 0, output
    assert frame["ok"] is True
    assert state.read_text(encoding="utf-8") == "3"
    assert "Baseline import probe failed (attempt 1/3," in output
    assert "Baseline import probe failed (attempt 2/3," in output
    assert "Baseline imports verified in venv" in output


def test_dependency_stage_ignores_checkout_and_inherited_pythonpath(
    tmp_path: Path,
) -> None:
    result, state, managed_python = _run_dependency_stage(
        tmp_path,
        failures=0,
        standins_in_managed_site_packages=False,
    )
    output = result.stdout + result.stderr
    frame = _stage_frame(result)
    reason = frame["reason"]

    assert result.returncode == 1, output
    assert frame["ok"] is False
    assert isinstance(reason, str)
    assert not state.exists(), "checkout dotenv.py was imported despite isolated mode"
    assert f"Exact interpreter: '{managed_python}'" in reason
    assert "Exit codes: [1, 1, 1]" in reason
    assert "ModuleNotFoundError: No module named 'dotenv'" in reason


def test_dependency_stage_reports_persistent_import_failure(tmp_path: Path) -> None:
    result, state, managed_python = _run_dependency_stage(tmp_path, failures=99)
    output = result.stdout + result.stderr
    frame = _stage_frame(result)
    reason = frame["reason"]

    assert result.returncode == 1, output
    assert frame["ok"] is False
    assert isinstance(reason, str)
    assert state.read_text(encoding="utf-8") == "3"
    assert "Reinstalling locked dependencies" not in output
    assert "Baseline imports failed after 3 attempts" in reason
    assert f"Exact interpreter: '{managed_python}'" in reason
    assert "Modules: dotenv, openai, rich, prompt_toolkit" in reason
    assert "Exit codes: [1, 1, 1]" in reason
    assert f"Environment: executable={managed_python}" in reason
    assert "Captured probe output: Traceback" in reason
    assert "synthetic baseline import failure 3" in reason
    assert "\n" in reason
    assert "dependencies are not in the venv" not in reason
