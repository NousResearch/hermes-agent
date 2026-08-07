"""Behavioral PowerShell installer coverage for managed runtime safety.

The tests execute ``install.ps1 -Stage`` through PowerShell.  They use a
contract-only disposable managed runtime and external uv/process fixtures;
PowerShell functions are never parsed, extracted, or replaced.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"
POWERSHELL = os.environ.get("POWERSHELL_UNDER_TEST") or next(
    (candidate for candidate in ("pwsh", "powershell") if shutil.which(candidate)),
    None,
)

pytestmark = [
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(
        os.name == "nt" or POWERSHELL is None,
        reason="behavioral install.ps1 coverage needs PowerShell on a POSIX host",
    ),
]


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _uv_stub(home: Path, *, python: Path, venv_rc: int = 0) -> Path:
    uv = home / "bin" / "uv.exe"
    _write_executable(
        uv,
        "#!/bin/bash\n"
        'printf \'%s\\n\' "$*" >> "$HERMES_TEST_UV_LOG"\n'
        'case "${1:-}" in\n'
        "  --version) echo 'uv 0.test'; exit 0 ;;\n"
        "  python)\n"
        '    case "${2:-}" in\n'
        f"      find) printf '%s\\n' {str(python)!r}; exit 0 ;;\n"
        "      install) exit 0 ;;\n"
        "    esac ;;\n"
        f"  venv) exit {venv_rc} ;;\n"
        "  sync) exit 0 ;;\n"
        "  pip) exit 0 ;;\n"
        "esac\n"
        "exit 0\n",
    )
    return uv


def _run_stage(
    root: Path,
    home: Path,
    stage: str,
    *,
    holder_probe: str = "skip",
    **extra_env: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ | {
        "OS": "HERMES_TEST" if holder_probe == "skip" else "Windows_NT",
        **extra_env,
    }
    command = [str(POWERSHELL), "-NoProfile", "-NonInteractive"]
    if holder_probe == "skip":
        command += ["-File", str(INSTALL_PS1)]
    else:
        host = root.parent / f"host-{holder_probe}.ps1"
        cim = (
            "throw 'fixture holder enumeration failed'"
            if holder_probe == "fail"
            else "return @()"
        )
        host.write_text(
            "param($Installer,$StageName,$Root,$HomeDir)\n"
            "function global:schtasks { return @() }\n"
            "function global:taskkill { return }\n"
            "function global:Start-Sleep { return }\n"
            f"function global:Get-CimInstance {{ {cim} }}\n"
            "& $Installer -Stage $StageName -Json -NonInteractive "
            "-InstallDir $Root -HermesHome $HomeDir\n"
            "exit $LASTEXITCODE\n",
            encoding="utf-8",
        )
        command += [
            "-File",
            str(host),
            "-Installer",
            str(INSTALL_PS1),
            "-StageName",
            stage,
            "-Root",
            str(root),
            "-HomeDir",
            str(home),
        ]
        return subprocess.run(
            command,
            cwd=root.parent,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    command += [
        "-Stage",
        stage,
        "-Json",
        "-NonInteractive",
        "-InstallDir",
        str(root),
        "-HermesHome",
        str(home),
    ]
    return subprocess.run(
        command,
        cwd=root.parent,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _frame(proc: subprocess.CompletedProcess[str]) -> dict[str, object]:
    frames: list[dict[str, object]] = []
    for line in proc.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("stage"):
            frames.append(value)
    assert len(frames) == 1, (proc.stdout, proc.stderr)
    return frames[0]


def _compatible_venv_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "checkout"
    python = root / "venv" / "Scripts" / "python.exe"
    _write_executable(
        python,
        "#!/bin/bash\n"
        "if [ \"${1:-}\" = '--version' ]; then "
        "echo 'Python 3.11.99'; else echo '3.11'; fi\n",
    )
    sentinel = root / "venv" / "keep-me"
    sentinel.write_text("preserved", encoding="utf-8")
    home = tmp_path / "home"
    uv_log = tmp_path / "uv.log"
    _uv_stub(home, python=python, venv_rc=97)
    return root, home, uv_log, sentinel


def test_venv_stage_preserves_compatible_runtime(tmp_path: Path) -> None:
    root, home, uv_log, sentinel = _compatible_venv_fixture(tmp_path)

    proc = _run_stage(
        root,
        home,
        "venv",
        holder_probe="empty",
        HERMES_TEST_UV_LOG=str(uv_log),
    )

    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert _frame(proc)["ok"] is True
    assert sentinel.read_text(encoding="utf-8") == "preserved"
    assert not any(
        line.startswith("venv ")
        for line in uv_log.read_text(encoding="utf-8").splitlines()
    )


def test_venv_stage_fails_closed_when_holder_sweep_is_unverifiable(
    tmp_path: Path,
) -> None:
    root, home, uv_log, sentinel = _compatible_venv_fixture(tmp_path)

    proc = _run_stage(
        root,
        home,
        "venv",
        holder_probe="fail",
        HERMES_TEST_UV_LOG=str(uv_log),
    )

    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    frame = _frame(proc)
    assert frame["ok"] is False
    assert "Cannot safely preserve virtual environment" in str(frame["reason"])
    assert sentinel.read_text(encoding="utf-8") == "preserved"
    assert not any(
        line.startswith("venv ")
        for line in uv_log.read_text(encoding="utf-8").splitlines()
    )


def _contract_runtime(root: Path) -> Path:
    builder = venv.EnvBuilder(with_pip=False, symlinks=True)
    builder.create(root / "venv")
    python = root / "venv" / "bin" / "python"
    windows_python = root / "venv" / "Scripts" / "python.exe"
    windows_python.parent.mkdir(parents=True, exist_ok=True)
    windows_python.symlink_to(Path("..") / "bin" / "python")

    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = root / "venv" / "lib" / version / "site-packages"
    package = site / "hermes_cli"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "main.py").write_text(
        "def _detect_venv_python_processes():\n    return []\n", encoding="utf-8"
    )
    (package / "managed_uv.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "class Result:\n"
        "    def __init__(self, status):\n"
        "        self.status = status\n"
        "        self.detail = 'fixture-' + status\n"
        "def repair_vulnerable_runtime(uv_bin, *, project_root):\n"
        "    status = os.environ['HERMES_TEST_RUNTIME_STATUS']\n"
        "    Path(os.environ['HERMES_TEST_REPAIR_LOG']).write_text(json.dumps({\n"
        "        'status': status, 'uv_bin': str(uv_bin), "
        "'project_root': str(project_root)\n"
        "    }), encoding='utf-8')\n"
        "    return Result(status)\n",
        encoding="utf-8",
    )
    for module in (
        "dotenv",
        "fastapi",
        "openai",
        "prompt_toolkit",
        "rich",
        "uvicorn",
    ):
        (site / f"{module}.py").write_text("", encoding="utf-8")
    return windows_python


@pytest.mark.parametrize(
    ("status", "expected_rc"), [("safe", 0), ("repaired", 0), ("failed", 1)]
)
def test_dependencies_stage_maps_runtime_repair_contract(
    tmp_path: Path, status: str, expected_rc: int
) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0'\n"
        "[project.optional-dependencies]\nall=[]\n",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("fixture", encoding="utf-8")
    source_package = root / "hermes_cli"
    source_package.mkdir()
    (source_package / "web_server.py").write_text("pass\n", encoding="utf-8")
    # install.ps1 deliberately uses a Windows-native backslash argument; Unix
    # PowerShell passes that spelling literally to Python.
    Path(str(root) + "\\hermes_cli\\web_server.py").write_text(
        "pass\n", encoding="utf-8"
    )
    event_log = tmp_path / "repair.json"
    python = _contract_runtime(root)
    sentinel = root / "venv" / "keep-me"
    sentinel.write_text("preserved", encoding="utf-8")
    home = tmp_path / "home"
    uv_log = tmp_path / "uv.log"
    uv = _uv_stub(home, python=python)

    proc = _run_stage(
        root,
        home,
        "dependencies",
        HERMES_TEST_UV_LOG=str(uv_log),
        HERMES_TEST_RUNTIME_STATUS=status,
        HERMES_TEST_REPAIR_LOG=str(event_log),
    )

    assert proc.returncode == expected_rc, (proc.stdout, proc.stderr)
    frame = _frame(proc)
    assert frame["ok"] is (expected_rc == 0)
    event = json.loads(event_log.read_text(encoding="utf-8"))
    assert event == {
        "status": status,
        "uv_bin": str(uv),
        "project_root": str(root),
    }
    assert sentinel.read_text(encoding="utf-8") == "preserved"
    if expected_rc:
        assert frame["reason"] == "Managed Python/SQLite runtime is not safe"
