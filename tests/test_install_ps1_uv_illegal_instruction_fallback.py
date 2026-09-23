"""Windows installer fallback when uv crashes with illegal instruction (#72518)."""

from __future__ import annotations

import ensurepip
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"
POWERSHELL = next(
    (candidate for candidate in ("pwsh", "powershell") if shutil.which(candidate)),
    None,
)


def _compile_uv_stub(output: Path, exit_code: int) -> None:
    """Build an executable stub with a controlled Windows exit status."""
    source = (
        "public static class Program { "
        "public static int Main(string[] args) { "
        f"return unchecked((int)0x{exit_code:08X}); }} }}"
    )
    command = (
        f"$code = '{source}'; "
        f"Add-Type -TypeDefinition $code -Language CSharp "
        f"-OutputAssembly '{output}' -OutputType ConsoleApplication"
    )
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-Command", command],
        env=os.environ | {
            "TEMP": str(output.parent),
            "TMP": str(output.parent),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    os.name != "nt" or POWERSHELL is None,
    reason="requires native Windows and PowerShell",
)
@pytest.mark.parametrize(
    ("uv_exit_code", "powershell_exit_code", "expects_fallback"),
    [
        pytest.param(0xC000001D, -1073741795, True, id="illegal-instruction"),
        pytest.param(1, 1, False, id="ordinary-error"),
    ],
)
def test_dependencies_only_fall_back_for_uv_illegal_instruction(
    tmp_path: Path,
    uv_exit_code: int,
    powershell_exit_code: int,
    expects_fallback: bool,
) -> None:
    install_dir = tmp_path / "hermes-agent"
    hermes_home = tmp_path / "hermes-home"
    managed_bin = hermes_home / "bin"
    install_dir.mkdir()
    managed_bin.mkdir(parents=True)

    # The real installer creates the venv in an earlier stage. Leave pip out so
    # this test also exercises the fallback's ensurepip bootstrap.
    venv_dir = install_dir / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv_dir)],
        check=True,
        capture_output=True,
        text=True,
    )

    # A local editable fixture avoids network access. The modules satisfy the
    # installer's real baseline-import gate after the core-only tier succeeds.
    (install_dir / "setup.py").write_text(
        "from setuptools import setup\n"
        "setup(name='hermes-installer-fixture', version='0.0.0', "
        "py_modules=['dotenv', 'openai', 'rich', 'prompt_toolkit', "
        "'fastapi', 'uvicorn'])\n",
        encoding="utf-8",
    )
    (install_dir / "pyproject.toml").write_text(
        "[build-system]\n"
        "requires = ['setuptools>=40.8.0']\n"
        "build-backend = 'setuptools.build_meta'\n"
        "\n"
        "[project]\n"
        "name = 'hermes-installer-fixture'\n"
        "version = '0.0.0'\n"
        "\n"
        "[project.optional-dependencies]\n"
        "all = []\n"
        "\n"
        "[project.scripts]\n",
        encoding="utf-8",
    )
    for module in ("dotenv", "openai", "rich", "prompt_toolkit", "fastapi", "uvicorn"):
        (install_dir / f"{module}.py").write_text("READY = True\n", encoding="utf-8")
    web_source = install_dir / "hermes_cli" / "web_server.py"
    web_source.parent.mkdir()
    web_source.write_text("READY = True\n", encoding="utf-8")

    # Force the hash-verified uv path first, matching the reported update flow.
    (install_dir / "uv.lock").write_text("", encoding="utf-8")
    _compile_uv_stub(managed_bin / "uv.exe", uv_exit_code)

    env = os.environ | {
        "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_FIND_LINKS": str(Path(ensurepip.__file__).parent / "_bundled"),
        "COMSPEC": r"C:\Windows\System32\cmd.exe",
        "NUMBER_OF_PROCESSORS": "1",
        "OS": "Windows_NT",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "PROCESSOR_ARCHITECTURE": "AMD64",
        "SYSTEMDRIVE": r"C:",
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
        "TEMP": str(tmp_path),
        "TMP": str(tmp_path),
        "WINDIR": r"C:\Windows",
    }
    uv_probe = subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-Command",
            f"& '{managed_bin / 'uv.exe'}'; "
            "Write-Output ('UV_TEST_EXIT=' + $LASTEXITCODE)",
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert f"UV_TEST_EXIT={powershell_exit_code}" in uv_probe.stdout, (
        f"invalid crash fixture: stdout={uv_probe.stdout!r}, "
        f"stderr={uv_probe.stderr!r}"
    )

    result = subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(INSTALL_PS1),
            "-Stage",
            "dependencies",
            "-NonInteractive",
            "-InstallDir",
            str(install_dir),
            "-HermesHome",
            str(hermes_home),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output = result.stdout + result.stderr
    if not expects_fallback:
        assert result.returncode != 0, output
        assert "pip fallback" not in output
        return

    assert result.returncode == 0, output
    assert "EXCEPTION_ILLEGAL_INSTRUCTION" in output
    assert "pip fallback" in output

    venv_python = venv_dir / "Scripts" / "python.exe"
    probe = subprocess.run(
        [
            str(venv_python),
            "-c",
            "import dotenv, openai, rich, prompt_toolkit; "
            "assert all(m.READY for m in (dotenv, openai, rich, prompt_toolkit))",
        ],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
