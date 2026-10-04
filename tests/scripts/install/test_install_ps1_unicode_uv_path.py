"""Behavioral regression for #128326: uv-managed Python paths under legacy Windows code pages."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


pytestmark = pytest.mark.platforms("windows")
ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / "scripts" / "install.ps1"
SETUP = ROOT / "setup-hermes.ps1"


FIXTURE_SOURCE = r"""
using System;
using System.IO;
using System.Text;

class UvFixture {
    static string Env(string name) {
        return Environment.GetEnvironmentVariable(name);
    }

    static void Log(string line) {
        File.AppendAllText(Env("HERMES_UTF8_PROBE_LOG"), line + "\n");
    }

    static int Main(string[] args) {
        if (args.Length > 0 && args[0] == "--version") {
            Console.WriteLine("uv 0.12.3 (fixture)");
            return 0;
        }

        if (args.Length > 1 && args[0] == "python" && args[1] == "install") {
            Log("install");
            return 0;
        }

        if (args.Length > 1 && args[0] == "python" && args[1] == "find") {
            Log("find");
            string marker = Env("HERMES_UTF8_PROBE_MARKER");
            if (Env("HERMES_UTF8_PROBE_MISS_ONCE") == "1" && !File.Exists(marker)) {
                File.WriteAllText(marker, "missed");
                Console.Error.WriteLine("fixture: managed interpreter not installed yet");
                return 1;
            }

            byte[] bytes = new UTF8Encoding(false).GetBytes(
                Env("HERMES_UTF8_PROBE_PYTHON") + "\r\n"
            );
            Stream stdout = Console.OpenStandardOutput();
            stdout.Write(bytes, 0, bytes.Length);
            stdout.Flush();
            return 0;
        }

        if (args.Length > 1 && args[0] == "-m" && args[1] == "pm.cli") {
            Log("pm");
            return 0;
        }

        return 23;
    }
}
"""


def _compile_native_fixture(powershell: str, tmp_path: Path, output: Path) -> None:
    source = tmp_path / "uv-fixture.cs"
    source.write_text(FIXTURE_SOURCE, encoding="ascii")
    compile_script = tmp_path / "compile.ps1"
    compile_script.write_text(
        "param([string]$Source, [string]$Output)\n"
        "Add-Type -Path $Source -OutputAssembly $Output -OutputType ConsoleApplication\n",
        encoding="utf-8-sig",
    )
    run = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(compile_script),
            "-Source",
            str(source),
            "-Output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert output.exists()


def _env(tmp_path: Path, fixture: Path, log: Path, miss_once: bool) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        HERMES_UTF8_PROBE_PYTHON=str(fixture),
        HERMES_UTF8_PROBE_LOG=str(log),
        HERMES_UTF8_PROBE_MARKER=str(tmp_path / "first-find-missed"),
        HERMES_UTF8_PROBE_MISS_ONCE="1" if miss_once else "0",
    )
    env.pop("PYTHONPATH", None)
    return env


@pytest.mark.parametrize("miss_once", [False, True])
def test_install_python_deps_executes_utf8_path_without_console_reencoding(tmp_path, miss_once):
    powershell = shutil.which("powershell")
    assert powershell

    # U+0142 reproduces #128326 exactly: UTF-8 C5 82 becomes CP437 mojibake
    # when PowerShell decodes native stdout through the ambient console table.
    unicode_root = tmp_path / "Pawe\u0142 profile"
    unicode_root.mkdir()
    fixture = unicode_root / "python.exe"
    compiled_fixture = tmp_path / "uv-fixture.exe"
    _compile_native_fixture(powershell, tmp_path, compiled_fixture)
    shutil.copy2(compiled_fixture, fixture)

    install_dir = unicode_root / "hermes-agent"
    pm_dir = install_dir / "pm"
    pm_dir.mkdir(parents=True)
    (pm_dir / "lock.json").write_text(
        json.dumps({"packages": {"python": {"version": "3.14.0+fixture"}}}),
        encoding="utf-8",
    )
    home = unicode_root / "home"
    log = tmp_path / "calls.log"
    state = tmp_path / "encoding.txt"
    wrapper = tmp_path / "install-boundary.ps1"
    wrapper.write_text(
        r"""
param([string]$Installer, [string]$InstallDir, [string]$HomeDir, [string]$State)
$ErrorActionPreference = 'Stop'
. $Installer -InstallDir $InstallDir -HermesHome $HomeDir
function Get-Uv { return $env:HERMES_UTF8_PROBE_PYTHON }
[Console]::OutputEncoding = [Text.Encoding]::GetEncoding(437)
$before = [Console]::OutputEncoding.CodePage
$script:BootstrapPython = $null
Stage-PythonDeps
$after = [Console]::OutputEncoding.CodePage
[IO.File]::WriteAllText($State, "$before,$after")
""",
        encoding="utf-8-sig",
    )

    run = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(wrapper),
            "-Installer",
            str(INSTALLER),
            "-InstallDir",
            str(install_dir),
            "-HomeDir",
            str(home),
            "-State",
            str(state),
        ],
        cwd=tmp_path,
        env=_env(tmp_path, fixture, log, miss_once),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    expected = ["find", "install", "find", "pm"] if miss_once else ["find", "pm"]
    assert log.read_text(encoding="utf-8-sig").splitlines() == expected
    assert state.read_text(encoding="utf-8-sig") == "437,437"


def test_setup_hermes_executes_utf8_path_without_console_reencoding(tmp_path):
    powershell = shutil.which("powershell")
    assert powershell

    unicode_root = tmp_path / "Pawe\u0142 setup"
    checkout = unicode_root / "checkout"
    pm_dir = checkout / "pm"
    pm_dir.mkdir(parents=True)
    fixture = unicode_root / "python.exe"
    unicode_root.mkdir(exist_ok=True)
    compiled_fixture = tmp_path / "uv-fixture.exe"
    _compile_native_fixture(powershell, tmp_path, compiled_fixture)
    shutil.copy2(compiled_fixture, fixture)

    lock = {
        "packages": {
            "python": {"version": "3.14.0+fixture"},
            "uv": {
                "version": "fixture",
                "artifacts": {
                    "any": {
                        "url": "https://example.invalid/unused.zip",
                        "sha256": "0" * 64,
                    }
                },
            },
        }
    }
    (pm_dir / "lock.json").write_text(json.dumps(lock), encoding="utf-8")
    setup_copy = checkout / "setup-hermes.ps1"
    setup_copy.write_text(SETUP.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")

    runtime = unicode_root / "runtime"
    for target in ("win32-x64", "win32-arm64"):
        entry = runtime / f"uv-fixture-{target}"
        entry.mkdir(parents=True)
        shutil.copy2(fixture, entry / "uv.exe")

    log = tmp_path / "setup-calls.log"
    state = tmp_path / "setup-encoding.txt"
    home = unicode_root / "home"
    wrapper = tmp_path / "setup-boundary.ps1"
    wrapper.write_text(
        r"""
param([string]$Setup, [string]$State)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::GetEncoding(437)
$before = [Console]::OutputEncoding.CodePage
& $Setup
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$after = [Console]::OutputEncoding.CodePage
[IO.File]::WriteAllText($State, "$before,$after")
""",
        encoding="utf-8-sig",
    )
    env = _env(tmp_path, fixture, log, False)
    env["HERMES_RUNTIME_DIR"] = str(runtime)
    env["HERMES_HOME"] = str(home)

    run = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(wrapper),
            "-Setup",
            str(setup_copy),
            "-State",
            str(state),
        ],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert log.read_text(encoding="utf-8-sig").splitlines() == ["install", "find", "pm"]
    assert state.read_text(encoding="utf-8-sig") == "437,437"
