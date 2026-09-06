"""Execute the SQLite bridge with real Python and native Windows directory swaps."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.windows_only


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


@pytest.mark.parametrize("host", ["powershell.exe", "pwsh.exe"])
def test_real_repair_host_is_external_and_probe_rejects_redirected_managed_root(
    tmp_path, host
):
    executable = shutil.which(host)
    assert executable, host
    root = tmp_path / "owned install"
    base = root / ".hermes-runtime/python/fixture"
    shutil.copytree(
        sys.base_prefix,
        base,
        ignore=shutil.ignore_patterns(
            "site-packages", "__pycache__", "include", "libs", "tcl"
        ),
    )
    completed = subprocess.run(
        [
            str(base / "python.exe"),
            "-I",
            "-m",
            "venv",
            "--without-pip",
            str(root / "venv"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (root / "venv/Scripts/python.exe").is_file(), list((root / "venv").iterdir())
    package = root / "hermes_cli"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "main.py").write_text("", encoding="utf-8")
    shutil.copyfile(
        REPO / "hermes_cli/sqlite_runtime.py", package / "sqlite_runtime.py"
    )
    # The native bridge and filesystem replacement are real. Provisioning a
    # network runtime is the one substituted edge; this fixture never downloads.
    (package / "managed_uv.py").write_text(
        "import json, shutil, sys\nfrom pathlib import Path\nfrom types import SimpleNamespace\n"
        "def repair_vulnerable_runtime(uv, *, project_root):\n"
        "    live = project_root / 'venv'\n"
        "    assert not Path(sys.executable).resolve().is_relative_to(live.resolve())\n"
        "    old = project_root / 'replacement-held-for-test'\n"
        "    live.rename(old)\n"
        "    shutil.copytree(old, live)\n"
        "    (project_root / 'repair.json').write_text(json.dumps({'host': sys.executable, 'replaced': True}))\n"
        "    return SimpleNamespace(status='repaired', detail='native fixture')\n",
        encoding="utf-8",
    )
    fixture = tmp_path / "native-bridge.ps1"
    fixture.write_text(
        f". {literal(REPO / 'scripts/install.ps1')} -InstallDir {literal(root)} -HermesHome {literal(tmp_path / 'home')}\n"
        "$UvCmd = 'unused-native-fixture'\n"
        "Write-Host ('Native fixture install=' + $InstallDir + '; python=' + (Test-Path -LiteralPath (Join-Path $InstallDir 'venv\\Scripts\\python.exe')))\n"
        "$runtime = Get-ManagedVenvRuntime\n"
        "if (-not $runtime) { throw 'Owned runtime was rejected' }\n"
        "Repair-ManagedRuntime\n"
        "if ($env:UV_PYTHON -ne (Join-Path $InstallDir 'venv\\Scripts\\python.exe')) { throw 'Live Python was not repinned' }\n",
        encoding="utf-8-sig",
    )

    def run():
        return subprocess.run(
            [
                executable,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(fixture),
            ],
            env={**os.environ, "TEMP": str(tmp_path), "TMP": str(tmp_path)},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

    completed = run()
    assert completed.returncode == 0, completed.stdout + completed.stderr
    evidence = json.loads((root / "repair.json").read_text())
    assert Path(evidence["host"]).resolve() == (base / "python.exe").resolve()
    assert evidence["replaced"]
    # A junction must not turn another application's runtime into owned Python.
    managed = root / ".hermes-runtime/python"
    outside = tmp_path / "outside-runtime"
    managed.rename(outside)
    fixture.write_text(
        f"New-Item -ItemType Junction -Path {literal(managed)} -Target {literal(outside)} | Out-Null\n"
        f". {literal(REPO / 'scripts/install.ps1')} -InstallDir {literal(root)} -HermesHome {literal(tmp_path / 'home')}\n"
        "if (Get-ManagedVenvRuntime) { throw 'Redirected managed root was admitted' }\n",
        encoding="utf-8-sig",
    )
    completed = run()
    assert completed.returncode == 0, completed.stdout + completed.stderr
