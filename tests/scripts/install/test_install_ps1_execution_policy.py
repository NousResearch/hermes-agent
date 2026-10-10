"""The iex'd installer must reach its runtime helper under the Restricted policy.

`iex (irm .../install.ps1)` is a string, so execution policy never checks it.
Loading a helper .ps1 from disk is a file load, which the default Restricted
policy (Windows Sandbox, fresh machines) refuses: "runtime.ps1 cannot be
loaded because running scripts is disabled on this system". Past the load,
the resolved runtime command must actually run (a `$command` local once
collided with Invoke-Native's `$Command` parameter and recursed forever).
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import pytest

from hermes_cli._launchers import runtime_command

pytestmark = pytest.mark.platforms("windows")
INSTALLER = Path(__file__).resolve().parents[3] / "scripts" / "install.ps1"


def test_installed_hermes_runs_under_restricted_policy(tmp_path):
    powershell = shutil.which("powershell")
    assert powershell
    helper = tmp_path / "install" / "scripts" / "desktop-update" / "runtime.ps1"
    helper.parent.mkdir(parents=True)
    helper.write_text("function Get-HermesRuntimeCommand([string]$InstallRoot) {\n"
                      "    @('cmd.exe', '/c', 'echo', 'runtime-from', $InstallRoot)\n}\n",
                      encoding="utf-8")
    install_dir = str(helper.parents[2]).replace("'", "''")
    # Definitions load under Bypass (the iex'd text is never policy-checked);
    # the session then drops to Restricted, the policy a fresh machine has.
    command = (f". '{str(INSTALLER).replace(chr(39), chr(39) * 2)}'; "
               "Set-ExecutionPolicy -Scope Process Restricted -Force; "
               f"$InstallDir = '{install_dir}'; "
               "Invoke-InstalledHermes @('setup')")
    result = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                             "-Command", command],
                            cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=False)
    output = result.stdout + result.stderr
    assert "running scripts is disabled" not in output, output
    assert f"runtime-from {helper.parents[2]} setup" in result.stdout, output


def test_installer_runs_runtime_command_from_apostrophe_account_path(tmp_path):
    """A Windows account apostrophe must survive launcher JSON -> PowerShell -> Python -c."""
    powershell = shutil.which("powershell")
    assert powershell
    install_root = tmp_path / "O'Brien" / "hermes-agent"
    helper = install_root / "scripts" / "desktop-update" / "runtime.ps1"
    helper.parent.mkdir(parents=True)
    (install_root / "hermes_bootstrap.py").write_text("", encoding="utf-8")
    helper.write_text(
        "function Get-HermesRuntimeCommand([string]$InstallRoot) {\n"
        "    $items = $env:HERMES_TEST_RUNTIME_COMMAND | ConvertFrom-Json\n"
        "    foreach ($item in $items) { Write-Output $item }\n"
        "}\n",
        encoding="utf-8",
    )
    selected = runtime_command(
        install_root, code="print('setup-runtime-ok')", python=sys.executable,
        home=install_root.parent / ".hermes",
    )
    env = os.environ.copy()
    env["HERMES_TEST_RUNTIME_COMMAND"] = json.dumps(selected)
    quoted_installer = str(INSTALLER).replace("'", "''")
    quoted_root = str(install_root).replace("'", "''")
    command = (f". '{quoted_installer}'; "
               f"$InstallDir = '{quoted_root}'; "
               "Invoke-InstalledHermes @('setup')")
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=60,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "setup-runtime-ok" in result.stdout, output
