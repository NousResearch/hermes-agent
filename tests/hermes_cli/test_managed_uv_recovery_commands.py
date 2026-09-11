"""Validate the printed recovery commands with Windows' actual shell parser."""

import json
import shutil
import subprocess

import pytest

from hermes_cli import managed_uv
from hermes_cli.sqlite_runtime import SQLiteRuntimeInfo


@pytest.mark.windows_only
@pytest.mark.parametrize("directory", ["Hermes checkout", "Hermes' $data $(Write-Output injected)"])
def test_recovery_commands_preserve_paths_and_share_updater_python(tmp_path, monkeypatch, capsys, directory):
    root = tmp_path / directory
    uv_bin = str(root / "tools" / "uv.exe")
    current = SQLiteRuntimeInfo(
        executable=root / "venv" / "Scripts" / "python.exe", base_prefix=root,
        python_version=(3, 11, 15), sqlite_version=(3, 50, 4),
        sqlite_version_string="3.50.4", sqlite_source_id="test",
    )
    # Isolate holder discovery: the subject is the recovery recipe it selects.
    monkeypatch.setattr(managed_uv, "_windows_runtime_holders", lambda: (False, ""))
    monkeypatch.setattr(managed_uv, "_windows_runtime_self_lock", lambda live: (True, "self-locked"))
    result = managed_uv._repair_windows_preflight(root, root / "venv", current, uv_bin)
    assert result.status == "skipped"
    recipe = "\n".join(line.strip() for line in capsys.readouterr().out.splitlines() if line.startswith("      "))
    recipe_file = tmp_path / "recovery.ps1"
    recipe_file.write_text(recipe, encoding="utf-8-sig")
    parser = tmp_path / "parse.ps1"
    # Parse, never execute: no installs, updates or deletion run in this test.
    parser.write_text(r'''
param([string]$Recipe)
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Recipe, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
$commands = @($ast.FindAll({param($node) $node -is [System.Management.Automation.Language.CommandAst]}, $true) |
    ForEach-Object {
        [pscustomobject]@{parts = @($_.CommandElements | ForEach-Object {
            if ($_ -is [System.Management.Automation.Language.StringConstantExpressionAst]) { $_.Value }
            else { $_.Extent.Text }
        })}
    })
ConvertTo-Json -InputObject $commands -Depth 5 -Compress
''', encoding="utf-8-sig")
    shell = shutil.which("pwsh") or shutil.which("powershell")
    assert shell, "Windows recovery must be validated by PowerShell"
    parsed = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(parser), str(recipe_file)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    commands = [entry["parts"] for entry in json.loads(parsed.stdout)]
    uv_calls = [parts for parts in commands if parts[0] == uv_bin]
    venv, install = uv_calls
    assert venv[1] == "venv"
    assert install[1:3] == ["pip", "install"]
    assert install[install.index("--editable") + 1] == str(root)
    update = next(parts for parts in commands if parts[1:] == ["-m", "hermes_cli.main", "update"])
    assert install[install.index("--python") + 1] == update[0]
    assert commands.index(venv) < commands.index(install) < commands.index(update)
    cd = next(parts for parts in commands if parts[0] == "Set-Location")
    assert cd[cd.index("-LiteralPath") + 1] == str(root)
    cleanup = next(parts for parts in commands if parts[0] == "Remove-Item")
    assert cleanup[cleanup.index("-LiteralPath") + 1] == venv[-1]
    assert commands.index(cleanup) > commands.index(update)
    assert all(parts[0] != "Write-Output" for parts in commands)
