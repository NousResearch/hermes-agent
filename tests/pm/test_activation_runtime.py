"""Activation syncs through setup before selecting or changing the caller's env."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import textwrap

import pytest

from tests.pm.test_activate_scripts import (
    _bash, _bash_env, _isolated_checkout, _posix, _powershell, _spawnable_python,
)


def _sync_checkout(tmp_path: Path):
    root = _isolated_checkout(tmp_path)
    env = _bash_env(tmp_path / "store")
    python = _spawnable_python()
    # The setup seam publishes a bootstrap and a selected generation. PM's
    # freshness algorithms have their own tests; here a warm setup call is a
    # no-op and activation must still call it, rather than cache its own answer.
    (root / "sync.py").write_text(textwrap.dedent('''\
        import json, os, pathlib, shutil, sys
        from hermes_cli.runtime_paths import runtime_facts_path, site_packages
        root = pathlib.Path(__file__).parent
        record = {"argv": sys.argv[1:], "python_env": {
            key: os.environ.get(key) for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")}}
        with (root / "calls.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\\n")
        assert all(value is None for value in record["python_env"].values()), record
        assert sys.argv[1:] == ["runtime-only"], record
        print("setup progress")
        if (root / "fail").exists():
            sys.exit(42)
        bootstrap = root / ".venv"
        if not bootstrap.exists():
            shutil.copytree(root / "prepared-bootstrap", bootstrap)
        generation = (root / "input").read_text(encoding="utf-8").strip()
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
    (root / "setup-hermes.sh").write_text(
        'test "$#" = 1 && test "$1" = --runtime-only || exit 2\n'
        f'cd {shlex.quote(str(root))} || exit 3\n'
        f'exec {shlex.quote(str(python))} sync.py runtime-only\n', encoding="utf-8",
    )
    (root / "setup-hermes.ps1").write_text(
        "param([switch]$RuntimeOnly)\n"
        "if (-not $RuntimeOnly) { exit 2 }\n"
        "$ErrorActionPreference = 'Stop'\n"
        "$record = @{pid=$PID; executable=(Get-Process -Id $PID).Path; "
        "argv=[Environment]::GetCommandLineArgs()}\n"
        "$record | ConvertTo-Json -Compress | Add-Content -LiteralPath \"$PSScriptRoot\\ps-calls.jsonl\"\n"
        "Set-Location -LiteralPath $PSScriptRoot\n"
        f"& '{python}' sync.py runtime-only\nexit $LASTEXITCODE\n", encoding="utf-8",
    )
    return root, env


def _assert_syncs(root: Path):
    calls = [json.loads(line) for line in (root / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert calls == [{"argv": ["runtime-only"], "python_env": {
        "PYTHONHOME": None, "PYTHONPATH": None, "VIRTUAL_ENV": None,
    }}] * 3
    assert (root / "builds").read_text(encoding="utf-8").splitlines() == ["first", "second"]


@pytest.mark.platforms("posix")
def test_bash_cold_sync_changed_input_and_warm_noop(tmp_path):
    root, env = _sync_checkout(tmp_path)
    assert not (root / ".venv").exists()
    script = f'''
        set -e
        export PYTHONPATH=caller-original VIRTUAL_ENV=caller-venv
        original_path="$PATH"
        source "{_posix(root / 'activate')}"
        printf '%s\\n' "$PYTHONPATH"
        printf second > "{_posix(root / 'input')}"
        source "{_posix(root / 'activate')}"
        printf '%s\\n' "$PYTHONPATH"
        source "{_posix(root / 'activate')}"
        printf '%s\\n' "$PYTHONPATH"
        test "$PWD" = "{_posix(tmp_path)}"
        deactivate
        test "$PATH" = "$original_path"
        test "$PYTHONPATH" = caller-original
        test "$VIRTUAL_ENV" = caller-venv
    '''
    run = subprocess.run([_bash(), "-c", script], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=40)
    assert run.returncode == 0, run.stdout + run.stderr
    first, second, warm = run.stdout.splitlines()
    assert first.startswith(str(root) + os.pathsep) and "/first/venv/" in first
    assert second.startswith(str(root) + os.pathsep) and "/second/venv/" in second
    assert warm == second
    assert run.stderr.count("setup progress") == 3
    _assert_syncs(root)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("already_active", [False, True])
def test_bash_setup_failure_preserves_caller(tmp_path, already_active):
    root, env = _sync_checkout(tmp_path)
    activate = shlex.quote(str(root / "activate"))
    script = f'''
        set -e
        {f'source {activate}' if already_active else ':'}
        export PYTHONHOME=caller-home PYTHONPATH=caller-path VIRTUAL_ENV=caller-venv
        before_env=$(export -p)
        before_function=$(declare -f deactivate || :)
        before_active=${{__HERMES_ACTIVATED-unset}}
        before_cwd="$PWD"
        touch {shlex.quote(str(root / 'fail'))}
        if source {activate}; then exit 9; fi
        test "$(export -p)" = "$before_env"
        test "$(declare -f deactivate || :)" = "$before_function"
        test "${{__HERMES_ACTIVATED-unset}}" = "$before_active"
        test "$PWD" = "$before_cwd"
        {"deactivate" if already_active else ':'}
        printf preserved
    '''
    run = subprocess.run([_bash(), "-c", script], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=40)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout == "preserved"
    calls = (root / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(calls) == (2 if already_active else 1)
    assert json.loads(calls[-1])["python_env"] == dict.fromkeys(("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"))


@pytest.mark.platforms("windows")
def test_powershell_cold_sync_changed_input_warm_and_failure(tmp_path):
    # Native child-process and venv semantics cannot be reproduced by faking win32.
    root, env = _sync_checkout(tmp_path)
    ps = _powershell()
    assert ps, "native Windows test requires PowerShell"
    script = tmp_path / "run.ps1"
    script.write_text(f'''
        $ErrorActionPreference = 'Stop'
        $env:PYTHONPATH = 'caller-original'
        $env:PYTHONHOME = 'caller-home'
        $env:VIRTUAL_ENV = 'caller-venv'
        $originalPath = $env:PATH
        $originalCwd = (Get-Location).Path
        . '{root / 'activate.ps1'}'
        Write-Output ('selection=' + $env:PYTHONPATH)
        Set-Content -LiteralPath '{root / 'input'}' -Value second
        . '{root / 'activate.ps1'}'
        Write-Output ('selection=' + $env:PYTHONPATH)
        . '{root / 'activate.ps1'}'
        Write-Output ('selection=' + $env:PYTHONPATH)
        Write-Output ('parent=' + $PID)
        $beforeEnv = Get-ChildItem env: | Sort-Object Name | ConvertTo-Json -Compress
        $beforeFunction = (Get-Item function:deactivate).Definition
        Set-Content -LiteralPath '{root / 'fail'}' -Value fail
        $failed = $false
        try {{ . '{root / 'activate.ps1'}' }} catch {{ $failed = $true }}
        if (-not $failed) {{ throw 'setup failure accepted' }}
        if ((Get-ChildItem env: | Sort-Object Name | ConvertTo-Json -Compress) -ne $beforeEnv) {{ throw 'env changed on failure' }}
        if ((Get-Item function:deactivate).Definition -ne $beforeFunction) {{ throw 'deactivation lost' }}
        if ((Get-Location).Path -ne $originalCwd) {{ throw 'cwd changed' }}
        deactivate
        if ($env:PATH -ne $originalPath -or $env:PYTHONPATH -ne 'caller-original' -or
            $env:PYTHONHOME -ne 'caller-home' -or $env:VIRTUAL_ENV -ne 'caller-venv') {{ throw 'restore failed' }}
    ''', encoding="utf-8")
    run = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                         cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    selected = [line.removeprefix("selection=") for line in run.stdout.splitlines() if line.startswith("selection=")]
    assert len(selected) == 3
    assert "\\first\\venv\\" in selected[0]
    assert "\\second\\venv\\" in selected[1]
    assert selected[1] == selected[2]
    assert all(value.startswith(str(root) + os.pathsep) for value in selected)
    calls = [json.loads(line) for line in (root / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(calls) == 4
    assert all(call["python_env"] == dict.fromkeys(("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")) for call in calls)
    assert (root / "builds").read_text(encoding="utf-8").splitlines() == ["first", "second"]
    parent = next(line.removeprefix("parent=") for line in run.stdout.splitlines() if line.startswith("parent="))
    ps_calls = [json.loads(line) for line in (root / "ps-calls.jsonl").read_text(encoding="utf-8-sig").splitlines()]
    assert len(ps_calls) == 4
    for call in ps_calls:
        assert str(call["pid"]) != parent
        assert Path(call["executable"]).samefile(ps)
        assert [arg.lower() for arg in call["argv"][1:]] == [
            "-noprofile", "-noninteractive", "-executionpolicy", "bypass", "-file",
            str(root / "setup-hermes.ps1").lower(), "-runtimeonly",
        ]
