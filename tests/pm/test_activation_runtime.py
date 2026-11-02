"""Activation syncs through setup before selecting or changing the caller's env."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import textwrap

import pytest

from tests.pm.activation_support import bash, fake_store, posix, powershell, sync_checkout

# Spawns children with a home it builds itself; the parent's must stay real.
pytestmark = pytest.mark.real_machine_home


def _assert_syncs(root: Path):
    calls = [json.loads(line) for line in (root / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert calls == [{"argv": ["runtime-only", "--trust-recorded", "--test-environment"], "python_env": {
        "PYTHONHOME": None, "PYTHONPATH": None, "VIRTUAL_ENV": None,
    }}] * 3
    assert (root / "builds").read_text(encoding="utf-8").splitlines() == ["first", "second"]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("canary", [None, "caller-canary"])
def test_bash_cold_sync_changed_input_and_warm_noop(tmp_path, canary):
    root, env = sync_checkout(tmp_path)
    assert not (root / ".venv").exists()
    if canary is not None:
        env["HERMES_PM_ACTIVATE_CANARY"] = canary
    from tests.pm.test_activate_scripts import fake_store
    fake_store(tmp_path)
    script = f'''
        set -e
        export PYTHONPATH=caller-original VIRTUAL_ENV=caller-venv
        original_path="$PATH"
        source "{posix(root / 'activate')}"
        printf '%s\\n' "$PYTHONPATH"
        test "$HERMES_PM_ACTIVATE_CANARY" = env-ok
        printf second > "{posix(root / 'input')}"
        source "{posix(root / 'activate')}"
        printf '%s\\n' "$PYTHONPATH"
        source "{posix(root / 'activate')}"
        printf '%s\\n' "$PYTHONPATH"
        test "$PWD" = "{posix(tmp_path)}"
        deactivate
        test "$PATH" = "$original_path"
        test "$PYTHONPATH" = caller-original
        test "$VIRTUAL_ENV" = caller-venv
        {('test "$HERMES_PM_ACTIVATE_CANARY" = caller-canary' if canary else 'test -z "${HERMES_PM_ACTIVATE_CANARY+set}"')}
        test -z "${{__HERMES_ACTIVATED+set}}"
        ! declare -F deactivate >/dev/null
    '''
    run = subprocess.run([bash(), "-c", script], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=40, check=False)
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
    root, env = sync_checkout(tmp_path)
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
    run = subprocess.run([bash(), "-c", script], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=40, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout == "preserved"
    calls = (root / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(calls) == (2 if already_active else 1)
    assert json.loads(calls[-1])["python_env"] == dict.fromkeys(("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"))


@pytest.mark.platforms("windows")
def test_powershell_cold_sync_changed_input_warm_and_failure(tmp_path):
    # Native child-process and venv semantics cannot be reproduced by faking win32.
    root, env = sync_checkout(tmp_path)
    ps = powershell()
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
                         cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90, check=False)
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
