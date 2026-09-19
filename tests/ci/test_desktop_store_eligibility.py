"""Execute the packaging workflow branches; nonstable never builds Store identities."""
import json
import os
import shlex
import shutil
import subprocess
import sys

import pytest

from tests.ci.test_desktop_release_tag_admission import _BASH, _child_env, _workflow


@pytest.mark.parametrize('tag,commit,store', [
    ('v0.28.0', '', True),
    ('v0.28.0-canary.20260818', '', False),
    ('', 'a' * 40, False),
])
def test_bundle_only_requests_store_for_stable(tmp_path, tag, commit, store):
    helper = tmp_path / 'bin'
    helper.mkdir()
    log = tmp_path / 'calls.jsonl'
    recorder = helper / 'record.py'
    recorder.write_text(
        'import json,os,sys\n'
        'with open(os.environ["CALL_LOG"],"a",encoding="utf-8") as file: file.write(json.dumps(sys.argv[1:])+"\\n")\n',
        encoding='utf-8',
    )
    for tool in ['python', 'node']:
        wrapper = helper / tool
        wrapper.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(recorder))} "$@"\n', encoding='utf-8')
        wrapper.chmod(0o755)
    jobs = _workflow()['jobs']
    env = _child_env(HERMES_PAYLOAD_TAG=tag, HERMES_BUILD_COMMIT=commit,
                     HERMES_PAYLOAD_VERSION='0.28.0', RELEASE_PHASE='candidate' if store else '', CALL_LOG=str(log))
    env['PATH'] = str(helper) + os.pathsep + env['PATH']
    job = jobs['assemble-win32-bundle']
    script = next(step['run'] for step in job['steps']
                  if step.get('name') == 'Assemble the signed MSIX bundle without publishing')
    result = subprocess.run([_BASH, '-e', '-o', 'pipefail', '-c', script], env=env, cwd=tmp_path,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in log.read_text(encoding='utf-8-sig').splitlines()]
    assert calls[0][:1] == ['scripts/stage-msixbundle.mjs']
    assert calls[0][calls[0].index('--variant') + 1] == 'bundled'
    assert ('--no-upload' in calls[0]) is bool(commit)
    assert ('--candidate' in calls[0]) is store
    assert any('scripts/bundle-store-msixbundle.mjs' in call for call in calls) is store
    assert not any('publish-appinstaller' in call for call in calls)


@pytest.mark.platforms('windows')
@pytest.mark.parametrize('tag,commit,store', [
    ('v0.28.0', '', True), ('v0.28.0-canary.20260818', '', False), ('', 'a' * 40, False),
])
def test_native_windows_build_selects_store_only_for_stable(tmp_path, tag, commit, store):
    jobs = _workflow()['jobs']
    selected = 'build-win32-commit' if commit else 'build-win32-release'
    script = next(step['run'] for step in jobs[selected]['steps'] if step.get('name') == 'Build and package')
    wrapper = tmp_path / 'run.ps1'
    # Only the build boundary is substituted; execute the actual PowerShell
    # branch and native-exit handling instead of interpreting it as Bash.
    wrapper.write_text('function python { $args -join " " | Add-Content $env:CALL_LOG -Encoding UTF8; $global:LASTEXITCODE = 0 }\n' + script, encoding='utf-8')
    log = tmp_path / 'calls.txt'
    powershell = shutil.which('powershell')
    assert powershell
    result = subprocess.run([powershell, '-NoProfile', '-File', str(wrapper)],
                            env=_child_env(HERMES_PAYLOAD_TAG=tag, HERMES_BUILD_COMMIT=commit,
                                           RUNNER_TEMP=str(tmp_path), CALL_LOG=str(log)),
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = log.read_text(encoding='utf-8-sig')
    assert '--variant bundled' in calls
    assert ('--variant store' in calls) is store
