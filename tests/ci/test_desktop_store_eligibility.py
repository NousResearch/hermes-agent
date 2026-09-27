"""Execute the packaging workflow branches; nonstable never builds Store identities."""
import json
import os
import shlex
import shutil
import subprocess
import sys

import pytest

from tests.ci.desktop_release_roles import native_builds, universal_assembler
from tests.ci.test_desktop_release_tag_admission import _BASH, _child_env, _workflow


@pytest.mark.parametrize('tag,commit,store', [
    ('v0.28.0', '', True),
    ('v0.28.0+canary.20260818T000000Z', '', False),
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
    job = jobs[universal_assembler(jobs)]
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
def test_native_windows_build_selects_store_only_for_stable(tmp_path):
    jobs = _workflow()['jobs']
    legs = native_builds(jobs)
    scripts = {
        kind: next(step['run'] for step in jobs[legs[('win32-x64', kind)]]['steps']
                   if step.get('name') == 'Build and package')
        for kind in ('release', 'commit')
    }
    cases = [
        ('v0.28.0', '', True, 'release'),
        ('v0.28.0+canary.20260818T000000Z', '', False, 'release'),
        ('', 'a' * 40, False, 'commit'),
    ]
    wrapper = tmp_path / 'run.ps1'
    # Substitute only the build boundary, then execute both workflow scripts
    # as real PowerShell in one session. Cold starts contend heavily in the
    # native OS lane; charging one to every parameter made startup latency part
    # of the assertion.
    body = [
        'function python { $args -join " " | Add-Content $env:CALL_LOG -Encoding UTF8; $global:LASTEXITCODE = 0 }',
        f'$releaseBuild = {{\n{scripts["release"]}\n}}',
        f'$commitBuild = {{\n{scripts["commit"]}\n}}',
    ]
    for index, (tag, commit, _, kind) in enumerate(cases):
        body.extend([
            f'$env:HERMES_PAYLOAD_TAG = \'{tag}\'',
            f'$env:HERMES_BUILD_COMMIT = \'{commit}\'',
            f'$env:CALL_LOG = Join-Path $env:RUNNER_TEMP \'calls-{index}.txt\'',
            f'& ${kind}Build',
        ])
    wrapper.write_text('\n'.join(body) + '\n', encoding='utf-8')
    powershell = shutil.which('powershell')
    assert powershell
    # The canonical per-file runner remains the deadlock guard; a nested
    # startup deadline only measures scheduler contention on the Windows host.
    result = subprocess.run(
        [powershell, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', str(wrapper)],
        env=_child_env(RUNNER_TEMP=str(tmp_path)), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for index, (_, _, store, _) in enumerate(cases):
        calls = (tmp_path / f'calls-{index}.txt').read_text(encoding='utf-8-sig')
        assert '--variant bundled' in calls
        assert ('--variant store' in calls) is store
