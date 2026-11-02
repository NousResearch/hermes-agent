"""Execute the packaging workflow branches; only release candidates build Store identities.

A canary candidate (Stable Release in canary mode) builds the Store bundle like any candidate and
never submits it; the legacy direct canary dispatch and commit builds never build one."""
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
    ('canary.20260818T000000Z-v0.28.0', '', False),
    ('canary.20260818T000000Z-v0.28.0', '', True),
    ('commit.20260818T000000Z-v0.28.0', 'a' * 40, False),
])
def test_bundle_only_requests_store_for_release_candidates(tmp_path, tag, commit, store):
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
                            capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in log.read_text(encoding='utf-8-sig').splitlines()]
    assert calls[0][:1] == ['scripts/stage-msixbundle.mjs']
    assert calls[0][calls[0].index('--variant') + 1] == 'bundled'
    # Every claim kind assembles from its claim tag; there is no tagless, upload-less mode.
    assert calls[0][calls[0].index('--tag') + 1] == tag and '--no-upload' not in calls[0]
    assert ('--candidate' in calls[0]) is store
    assert any('scripts/bundle-store-msixbundle.mjs' in call for call in calls) is store
    assert not any('publish-appinstaller' in call for call in calls)


@pytest.mark.platforms('windows')
def test_native_windows_build_selects_store_only_for_stable(tmp_path):
    jobs = _workflow()['jobs']
    legs = native_builds(jobs)
    script = next(step['run'] for step in jobs[legs['win32-x64']]['steps'] if step.get('name') == 'Build and package')
    # (payload tag, commit identity, builds the Store variant): every claim kind builds on this one leg.
    cases = [
        ('v0.28.0', '', True),
        ('canary.20260818T000000Z-v0.28.0', '', False),
        ('commit.20260818T000000Z-v0.28.0', 'a' * 40, False),
    ]
    wrapper = tmp_path / 'run.ps1'
    # Substitute only the build boundary, then execute both workflow scripts
    # as real PowerShell in one session. Cold starts contend heavily in the
    # native OS lane; charging one to every parameter made startup latency part
    # of the assertion.
    body = [
        'function python { $args -join " " | Add-Content $env:CALL_LOG -Encoding UTF8; $global:LASTEXITCODE = 0 }',
        f'$build = {{\n{script}\n}}',
    ]
    for index, (tag, commit, _) in enumerate(cases):
        body.extend([
            f'$env:HERMES_PAYLOAD_TAG = \'{tag}\'',
            f'$env:HERMES_BUILD_COMMIT = \'{commit}\'',
            f'$env:CALL_LOG = Join-Path $env:RUNNER_TEMP \'calls-{index}.txt\'',
            '& $build',
        ])
    wrapper.write_text('\n'.join(body) + '\n', encoding='utf-8')
    powershell = shutil.which('powershell')
    assert powershell
    # The canonical per-file runner remains the deadlock guard; a nested
    # startup deadline only measures scheduler contention on the Windows host.
    result = subprocess.run(
        [powershell, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', str(wrapper)],
        env=_child_env(RUNNER_TEMP=str(tmp_path)), capture_output=True, text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for index, (_, _, store) in enumerate(cases):
        calls = (tmp_path / f'calls-{index}.txt').read_text(encoding='utf-8-sig')
        assert '--variant bundled' in calls
        assert ('--variant store' in calls) is store
