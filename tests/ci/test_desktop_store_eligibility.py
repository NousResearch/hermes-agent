"""Execute the packaging workflow branches; nonstable never builds Store identities."""
import json
import os
import shlex
import subprocess
import sys

import pytest

from tests.ci.test_desktop_release_tag_admission import _BASH, _child_env, _workflow


@pytest.mark.parametrize('tag,commit,store', [
    ('v0.28.0', '', True),
    ('v0.28.0-canary.20260818', '', False),
    ('', 'a' * 40, False),
])
def test_windows_build_and_bundle_only_request_store_for_stable(tmp_path, tag, commit, store):
    helper = tmp_path / 'bin'
    helper.mkdir()
    log = tmp_path / 'calls.jsonl'
    recorder = helper / 'record.py'
    recorder.write_text(
        'import json,os,sys\n'
        'with open(os.environ["CALL_LOG"],"a") as file: file.write(json.dumps(sys.argv[1:])+"\\n")\n',
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
    for job, name in [('build-win32', 'Build and package'),
                      ('publish-win32-updater', 'Bundle + stage the MSIX feeds')]:
        script = next(step['run'] for step in jobs[job]['steps'] if step.get('name') == name)
        result = subprocess.run([_BASH, '-e', '-o', 'pipefail', '-c', script], env=env, cwd=tmp_path,
                                capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert any('--variant=bundled' in call for call in calls)
    assert any('--variant=store' in call for call in calls) is store
    assert any('scripts/bundle-store-msixbundle.mjs' in call for call in calls) is store
