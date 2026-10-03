"""Real inert executables distinguish launcher exit from failure to execute it.

Only service-membership discovery is synthetic; availability/version probes,
launch, wait, receipt publication and correlation run in real subprocesses.
No updater handler, systemd unit or source mutation is invoked here.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
HARNESS = r'''
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
original = Path.read_text
def read(path, *args, **kwargs):
    if str(path) == '/proc/self/cgroup':
        return '0::/user.slice/user@1000.service/app.slice/inert-test.service\n'
    return original(path, *args, **kwargs)
Path.read_text = read
from hermes_cli.update_process import isolate_update_process
isolate_update_process()
Path(os.environ['OWNED_ROOT'], 'unsafe-handler').write_text('forbidden')
'''
LAUNCHER = r'''
import json, os, sys
from pathlib import Path
root = Path(os.environ['OWNED_ROOT'])
phase = 'version' if sys.argv[1:] == ['--version'] else 'probe' if '/bin/sh' in sys.argv else 'launch'
with (root / 'launcher.jsonl').open('a') as stream:
    stream.write(json.dumps({'phase': phase, 'argv': sys.argv, 'cwd': os.getcwd(),
        'home': os.environ['HERMES_HOME'], 'action': os.environ.get('HERMES_ACTION_ID'),
        'stdout': os.readlink('/proc/self/fd/1'), 'stderr': os.readlink('/proc/self/fd/2')}) + '\n')
if phase == 'version':
    print('systemd 255 (inert-control)')
    sys.exit(0)
if phase == 'probe':
    sys.exit(0)
if os.environ['OWNED_MODE'] == 'terminal-success':
    sys.path.insert(0, os.environ['OWNED_REPO'])
    from hermes_cli.update_receipt import begin_update_receipt, finalize_pending_update_receipt
    begin_update_receipt(correlation_id=os.environ['HERMES_ACTION_ID'])
    path = finalize_pending_update_receipt(0, 'inert child terminal success')
    (root / 'child-receipt.json').write_bytes(path.read_bytes())
    # Move the shared pointer to an unrelated PM payload. Only the exact archive
    # proves that this action already has a terminal receipt.
    (path.parent / 'latest.json').write_text(json.dumps({'kind': 'pm_sync', 'outcome': 'success'}))
print('INERT_LAUNCHER_EXIT_17', flush=True)
print('INERT_LAUNCHER_STDERR', file=sys.stderr, flush=True)
sys.exit(17)
'''


def _run(tmp_path, action_id, mode='failure'):
    binary = tmp_path / 'systemd-run'
    binary.write_text('#!' + sys.executable + '\n' + LAUNCHER)
    binary.chmod(0o700)
    harness = tmp_path / 'harness.py'
    harness.write_text(HARNESS)
    home = tmp_path / 'profile home'
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ['PATH'],
               OWNED_ROOT=str(tmp_path), HERMES_HOME=str(home),
               HERMES_ACTION_ID=action_id, INVOCATION_ID='owned-inert-service',
               OWNED_MODE=mode, OWNED_REPO=str(REPO))
    command = [sys.executable, '-u', str(harness), str(REPO), '${HOME}', '$HOME', 'two ${HOME} spaces', '$$', '']
    out_path, err_path = tmp_path / 'stdout.log', tmp_path / 'stderr.log'
    with out_path.open('w') as out, err_path.open('w') as err:
        result = subprocess.run(command, cwd=tmp_path, env=env, stdout=out, stderr=err, timeout=20)
    records = [json.loads(line) for line in (tmp_path / 'launcher.jsonl').read_text().splitlines()]
    observations = {'command': command, 'exit_code': result.returncode, 'records': records,
                    'stdout': out_path.read_text(), 'stderr': err_path.read_text(),
                    'archives': [str(p) for p in home.glob('logs/update_receipts/update_*.json')]}
    (tmp_path / 'observations.json').write_text(json.dumps(observations, indent=2))
    artifacts = os.environ.get('HERMES_E2E_ARTIFACTS')
    if artifacts:
        import shutil
        target = Path(artifacts) / tmp_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_path, target)
        print(f'LAUNCHER_EVIDENCE={target}')
    return result, home, records, command


@pytest.mark.platforms('linux')
def test_executed_launcher_nonzero_publishes_correlated_failure(tmp_path):
    action_id = 'e' * 32
    result, home, records, command = _run(tmp_path, action_id)
    assert result.returncode == 17
    assert [record['phase'] for record in records] == ['probe', 'version', 'launch']
    launch = records[-1]
    assert launch['argv'][launch['argv'].index('--') + 1:] == command
    assert '--expand-environment=no' in launch['argv']
    assert launch['cwd'] == str(tmp_path) and launch['home'] == str(home)
    assert launch['action'] == action_id
    assert launch['stdout'] == str(tmp_path / 'stdout.log')
    assert launch['stderr'] == str(tmp_path / 'stderr.log')
    assert not (tmp_path / 'unsafe-handler').exists()
    assert not (home / '.hermes-update-in-progress').exists()
    archives = list(home.glob(f'logs/update_receipts/update_*_{action_id}.json'))
    assert len(archives) == 1, 'executed launcher exit must have a durable failure owner'
    receipt = json.loads(archives[0].read_text())
    assert receipt['update_id'] == action_id
    assert receipt['exit_code'] == 17 and receipt['outcome'] == 'failed' and receipt['finished_at']
    assert 'scope' in receipt['stop_reason'] and '17' in receipt['stop_reason']
    assert json.loads((home / 'logs/update_receipts/latest.json').read_text()) == receipt


@pytest.mark.platforms('linux')
def test_launcher_nonzero_does_not_overwrite_correlated_child_terminal_success(tmp_path):
    action_id = 'f' * 32
    result, home, records, _ = _run(tmp_path, action_id, mode='terminal-success')
    assert result.returncode == 17  # launcher status is not the receipt outcome
    assert [record['phase'] for record in records] == ['probe', 'version', 'launch']
    archives = list(home.glob(f'logs/update_receipts/update_*_{action_id}.json'))
    assert len(archives) == 1
    assert archives[0].read_bytes() == (tmp_path / 'child-receipt.json').read_bytes()
    receipt = json.loads(archives[0].read_text())
    assert receipt['outcome'] == 'success' and receipt['exit_code'] == 0 and receipt['finished_at']
    assert json.loads((home / 'logs/update_receipts/latest.json').read_text())['kind'] == 'pm_sync'
    assert not (tmp_path / 'unsafe-handler').exists()
    assert not (home / '.hermes-update-in-progress').exists()


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('action_id', ['', '../bad', 'A' * 32])
def test_launcher_failure_creates_valid_child_correlation_when_input_invalid(tmp_path, action_id):
    result, home, records, _ = _run(tmp_path, action_id)
    assert result.returncode == 17
    generated = records[-1]['action']
    assert len(generated) == 32 and all(char in '0123456789abcdef' for char in generated)
    assert generated != action_id
    archives = list(home.glob(f'logs/update_receipts/update_*_{generated}.json'))
    assert len(archives) == 1
    receipt = json.loads(archives[0].read_text())
    assert receipt['update_id'] == generated and receipt['exit_code'] == 17
    assert receipt['finished_at'] and receipt['outcome'] == 'failed'
    assert not (tmp_path / 'unsafe-handler').exists()
