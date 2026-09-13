"""Real Hermes argparse dispatch, processes and exit status; no model or input."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.linux_only
def test_real_host_parser_keeps_root_command_separate_from_exec_argv(tmp_path):
    code = r'''
import json, sys, types
from pathlib import Path
from hermes_constants import get_hermes_home
from hermes_cli.plugins_discovery import collect_directory_manifests
from hermes_cli.plugins_manifest import manifest_key
home = get_hermes_home()
home.mkdir(parents=True, exist_ok=True)
other = [manifest_key(m) for m in collect_directory_manifests() if m.name != 'hermes-realms']
(home / 'config.yaml').write_text(json.dumps({'plugins': {
    'enabled': ['hermes-realms'], 'disabled': other}}))
foreign = types.ModuleType('realms')
sys.modules['realms'] = foreign
Path('realms.py').write_text("raise AssertionError('project runtime imported')")
def no_process_or_network(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'socket.bind', 'socket.connect'):
        raise AssertionError('parser must not launch processes or network: ' + event)
sys.addaudithook(no_process_or_network)
sys.argv = ['hermes', 'realms', 'exec', 'missing-owner', '--', 'printf', 'never-executed']
from hermes_cli.main import _build_cli_parser
# The host entry module prepares its own import path; plugin registration must not.
before = list(sys.path)
parser, _ = _build_cli_parser()
args = parser.parse_args(sys.argv[1:])
assert args.command == 'realms', args
assert args.operation == 'exec', args
assert args.realm_command == ['printf', 'never-executed'], args
assert callable(args.func)
assert sys.modules['realms'] is foreign
assert sys.path == before
'''
    env = {key: os.environ[key] for key in ("LANG", "TZ") if key in os.environ}
    # No systemd, desktop tools, or other executable can be found in this lane.
    env.update(PATH=str(tmp_path / "no-executables"), HOME=str(tmp_path),
               HERMES_HOME=str(tmp_path / "profile"), PYTHONPATH=str(ROOT))
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.linux_only
@pytest.mark.integration
def test_native_cli_exec_preserves_host_dispatch_and_child_exit(tmp_path):
    code = r'''
import json, runpy, subprocess, sys, time
from pathlib import Path
from hermes_constants import get_hermes_home
from hermes_cli.plugins_discovery import collect_directory_manifests
from hermes_cli.plugins_manifest import manifest_key
root = Path(sys.argv[1])
home = get_hermes_home()
home.mkdir(parents=True, exist_ok=True)
other = [manifest_key(m) for m in collect_directory_manifests() if m.name != 'hermes-realms']
(home / 'config.yaml').write_text(json.dumps({'plugins': {
    'enabled': ['hermes-realms'], 'disabled': other,
    'realms': {'renderer': 'pixman', 'size': '320x240', 'overlay': False}}}))
Path('realms.py').write_text("from pathlib import Path\nPath('canary').touch()")
def cli(*args):
    return subprocess.run([sys.executable, '-m', 'hermes_cli.main', 'realms', *args],
                          capture_output=True, text=True, timeout=40)
lifecycle = runpy.run_path(str(root / 'plugins/hermes-realms/realms/_binding.py'))['load_runtime']('lifecycle')
record = None
try:
    started = cli('start', 'native-cli-exec')
    assert started.returncode == 0, started.stdout + started.stderr
    record = json.loads(started.stdout)
    rid = record['id']
    success = cli('exec', rid, '--', '/usr/bin/printf', 'cli-success')
    assert success.returncode == 0, success.stdout + success.stderr
    assert success.stdout == 'cli-success'
    failure = cli('exec', rid, '--', '/usr/bin/sh', '-c', 'printf cli-failure; printf cli-stderr >&2; exit 7')
    assert failure.returncode == 7, failure.stdout + failure.stderr
    assert failure.stdout == 'cli-failure'
    assert failure.stderr == 'cli-stderr'
finally:
    if record:
        stopped = cli('stop', record['id'])
        assert stopped.returncode == 0, stopped.stdout + stopped.stderr
        assert json.loads(stopped.stdout)['stopped'] is True
        assert not Path(record['runtime_dir']).exists()
        assert not any(lifecycle.alive(p) for p in record['processes'].values())
        for key in ('scope', 'guardian_unit'):
            deadline = time.monotonic() + 8
            while lifecycle.scope_info(record[key])['ActiveState'] not in ('inactive', 'failed'):
                assert time.monotonic() < deadline, key + ' did not stop'
                time.sleep(0.05)
    listed = cli('list')
    assert listed.returncode == 0 and json.loads(listed.stdout) == []
    assert not Path('canary').exists()
print('native CLI: stdout/stderr, exit 0/7, runtime/process/unit/registry cleanup verified')
'''
    env = {key: os.environ[key] for key in ("PATH", "LANG", "TZ") if key in os.environ}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(tmp_path / "profile"), PYTHONPATH=str(ROOT))
    result = subprocess.run([sys.executable, "-c", code, str(ROOT)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())
