"""Authorized native lane: private desktops, no GUI app or input requests."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

PLUGIN = Path(__file__).resolve().parents[2] / "plugins/hermes-realms"


@pytest.mark.linux_only
@pytest.mark.integration
def test_canonical_runtime_launches_jobs_and_contained_driver(tmp_path):
    code = r'''
import json, os, runpy, subprocess, sys, time, types
from pathlib import Path
root = Path(sys.argv[1])
foreign = types.ModuleType('realms')
sys.modules['realms'] = foreign
load = runpy.run_path(str(root / 'realms/_binding.py'))['load_runtime']
manager = load('manager').Manager()
lifecycle = load('lifecycle')
record = None
try:
    record = manager.start('import-isolation')
    rid = record['id']
    # Running helpers consume the owned launch spec, not a newly edited host
    # config, and need no host-core imports on their payload-only -P path.
    (manager.home / 'config.yaml').write_text('plugins: [broken', encoding='utf-8')
    frozen_code = """import runpy, sys
from pathlib import Path
load = runpy.run_path(sys.argv[1])['load_runtime']
before = list(sys.path)
manager = load('manager').Manager(sys.argv[2], realm_id=sys.argv[3])
assert manager.config.size == '320x240'
assert manager.config.renderer == 'pixman' and manager.config.overlay is False
assert sys.path == before
assert 'hermes_cli.config' not in sys.modules
"""
    child_env = dict(manager.env(rid), PYTHONPATH=str(root))
    frozen = subprocess.run([sys.executable, '-P', '-c', frozen_code,
                             str(root / 'realms/_binding.py'), str(manager.home), rid],
                            env=child_env, capture_output=True, text=True, timeout=20)
    assert frozen.returncode == 0, frozen.stdout + frozen.stderr
    result = manager.exec(rid, ['/usr/bin/printf', 'canonical-job'], wait=True)
    assert result['returncode'] == 0 and result['stdout'] == 'canonical-job', result
    run = subprocess.run([*manager.command_prefix(rid), '/usr/bin/printf', 'canonical-fd'],
                         env=manager.env(rid), capture_output=True, text=True, timeout=20)
    assert run.returncode == 0 and run.stdout == 'canonical-fd', (run.stdout, run.stderr)
    launcher = load('driver').create_driver_launcher(manager, rid, '/usr/bin/true')
    driver_code = "import runpy,sys,types;sys.modules['realms']=types.ModuleType('realms');runpy.run_path(sys.argv[1],run_name='__main__')"
    run = subprocess.run([sys.executable, '-c', driver_code, launcher],
                         env=manager.env(rid), capture_output=True, text=True, timeout=20)
    assert run.returncode == 0, run.stdout + run.stderr
    direct = subprocess.run([launcher], env=manager.env(rid), capture_output=True, text=True, timeout=20)
    assert direct.returncode == 0, direct.stdout + direct.stderr
    shot = Path.cwd() / 'frame.png'
    manager.shot(rid, shot)
    assert shot.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert sys.modules['realms'] is foreign
finally:
    if record:
        manager.stop(record['id'])
        assert not Path(record['runtime_dir']).exists()
        assert not any(lifecycle.alive(p) for p in record['processes'].values())
        assert lifecycle.scope_info(record['scope'])['ActiveState'] in ('inactive', 'failed')
        deadline = time.monotonic() + 8
        while lifecycle.scope_info(record['guardian_unit'])['ActiveState'] not in ('inactive', 'failed'):
            assert time.monotonic() < deadline, 'guardian cleanup did not finish'
            time.sleep(0.05)
    assert manager.list() == []
if record:
    stale = subprocess.run([sys.executable, '-P', '-c', frozen_code,
                            str(root / 'realms/_binding.py'), str(manager.home), record['id']],
                           env=child_env, capture_output=True, text=True, timeout=20)
    assert stale.returncode != 0 and 'realm not found' in stale.stderr, stale.stdout + stale.stderr
print(json.dumps({'source': str(root), 'jobs': 'captured and FD', 'driver': 'generated launcher /usr/bin/true inside bubblewrap', 'capture': 'PNG', 'config': 'frozen owned spec; corrupt host config not read; stale owner rejected', 'cleanup': 'runtime, processes, scope, guardian, registry removed'}))
'''
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(json.dumps({"plugins": {"realms": {
        "renderer": "pixman", "size": "320x240", "overlay": False
    }}}))
    env = {key: os.environ[key] for key in ("PATH", "LANG", "TZ") if key in os.environ}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(home), PYTHONPATH=str(PLUGIN.parents[1]))
    result = subprocess.run([sys.executable, "-c", code, str(PLUGIN)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())
