"""Desktop update custody uses the live marker owner and still excludes a competing update."""

from pathlib import Path
import os
import shlex
import subprocess
import sys

import pytest

pytestmark = pytest.mark.platforms("posix")
REPO = Path(__file__).resolve().parents[3]


def test_desktop_handoff_adopts_custodian_and_blocks_another_updater(tmp_path):
    home = tmp_path / 'home'
    install = home / 'hermes-agent'
    bin_dir = install / '.hermes' / 'bin'
    bin_dir.mkdir(parents=True)
    (install / 'pm').mkdir()
    worker = tmp_path / 'worker.py'
    worker.write_text('''
import os, sys, subprocess
from pathlib import Path
if '--help' in sys.argv:
    print('--keep-stash'); raise SystemExit(0)
sys.path.insert(0, os.environ['HERMES_REGRESSION_SOURCE'])
from hermes_cli.update_lock import UpdateLock, MARKER_NAME
marker = Path(os.environ['HERMES_HOME']) / MARKER_NAME
assert int(os.environ['HERMES_UPDATE_HANDOFF_PID']) == int(marker.read_text().splitlines()[0]), 'handoff must name the live marker owner'
lock = UpdateLock(path=marker, install_root=Path.cwd())
if not lock.acquire():
    print('FAILED_TO_ADOPT', lock.holder); raise SystemExit(2)
print('ADOPTED_CUSTODIAN')
other_env = dict(os.environ)
other_env.pop('HERMES_UPDATE_HANDOFF_PID', None)
code = "from pathlib import Path; from hermes_cli.update_lock import UpdateLock; import sys; x=UpdateLock(path=Path(sys.argv[1]),install_root=Path.cwd()); ok=x.acquire(); print('COMPETITOR_ACQUIRED' if ok else 'COMPETITOR_REFUSED'); x.release(); sys.exit(1 if ok else 0)"
other_env['PYTHONPATH'] = os.environ['HERMES_REGRESSION_SOURCE']
other = subprocess.run([sys.executable, '-c', code, str(marker)],env=other_env,capture_output=True,text=True,timeout=30)
print(other.stdout)
print(other.stderr)
lock.release()
raise SystemExit(other.returncode)
''')
    launcher = bin_dir / 'hermes'
    launcher.write_text(f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} {shlex.quote(str(worker))} "$@"\n')
    launcher.chmod(0o755)
    env = {**os.environ, 'HOME': str(tmp_path), 'TMPDIR': str(tmp_path),
           'HERMES_HOME': str(home), 'HERMES_RUNTIME_DIR': str(tmp_path / 'runtime'),
           'HERMES_REGRESSION_SOURCE': str(REPO), 'HERMES_UPDATE_SHIM_GRACE_SECONDS': '0'}
    for name in ('PYTHONPATH','PYTHONHOME','HERMES_UPDATE_HANDOFF_PID','HERMES_UPDATE_STARTED_AT'):
        env.pop(name,None)
    result = subprocess.run(['bash', str(REPO/'scripts/desktop-update/posix.sh'),
        '--daemonized','--no-ui','--install-root',str(install)],env=env,cwd=tmp_path,
        capture_output=True,text=True,timeout=120)
    log = (home/'logs/desktop-update-handoff.log').read_text()
    assert result.returncode == 0, log + result.stderr
    assert 'ADOPTED_CUSTODIAN' in log
    assert 'COMPETITOR_REFUSED' in log
    assert not (home/'.hermes-update-in-progress').exists()
