"""Importing Hermes from an auxiliary script must not take installer custody."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

PROBE = '''
import json, sys
import pm.environments
import hermes_cli.venv_sync as sync
import hermes_cli._early_recovery as recovery
events=[]
pm.environments.activate_dependencies=lambda root:events.append('activate')
sync.prepare_launch=lambda root,argv:events.append('prepare')
recovery.recover_if_needed=lambda root:events.append('repair')
recovery.restore_interrupted_pull=lambda root:events.append('restore')
# No cron/no-lazy protection: an ordinary repair helper importing the bootstrap.
import hermes_bootstrap
print(json.dumps(events))
'''


@pytest.mark.parametrize('as_file', [False, True])
@pytest.mark.parametrize('managed', [False, True])
def test_helper_import_never_starts_automatic_maintenance(tmp_path, as_file, managed):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root), HOME=str(tmp_path),
               HERMES_HOME=str(tmp_path / 'home'))
    for key in ('_HERMES_CRON_WORKER_BOOT', 'HERMES_CRON_SESSION', 'HERMES_DISABLE_LAZY_INSTALLS'):
        env.pop(key, None)
    probe = PROBE.replace('import hermes_bootstrap',
                          f'sys._hermes_pin_default_home = {managed!r}\nimport hermes_bootstrap')
    if as_file:
        helper = tmp_path / 'repair_helper.py'
        helper.write_text(probe)
        args = [str(helper)]
    else:
        args = ['-c', probe]
    result = subprocess.run([sys.executable, *args], cwd=root, env=env,
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    expected = ['restore', 'prepare', 'repair', 'activate'] if managed else ['restore', 'activate']
    assert json.loads(result.stdout.strip().splitlines()[-1]) == expected
