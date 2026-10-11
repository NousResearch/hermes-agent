"""One launch's hook consent and Kanban/terminal pins never reach the auto-started shared daemon."""
import json
import os
from pathlib import Path
import sys
import time

import pytest

LEAKS = {'HERMES_ACCEPT_HOOKS': '1', 'HERMES_KANBAN_TASK': 't_1', 'HERMES_KANBAN_BOARD': 'other',
         'HERMES_KANBAN_DB': '/x/kanban.db', 'HERMES_SESSION_SOURCE': 'kanban', 'TERMINAL_CWD': '/x'}


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike  # stub interpreter records env then exits; reaped below
def test_daemon_does_not_inherit_hook_consent_or_worker_pins(tmp_path, monkeypatch):
    from hermes_cli import gateway_runtime_start as start
    home = tmp_path / 'home'
    home.mkdir(mode=0o700)
    witness = home / 'env.json'
    executable = tmp_path / 'owned-interpreter'
    executable.write_text(
        '#!' + sys.executable + '\nimport json, os\n'
        + f'open({str(witness)!r}, "w").write(json.dumps('
        + '{k: os.environ.get(k) for k in ' + repr([*LEAKS, 'HERMES_INFERENCE_MODEL']) + '}))\n',
        encoding='utf-8')
    executable.chmod(0o700)
    monkeypatch.setattr(sys, 'executable', str(executable))
    for key, value in {**LEAKS, 'HERMES_INFERENCE_MODEL': 'fallback'}.items():
        monkeypatch.setenv(key, value)
    child = start.spawn_unmanaged_gateway(home, deadline=time.monotonic() + 5)
    try:
        assert child.wait(timeout=10) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
    seen = json.loads(Path(witness).read_text())
    assert seen == {**dict.fromkeys(LEAKS), 'HERMES_INFERENCE_MODEL': 'fallback'}
    assert os.environ['HERMES_ACCEPT_HOOKS'] == '1'  # the launching process keeps its own policy
