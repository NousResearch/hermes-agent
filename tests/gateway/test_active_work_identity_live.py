"""End-to-end: a live gateway names each managed turn it would interrupt.

The fixture (``fixtures/kanban_active_work_probe.py``) holds one real managed turn open behind a
blocking model, then asks the live control socket for ``active-work`` — the same answer
``hermes gateway restart`` prints its refusal from.

The unit must carry the turn's SESSION, its ADMISSION and the CHILD's pid. Printing the gateway's
own pid once per turn is what made a blind ``--yes`` restart possible on 2026-11-05
(t_925a816f): two real worker turns looked like a leak.
"""
import json
import subprocess
import sys
from pathlib import Path

from tests.gateway.fixtures.local_recovery_probe import child_env


def test_active_work_names_the_live_managed_turn(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    env = child_env()
    env.update(HOME=str(tmp_path / 'home'), USERPROFILE=str(tmp_path / 'home'),
               HERMES_HOME=str(tmp_path / 'state'), PYTHONPATH=str(repo))
    result = subprocess.run([sys.executable, str(Path(__file__).parent / 'fixtures' / 'kanban_active_work_probe.py')],
                            env=env, cwd=repo, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                            timeout=180, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / 'state' / 'receipt.json').read_text())
    print(json.dumps(receipt, indent=2))

    units = receipt['managed_units']
    assert len(units) == 1, receipt
    unit = units[0]
    assert unit['session'].startswith('local-'), receipt
    assert unit['admission_id'], receipt
    assert unit['pid'] == receipt['bound_pid'], receipt         # the child the run bound
    assert unit['pid'] != receipt['gateway_pid'], receipt       # never the gateway's own pid
    assert unit['profile'].replace('\\', '/').endswith('/state'), receipt
    assert unit.get('elapsed_s') is not None, receipt

    rendered = receipt['rendered'][0]
    assert rendered.startswith(f"managed turn {unit['session']} pid {unit['pid']}"), rendered
    assert f"admission {unit['admission_id']}" in rendered and 'profile ' in rendered, rendered
    assert f"pid {receipt['gateway_pid']}" not in rendered, rendered
