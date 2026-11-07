"""A later generation of a POSIX Desktop hand-off's update adopts the hand-off's own claim.

``hermes update`` adopts an existing claim only when a live partner is its own pid, the
``HERMES_UPDATE_HANDOFF_PID`` value, or an ancestor (``update_lock.UpdateLock._is_partner``).
``posix.sh`` hands marker line 1 to a custodian before any update work starts; the custodian is a
sibling of the update, never an ancestor. The first ``hermes update`` is named on the delegate
line, but a later generation (a relaunch or a takeover detached from it) has a new pid, so the
hand-off pid is its only way to recognise the custodian. Naming the script there made those
updates refuse their own claim with exit 2 (#134268, #134602).

Runs on macOS too (the CI macOS lane), where the marker's creation times come from ``ps`` in
whole seconds: that host is where the refusal was reported.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.scripts.desktop_update.test_desktop_update_posix_marker import FAKE_CLI, _custodian, _install, _run

pytestmark = pytest.mark.platforms("posix")

REPOSITORY = Path(__file__).resolve().parents[3]

# The update child (generation 1) starts generation 2 through an intermediate that exits at once,
# so generation 2 is reparented away from it (no ancestry to lean on), then waits for its verdict.
# Generation 2 runs this repository's real UpdateLock against the marker the real posix.sh wrote.
SECOND_GENERATION = """
import os, sys, json, subprocess, time
from pathlib import Path
if sys.argv[1:2] == ['update'] and '--help' not in sys.argv:  # not the --keep-stash probe
    out = Path(os.environ['GEN2_OUT'])
    probe = (
        "import json, os, sys; from pathlib import Path\\n"
        "from hermes_cli.update_lock import UpdateLock, update_marker_path\\n"
        "lock = UpdateLock(path=update_marker_path())\\n"
        "ok = lock.acquire()\\n"
        "Path(sys.argv[1]).write_text(json.dumps({'adopted': ok, 'pid': os.getpid(), "
        "'holder': lock.holder.pid if lock.holder else None, "
        "'told': os.environ.get('HERMES_UPDATE_HANDOFF_PID')}), encoding='utf-8')\\n"
        "ok and lock.release()\\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
    env['PYTHONPATH'] = os.environ['GEN2_REPO']
    subprocess.run([sys.executable, '-c',
                    'import subprocess, sys; subprocess.Popen(sys.argv[1:], start_new_session=True)',
                    sys.executable, '-c', probe, str(out)],
                   cwd=os.environ['GEN2_REPO'], env=env, check=True)
    deadline = time.monotonic() + 60
    while not out.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    sys.exit(0 if out.exists() and json.loads(out.read_text(encoding='utf-8-sig'))['adopted'] else 2)
"""


def test_a_later_update_generation_adopts_the_handoffs_claim(tmp_path):
    home, install = _install(tmp_path)
    (install / "hermes_cli" / "main.py").write_text(SECOND_GENERATION + FAKE_CLI, encoding="utf-8")
    out = tmp_path / "gen2.json"

    result = _run(tmp_path, home, install, GEN2_OUT=str(out), GEN2_REPO=str(REPOSITORY))

    custodian = _custodian(home)
    assert custodian, "the hand-off never named a custodian"
    verdict = json.loads(out.read_text(encoding="utf-8-sig"))
    assert verdict["adopted"], f"generation 2 refused its own hand-off's claim (holder {verdict['holder']})"
    assert verdict["told"] == custodian, "the update must be told the marker's owner, not the hand-off script"
    assert result.returncode == 0, result.stdout + result.stderr
