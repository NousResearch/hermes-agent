"""The paused-gateway record survives an update that runs under a partner's claim.

A real checkout (a clone of the commit under test) and real processes: a ``hermes update``
stand-in that recorded a paused set is SIGKILLed; the next update runs the way Desktop/Tauri
drives it — the orchestrator holds the marker and the ``hermes update`` child takes the checkout
lock and adopts the orchestrator's claim. That update must adopt the orphaned set and fold it into
its own record, never treat its own partner as "another update" and overwrite the record.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I

pytestmark = [pytest.mark.platforms("linux"), pytest.mark.live_system_guard_bypass]

_ORPHAN = """
import time
from hermes_cli import update_pause_record as r
r.write(r.stamp_tree({"resume_needed": True, "profiles": {"orphan": 4242}}), owner=r.identity())
print("written", flush=True)
time.sleep(120)
"""
# The orchestrator (Tauri ``hermes-setup --update``) claims the marker, then runs the update child.
_ORCHESTRATOR = """
import subprocess, sys
from hermes_cli.update_lock import UpdateLock
lock = UpdateLock()
assert lock.acquire() and lock.acquired
sys.exit(subprocess.call([sys.executable, "-c", sys.argv[1], sys.argv[2]]))
"""
_UPDATE_CHILD = """
import json, sys
from hermes_cli import update_pause_record as r
from hermes_cli.update_lock import UpdateLock
lock = UpdateLock(install_root=sys.argv[1])
adopted_claim = lock.acquire() and not lock.acquired
saw_orphan = r.orphaned_record() is not None
r.write(r.stamp_tree({"resume_needed": True, "profiles": {"mine": 1}}), owner=r.identity())
print(json.dumps({"adopted_claim": adopted_claim, "saw_orphan": saw_orphan,
                  "recorded": sorted(r.read()["token"]["profiles"])}))
lock.release()
"""


def _py(code: str, *args: str, env: dict, cwd: Path) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", textwrap.dedent(code), *args], env=env, cwd=cwd,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_update_under_a_partner_claim_merges_an_orphaned_pause(tmp_path):
    origin = I.make_origin(tmp_path, I.head_sha())
    checkout = tmp_path / "checkout"
    I.git("clone", "-q", "--shared", "-b", "main", str(origin), str(checkout), cwd=tmp_path)
    env = H.isolated_env(tmp_path / "env", pythonpath=checkout)

    owner = _py(_ORPHAN, env=env, cwd=checkout)
    assert owner.stdout.readline().strip() == "written", owner.stderr.read()
    owner.send_signal(signal.SIGKILL)  # windows-footgun: ok — linux-only cell
    owner.wait(timeout=10)

    run = _py(_ORCHESTRATOR, textwrap.dedent(_UPDATE_CHILD), str(checkout), env=env, cwd=checkout)
    out, err = run.communicate(timeout=120)
    assert run.returncode == 0, f"{out}\n{err}"
    result = json.loads(out.strip().splitlines()[-1])
    assert result["adopted_claim"], f"premise: the update child did not run under the orchestrator's claim: {result}"
    assert result["saw_orphan"], "the update treated its own orchestrator as another update and skipped the orphan"
    assert result["recorded"] == ["mine", "orphan"], f"the orphaned paused set was overwritten: {result}"
