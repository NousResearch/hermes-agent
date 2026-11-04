"""Real-OS-process contention and crash recovery for the admission store.

Children are separate CPython processes hammering one SQLite store; the
parent asserts exactly one winner per round, strictly increasing epochs,
and SIGKILL recovery that never presumes memory was freed.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from hermes_cli.local_runtime import admission
from hermes_cli.local_runtime.admission import AdmissionStore, Grant

REPO_ROOT = str(Path(admission.__file__).resolve().parents[2])
BOOT = "boot-proc"

# One child: acquire once against the shared store, print the verdict as a
# JSON line, optionally hold the grant for `hold_s` then release and exit.
CHILD = """
import json, sys, time
sys.path.insert(0, sys.argv[1])
from hermes_cli.local_runtime.admission import AdmissionStore
store = AdmissionStore(sys.argv[2])
grant = store.acquire(
    sys.argv[3], owner_scope="scope-proc", controller_id=sys.argv[4],
    pid=os.getpid(), process_start_identity=sys.argv[5], engine="llama.cpp",
    model="M", boot_id=sys.argv[6], now=time.monotonic(), lease_seconds=300.0)
payload = {"granted": hasattr(grant, "epoch"),
           "epoch": getattr(grant, "epoch", None),
           "reason": getattr(grant, "reason", None),
           "grant": getattr(grant, "__dict__", None)}
print("RESULT " + json.dumps(payload), flush=True)
hold = float(sys.argv[7])
if hold > 0 and payload["granted"]:
    time.sleep(hold)
    store.release(grant, now=time.monotonic())
    print("RELEASED", flush=True)
"""
CHILD = CHILD.replace("pid=os.getpid()", "pid=__import__('os').getpid()")


def _child(store_root, resource, controller, ident, hold_s=0.0):
    return subprocess.Popen(
        [sys.executable, "-u", "-c", CHILD, REPO_ROOT, str(store_root),
         resource, controller, ident, BOOT, str(hold_s)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _read_result(proc) -> dict:
    line = proc.stdout.readline()
    assert line.startswith("RESULT "), f"child said: {line!r} " \
        f"stderr={proc.stderr.read()!r}"
    return json.loads(line[len("RESULT "):])


def test_one_winner_per_round_and_strictly_increasing_epochs(tmp_path):
    store_root = tmp_path / "home"
    resource = "llama:hammer"
    seen_epochs = []
    for round_no in range(2):
        procs = [_child(store_root, resource, f"ctrl-{round_no}-{i}",
                        f"ident-{round_no}-{i}", hold_s=5.0) for i in range(6)]
        results = [_read_result(p) for p in procs]
        winners = [r for r in results if r["granted"]]
        losers = [r for r in results if not r["granted"]]
        assert len(winners) == 1, f"round {round_no}: {results}"
        assert all(r["reason"] == "busy" for r in losers), results
        seen_epochs.append(winners[0]["epoch"])
        # The winner releases after its hold; only then may the next round run.
        for p in procs:
            p.wait(timeout=60)
        assert winners[0]["epoch"] == round_no + 1
    assert seen_epochs == sorted(seen_epochs) and len(set(seen_epochs)) == 2


def test_sigkilled_holder_stays_quarantined_until_verified_exit(tmp_path):
    store_root = tmp_path / "home"
    store = AdmissionStore(store_root)
    resource = "llama:victim"
    proc = _child(store_root, resource, "ctrl-victim", "ident-victim",
                  hold_s=600.0)
    try:
        result = _read_result(proc)
        assert result["granted"] and result["epoch"] == 1
        killed_epoch = result["epoch"]
        # The test kills the child; the admission module itself never does.
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)

    now = time.monotonic()
    # Recovery without proof of exit: quarantined, next grant refused.
    assert store.recover(resource, now, BOOT) == "quarantined"
    assert store.recover(resource, now + 1, BOOT) == "quarantined"
    refusal = store.acquire(
        resource, owner_scope="scope-new", controller_id="ctrl-new",
        pid=os.getpid(), process_start_identity="ident-new",
        engine="llama.cpp", model="M", boot_id=BOOT, now=now + 2,
        lease_seconds=300.0)
    assert refusal.reason == "quarantined"
    assert "ctrl-victim" in refusal.detail

    # Verified exit unlocks the next grant at a strictly higher epoch.
    assert store.recover(resource, now + 3, BOOT, verified_exit=True,
                         expected_epoch=killed_epoch) == "released"
    grant = store.acquire(
        resource, owner_scope="scope-new", controller_id="ctrl-new",
        pid=os.getpid(), process_start_identity="ident-new",
        engine="llama.cpp", model="M", boot_id=BOOT, now=now + 4,
        lease_seconds=300.0)
    assert isinstance(grant, Grant)
    assert grant.epoch > killed_epoch


def test_cross_process_concurrent_lease_idempotency(tmp_path):
    store_root = tmp_path / "home"
    store = AdmissionStore(store_root)
    resource = "llama:leases"
    grant = store.acquire(
        resource, owner_scope="scope-x", controller_id="ctrl-x",
        pid=os.getpid(), process_start_identity="ident-x",
        engine="llama.cpp", model="M", boot_id=BOOT,
        now=time.monotonic(), lease_seconds=300.0)
    assert isinstance(grant, Grant)
    procs = [subprocess.Popen(
        [sys.executable, "-u", "-c",
         "import sys, json;"
         f"sys.path.insert(0, {REPO_ROOT!r});"
         "from hermes_cli.local_runtime.admission import AdmissionStore, Grant;"
         f"g = Grant(**json.loads({json.dumps(json.dumps(grant.__dict__))}));"
         f"s = AdmissionStore({str(store_root)!r});"
         "print(json.dumps(s.add_lease(g, 'job-shared', kind='job').__dict__), "
         "flush=True)"],
        stdout=subprocess.PIPE, text=True) for _ in range(4)]
    states = []
    for p in procs:
        out, err = p.communicate(timeout=60)
        assert p.returncode == 0, err
        states.append(json.loads(out.strip().splitlines()[-1]))
    assert all(s["state"] == "active" and s["epoch"] == grant.epoch
               for s in states), states
    assert store.status(resource).active_leases == ("job-shared",)
