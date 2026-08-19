"""Conformance: kill -9 the workload at random points; verify on restart:

  I1  hash chain valid (possibly after torn-tail repair)
  I2  replay state contains only committed transactions, contiguous turns
  I3  every delivered receipt exists exactly once (idempotency)
  I4  every receipt on the receiver corresponds to a committed enqueue
  I5  recovery drains any send that committed but hadn't delivered
"""

import json
import os
import random
import signal
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_durability import DurableRuntime, USER_MESSAGE

CHILD = os.path.join(os.path.dirname(__file__), "crash_child.py")
ROUNDS = int(os.environ.get("CRASH_ROUNDS", "12"))


def run_and_kill(db, recv_dir, delay: float) -> None:
    proc = subprocess.Popen([sys.executable, CHILD, db, recv_dir],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    time.sleep(delay)
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()


def verify_invariants(db, recv_dir) -> dict:
    rt = DurableRuntime(db, start_worker=False)
    ok, bad = rt.journal.verify_chain(repair=True)
    ok2, _ = rt.journal.verify_chain()
    assert ok2, "chain must be valid after repair"

    state = rt.replay_state("s1")
    turns = sorted(m["n"] for m in state.get("messages", [])
                   if m.get("type") == USER_MESSAGE)
    # I2: committed turns are contiguous from the last compaction point
    if turns:
        assert turns == list(range(turns[0], turns[0] + len(turns))), \
            f"non-contiguous committed turns: {turns}"

    receipts = sorted(os.listdir(recv_dir))
    # I3: receiver files are keyed by outbox_id -> inherently unique
    payload_ns = set()
    for r in receipts:
        if r.endswith(".tmp"):
            continue
        with open(os.path.join(recv_dir, r)) as f:
            payload_ns.add(json.load(f)["n"])
    assert len(payload_ns) == len([r for r in receipts if r.endswith(".json")])

    # I4: every receipt corresponds to a committed enqueue in the journal
    enqueued = set()
    for rec in rt.journal.records("s1"):
        if rec.record_type == "OutboxEnqueued":
            enqueued.add(rec.payload["outbox_id"])
    committed_outbox = {
        row[0] for row in rt.journal._conn.execute(
            "SELECT outbox_id FROM outbox").fetchall()}
    for r in receipts:
        if r.endswith(".json"):
            key = r[:-5]
            assert key in committed_outbox or key in enqueued, \
                f"receiver has receipt {key} with no committed enqueue (ghost send)"
    n_msgs = len(state.get("messages", []))
    rt.close()
    return {"turns": len(turns), "receipts": len(receipts), "chain_ok": ok,
            "messages": n_msgs}


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("seed", range(ROUNDS))
def test_kill9_at_random_points(tmp_path, seed):
    random.seed(seed)
    db = str(tmp_path / "crash.db")
    recv_dir = str(tmp_path / "recv")
    os.makedirs(recv_dir, exist_ok=True)
    # several crash/restart cycles per scenario
    for cycle in range(3):
        delay = random.uniform(0.05, 1.2)
        run_and_kill(db, recv_dir, delay)
        verify_invariants(db, recv_dir)
    # final full recovery: pending sends must drain (I5)
    rt = DurableRuntime(db, start_worker=False)

    class Recv:
        def __init__(self, root):
            self.root = root

        def send(self, envelope, idempotency_key):
            path = os.path.join(self.root, idempotency_key + ".json")
            if not os.path.exists(path):
                with open(path, "w") as f:
                    json.dump(envelope.payload, f)
            return {"id": idempotency_key}

    rt.worker.adapters["file"] = Recv(recv_dir)
    rt.recover()
    pending = rt.journal._conn.execute(
        "SELECT COUNT(*) FROM outbox WHERE status='pending'").fetchone()[0]
    assert pending == 0, "recovery must drain all committed pending sends"
    rt.close()
