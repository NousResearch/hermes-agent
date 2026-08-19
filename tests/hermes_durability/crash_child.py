"""Workload child for crash-injection tests.

Runs agent-like turns forever: journal a user message + tool call, enqueue
an outbound message, commit, deliver via a file-based idempotent receiver.
The parent kill -9s this process at a random moment.

Usage: python crash_child.py <db_path> <receiver_dir>
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_durability import (DurableRuntime, TOOL_CALL_INVOKED, USER_MESSAGE)
from hermes_durability.guardrail import Envelope


class FileReceiver:
    """External service simulated on disk; idempotent on key.

    Writes are atomic (tmp + rename) so the receiver itself can't be torn.
    """

    def __init__(self, root: str):
        self.root = root

    def send(self, envelope: Envelope, idempotency_key: str) -> dict:
        path = os.path.join(self.root, idempotency_key + ".json")
        if not os.path.exists(path):
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(envelope.payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp, path)
        return {"id": idempotency_key}


def main() -> None:
    db_path, receiver_dir = sys.argv[1], sys.argv[2]
    rt = DurableRuntime(db_path, adapters={"file": FileReceiver(receiver_dir)},
                        start_worker=False)
    report = rt.recover()
    print("RECOVERED", json.dumps(report), flush=True)

    state = rt.replay_state("s1")
    n = sum(1 for m in state.get("messages", [])
            if m.get("type") == USER_MESSAGE)
    while True:
        with rt.transaction("s1") as txn:
            txn.record(USER_MESSAGE, {"text": f"user turn {n}", "n": n})
            txn.record(TOOL_CALL_INVOKED, {"tool": "search", "n": n})
            txn.enqueue_outbound("file", {"text": f"reply {n}", "n": n},
                                 outbox_id=f"turn-{n}")
        rt.worker.drain_once()
        print(f"TURN {n}", flush=True)
        n += 1
        if n % 25 == 0:  # exercise compaction under crashes too
            rt.journal.compact("s1", rt.replay_state("s1"))
            print(f"COMPACTED {n}", flush=True)


if __name__ == "__main__":
    main()
