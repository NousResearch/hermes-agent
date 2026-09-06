"""Multi-process concurrency tests for session storage (Issue #23717).

The ticket's failure modes are all CROSS-PROCESS: the CLI, gateway, cron scheduler, TUI and
API server share one ``state.db``; SQLite WAL serializes writers at the OS level; and the
"hot-update death spiral" is a process killed mid-write leaving a torn WAL. The in-process
thread test in ``test_session_provider_tdd.py`` cannot see any of that — this module spawns
real worker processes against one database and pins the two contracts the RFC must not
regress:

- N concurrent process writers commit EVERYTHING (jittered retry + BEGIN IMMEDIATE absorb
  lock contention; no lost writes, integrity intact).
- A process hard-killed mid-write (kill -9 / TerminateProcess — the update path's SIGTERM
  made worse) leaves a consistent database: every commit the worker acknowledged is durable,
  the in-flight write never tears the file, and the next process recovers and keeps writing.
"""

import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from hermes_state_provider import get_session_db_provider

_WORKER_SCRIPT = """\
import sys
import time
from pathlib import Path

from hermes_state import SessionDB

db_path = Path(sys.argv[1])
worker_id = sys.argv[2]
count = int(sys.argv[3])
payload_bytes = int(sys.argv[4])
go_file = Path(sys.argv[5])

# Start barrier across processes: wait for the parent's go signal.
deadline = time.monotonic() + 30
while not go_file.exists() and time.monotonic() < deadline:
    time.sleep(0.01)

db = SessionDB(db_path=db_path)
session_id = f"mp-{worker_id}"
db.create_session(session_id, source="cli")
payload = "x" * payload_bytes
for i in range(count):
    db.append_message(session_id, role="user", content=payload)
    print(f"committed:{i}", flush=True)  # printed only after the commit returned
db.close()
print("done", flush=True)
"""


def _spawn_worker(db_path: Path, worker_id: str, count: int, payload_bytes: int,
                  go_file: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _WORKER_SCRIPT, str(db_path), worker_id, str(count),
         str(payload_bytes), str(go_file)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def _drain_lines(stream, out: "queue.Queue") -> None:
    for line in iter(stream.readline, ""):
        out.put(line.rstrip("\n"))
    out.put(None)  # EOF


class TestMultiProcessContention:
    WORKERS = 4
    MESSAGES_PER_WORKER = 20
    PAYLOAD_BYTES = 1024 * 1024

    def test_concurrent_process_writers_commit_everything(self, tmp_path):
        db_path = tmp_path / "state.db"
        go_file = tmp_path / "go"
        procs = [
            _spawn_worker(db_path, str(w), self.MESSAGES_PER_WORKER, self.PAYLOAD_BYTES, go_file)
            for w in range(self.WORKERS)
        ]
        go_file.touch()  # release all workers at once
        for proc in procs:
            out, _ = proc.communicate(timeout=180)
            assert proc.returncode == 0, f"worker failed (rc={proc.returncode}): {out[-2000:]}"

        db = get_session_db_provider(db_path=db_path)
        try:
            for w in range(self.WORKERS):
                assert db.message_count(f"mp-{w}") == self.MESSAGES_PER_WORKER
            assert db.session_count() >= self.WORKERS
            assert db._read_one("PRAGMA integrity_check")[0] == "ok"
        finally:
            db.close()


class TestMidWriteProcessKill:
    PAYLOAD_BYTES = 8 * 1024 * 1024  # wide commit window: the kill lands mid-write

    def test_sigkill_mid_write_leaves_database_consistent(self, tmp_path):
        db_path = tmp_path / "state.db"
        go_file = tmp_path / "go"
        go_file.touch()
        proc = _spawn_worker(db_path, "victim", 50, self.PAYLOAD_BYTES, go_file)

        lines: "queue.Queue" = queue.Queue()
        reader = threading.Thread(target=_drain_lines, args=(proc.stdout, lines), daemon=True)
        reader.start()

        # Wait until the worker has ACKNOWLEDGED at least two commits, then kill it hard —
        # with an 8MB payload per append the next write is in flight right now.
        markers = []
        deadline = time.monotonic() + 120
        while len(markers) < 2 and time.monotonic() < deadline:
            line = lines.get(timeout=max(deadline - time.monotonic(), 0.001))
            if line is not None and line.startswith("committed:"):
                markers.append(line)
        assert len(markers) >= 2, "worker never committed; cannot test a mid-write kill"
        proc.kill()  # SIGKILL / TerminateProcess — no cleanup handlers run
        assert proc.wait(timeout=30) != 0
        reader.join(timeout=10)

        # A fresh handle (new connection → WAL recovery path) must find the database
        # consistent, every acknowledged commit durable, and must keep accepting writes.
        db = get_session_db_provider(db_path=db_path)
        try:
            assert db._read_one("PRAGMA integrity_check")[0] == "ok"
            assert db.message_count("mp-victim") >= len(markers)
            db.append_message("mp-victim", role="user", content="post-crash write")
            assert db.message_count("mp-victim") >= len(markers) + 1
        finally:
            db.close()
