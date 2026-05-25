"""Multiprocess integrity stress tests for the Kanban SQLite backend."""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import sqlite3
import sys
import time
import traceback
from pathlib import Path


def _stress_worker(repo_root: str, db_path: str, idx: int, iterations: int, q) -> None:
    sys.path.insert(0, repo_root)
    from hermes_cli import kanban_db as kb

    try:
        random.seed(os.getpid() + idx)
        conn = kb.connect(Path(db_path))
        made: list[str] = []
        for i in range(iterations):
            op = i % 5
            if op == 0 or not made:
                tid = kb.create_task(
                    conn,
                    title=f"stress task {idx}-{i}",
                    body="temp stress regression",
                    assignee="default",
                    created_by=f"worker-{idx}",
                    initial_status="running",
                )
                made.append(tid)
            else:
                tid = random.choice(made)
                if op == 1:
                    kb.add_comment(conn, tid, f"worker-{idx}", f"comment {i}")
                elif op == 2:
                    kb.complete_task(conn, tid, result=f"done {idx}-{i}")
                elif op == 3:
                    kb.block_task(conn, tid, reason=f"blocked {idx}-{i}")
                else:
                    kb.add_comment(conn, tid, f"worker-{idx}", f"post {i}")
            if i % 11 == 0:
                time.sleep(random.random() / 1000)
        conn.close()
        q.put(("ok", idx, len(made)))
    except Exception:
        q.put(("err", idx, traceback.format_exc()))


def test_multiprocess_writers_preserve_sqlite_integrity(tmp_path):
    from hermes_cli import kanban_db as kb

    db_path = tmp_path / "kanban.db"
    conn = kb.connect(db_path)
    conn.close()

    procs = 4
    iterations = 80
    q = mp.Queue()
    repo_root = str(Path(__file__).resolve().parents[2])
    workers = [
        mp.Process(target=_stress_worker, args=(repo_root, str(db_path), i, iterations, q))
        for i in range(procs)
    ]
    for proc in workers:
        proc.start()
    for proc in workers:
        proc.join(60)
    for proc in workers:
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            raise AssertionError(f"stress worker timed out: pid={proc.pid}")

    results = [q.get(timeout=5) for _ in workers]
    failures = [r for r in results if r[0] != "ok"]
    assert not failures, failures

    raw = sqlite3.connect(db_path)
    try:
        assert raw.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert raw.execute("SELECT count(*) FROM tasks").fetchone()[0] == procs * (iterations // 5)
        assert raw.execute("SELECT count(*) FROM tasks WHERE id IS NULL OR created_at IS NULL").fetchone()[0] == 0
    finally:
        raw.close()
