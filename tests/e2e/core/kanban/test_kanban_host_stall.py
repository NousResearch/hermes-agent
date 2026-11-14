"""Integration — a host-side stall on a REAL card must book through the infrastructure seam:
the card is released without spending ``consecutive_failures``, never blocked, while the
``infrastructure`` marker keeps the outage visible in ``hermes kanban runs``.

Everything except the stall itself is the shipped code running for real: the card is created
and claimed through the CLI's own functions, bound to a real submitter process with the
dispatcher's own ``_set_worker_pid``, that submitter runs ``kanban_worker_client.main()`` to
completion (exit 74 + its log trailer), the booking happens in a real ``hermes kanban
dispatch`` tick, and the verdict is read back with the real ``hermes kanban runs`` /
``archive`` commands.

The ONLY seam patched is the transport: ``connect_gateway`` raises the owner's own transient
``GatewayUnavailableError`` and the submitter's wait is pinned to zero — the "owner never
answered" simulation the fix targets. Every line below that seam (exit-code verdict,
trailer, classification, reclaim, bookkeeping) is the code under review.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from tests.e2e.core.kanban._helpers import Board

pytestmark = [
    pytest.mark.skipif(sys.platform != "linux", reason="worker liveness reads /proc"),
    pytest.mark.live_system_guard_bypass,  # teardown may signal the spawned submitter
]

PY = sys.executable
INFRASTRUCTURE_EXIT = 74  # KANBAN_INFRASTRUCTURE_EXIT_CODE

# Bookkeeping on the scratch board through the real kanban_db functions, from a child process so
# no test process ever holds the board the dispatcher ticks against.
HELPER = """
import sqlite3, sys
from hermes_cli import kanban_db as kb

mode = sys.argv[1]
conn = sqlite3.connect(sys.argv[2])
conn.row_factory = sqlite3.Row
if mode == "claim":
    task = kb.claim_task(conn, sys.argv[3])
    assert task is not None, "claim refused"
    print("%s\\t%s" % (task.current_run_id, task.claim_lock))
elif mode == "bind":
    from hermes_cli.kanban_db_dispatch import _set_worker_pid
    # _set_worker_pid opens its own write transaction.
    _set_worker_pid(conn, sys.argv[3], int(sys.argv[4]),
                    run_id=int(sys.argv[5]), claim_lock=sys.argv[6])
    print("bound")
elif mode == "where":
    print(kb.kanban_db_path())
    print(kb.worker_log_path(sys.argv[3]))
    print(kb.get_current_board())
"""

# The real submitter, with the transport seam replaced: the owner drains forever and the
# wait is zero, so the run dies exactly like a stalled owner does in production.
DRIVER = """
import sys, time
from contextlib import asynccontextmanager

from hermes_cli import gateway_client, kanban_worker_client

# A window for the parent to record this pid as the claim's worker before we exit.
time.sleep(2.0)


@asynccontextmanager
async def _stall(*_args, **_kwargs):
    raise gateway_client.GatewayUnavailableError(
        "Gateway draining: simulated owner stall", "draining")
    yield  # unreachable: the raise above is what makes this an async generator


gateway_client.connect_gateway = _stall
kanban_worker_client.OWNER_WAIT_S = 0.0  # the owner's silence is total: the wait ends at once
sys.exit(kanban_worker_client.main())
"""


def _run(board: Board, args: list[str], timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run([PY, *args], cwd=str(board.root), env=board.env(),
                          capture_output=True, text=True, timeout=timeout, check=False)


def _helper(board: Board, *args: str) -> list[str]:
    proc = _run(board, ["-c", HELPER, *args])
    assert proc.returncode == 0, f"helper {args[0]} rc={proc.returncode}\n{proc.stderr}"
    return proc.stdout.strip().splitlines()


def test_a_host_stall_books_through_the_infrastructure_seam(tmp_path: Path) -> None:
    board = Board(tmp_path, "http://127.0.0.1:1/v1")
    # An assignee no profile answers for: the tick must book the reclaim, never spawn a worker
    # that would race the assertions below.
    tid = board.create("throwaway: host stall must not spend the retry budget",
                       "--assignee", "no-such-profile-x1")
    db, log_path, current_board = _helper(board, "where", str(board.db_path), tid)
    log_path = Path(log_path)

    # 1. Claim the card the real way, then bind a real submitter pid to that claim.
    run_id, claim_lock = _helper(board, "claim", db, tid).pop().split("\t")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    driver = tmp_path / "host_stall_driver.py"
    driver.write_text(DRIVER, encoding="utf-8")
    env = board.env()
    env.update({
        "HERMES_KANBAN_DB": db,
        "HERMES_KANBAN_BOARD": current_board,
        "HERMES_KANBAN_TASK": tid,
        "HERMES_KANBAN_RUN_ID": str(run_id),
        "HERMES_KANBAN_CLAIM_LOCK": claim_lock,
    })
    with open(log_path, "ab") as log:
        proc = subprocess.Popen([PY, str(driver)], cwd=str(board.root), env=env,
                                stdout=subprocess.DEVNULL, stderr=log)
        _helper(board, "bind", db, tid, str(proc.pid), run_id, claim_lock)
        rc = proc.wait(timeout=120)

    # 2. The submitter's own verdict: exit 74 and its trailer, with the host's — not the
    # owner's — label on the wire.
    log_text = board.worker_log(tid)
    assert rc == INFRASTRUCTURE_EXIT, f"submitter exited {rc}\n--- log ---\n{log_text[-4000:]}"
    assert f"[kanban-worker-exit] rc={INFRASTRUCTURE_EXIT}" in log_text, log_text[-2000:]
    assert "Kanban host unavailable" in log_text
    assert "Kanban owner refused" not in log_text

    # 3. A REAL dispatch tick books it: released, never blocked, and the card's budget intact.
    tick = board.dispatch()
    assert tick.get("crashed") == [tid], tick
    assert tick.get("auto_blocked") == [], tick
    assert not board.events(tid, "gave_up"), board.diag(tid)
    assert not board.events(tid, "blocked"), board.diag(tid)

    task = board.task(tid)
    assert task["status"] == "ready", task["status"]      # released, NOT left running
    assert task["consecutive_failures"] == 0, task          # the budget was not spent
    assert "requeued without counting a failure" in (task["last_failure_error"] or "")

    # 4. The outage is visible: the run carries the infrastructure marker and its exit class.
    run = board.runs(tid)[-1]
    assert run["outcome"] == "crashed", run
    meta = json.loads(run["metadata"] or "{}")
    assert meta.get("infrastructure") is True, meta
    assert meta.get("exit_kind") == "infrastructure_failure", meta
    runs_out = board.cli("runs", tid).stdout
    assert re.search(r"\bcrashed\b", runs_out), runs_out
    board.cli("archive", tid)
    assert board.task(tid)["status"] == "archived"
