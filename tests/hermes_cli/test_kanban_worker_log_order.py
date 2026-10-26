"""A worker's exit trailer must be the LAST line of its log.

The dispatcher opens the card's log once and hands it to the worker as both streams
(``stdout=log_f, stderr=subprocess.STDOUT``). The reply goes to stdout, which is
block-buffered when it is a file; the trailer goes to stderr with ``flush=True``. An
unflushed reply therefore reached the file only at interpreter exit, after the trailer,
and the log read out of order:

    Session: local-…
    [kanban-worker-exit] rc=0
    <the reply>

The trailer is the dead-worker sweep's witness of how the worker exited (it never reaps
the worker), so a reader that checks the log's tail missed it exactly then.

Both exit paths are covered: the quiet one-shot worker (``exit_single_query``) and the
owner-admission submitter (``kanban_worker_client.main``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER

REPO_ROOT = Path(__file__).resolve().parents[2]

# The reply is written the way each path writes it, then the trailer, with stdout and
# stderr pointed at ONE file — the dispatcher's arrangement, and the only one that shows
# the buffering difference between the two streams.
_ONE_SHOT = """
import os

os.environ["HERMES_KANBAN_TASK"] = "t_order"
from hermes_cli.quiet_single_query import exit_single_query

print("the attempt reply")
exit_single_query(0)
"""

_WORKER_CLIENT = """
from hermes_cli import kanban_worker_client


def _fake_submit():
    print("the attempt reply")
    return 0


kanban_worker_client._submit = _fake_submit
raise SystemExit(kanban_worker_client.main())
"""


def _log_lines(program: str, tmp_path: Path) -> list[str]:
    """Run *program* with both streams on one file; return its non-empty lines."""
    log = tmp_path / "worker.log"
    env = {k: v for k, v in os.environ.items() if not k.startswith("HERMES_KANBAN_")}
    env["PYTHONPATH"] = str(REPO_ROOT)
    with open(log, "wb") as fh:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
    assert proc.returncode == 0
    return [line for line in log.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def test_one_shot_worker_exit_trailer_is_the_logs_last_line(tmp_path):
    lines = _log_lines(_ONE_SHOT, tmp_path)
    assert lines.index("the attempt reply") == len(lines) - 2
    assert lines[-1].startswith(f"{KANBAN_WORKER_EXIT_TRAILER}0")


def test_managed_worker_client_exit_trailer_is_the_logs_last_line(tmp_path):
    lines = _log_lines(_WORKER_CLIENT, tmp_path)
    assert lines.index("the attempt reply") == len(lines) - 2
    assert lines[-1].startswith(f"{KANBAN_WORKER_EXIT_TRAILER}0")
