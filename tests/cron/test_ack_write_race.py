"""Regression test for t_12a8ecc4: cron external-worker ack file partial-read race.

Root cause: `_run_external_worker_payload` created the ack file with
O_CREAT|O_EXCL directly at its final path, so the directory entry (and thus
`Path.exists()`) became visible to a polling reader before any JSON bytes were
written. `_launch_external_cron_worker`'s poll loop treats `exists()` as
"ready" and immediately `json.loads(ack_path.read_text())`s it, so a reader
that wins the race against the writer's own scheduling latency sees a
zero-length file and raises `json.decoder.JSONDecodeError`.

Fix: write to a sibling `<ack_path>.tmp-<pid>` and `os.replace()` it into
place. `os.replace`/rename is atomic on POSIX, so a concurrent reader can only
ever observe "no file yet" or "the fully written file" -- never a partial one.

This test exercises the *actual* writer/reader shapes as separate OS
processes (mirroring the production architecture: the ack writer is a
spawned `python -m cron.scheduler --external-worker-file ...` child, and the
poller is the parent gateway process), because the bug is a cross-process
filesystem-visibility race that cannot be observed with in-process
monkeypatching alone.
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Mirrors cron/scheduler.py's `_run_external_worker_payload` ack-write block
# (post-fix, lines ~3255-3280): write-temp-then-replace.
_WRITER_SRC = r"""
import os, json, sys, time
ack_path = sys.argv[1]
delay = float(sys.argv[2])
ack_tmp_path = ack_path + f".tmp-{os.getpid()}"
fd = os.open(ack_tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as ack_file:
    time.sleep(delay)  # models scheduler preemption between open() and dump()
    json.dump({"pid": os.getpid(), "execution_id": "EXEC123"}, ack_file)
    ack_file.flush()
    os.fsync(ack_file.fileno())
os.replace(ack_tmp_path, ack_path)
"""


def _poll_ack(ack_path: Path, deadline_s: float = 5.0):
    """Mirrors cron/scheduler.py `_launch_external_cron_worker`'s poll body
    (lines ~3134-3152): exists() -> read_text() -> json.loads(), unchanged by
    this fix (the fix is entirely on the writer side)."""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if ack_path.exists():
            try:
                return ("OK", json.loads(ack_path.read_text(encoding="utf-8")))
            except Exception as e:
                return ("EXC", e)
            finally:
                ack_path.unlink(missing_ok=True)
        time.sleep(0.01)
    return ("TIMEOUT", None)


@pytest.mark.parametrize("trial", range(25))
def test_ack_writer_never_exposes_partial_file(tmp_path, trial):
    """GREEN: with the temp+rename writer, a reader racing the writer at a
    15ms open-to-flush delay never observes a partial/unreadable ack file."""
    ack_path = tmp_path / f"trial-{trial}.ready"
    writer = subprocess.Popen([sys.executable, "-c", _WRITER_SRC, str(ack_path), "0.015"])
    try:
        outcome, detail = _poll_ack(ack_path, deadline_s=5.0)
    finally:
        writer.wait(timeout=5)

    assert outcome == "OK", (
        f"expected a clean, fully-formed ack read but got {outcome!r} ({detail!r}); "
        "a partial read here means the atomic-rename fix regressed"
    )
    assert detail == {"pid": writer.pid, "execution_id": "EXEC123"}
