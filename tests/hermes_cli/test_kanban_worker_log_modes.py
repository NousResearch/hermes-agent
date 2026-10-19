"""Worker logs and their directory must be created private, not at the umask.

Background
==========
A kanban worker log echoes everything the worker prints — prompts, tool
output, and files it is authoring — so it routinely carries live credentials.
``_open_worker_log`` used to ``open(log_path, "ab")`` with no mode into a
directory made by ``mkdir(parents=True, exist_ok=True)``, which lands the log
at 0o644 and the directory at 0o755 on a typical umask 0o022: any local user
can read a worker's pasted secrets. The dispatcher now creates both at the
private posture the rest of the credential state under HERMES_HOME already
uses (0o600 / 0o700 — same as ``hermes_cli/auth.py``), sets the file mode at
creation time rather than chmod'ing after, and tightens logs and directories
an older release already wrote when it next opens them.

These tests stay green only while a fresh log and directory end up 0o600 /
0o700 under umask 0o022, and an opened legacy 0o644 log is tightened without
losing its append history. POSIX-only — the mode-bit enforcement does not
exist on Windows.
"""

from __future__ import annotations

import os
import stat

import pytest

pytestmark = pytest.mark.platforms("posix")  # POSIX mode bits not enforced on Windows


def _task(tid: str = "t_0000abcd"):
    from hermes_cli import kanban_db as kb

    return kb.Task(
        id=tid,
        title="log modes",
        body=None,
        assignee="default",
        status="in_progress",
        priority=0,
        created_by=None,
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )


@pytest.fixture
def logs_dir(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb

    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    d = home / "kanban" / "logs"
    monkeypatch.setattr(kb, "worker_logs_dir", lambda board=None: d)
    return d


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_new_worker_log_and_dir_are_private(logs_dir):
    """A fresh log opens at 0o600 with its directory at 0o700 under umask 0o022."""
    from hermes_cli import kanban_db_dispatch as kbd

    old_umask = os.umask(0o022)  # make the exposure observable if it regresses
    try:
        f = kbd._open_worker_log(_task(), None)
        try:
            f.write(b"worker line\n")
        finally:
            f.close()
    finally:
        os.umask(old_umask)

    log = logs_dir / "t_0000abcd.log"
    assert _mode(log) == 0o600, (
        f"worker log mode 0o{_mode(log):o} != 0o600 — a log carrying a worker's "
        "pasted credentials is world-readable"
    )
    assert _mode(logs_dir) == 0o700, (
        f"worker log dir mode 0o{_mode(logs_dir):o} != 0o700 — siblings can list "
        "and open every task log"
    )
    assert log.read_bytes() == b"worker line\n"


def test_opening_a_legacy_log_tightens_it_and_keeps_the_history(logs_dir):
    """A 0o644 log/directory from an older release is tightened on open, append-only."""
    from hermes_cli import kanban_db_dispatch as kbd

    logs_dir.mkdir(parents=True)
    os.chmod(logs_dir, 0o755)
    log = logs_dir / "t_0000abcd.log"
    log.write_bytes(b"old run\n")
    os.chmod(log, 0o644)

    f = kbd._open_worker_log(_task(), None)
    try:
        f.write(b"new run\n")
    finally:
        f.close()

    assert log.read_bytes() == b"old run\nnew run\n", (
        "re-opening a worker log must append, never truncate — the rotate-first, "
        "open-after order is the existing contract"
    )
    assert _mode(log) == 0o600
    assert _mode(logs_dir) == 0o700
