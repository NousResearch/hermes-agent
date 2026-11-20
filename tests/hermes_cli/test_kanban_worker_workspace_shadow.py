"""A dispatched worker client must import Hermes from the pinned install, never the workspace.

Regression: dispatcher spawn onto a *worktree* workspace of the Hermes checkout. The
worker client is spawned as ``sys.executable -m hermes_cli.kanban_worker_client`` with
cwd=task workspace, and the spawn pins the dispatcher's checkout on PYTHONPATH
(#122299, #122487, #122500) — but a bare ``-m`` also prepends the child's cwd, and the
cwd is the task workspace. A worktree workspace is a
full copy of the source, so it shadowed the pin: the child booted the WORKTREE's
``hermes_bootstrap``, which resolves a fresh install root with no committed dependency
environment, and on a PM store interpreter the boot died with
"no dependency environment is committed for this install"
(``pm/environments.py::_require_own_dependencies``) before the claim was ever submitted —
every worktree card crashed at bootstrap with zero worker turns.

The contract: the PYTHONPATH pin is the whole import contract for the module child;
the workspace directory must never enter its ``sys.path``. The spawn therefore uses the
safe-path flag (``-P``), the same guarantee ``hermes_bootstrap.harden_import_path``
applies to the modules that follow it.
"""
from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest

from tests.waiting import wait_until

_DECOY = (
    "import pathlib\n"
    "pathlib.Path({sentinel!r}).write_text({name!r} + ' booted from the workspace tree')\n"
    "raise SystemExit(9)\n"
)


def _make_task(kb, **over):
    fields = dict(
        id="t_shadow", title="workspace shadow", body=None, assignee="shadow", status="running",
        priority=0, created_by="test", created_at=0, started_at=0, completed_at=None,
        workspace_kind="worktree", claim_lock=None, claim_expires=None, tenant=None, branch_name=None,
    )
    fields.update(over)
    return kb.Task(**fields)


@pytest.mark.platforms("posix")
def test_dispatch_workspace_cannot_shadow_the_pinned_hermes_tree(tmp_path, monkeypatch):
    """The real module child spawned into a workspace carrying a decoy Hermes tree must
    never import it: the decoy boot would exit 9 and leave a sentinel; the real client
    boots from the pinned tree and exits on its missing claim instead."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("HERMES_BIN", raising=False)

    workspace = tmp_path / "worktree"
    (workspace / "hermes_cli").mkdir(parents=True)
    sentinel = tmp_path / "decoy-booted"
    (workspace / "hermes_bootstrap.py").write_text(
        _DECOY.format(sentinel=str(sentinel), name="hermes_bootstrap"), encoding="utf-8")
    (workspace / "hermes_cli" / "__init__.py").write_text(
        _DECOY.format(sentinel=str(sentinel), name="hermes_cli"), encoding="utf-8")
    (workspace / "hermes_cli" / "kanban_worker_client.py").write_text(
        _DECOY.format(sentinel=str(sentinel), name="kanban_worker_client"), encoding="utf-8")

    log_path = tmp_path / "worker.log"
    monkeypatch.setattr(kbd, "_open_worker_log", lambda *_a, **_k: open(log_path, "wb"))
    # The systemd scope wrap is orthogonal to import resolution — and it refuses a fixture
    # claim with no run id, which is exactly how this test makes the real client exit fast.
    monkeypatch.setattr(kbd, "_restart_safe_worker_argv", lambda _task, command: command)

    # No run id / claim lock in the env: the REAL client boots fully, then exits 1 on the
    # missing identity before any gateway dial — the fastest honest proof it got past import.
    pid = kbd._default_spawn(_make_task(kb, workspace_path=str(workspace)), str(workspace))
    assert pid is not None

    def _reaped():
        # Tuple-wrapped: exit status 0 is falsy and would never end the wait.
        wpid, st = os.waitpid(pid, os.WNOHANG)
        return (st,) if wpid == pid else None

    try:
        result = wait_until(_reaped, timeout=120, what="worker child to exit")
    except AssertionError:
        os.killpg(pid, signal.SIGKILL)  # start_new_session=True: the pgid is the child pid
        os.waitpid(pid, 0)
        raise
    assert result is not None  # wait_until returns only a truthy value
    (status,) = result

    assert not sentinel.exists(), (
        "the worker client imported Hermes from the task workspace instead of the pinned "
        "install root; a worktree workspace reboots a fresh install root with no committed "
        "dependency environment and dies at pm/environments.py before reading the card")
    assert os.waitstatus_to_exitcode(status) == 1, log_path.read_text()
    log = log_path.read_text()
    assert "Kanban owner refused" in log, log  # the real client ran and answered with its own words
    # The dead-worker sweep books the exit from this trailer — the client's boot contract,
    # asserted through its own constant.
    from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER
    assert f"{KANBAN_WORKER_EXIT_TRAILER}1" in log, log
