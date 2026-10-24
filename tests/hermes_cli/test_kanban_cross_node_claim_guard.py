"""Regression: the cross-node claim guard must be None-safe and fail-open.

Fleet outage reproduced 2026-10-02 09:12-09:20 on board ``hermes-orchestration``
(node CobradorWave).  The gateway-resident dispatcher tick calls
``_dispatch_once_locked(board=None)``; when ``KANBAN_SPAWN_CLAIM_CMD`` is set
(role ``59-dispatch-health`` writes it into the gateway systemd drop-in), the
guard string-built the claim command with ``board=None``::

    File ".../hermes_cli/kanban_db.py", line 109, in _cross_node_claim_allows
        cmd = tmpl.replace("{board}", board).replace("{task_id}", task_id)
    TypeError: replace() argument 2 must be str, not None

The ``TypeError`` is raised *before* the guarded ``subprocess.run``, so the
function's own fail-open ``except Exception`` never sees it.  Every tick then
claimed a card and died before spawning, leaving the card stuck ``running``
with a dead pid (reclaim-backoff loop).  The docstring contract is explicit:
"Fail-open on error: the guard must never wedge dispatch."
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


# ---------------------------------------------------------------------------
# None / empty identities must never reach the template substitution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "board, task_id",
    [
        (None, "t_x"),          # the live outage: gateway tick passes board=None
        ("b", None),
        (None, None),
        ("", "t_x"),
        ("b", ""),
    ],
)
def test_guard_allows_when_identity_is_missing(monkeypatch, board, task_id):
    """A guard with nothing to claim on must return True, not raise."""
    monkeypatch.setenv("KANBAN_SPAWN_CLAIM_CMD", "true {board} {task_id}")
    assert kbd._cross_node_claim_allows(board, task_id) is True


def test_guard_allows_when_env_unset(monkeypatch):
    monkeypatch.delenv("KANBAN_SPAWN_CLAIM_CMD", raising=False)
    assert kbd._cross_node_claim_allows("boardA", "t_1") is True


# ---------------------------------------------------------------------------
# The real (str, str) path must still actually run the claim command
# ---------------------------------------------------------------------------

def test_guard_substitutes_ids_and_runs_the_command(monkeypatch, tmp_path):
    out = tmp_path / "claim.log"
    monkeypatch.setenv(
        "KANBAN_SPAWN_CLAIM_CMD", "echo {board} {task_id} >> " + str(out)
    )
    assert kbd._cross_node_claim_allows("boardA", "t_1") is True
    assert out.read_text().split() == ["boardA", "t_1"]


def test_guard_declines_when_claim_command_exits_nonzero(monkeypatch):
    """Non-zero exit still means 'a peer holds it' — fail-closed on that path."""
    monkeypatch.setenv("KANBAN_SPAWN_CLAIM_CMD", "exit 3")
    assert kbd._cross_node_claim_allows("boardA", "t_1") is False


def test_guard_never_wedges_on_broken_template(monkeypatch):
    """Any unexpected failure building/running the command is fail-open."""
    monkeypatch.setenv("KANBAN_SPAWN_CLAIM_CMD", "true {board} {task_id}")
    monkeypatch.setattr(
        kbd.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert kbd._cross_node_claim_allows("boardA", "t_1") is True


# ---------------------------------------------------------------------------
# End-to-end: the dispatcher tick itself must survive board=None
# ---------------------------------------------------------------------------

def test_dispatch_tick_survives_guard_with_board_none(conn, monkeypatch):
    """``dispatch_once`` defaults to board=None — the exact outage path."""
    monkeypatch.setenv("KANBAN_SPAWN_CLAIM_CMD", "true {board} {task_id}")
    # The test HERMES_HOME is a tmp dir, so the assignee must be force-treated
    # as a real profile or the row is skipped as non-spawnable before the
    # guard is ever reached (which is how an earlier draft of this test
    # produced a false red).
    import hermes_cli.profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "profile_exists", lambda name: True)

    kb.create_task(conn, title="t", assignee="w")

    spawned: list = []

    def spy_spawn(task, workspace_path, board=None):
        spawned.append(getattr(task, "id", task))
        return 424242

    result = kbd.dispatch_once(conn, spawn_fn=spy_spawn, dry_run=False)

    assert spawned, "the guard must not wedge the tick when board is None"
    assert len(result.spawned) == 1
    assert result.spawned[0][0] == spawned[0]
    assert result.spawned[0][1] == "w"
