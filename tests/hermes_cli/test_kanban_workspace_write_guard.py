"""Regression: a ``kind:``-scheme-prefixed ``workspace_path`` must be refused at WRITE.

Seven estate cards died silently because ``--workspace dir:C:/...`` was stored
verbatim in ``tasks.workspace_path`` (the combined ``kind:path`` CLI argument was
never split). On Windows the scheme occupies position 0, so ``Path(...).is_absolute()``
is False even though the path underneath is absolute; the dispatcher raised
``non-absolute workspace_path`` on every tick, burned the retry budget, and blocked
the card permanently. The write reported CREATED -- the card was dead on the board.

What this file adds on top of the existing non-absolute creation guard:

* the refusal NAMES the scheme prefix (the old message pointed at a CWD problem
  that did not exist, which is what four operators had to reverse-engineer);
* the guard is exercised on BOTH write boundaries (``create_task`` and
  ``set_workspace_path``), because a check that only fires at creation cannot
  protect a writer that reaches the setter directly;
* a project-scoped card is asserted to still be creatable, pinning the guard's
  load-bearing placement AFTER project materialisation (see ``cf0eb4b59acc``:
  validating before it raised ValueError for every project card in the estate).

Everything goes through the real storage layer against a temp ``HERMES_HOME`` --
no source-text assertions (AGENTS.md: never read source code in tests).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _conn():
    """Open a connection to the isolated board DB (callers use it as ``with _conn()``)."""
    return kbc.connect_closing()


# The values that actually caused the incident. ``C:/...`` is absolute; the SAME
# string with a scheme prefix is not, on Windows -- that is the entire bug.
_WINDOWS_SCHEME_VALUES = [
    "dir:C:/Users/jON/virtengine-ops",
    "worktree:C:/Users/jON/virtengine-ops/virtengine",
    "scratch:C:/Users/jON/AppData/Local/hermes",
    "file:C:/Users/jON/virtengine-ops",
    # Case-insensitivity: the writer may normalise the scheme.
    "DIR:C:/Users/jON/virtengine-ops",
]


def test_scheme_prefix_makes_absolute_path_look_relative():
    """Pin the platform fact the whole defect rests on.

    A POSIX-only CI run can never observe this: ``dir:/x`` is a legal RELATIVE path
    there (a colon is an ordinary filename character), so an absolute-path check
    refuses the value for the WRONG REASON -- it looks like an ordinary relative path.
    That is why the guard must also match the prefix BY NAME.
    """
    absolute = Path("C:/Users/jON/virtengine-ops")
    assert absolute.is_absolute()

    scheme = Path("dir:C:/Users/jON/virtengine-ops")
    if sys.platform == "win32":
        # The Windows behaviour that made this a fleet-wide incident.
        assert not scheme.is_absolute()
        assert not Path("dir:C:/Users/jON/virtengine-ops").expanduser().is_absolute()
    else:
        # On POSIX the absolute-path check alone cannot identify the cause.
        assert os.path.isabs("dir:C:/Users/jON/virtengine-ops") is False


@pytest.mark.parametrize("value", _WINDOWS_SCHEME_VALUES)
def test_create_task_refuses_scheme_prefixed_workspace_path(kanban_home, value):
    """The write boundary rejects it -- before any row exists, not at spawn."""
    with pytest.raises(ValueError) as exc:
        with _conn() as conn:
            kb.create_task(
                conn, title="dead card", assignee="coder",
                workspace_kind="dir", workspace_path=value,
            )
    msg = str(exc.value)
    # The message must name the fix, not just refuse: this string is what an
    # operator sees instead of a card that silently blocks.
    assert "scheme prefix" in msg
    assert "workspace_kind" in msg
    # And it must point at the repaired value.
    kind, _, repaired = value.partition(":")
    assert repr(repaired) in msg, f"message does not name the fixed value: {msg}"
    assert repr(kind.lower()) in msg

    # Nothing was written: a refused card leaves no dead-on-arrival row behind.
    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE title = 'dead card'"
        ).fetchone()
    assert row["n"] == 0


def test_create_task_refuses_relative_workspace_path(kanban_home):
    """The sibling defect: any non-absolute path, refused at write for the same reason.

    This is the shape actually present on the live boards (a bare repo name such as
    ``virtengine``), so it is the more frequent half of the class.
    """
    with pytest.raises(ValueError, match="not absolute"):
        with _conn() as conn:
            kb.create_task(
                conn, title="relative card", assignee="coder",
                workspace_kind="dir", workspace_path="relative/dir",
            )
    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE title = 'relative card'"
        ).fetchone()
    assert row["n"] == 0


def test_set_workspace_path_refuses_scheme_prefixed_path(kanban_home):
    """The OTHER writer. ``set_workspace_path`` persists a resolved workspace onto a
    LIVE card, so a scheme arriving here would strand it on the next tick."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="live card", assignee="coder")
        with pytest.raises(ValueError, match="scheme prefix"):
            kbw.set_workspace_path(conn, tid, "dir:C:/Users/jON/virtengine-ops")

        row = conn.execute(
            "SELECT workspace_path FROM tasks WHERE id = ?", (tid,)
        ).fetchone()

    assert row["workspace_path"] is None, "the refused value was written anyway"


def test_set_workspace_path_refuses_relative_path(kanban_home):
    """Same defect, sibling shape: the setter is a write boundary too."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="live card 2", assignee="coder")
        with pytest.raises(ValueError, match="not absolute"):
            kbw.set_workspace_path(conn, tid, "virtengine")
        row = conn.execute(
            "SELECT workspace_path FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
    assert row["workspace_path"] is None


def test_set_workspace_path_accepts_a_valid_absolute_path(kanban_home, tmp_path):
    """The guard must not block legitimate writes -- a repaired card has to land."""
    target = tmp_path / "workspaces" / "t_real"
    target.mkdir(parents=True)
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="ok card", assignee="coder")
        kbw.set_workspace_path(conn, tid, target)
        row = conn.execute(
            "SELECT workspace_path FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
    assert Path(row["workspace_path"]) == target


def test_create_task_accepts_the_split_form_the_cli_actually_produces(kanban_home, tmp_path):
    """The legitimate end-to-end path: what ``_parse_workspace_flag`` stores.

    Guards against the write-boundary check being over-eager and killing every
    ``dir:``/``worktree:`` task the estate can create.
    """
    target = tmp_path / "ops"
    target.mkdir()
    with kbc.connect_closing() as conn:
        tid = kb.create_task(
            conn, title="good card", assignee="coder",
            workspace_kind="dir", workspace_path=str(target),
        )
        row = conn.execute(
            "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
    assert row["workspace_kind"] == "dir"
    assert Path(row["workspace_path"]).is_absolute()
    assert "dir:" not in row["workspace_path"]


def test_project_card_is_still_creatable(kanban_home, tmp_path):
    """Placement guard. A project-linked card legitimately presents
    ``kind='worktree', workspace_path=None`` before the txn materialises it
    (``_resolve_project_link`` defers the concrete path to the insert loop on
    purpose). Validating BEFORE materialisation raised ValueError for EVERY
    project-scoped card in the estate -- measured 2026-10-03 in ``cf0eb4b59acc``.
    This pins that the guard stays inside the txn, after materialisation.
    """
    from hermes_cli import projects_db as pdb

    repo = tmp_path / "projrepo"
    (repo / ".git").mkdir(parents=True)
    with pdb.connect_closing() as pconn:
        pid = pdb.create_project(
            pconn, name="Proj Repo", primary_path=str(repo),
        )
    with kbc.connect_closing() as conn:
        tid = kb.create_task(
            conn, title="project card", assignee="coder",
            workspace_kind="worktree", project_id=pid,
        )
        row = conn.execute(
            "SELECT workspace_kind, workspace_path, project_id FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
    assert row["project_id"] == pid
    assert row["workspace_kind"] == "worktree"
    # Materialised to a concrete <repo>/.worktrees/<id> path, or left for the
    # resolver -- either way never a refused value.
    assert row["workspace_path"] is None or Path(row["workspace_path"]).is_absolute()


def test_resolve_workspace_error_names_the_scheme_cause(kanban_home):
    """DONE WHEN: the spawn-time string an operator reverse-engineered now names the
    likely cause. Asserts on the REAL raised message from the real resolver with a
    scheme-prefixed row seeded by raw SQL (the bypass path the guard cannot see)."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="legacy dead card", assignee="coder")
        # Raw UPDATE reproduces the historical bypass: the defect predates the guard.
        conn.execute(
            "UPDATE tasks SET workspace_kind='dir', workspace_path=? WHERE id=?",
            ("dir:C:/Users/jON/virtengine-ops", tid),
        )
        conn.commit()
        task = kb.get_task(conn, tid)
        assert task.workspace_path == "dir:C:/Users/jON/virtengine-ops"

        with pytest.raises(ValueError) as exc:
            kbw.resolve_workspace(task)

    msg = str(exc.value)
    assert "non-absolute workspace_path" in msg
    assert "scheme prefix" in msg
    assert "workspace_kind" in msg


def test_board_default_workdir_with_a_scheme_prefix_is_refused(kanban_home):
    """A corrupt ``default_workdir`` in board meta would otherwise be inherited
    silently by every dir/worktree task on the board."""
    slug = "badboard"
    kb.create_board(slug, name="Bad Board")
    kb.write_board_metadata(slug, default_workdir="dir:C:/Users/jON/virtengine-ops")

    with kbc.connect_closing(board=slug) as conn:
        with pytest.raises(ValueError, match="scheme prefix"):
            kb.create_task(conn, title="inherits garbage", assignee="coder",
                           workspace_kind="dir", board=slug)


def test_guard_does_not_reject_unset_workspace(kanban_home):
    """Sanity: an unset workspace (the common scratch case) always passes, so the
    guard cannot regress normal task creation."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="scratch card", assignee="coder")
        row = conn.execute(
            "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
    assert row["workspace_kind"] == "scratch"
    assert row["workspace_path"] is None


def test_require_absolute_workspace_path_is_platform_independent(kanban_home, tmp_path):
    """The scheme test must not depend on ``is_absolute()``: on POSIX a scheme path
    is a legal relative path, so only the explicit prefix check names the cause."""
    for value in _WINDOWS_SCHEME_VALUES:
        with pytest.raises(ValueError, match="scheme prefix"):
            kbw.require_absolute_workspace_path(value, kind="dir", where="test")

    # A legitimate absolute path passes on both platforms.
    kbw.require_absolute_workspace_path(str(tmp_path), kind="dir", where="test")
    # Unset passes for the kinds whose resolver can supply one (scratch
    # materialises it, worktree falls back to the board's default_workdir).
    kbw.require_absolute_workspace_path(None, kind="scratch", where="test")
    kbw.require_absolute_workspace_path(None, kind="worktree", where="test")
    # BLANK is not unset: no resolver serves "" for any kind, so it is refused
    # for all three (pinned by the estate guard verify-workspace-path-creation.py
    # for dir AND worktree; measured RED 37/40 when scratch blank was allowed).
    for kind in ("scratch", "dir", "worktree"):
        with pytest.raises(ValueError, match="non-empty"):
            kbw.require_absolute_workspace_path("", kind=kind, where="test")
        with pytest.raises(ValueError, match="non-empty"):
            kbw.require_absolute_workspace_path("   ", kind=kind, where="test")


def test_deprecated_alias_delegates_to_the_same_guard(kanban_home):
    """One implementation, two public names: the alias must refuse identically, so a
    caller still using the old spelling cannot slip past the scheme check."""
    with pytest.raises(ValueError, match="scheme prefix"):
        kbw.require_spawnable_workspace_path(
            "dir:C:/Users/jON/virtengine-ops", kind="dir", where="test",
        )


def test_worktree_anchor_rejects_a_scheme_default_workdir_at_create(kanban_home):
    """The THIRD way the scheme defect reaches a card -- and the one that proves the
    guard covers a value NOBODY passed as an argument.

    ``create_task`` inherits a board ``default_workdir`` into ``workspace_path``
    (kanban_db.py:1341) when the caller passes none, so the only scheme-prefixed
    value here lives in board META, never in a ``--workspace`` argument. The card
    must therefore be refused at CREATE, by the same guard -- not born with the
    board's garbage in its own column.

    This is strictly stronger than the resolver message: the existing resolver-side
    branch (:624) can only be reached by a raw-SQL bypass, because the write guard
    now intercepts every create path. It is still patched below because a legacy
    row -- or any future writer that bypasses create_task -- reaches it, and the
    operator's only route out is that string.
    """
    kb.create_board("wdboard", name="WD Board")
    kb.write_board_metadata("wdboard", default_workdir="dir:C:/Users/jON/virtengine-ops")

    # Refused at create, naming the board value it inherited.
    with kbc.connect_closing(board="wdboard") as conn:
        with pytest.raises(ValueError) as exc:
            kb.create_task(conn, title="anchored card", assignee="coder",
                           workspace_kind="worktree", board="wdboard")
    assert "scheme prefix" in str(exc.value)
    assert "workspace_kind" in str(exc.value)

    # And the board's own anchor branch says the same thing, for the bypass path:
    # seed board.json the way the historical defect arrived (board metadata is a
    # JSON file, not a table). NOTE a clean board is used here because
    # ``write_board_metadata(..., default_workdir=None)`` means UNCHANGED (:589),
    # not cleared -- so a corrupt value cannot be removed through that API, which
    # is exactly why it is worth an explicit guard.
    kb.create_board("cleanboard", name="Clean Board")
    with kbc.connect_closing(board="cleanboard") as conn:
        tid = kb.create_task(conn, title="legacy anchor", assignee="coder",
                             workspace_kind="worktree", board="cleanboard")
    # Bypass the writer entirely, as the historical ad-hoc INSERTs did.
    board_json = kb.board_dir("cleanboard") / "board.json"
    meta = json.loads(board_json.read_text(encoding="utf-8"))
    meta["default_workdir"] = "dir:C:/Users/jON/virtengine-ops"
    board_json.write_text(json.dumps(meta), encoding="utf-8")

    with kbc.connect_closing(board="cleanboard") as conn:
        task = kb.get_task(conn, tid)
        assert task.workspace_path is None, "the anchor must be resolved at spawn"
        with pytest.raises(ValueError) as exc:
            kbw.resolve_workspace(task, board="cleanboard")

    msg = str(exc.value)
    assert "not " in msg and "absolute" in msg
    assert "scheme prefix" in msg
    assert "workspace_kind" in msg


def test_unset_worktree_path_is_still_creatable(kanban_home):
    """The over-refusal half of the guard. ``kind='worktree'`` with NO path is LEGAL:
    ``_resolve_worktree_workspace`` (:613) falls back to the board's
    ``default_workdir``, so the resolver would serve it. A guard that refuses it
    would kill every board-default-anchored worktree card in the estate -- an
    outage shipped as the fix, the same shape as ``cf0eb4b59acc``.

    ``kind='dir'`` with no path is genuinely an error (the resolver raises), so it
    is pinned as still refused -- the two kinds must not be collapsed.
    """
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="anchored worktree", assignee="coder",
                             workspace_kind="worktree")
        row = conn.execute(
            "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
    assert row["workspace_kind"] == "worktree"

    with kbc.connect_closing() as conn:
        with pytest.raises(ValueError, match="needs a non-empty workspace_path"):
            kb.create_task(conn, title="bare dir", assignee="coder",
                           workspace_kind="dir")


def test_legacy_scheme_row_still_fails_at_spawn_and_is_not_rewritten(kanban_home):
    """An existing row keeps failing at spawn until repaired -- the guard does not
    rewrite board state (that is a separate migration, per the card). The fix is
    discoverable from the error, which is the operator's route out."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="legacy", assignee="coder")
        conn.execute(
            "UPDATE tasks SET workspace_kind='dir', workspace_path=? WHERE id=?",
            ("dir:C:/Users/jON/virtengine-ops", tid),
        )
        conn.commit()
        task = kb.get_task(conn, tid)
        with pytest.raises(ValueError) as exc:
            kbw.resolve_workspace(task)
    # The repair is named in the message, not silently applied.
    assert "dir:" in str(exc.value)
    assert task.workspace_path == "dir:C:/Users/jON/virtengine-ops"