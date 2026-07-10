"""Repro: auto-decompose ignores the board's ``default_workdir``.

Reported by @SenorStefan (X, 2026-05-30, replying to @tonysimons_'s
"How To Dominate Projects With Hermes Agent Kanban Board" article):

  "Still believe there is a live bug though with the auto decomposer
   creating everything as scratch and dependent tasks are losing access
   to outputs from predecessor tasks!"

Failure chain (all confirmed against current main):

1. ``hermes kanban create --triage <title>`` defaults to
   ``--workspace scratch`` (kanban.py p_create default), so the triage
   root row is ``workspace_kind='scratch'`` / ``workspace_path=NULL``.

2. ``create_task`` only inherits the board's ``default_workdir`` for the
   PERSISTENT workspace kinds (``dir`` / ``worktree``) — the scratch kind
   is deliberately excluded to avoid #28818 (rmtree-ing the user's source
   tree on completion). So a triage root NEVER picks up the board default.

3. ``decompose_triage_task`` copies the *root's* workspace onto every
   child (scratch / NULL), NOT the board default.

4. ``resolve_workspace`` maps each scratch child to its own isolated
   ``<board-root>/workspaces/<child-id>/`` dir. Sibling A's output lives
   in a directory sibling B never sees → the dependent child "loses
   access to outputs from predecessor tasks."

The board-level ``--default-workdir`` knob that @SenorStefan found "would
have saved my kanban board" is exactly the intended cure — but the
decompose path bypasses it. These tests assert the CURED behavior and
therefore FAIL on current main.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_decompose_inherits_board_default_workdir_for_scratch_root(
    kanban_home, tmp_path
):
    """A scratch triage root on a board WITH a default_workdir must fan
    out into children anchored on that shared project dir — not isolated
    per-task scratch dirs.

    This is the core @SenorStefan repro: the operator set a board
    default_workdir precisely so decomposed work lands in one visible,
    shared tree, but the fan-out ignores it.
    """
    project = tmp_path / "myproject"
    project.mkdir()
    kb.create_board(
        "proj", name="Project", default_workdir=str(project)
    )

    with kb.scoped_current_board("proj"):
        with kb.connect(board="proj") as conn:
            # Exactly what `hermes kanban create --triage` produces:
            # default --workspace scratch, no explicit path.
            tid = kb.create_task(
                conn,
                title="ship a feature",
                workspace_kind="scratch",
                triage=True,
                board="proj",
            )
        with kb.connect(board="proj") as conn:
            child_ids = kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orchestrator",
                children=[
                    {"title": "survey", "body": "gather facts"},
                    {"title": "draft", "body": "write it up", "parents": [0]},
                ],
                author="decomposer",
            )
        assert child_ids and len(child_ids) == 2

        with kb.connect(board="proj") as conn:
            survey = kb.get_task(conn, child_ids[0])
            draft = kb.get_task(conn, child_ids[1])

        # Both children must be anchored on the board's shared project
        # tree so the draft can read the survey's output.
        survey_ws = kb.resolve_workspace(survey, board="proj")
        draft_ws = kb.resolve_workspace(draft, board="proj")

        assert survey_ws == draft_ws == project, (
            "decomposed children ignored the board default_workdir: "
            f"survey={survey_ws} draft={draft_ws} board_default={project}"
        )


def test_decompose_scratch_children_share_a_workspace_when_board_default_set(
    kanban_home, tmp_path
):
    """The observable symptom, stated as a workspace-equality invariant.

    Whatever the mechanism, when the board has a default_workdir the
    predecessor and its dependent successor must resolve to the SAME
    directory so the handoff artifact is visible. On current main they
    resolve to disjoint per-task scratch dirs.
    """
    project = tmp_path / "shared"
    project.mkdir()
    kb.create_board("shared", default_workdir=str(project))

    with kb.scoped_current_board("shared"):
        with kb.connect(board="shared") as conn:
            tid = kb.create_task(
                conn, title="pipeline", workspace_kind="scratch",
                triage=True, board="shared",
            )
        with kb.connect(board="shared") as conn:
            child_ids = kb.decompose_triage_task(
                conn, tid, root_assignee="orch",
                children=[
                    {"title": "predecessor"},
                    {"title": "successor", "parents": [0]},
                ],
                author="decomposer",
            )
        with kb.connect(board="shared") as conn:
            pred = kb.get_task(conn, child_ids[0])
            succ = kb.get_task(conn, child_ids[1])

        pred_ws = kb.resolve_workspace(pred, board="shared")
        succ_ws = kb.resolve_workspace(succ, board="shared")
        assert pred_ws == succ_ws, (
            "dependent successor cannot see predecessor output: "
            f"predecessor workspace {pred_ws} != successor workspace {succ_ws}"
        )


def test_decompose_no_board_default_still_scratch(kanban_home):
    """Guard the non-regression: a board with NO default_workdir keeps the
    current scratch fan-out behavior (nothing to inherit)."""
    kb.create_board("plain")
    with kb.scoped_current_board("plain"):
        with kb.connect(board="plain") as conn:
            tid = kb.create_task(
                conn, title="x", workspace_kind="scratch",
                triage=True, board="plain",
            )
        with kb.connect(board="plain") as conn:
            child_ids = kb.decompose_triage_task(
                conn, tid, root_assignee="orch",
                children=[{"title": "s1"}], author="decomposer",
            )
        with kb.connect(board="plain") as conn:
            t = kb.get_task(conn, child_ids[0])
    assert t.workspace_kind == "scratch"
    assert t.workspace_path is None


def test_decompose_explicit_board_beats_ambient_current_board(
    kanban_home, tmp_path
):
    """P1 (Greptile #276): the board used for the default_workdir lookup
    must be the one the caller's connection is scoped to, NOT whatever
    get_current_board() ambiently resolves to.

    Reproduce the context leak: the caller operates on board 'proj'
    (with a default_workdir), but the ambient current board points at a
    DIFFERENT board 'other'. Passing board='proj' explicitly must upgrade
    children into proj's dir; the ambient 'other' must not leak in.
    """
    proj = tmp_path / "proj_tree"
    proj.mkdir()
    other = tmp_path / "other_tree"
    other.mkdir()
    kb.create_board("proj", default_workdir=str(proj))
    kb.create_board("other", default_workdir=str(other))

    # Ambient current board is 'other' — the wrong one for this caller.
    with kb.scoped_current_board("other"):
        with kb.connect(board="proj") as conn:
            tid = kb.create_task(
                conn, title="scoped root", workspace_kind="scratch",
                triage=True, board="proj",
            )
        with kb.connect(board="proj") as conn:
            child_ids = kb.decompose_triage_task(
                conn, tid, root_assignee="orch",
                children=[{"title": "c1"}, {"title": "c2", "parents": [0]}],
                author="decomposer",
                board="proj",  # explicit — must win over ambient 'other'
            )
        with kb.connect(board="proj") as conn:
            for cid in child_ids:
                t = kb.get_task(conn, cid)
                assert t.workspace_kind == "dir", (
                    f"child {cid} not upgraded to dir: {t.workspace_kind}"
                )
                assert t.workspace_path == str(proj), (
                    f"child {cid} leaked to wrong board: {t.workspace_path} "
                    f"(expected proj tree {proj}, ambient was 'other' -> {other})"
                )
