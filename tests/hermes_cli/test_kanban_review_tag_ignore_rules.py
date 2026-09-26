"""Regression test for t_e2cf891c: review/verifier-tagged kanban tasks must
dispatch their worker subprocess with --ignore-rules so it does not load the
profile's own MEMORY.md/USER.md/preloaded skills — the same isolation
delegate_task hardcodes for its children (tools/delegate_tool.py). This is
isolation only; it does not remove the task's own explicitly-forced --skills.
"""
from __future__ import annotations

from unittest.mock import patch

from hermes_cli.kanban_db import Task
from hermes_cli.kanban_db_dispatch import _is_review_tagged, _worker_argv, apply_review_tag


def _isolated_board(tmp_path, monkeypatch):
    """Point kanban at a fresh board under ``tmp_path``, unpinned from whatever
    board/DB/delegation env the caller happens to be running inside."""
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK",
                "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(var, raising=False)
    kb.init_db()


def _make_task(skills):
    return Task(
        id="t_test0001",
        title="test task",
        body=None,
        assignee="default",
        status="running",
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
        skills=skills,
    )


def _build_argv(task):
    with patch(
        "hermes_cli.kanban_db_dispatch._resolve_hermes_argv",
        return_value=["hermes"],
    ), patch(
        "hermes_cli.kanban_db_dispatch._resolve_worker_cli_toolsets",
        return_value=None,
    ):
        return _worker_argv(task, "default", None)


def test_review_tagged_task_gets_ignore_rules():
    task = _make_task(["kanban-independent-verification"])
    assert _is_review_tagged(task) is True

    argv = _build_argv(task)
    assert "--ignore-rules" in argv
    # --ignore-rules must not drop the task's own forced --skills.
    assert "--skills" in argv
    assert "kanban-independent-verification" in argv


def test_requesting_code_review_tag_also_gets_ignore_rules():
    task = _make_task(["requesting-code-review"])
    assert _is_review_tagged(task) is True
    assert "--ignore-rules" in _build_argv(task)


def test_untagged_task_does_not_get_ignore_rules():
    task = _make_task(["some-other-skill"])
    assert _is_review_tagged(task) is False

    argv = _build_argv(task)
    assert "--ignore-rules" not in argv
    assert "--skills" in argv


def test_task_with_no_skills_does_not_get_ignore_rules():
    task = _make_task(None)
    assert _is_review_tagged(task) is False

    argv = _build_argv(task)
    assert "--ignore-rules" not in argv
    assert "--skills" not in argv


# --- t_38792276: the tag a caller WRITES must be one the dispatcher READS -----
# Every producer of the tag (``hermes kanban create --review``, the swarm
# verifier node) goes through ``apply_review_tag``. The contract under test is
# the round-trip, not the literal: whatever that helper writes must make
# ``_is_review_tagged`` true and earn --ignore-rules at dispatch. That closes
# the drift hole where swarm hardcoded its own copy of the string.


def test_apply_review_tag_output_round_trips_to_ignore_rules():
    task = _make_task(apply_review_tag(None))
    assert _is_review_tagged(task) is True
    assert "--ignore-rules" in _build_argv(task)


def test_apply_review_tag_rejects_a_tag_the_dispatcher_does_not_read():
    """The helper exists to prevent drift, so it must refuse to write a tag
    that grants no isolation rather than silently producing a fake reviewer."""
    import pytest

    with pytest.raises(ValueError):
        apply_review_tag(None, tag="totally-not-a-review-skill")


def test_apply_review_tag_preserves_other_skills_and_is_idempotent():
    tagged = apply_review_tag(["humanizer"])
    assert tagged[0] == "humanizer"
    assert _is_review_tagged(_make_task(tagged)) is True
    # Re-tagging an already-review task must not stack duplicate tags.
    assert apply_review_tag(tagged) == tagged
    # An alternative existing review tag is respected rather than doubled up.
    assert apply_review_tag(["kanban-independent-verification"]) == ["kanban-independent-verification"]


def test_swarm_verifier_is_isolated_through_the_shared_tag(tmp_path, monkeypatch):
    """Regression: the swarm verifier's isolation must not depend on
    kanban_swarm repeating the tag literal. Real board, real create_swarm."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_swarm as ks

    _isolated_board(tmp_path, monkeypatch)
    with kbc.connect_closing() as conn:
        created = ks.create_swarm(
            conn, goal="probe the verifier tag",
            workers=[ks.SwarmWorkerSpec(profile="w", title="work", body="do it")],
            verifier_assignee="v", synthesizer_assignee="s",
        )
        verifier = kb.get_task(conn, created.verifier_id)

    assert verifier is not None
    assert _is_review_tagged(verifier) is True
    assert "--ignore-rules" in _build_argv(verifier)


def test_create_review_flag_produces_an_isolated_worker(tmp_path, monkeypatch):
    """End-to-end contract for ``hermes kanban create --review``: the flag must
    land a tag on the row that the dispatcher turns into --ignore-rules."""
    import argparse

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_parser
    from hermes_cli.kanban import _cmd_create

    _isolated_board(tmp_path, monkeypatch)
    root = argparse.ArgumentParser()
    kanban_parser.build_parser(root.add_subparsers(dest="command"))
    assert _cmd_create(root.parse_args(
        ["kanban", "create", "Independent review: probe", "--assignee", "default", "--review"])) == 0

    with kbc.connect_closing() as conn:
        row = conn.execute("SELECT id FROM tasks").fetchone()
        task = kb.get_task(conn, row["id"])

    assert task is not None
    assert _is_review_tagged(task) is True
    assert "--ignore-rules" in _build_argv(task)


def test_create_review_flag_unions_with_explicit_skills(tmp_path, monkeypatch):
    """``--review`` must ADD isolation, never replace the operator's own
    ``--skill`` choices — a reviewer that silently lost its domain skill would
    be isolated but useless."""
    import argparse

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_parser
    from hermes_cli.kanban import _cmd_create

    _isolated_board(tmp_path, monkeypatch)
    root = argparse.ArgumentParser()
    kanban_parser.build_parser(root.add_subparsers(dest="command"))
    assert _cmd_create(root.parse_args(
        ["kanban", "create", "Audit the thing", "--assignee", "default",
         "--skill", "humanizer", "--review"])) == 0

    with kbc.connect_closing() as conn:
        row = conn.execute("SELECT id FROM tasks").fetchone()
        task = kb.get_task(conn, row["id"])

    assert task is not None
    assert "humanizer" in (task.skills or [])
    assert _is_review_tagged(task) is True
    argv = _build_argv(task)
    assert "--ignore-rules" in argv
    assert "humanizer" in argv


# --- t_38792276: the NATIVE review lane must be isolated too ------------------
# The regression that motivated this card's rescope. `kanban_request_review`
# moves a card to the `review` column; the dispatcher force-loads `sdlc-review`
# for that lane. While `sdlc-review` was NOT in REVIEW_TAG_SKILLS,
# `_is_review_tagged` returned False for every such worker and the whole native
# review path ran WITHOUT --ignore-rules — a reviewer loading the implementer's
# own MEMORY.md/USER.md. No test covered the lane, so CI could not see it.


def test_review_lane_skill_is_a_recognised_review_tag():
    """The skill the lane force-loads must be one the isolation check reads."""
    from hermes_cli.kanban_db_dispatch import REVIEW_TAG_SKILL_LANE, REVIEW_TAG_SKILLS

    assert REVIEW_TAG_SKILL_LANE in REVIEW_TAG_SKILLS
    assert _is_review_tagged(_make_task([REVIEW_TAG_SKILL_LANE])) is True
    assert "--ignore-rules" in _build_argv(_make_task([REVIEW_TAG_SKILL_LANE]))


def test_review_lane_dispatch_spawns_an_isolated_worker(tmp_path, monkeypatch):
    """Drive the REAL dispatch path: a card in the review column must reach
    ``_worker_argv`` carrying --ignore-rules. Asserts on the argv the spawn
    actually receives, not on a reconstruction of it."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    _isolated_board(tmp_path, monkeypatch)
    captured: dict = {}

    def _spy_spawn(task, workspace, *, board=None):
        captured["skills"] = list(task.skills or [])
        captured["is_review_tagged"] = kbd._is_review_tagged(task)
        captured["argv"] = _build_argv(task)
        return 4242

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="implement the thing", assignee="default", workspace_kind="scratch")
        kb.claim_task(conn, task_id)
        assert kb.request_review(conn, task_id, summary="please check", force=True)
        assert kb.get_task(conn, task_id).status == "review"

        with patch("hermes_cli.kanban_db_dispatch._profile_exists_fn", return_value=None):
            kbd.dispatch_once(conn, spawn_fn=_spy_spawn, reconcile_orphans=False)

    assert captured, "review-lane task was never dispatched"
    assert kbd.REVIEW_TAG_SKILL_LANE in captured["skills"]
    assert captured["is_review_tagged"] is True, (
        "native review lane dispatched a worker that is not review-tagged — it "
        "would load the implementer's own MEMORY.md/USER.md")
    assert "--ignore-rules" in captured["argv"]


def test_review_lane_force_loads_its_skill_even_when_already_review_tagged(tmp_path, monkeypatch):
    """A card hand-tagged with another review skill must STILL get the lane's
    own ``sdlc-review`` playbook — isolation and force-load are separate jobs
    and an idempotent 'already tagged' shortcut must not skip the latter."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    _isolated_board(tmp_path, monkeypatch)
    captured: dict = {}

    def _spy_spawn(task, workspace, *, board=None):
        captured["skills"] = list(task.skills or [])
        return 4243

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="implement the thing", assignee="default", workspace_kind="scratch",
            skills=[kbd.REVIEW_TAG_SKILL_DEFAULT])
        kb.claim_task(conn, task_id)
        assert kb.request_review(conn, task_id, summary="please check", force=True)

        with patch("hermes_cli.kanban_db_dispatch._profile_exists_fn", return_value=None):
            kbd.dispatch_once(conn, spawn_fn=_spy_spawn, reconcile_orphans=False)

    assert captured, "review-lane task was never dispatched"
    assert kbd.REVIEW_TAG_SKILL_LANE in captured["skills"]
    assert kbd.REVIEW_TAG_SKILL_DEFAULT in captured["skills"]
