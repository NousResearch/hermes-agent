"""Per-task approval policy (#29457): flagged completions park in review.

The contract these tests pin:

* ``complete_task`` on a task carrying ``approval_required`` does NOT mark
  it done — it routes through the SAME first-class review machinery
  (:func:`request_review`) so the work parks in ``review``, reassigned to
  the designated approver (or left for a human when none is named).
* Completing a task already IN ``review`` is the sign-off act itself and
  is never re-gated — the human/agent approval verbs are unchanged.
* The completer identity check: the designated approver completing gets
  normal completion; anyone else (implementer, anonymous user) triggers
  the park. With no ``actor`` passed, the task's current assignee is the
  proxy — before parking that is always the implementer, once parked it
  is the approver.
* Unflagged tasks are untouched: same states, same events, same outcomes.
* The policy is editable mid-flight (``set_approval_policy`` / CLI
  ``set-approval`` / dashboard PATCH) and is consulted at completion time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty board, one approver profile and
    one installable approver skill."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # A real profile dir so create-time validation accepts the approver.
    (home / "profiles" / "reviewer-a").mkdir(parents=True)
    # A real skill dir so approver_skill validation resolves it.
    skill = home / "skills" / "review-checklist"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: review-checklist\ndescription: Walk the sign-off checklist.\n---\n\nChecklist.\n",
        encoding="utf-8",
    )
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


def _row(conn, tid):
    return conn.execute(
        "SELECT status, assignee, approval_required, approver_profile, "
        "approver_skill, current_run_id FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()


def _events(conn, tid, kind=None):
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
        (tid,),
    ).fetchall()
    out = [
        (r["kind"], json.loads(r["payload"]) if r["payload"] else None)
        for r in rows
    ]
    if kind is not None:
        out = [e for e in out if e[0] == kind]
    return out


def _last_run(conn, tid):
    return conn.execute(
        "SELECT status, outcome, summary, metadata FROM task_runs "
        "WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()


def _claimed_flagged_task(
    conn,
    *,
    title="impl a feature",
    assignee="builder",
    approver="reviewer-a",
    skill=None,
):
    tid = kb.create_task(
        conn,
        title=title,
        assignee=assignee,
        approval_required=True,
        approver_profile=approver,
        approver_skill=skill,
    )
    task = kb.claim_task(conn, tid)
    assert task is not None
    return tid


# ---------------------------------------------------------------------------
# The gate: flagged completion parks in review instead of done
# ---------------------------------------------------------------------------


def test_flagged_complete_parks_in_review_assigned_to_approver(kanban_home):
    with kb.connect() as conn:
        tid = _claimed_flagged_task(conn)
        run_id = kb.get_task(conn, tid).current_run_id

        ok = kb.complete_task(
            conn, tid,
            summary="Implementation complete",
            expected_run_id=run_id,
        )
        assert ok is True

        row = _row(conn, tid)
        assert row["status"] == "review"
        # Reassigned to the designated approver — this is what makes the
        # dispatcher spawn them as the reviewer.
        assert row["assignee"] == "reviewer-a"

        # Rode the existing review machinery: closed run + standard event.
        run = _last_run(conn, tid)
        assert run["outcome"] == "review_requested"
        rr = _events(conn, tid, kind="review_requested")
        assert len(rr) == 1
        payload = rr[0][1]
        assert payload["implementer"] == "builder"
        assert payload["reviewer"] == "reviewer-a"
        assert payload["summary"] == "Implementation complete"

        # The park reason is auditable on the handoff run's metadata.
        run_meta = json.loads(run["metadata"]) if run["metadata"] else {}
        assert run_meta["approval_policy"]["approver_profile"] == "reviewer-a"

        # NOT completed: no completed event, nothing marked done.
        assert _events(conn, tid, kind="completed") == []
        assert kb.get_task(conn, tid).completed_at is None


def test_park_without_approver_waits_for_human_and_keeps_assignee(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="human sign-off", assignee="builder",
            approval_required=True,
        )
        task = kb.claim_task(conn, tid)
        assert task is not None

        assert kb.complete_task(
            conn, tid, expected_run_id=task.current_run_id
        )

        row = _row(conn, tid)
        assert row["status"] == "review"
        # No designated agent: no reassignment, human pulls the lane.
        assert row["assignee"] == "builder"


def test_unflagged_task_is_byte_identical(kanban_home):
    with kb.connect() as conn:
        plain = kb.create_task(conn, title="plain work", assignee="builder")
        task = kb.claim_task(conn, plain)
        assert task is not None

        ok = kb.complete_task(
            conn, plain,
            summary="done directly",
            expected_run_id=task.current_run_id,
        )
        assert ok is True

        row = _row(conn, plain)
        assert row["status"] == "done"
        assert row["assignee"] == "builder"
        run = _last_run(conn, plain)
        assert run["outcome"] == "completed"
        assert len(_events(conn, plain, kind="completed")) == 1
        assert _events(conn, plain, kind="review_requested") == []


# ---------------------------------------------------------------------------
# Approval verbs: approver completes -> done; human approve unchanged
# ---------------------------------------------------------------------------


def test_designated_approver_completing_marks_done(kanban_home):
    with kb.connect() as conn:
        tid = _claimed_flagged_task(conn)
        implementer_run = kb.get_task(conn, tid).current_run_id
        assert kb.complete_task(
            conn, tid, summary="work handed off",
            expected_run_id=implementer_run,
        )
        assert _row(conn, tid)["status"] == "review"

        # Approver claims from the review lane (review -> running).
        review_claim = kb.claim_review_task(conn, tid)
        assert review_claim is not None
        assert review_claim.assignee == "reviewer-a"

        ok = kb.complete_task(
            conn, tid,
            summary="sign-off",
            actor="reviewer-a",
            expected_run_id=review_claim.current_run_id,
        )
        assert ok is True
        assert _row(conn, tid)["status"] == "done"
        # Exactly one completion, from the approver's pass.
        assert len(_events(conn, tid, kind="completed")) == 1


def test_human_approval_from_review_lane_unchanged(kanban_home):
    with kb.connect() as conn:
        tid = _claimed_flagged_task(conn)
        implementer_run = kb.get_task(conn, tid).current_run_id
        assert kb.complete_task(
            conn, tid, expected_run_id=implementer_run,
        )
        assert _row(conn, tid)["status"] == "review"

        # Completing FROM review is the approval act — no actor needed,
        # never re-gated, straight to done.
        assert kb.complete_task(conn, tid, result="approved by human")
        assert _row(conn, tid)["status"] == "done"


def test_completer_identity_is_canonicalized(kanban_home):
    with kb.connect() as conn:
        tid = _claimed_flagged_task(conn, assignee="reviewer-a")
        task = kb.get_task(conn, tid)

        # Same profile, different casing — still counts as the approver.
        assert kb.complete_task(
            conn, tid,
            actor="Reviewer-A",
            expected_run_id=task.current_run_id,
        )
        assert _row(conn, tid)["status"] == "done"


# ---------------------------------------------------------------------------
# The changes loop keeps working end-to-end under the policy
# ---------------------------------------------------------------------------


def test_request_changes_loop_still_works(kanban_home):
    with kb.connect() as conn:
        tid = _claimed_flagged_task(conn)
        implementer_run = kb.get_task(conn, tid).current_run_id
        assert kb.complete_task(conn, tid, expected_run_id=implementer_run)

        review_claim = kb.claim_review_task(conn, tid)
        assert review_claim is not None
        ok, implementer = kb.request_changes(
            conn, tid, reason="missing boundary test",
        )
        assert ok is True
        assert implementer == "builder"

        # Back with the implementer, out of review.
        row = _row(conn, tid)
        assert row["status"] == "ready"
        assert row["assignee"] == "builder"

        # Redo + re-complete: the policy parks it again.
        redo = kb.claim_task(conn, tid)
        assert redo is not None
        assert kb.complete_task(
            conn, tid, summary="addressed feedback",
            expected_run_id=redo.current_run_id,
        )
        assert _row(conn, tid)["status"] == "review"
        assert _row(conn, tid)["assignee"] == "reviewer-a"
        assert len(_events(conn, tid, kind="review_requested")) == 2


# ---------------------------------------------------------------------------
# Mid-flight edits
# ---------------------------------------------------------------------------


def test_midflight_flip_on_respected_at_completion(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="flip me on", assignee="builder")
        task = kb.claim_task(conn, tid)

        # Policy attached AFTER the work started — still enforced, because
        # the flag is consulted when the task naturally finishes.
        assert kb.set_approval_policy(
            conn, tid,
            approval_required=True, approver_profile="reviewer-a",
        )

        assert kb.complete_task(
            conn, tid, expected_run_id=task.current_run_id,
        )
        assert _row(conn, tid)["status"] == "review"


def test_midflight_flip_off_respected_at_completion(kanban_home):
    with kb.connect() as conn:
        tid = _claimed_flagged_task(conn)
        assert kb.set_approval_policy(conn, tid, approval_required=False)
        task = kb.get_task(conn, tid)

        assert kb.complete_task(
            conn, tid, expected_run_id=task.current_run_id,
        )
        assert _row(conn, tid)["status"] == "done"


def test_set_approval_policy_records_event_and_fields(conn):
    tid = kb.create_task(conn, title="policy audit")

    assert kb.set_approval_policy(
        conn, tid,
        approval_required=True,
        approver_profile="reviewer-a",
        approver_skill="review-checklist",
    )
    row = _row(conn, tid)
    assert row["approval_required"] == 1
    assert row["approver_profile"] == "reviewer-a"
    assert row["approver_skill"] == "review-checklist"

    events = _events(conn, tid, kind="approval_policy_set")
    assert len(events) == 1
    assert events[0][1] == {
        "approval_required": True,
        "approver_profile": "reviewer-a",
        "approver_skill": "review-checklist",
    }


def test_set_approval_policy_refuses_terminal_tasks(conn):
    tid = kb.create_task(conn, title="already done", assignee="builder")
    assert kb.complete_task(conn, tid)

    with pytest.raises(RuntimeError, match="done"):
        kb.set_approval_policy(conn, tid, approval_required=True)


def test_set_approval_policy_unknown_task(conn):
    assert kb.set_approval_policy(
        conn, "t_nope", approval_required=True,
    ) is False


# ---------------------------------------------------------------------------
# Creation-time validation of approver references
# ---------------------------------------------------------------------------


def test_create_rejects_unknown_approver_profile(conn):
    with pytest.raises(ValueError, match="not an existing profile"):
        kb.create_task(
            conn, title="ghost approver",
            approval_required=True, approver_profile="who-is-this",
        )


def test_create_rejects_unknown_approver_skill(conn):
    with pytest.raises(ValueError, match="does not match any installed skill"):
        kb.create_task(
            conn, title="ghost skill",
            approval_required=True, approver_profile="reviewer-a",
            approver_skill="never-installed",
        )


def test_created_policy_round_trips_through_get_task(conn):
    tid = kb.create_task(
        conn, title="flagged", assignee="builder",
        approval_required=True,
        approver_profile="reviewer-a",
        approver_skill="review-checklist",
    )
    task = kb.get_task(conn, tid)
    assert task.approval_required is True
    assert task.approver_profile == "reviewer-a"
    assert task.approver_skill == "review-checklist"


# ---------------------------------------------------------------------------
# Migration: fresh AND legacy boards gain the columns without data loss
# ---------------------------------------------------------------------------


def test_fresh_db_has_approval_columns(kanban_home):
    cols = {
        r[1]
        for r in kb.connect().execute("PRAGMA table_info(tasks)")
    }
    assert {"approval_required", "approver_profile", "approver_skill"} <= cols


def test_legacy_db_gains_approval_columns_without_data_loss(tmp_path):
    """A pre-#29457 board opens safely: columns are added, rows keep their
    values, defaults preserve old behaviour (flag off)."""
    import sqlite3

    db_path = tmp_path / ".hermes"
    db_path.mkdir(parents=True)
    legacy = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(legacy))
    conn.row_factory = sqlite3.Row
    # Pared-down schema WITHOUT any approval column.
    conn.execute("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('legacy', 'old work', 'ready', 123)"
    )
    conn.commit()

    before_cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    assert "approval_required" not in before_cols

    kb._migrate_add_optional_columns(conn)
    # Idempotent: running again must not raise or duplicate.
    kb._migrate_add_optional_columns(conn)

    after_cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    assert {
        "approval_required", "approver_profile", "approver_skill"
    } <= after_cols

    row = conn.execute(
        "SELECT * FROM tasks WHERE id = 'legacy'"
    ).fetchone()
    keys = set(row.keys())
    assert "approval_required" in keys
    # Defaults preserve the behaviour the row had before the column existed.
    assert row["approval_required"] == 0
    assert row["approver_profile"] is None
    assert row["approver_skill"] is None
    # No data loss.
    assert row["title"] == "old work"
    assert row["status"] == "ready"
    assert row["created_at"] == 123
    conn.close()


# ---------------------------------------------------------------------------
# Worker tool surface: kanban_create flags + honest kanban_complete report
# ---------------------------------------------------------------------------


def test_kanban_create_tool_accepts_approval_flags(
    kanban_home, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "orchestrator")
    from tools import kanban_tools as tools

    created = json.loads(tools._handle_create({
        "title": "needs sign-off",
        "assignee": "builder",
        "approval_required": True,
        "approver_profile": "reviewer-a",
        "approver_skill": "review-checklist",
    }))
    assert created["ok"] is True

    with kb.connect() as conn:
        task = kb.get_task(conn, created["task_id"])
        assert task.approval_required is True
        assert task.approver_profile == "reviewer-a"
        assert task.approver_skill == "review-checklist"


def test_kanban_complete_tool_reports_the_park(
    kanban_home, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    from tools import kanban_tools as tools

    with kb.connect() as conn:
        tid = _claimed_flagged_task(conn)
    run_id = kb.get_task(kb.connect(), tid).current_run_id
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    result = json.loads(tools._handle_complete({"summary": "finished up"}))
    assert result["ok"] is True
    assert result["status"] == "review"
    assert "reviewer-a" in result["approval"]


# ---------------------------------------------------------------------------
# Dashboard PATCH surface
# ---------------------------------------------------------------------------


def test_dashboard_patch_edits_policy_mid_flight(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import importlib.util
    import sys

    home = tmp_path / ".hermes"
    (home / "profiles" / "reviewer-a").mkdir(parents=True)
    (home / "skills" / "review-checklist").mkdir(parents=True)
    (home / "skills" / "review-checklist" / "SKILL.md").write_text(
        "---\nname: review-checklist\ndescription: Checklist.\n---\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = (
        repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_approval_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    app = __import__("fastapi").FastAPI()
    app.include_router(mod.router, prefix="/api/plugins/kanban")
    client = TestClient(app)

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="board card", assignee="builder")

    # Flip the whole policy on from the board.
    r = client.patch(
        f"/api/plugins/kanban/tasks/{tid}",
        json={
            "approval_required": True,
            "approver_profile": "reviewer-a",
            "approver_skill": "review-checklist",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()["task"]
    assert body["approval_required"] is True
    assert body["approver_profile"] == "reviewer-a"
    assert body["approver_skill"] == "review-checklist"

    # Clear just the approver pair; the flag stays on.
    r = client.patch(
        f"/api/plugins/kanban/tasks/{tid}", json={"clear_approver": True},
    )
    assert r.status_code == 200, r.text
    body = r.json()["task"]
    assert body["approval_required"] is True
    assert body["approver_profile"] is None
    assert body["approver_skill"] is None

    # Flip the flag back off.
    r = client.patch(
        f"/api/plugins/kanban/tasks/{tid}",
        json={"approval_required": False},
    )
    assert r.status_code == 200, r.text
    assert r.json()["task"]["approval_required"] is False

    # Invalid approver surfaces as a 400, not a silent write.
    r = client.patch(
        f"/api/plugins/kanban/tasks/{tid}",
        json={"approval_required": True, "approver_profile": "ghost"},
    )
    assert r.status_code == 400

    with kb.connect() as conn:
        kinds = [e.kind for e in kb.list_events(conn, tid)]
        assert kinds.count("approval_policy_set") == 3


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_create_with_approval_flags_round_trips(kanban_home):
    from hermes_cli import kanban as kc

    kc.run_slash(
        'create "flagged card" --assignee builder --approval-required '
        "--approver reviewer-a --approver-skill review-checklist"
    )
    listing = json.loads(kc.run_slash("list --json"))
    entry = next(r for r in listing if r["title"] == "flagged card")
    assert entry["approval_required"] is True
    assert entry["approver_profile"] == "reviewer-a"
    assert entry["approver_skill"] == "review-checklist"


def test_cli_set_approval_flips_policy(kanban_home):
    from hermes_cli import kanban as kc

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="cli flip target", assignee="builder")

    kc.run_slash(f"set-approval {tid} on --approver reviewer-a")
    task = kb.get_task(kb.connect(), tid)
    assert task.approval_required is True
    assert task.approver_profile == "reviewer-a"

    kc.run_slash(f"set-approval {tid} off --approver none")
    task = kb.get_task(kb.connect(), tid)
    assert task.approval_required is False
    assert task.approver_profile is None


def test_cli_complete_reports_the_park(kanban_home):
    from hermes_cli import kanban as kc

    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="cli park target", assignee="builder",
            approval_required=True, approver_profile="reviewer-a",
        )

    out = kc.run_slash(f"complete {tid}")
    assert "review" in out
    assert "reviewer-a" in out
    assert kb.get_task(kb.connect(), tid).status == "review"


# ---------------------------------------------------------------------------
# Review-lane dispatch loads the approver skill
# ---------------------------------------------------------------------------


def test_review_spawn_carries_sdlc_review_plus_approver_skill(
    kanban_home,
):
    """A policy-parked task dispatches through the real review lane: the
    spawned reviewer is the designated approver and its force-loaded
    skills are sdlc-review PLUS the task's approver_skill."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="policy parked", assignee="builder",
            approval_required=True,
            approver_profile="reviewer-a",
            approver_skill="review-checklist",
        )
        task = kb.claim_task(conn, tid)
        assert task is not None
        assert kb.complete_task(
            conn, tid, expected_run_id=task.current_run_id,
        )
        assert _row(conn, tid)["status"] == "review"

        spawned = []

        def _fake_spawn(claimed, workspace, *args, **kwargs):
            spawned.append((claimed, workspace))
            return 4242

        result = kb.dispatch_once(conn, spawn_fn=_fake_spawn)
        assert [(tid, "reviewer-a")] == [
            (t.id, t.assignee) for t, _ in spawned
        ]
        claimed = spawned[0][0]
        assert "sdlc-review" in claimed.skills
        assert "review-checklist" in claimed.skills
        assert result.spawned  # the review lane produced this spawn
