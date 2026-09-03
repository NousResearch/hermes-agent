"""Authority boundaries for typed exact-head pull-request tasks."""

import json
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


def _write_profile(home: Path, name: str, authority: str) -> Path:
    profile_dir = home / "profiles" / name
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_yaml = profile_dir / "profile.yaml"
    profile_yaml.write_text(
        f"name: {name}\nauthority: {authority}\n",
        encoding="utf-8",
    )
    return profile_yaml


def _pr_body(
    *,
    action: str = "repair_and_push",
    repository: str = "acme/widgets",
    expected_head_sha: str = "a" * 40,
) -> str:
    return json.dumps(
        {
            "repository": repository,
            "pr_number": 17,
            "expected_head_sha": expected_head_sha,
            "action": action,
        },
        sort_keys=True,
    )


def test_create_rejects_read_only_owner_for_exact_head_pr_write(kanban_home):
    _write_profile(kanban_home, "read-only-reviewer", "read_only")

    with kb.connect() as conn, pytest.raises(ValueError, match="read-only profile"):
        kb.create_task(
            conn,
            title="Repair pull request 17",
            body=_pr_body(),
            assignee="read-only-reviewer",
        )


def test_create_rejects_unproven_owner_for_exact_head_pr_write(kanban_home):
    with kb.connect() as conn, pytest.raises(
        ValueError, match="cannot verify write authority"
    ):
        kb.create_task(
            conn,
            title="Repair pull request 17",
            body=_pr_body(),
            assignee="missing-maintainer",
        )


def test_create_rejects_unrecognized_authority_metadata(kanban_home):
    _write_profile(kanban_home, "write-maintainer", "sometimes")

    with kb.connect() as conn, pytest.raises(
        ValueError, match="cannot verify write authority"
    ):
        kb.create_task(
            conn,
            title="Repair pull request 17",
            body=_pr_body(),
            assignee="write-maintainer",
        )


@pytest.mark.parametrize(
    ("repository", "expected_head_sha"),
    [
        ("not-a-repository", "a" * 40),
        ("acme/widgets", "short-head"),
    ],
)
def test_create_rejects_inexact_pr_write_identity(
    kanban_home, repository, expected_head_sha
):
    _write_profile(kanban_home, "write-maintainer", "write")

    with kb.connect() as conn, pytest.raises(ValueError, match="exact PR identity"):
        kb.create_task(
            conn,
            title="Repair pull request 17",
            body=_pr_body(
                repository=repository,
                expected_head_sha=expected_head_sha,
            ),
            assignee="write-maintainer",
        )


def test_reassign_rejects_read_only_owner_and_preserves_current_owner(kanban_home):
    _write_profile(kanban_home, "read-only-reviewer", "read_only")
    _write_profile(kanban_home, "write-maintainer", "write")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Resolve a merge conflict for pull request 17",
            body=_pr_body(action="resolve_merge_conflict"),
            assignee="write-maintainer",
        )
        with pytest.raises(ValueError, match="read-only profile"):
            kb.reassign_task(conn, task_id, "read-only-reviewer")
        assert kb.get_task(conn, task_id).assignee == "write-maintainer"


def test_claim_rejects_authority_revoked_after_admission(kanban_home):
    profile_yaml = _write_profile(kanban_home, "write-maintainer", "write")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Repair pull request 17",
            body=_pr_body(),
            assignee="write-maintainer",
        )
        profile_yaml.write_text(
            "name: write-maintainer\nauthority: read_only\n",
            encoding="utf-8",
        )

        assert kb.claim_task(conn, task_id) is None
        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
        assert task.block_kind == "capability"
        assert task.current_run_id is None


def test_claim_rejects_malformed_authority_metadata(kanban_home):
    profile_yaml = _write_profile(kanban_home, "write-maintainer", "write")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Repair pull request 17",
            body=_pr_body(),
            assignee="write-maintainer",
        )
        profile_yaml.write_text("authority: [not valid yaml", encoding="utf-8")

        assert kb.claim_task(conn, task_id) is None
        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
        assert task.block_kind == "capability"
        assert task.current_run_id is None


def test_request_review_rejects_read_only_reassignment_transactionally(kanban_home):
    _write_profile(kanban_home, "write-maintainer", "write")
    _write_profile(kanban_home, "read-only-reviewer", "read_only")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Repair pull request 17",
            body=_pr_body(),
            assignee="write-maintainer",
        )

        ok, reason = kb.request_review(
            conn,
            task_id,
            reviewer="read-only-reviewer",
            with_reason=True,
        )

        assert ok is False
        assert "read-only profile" in reason
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.assignee == "write-maintainer"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'review_requested'",
            (task_id,),
        ).fetchone()[0] == 0


def test_review_claim_rechecks_legacy_read_only_assignment(kanban_home):
    _write_profile(kanban_home, "write-maintainer", "write")
    _write_profile(kanban_home, "read-only-reviewer", "read_only")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Repair pull request 17",
            body=_pr_body(),
            assignee="write-maintainer",
        )
        conn.execute(
            "UPDATE tasks SET status = 'review', assignee = ? WHERE id = ?",
            ("read-only-reviewer", task_id),
        )
        conn.commit()

        assert kb.claim_review_task(conn, task_id, claimer="reviewer:1") is None
        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
        assert task.block_kind == "capability"
        assert task.current_run_id is None
        assert conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0] == 0


def test_request_changes_rejects_untrusted_read_only_implementer_provenance(
    kanban_home,
):
    _write_profile(kanban_home, "write-maintainer", "write")
    _write_profile(kanban_home, "write-reviewer", "write")
    _write_profile(kanban_home, "read-only-reviewer", "read_only")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Repair pull request 17",
            body=_pr_body(),
            assignee="write-maintainer",
        )
        assert kb.request_review(conn, task_id, reviewer="write-reviewer")
        claimed = kb.claim_review_task(conn, task_id, claimer="reviewer:1")
        assert claimed is not None
        event_row = conn.execute(
            "SELECT id, payload FROM task_events "
            "WHERE task_id = ? AND kind = 'review_requested' "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        payload = json.loads(event_row["payload"])
        payload["implementer"] = "read-only-reviewer"
        conn.execute(
            "UPDATE task_events SET payload = ? WHERE id = ?",
            (json.dumps(payload), event_row["id"]),
        )
        conn.commit()

        ok, reason = kb.request_changes(
            conn,
            task_id,
            reason="Add a regression.",
            expected_run_id=claimed.current_run_id,
        )

        assert ok is False
        assert "read-only profile" in reason
        task = kb.get_task(conn, task_id)
        assert task.status == "running"
        assert task.assignee == "write-reviewer"
        assert task.current_run_id == claimed.current_run_id


def test_reopen_review_rejects_untrusted_read_only_implementer_provenance(
    kanban_home,
):
    _write_profile(kanban_home, "write-maintainer", "write")
    _write_profile(kanban_home, "write-reviewer", "write")
    _write_profile(kanban_home, "read-only-reviewer", "read_only")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Repair pull request 17",
            body=_pr_body(),
            assignee="write-maintainer",
        )
        assert kb.request_review(conn, task_id, reviewer="write-reviewer")
        event_row = conn.execute(
            "SELECT id, payload FROM task_events "
            "WHERE task_id = ? AND kind = 'review_requested' "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        payload = json.loads(event_row["payload"])
        payload["implementer"] = "read-only-reviewer"
        conn.execute(
            "UPDATE task_events SET payload = ? WHERE id = ?",
            (json.dumps(payload), event_row["id"]),
        )
        conn.commit()

        assert kb.reopen_review_task(conn, task_id) is False
        task = kb.get_task(conn, task_id)
        assert task.status == "review"
        assert task.assignee == "write-reviewer"


def test_dispatch_default_assignment_does_not_persist_invalid_write_owner(
    kanban_home,
):
    _write_profile(kanban_home, "read-only-reviewer", "read_only")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Legacy unassigned task",
            body="Legacy body.",
        )
        conn.execute(
            "UPDATE tasks SET title = ?, body = ? WHERE id = ?",
            ("Repair pull request 17", _pr_body(), task_id),
        )
        conn.commit()

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("unauthorized task must not spawn")
            ),
            default_assignee="read-only-reviewer",
            max_spawn=1,
            reconcile_orphans=False,
        )

        assert result.spawned == []
        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
        assert task.block_kind == "capability"
        assert task.assignee is None
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'assigned'",
            (task_id,),
        ).fetchone()[0] == 0


def test_dispatch_dry_run_does_not_report_unauthorized_default_spawn(kanban_home):
    _write_profile(kanban_home, "read-only-reviewer", "read_only")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Legacy unassigned task",
            body="Legacy body.",
        )
        conn.execute(
            "UPDATE tasks SET title = ?, body = ? WHERE id = ?",
            ("Repair pull request 17", _pr_body(), task_id),
        )
        conn.commit()

        result = kb.dispatch_once(
            conn,
            dry_run=True,
            default_assignee="read-only-reviewer",
            max_spawn=1,
            reconcile_orphans=False,
        )

        assert result.spawned == []
        assert result.auto_assigned_default == []
        assert result.skipped_unassigned == [task_id]
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.assignee is None
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind IN ('assigned', 'claim_rejected')",
            (task_id,),
        ).fetchone()[0] == 0


def test_specify_rejects_typed_write_body_with_read_only_owner_atomically(
    kanban_home,
):
    _write_profile(kanban_home, "read-only-reviewer", "read_only")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Investigate report",
            body="Unspecified report.",
            assignee="read-only-reviewer",
            triage=True,
        )

        with pytest.raises(ValueError, match="read-only profile"):
            kb.specify_triage_task(
                conn,
                task_id,
                title="Repair pull request 17",
                body=_pr_body(),
                assignee="read-only-reviewer",
                author="specifier",
            )

        task = kb.get_task(conn, task_id)
        assert task.status == "triage"
        assert task.title == "Investigate report"
        assert task.body == "Unspecified report."
        assert task.assignee == "read-only-reviewer"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'specified'",
            (task_id,),
        ).fetchone()[0] == 0


def test_specify_cannot_erase_admitted_pr_identity_before_reader_assignment(
    kanban_home,
):
    _write_profile(kanban_home, "write-maintainer", "write")
    _write_profile(kanban_home, "read-only-reviewer", "read_only")
    original_body = _pr_body(action="repair_and_push")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Repair pull request 17",
            body=original_body,
            assignee="write-maintainer",
            triage=True,
        )

        with pytest.raises(ValueError, match="preserve exact pull-request identity"):
            kb.specify_triage_task(
                conn,
                task_id,
                title="Review local evidence",
                body="Untyped review prose.",
                assignee="read-only-reviewer",
                author="specifier",
            )

        task = kb.get_task(conn, task_id)
        assert task.status == "triage"
        assert task.title == "Repair pull request 17"
        assert task.body == original_body
        assert task.assignee == "write-maintainer"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'specified'",
            (task_id,),
        ).fetchone()[0] == 0


def test_specify_cannot_change_admitted_pr_action(kanban_home):
    _write_profile(kanban_home, "write-maintainer", "write")
    original_body = _pr_body(action="repair_and_push")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Repair pull request 17",
            body=original_body,
            assignee="write-maintainer",
            triage=True,
        )

        with pytest.raises(ValueError, match="preserve exact pull-request identity"):
            kb.specify_triage_task(
                conn,
                task_id,
                body=_pr_body(action="verify_ci_receipt"),
                author="specifier",
            )

        task = kb.get_task(conn, task_id)
        assert task.status == "triage"
        assert task.body == original_body


def test_specify_may_add_metadata_without_changing_admitted_pr_identity(kanban_home):
    _write_profile(kanban_home, "write-maintainer", "write")
    original_body = _pr_body(action="repair_and_push")
    replacement = json.loads(original_body)
    replacement["diagnostic_note"] = "bounded local evidence"

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Repair pull request 17",
            body=original_body,
            assignee="write-maintainer",
            triage=True,
        )

        assert kb.specify_triage_task(
            conn,
            task_id,
            body=json.dumps(replacement, sort_keys=True),
            author="specifier",
        )
        task = kb.get_task(conn, task_id)
        assert task.status in {"todo", "ready"}
        assert json.loads(task.body)["diagnostic_note"] == "bounded local evidence"


def test_decompose_rejects_read_only_typed_write_child_and_rolls_back_graph(
    kanban_home,
):
    _write_profile(kanban_home, "write-maintainer", "write")
    _write_profile(kanban_home, "read-only-reviewer", "read_only")

    with kb.connect() as conn:
        root_id = kb.create_task(
            conn,
            title="Plan bounded work",
            body="Split this task.",
            assignee="write-maintainer",
            triage=True,
        )
        before_count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

        with pytest.raises(ValueError, match="read-only profile"):
            kb.decompose_triage_task(
                conn,
                root_id,
                root_assignee="write-maintainer",
                children=[
                    {
                        "title": "Collect local evidence",
                        "body": "Run a local verification.",
                        "assignee": "read-only-reviewer",
                        "parents": [],
                    },
                    {
                        "title": "Repair pull request 17",
                        "body": _pr_body(),
                        "assignee": "read-only-reviewer",
                        "parents": [0],
                    },
                ],
                author="decomposer",
            )

        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == before_count
        root = kb.get_task(conn, root_id)
        assert root.status == "triage"
        assert root.assignee == "write-maintainer"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_links WHERE child_id = ?",
            (root_id,),
        ).fetchone()[0] == 0


def test_read_only_profile_may_own_exact_head_verification(kanban_home):
    _write_profile(kanban_home, "read-only-reviewer", "read_only")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Verify exact-head CI evidence for pull request 17",
            body=_pr_body(action="verify_ci_receipt"),
            assignee="read-only-reviewer",
        )

        assert kb.get_task(conn, task_id).assignee == "read-only-reviewer"


def test_read_only_exact_head_review_can_transition_and_claim(kanban_home):
    _write_profile(kanban_home, "write-maintainer", "write")
    _write_profile(kanban_home, "read-only-reviewer", "read_only")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Review exact-head evidence for pull request 17",
            body=_pr_body(action="review_exact_head_evidence"),
            assignee="write-maintainer",
        )

        assert kb.request_review(
            conn,
            task_id,
            reviewer="read-only-reviewer",
        )
        claimed = kb.claim_review_task(conn, task_id, claimer="reviewer:1")
        assert claimed is not None
        assert claimed.status == "running"
        assert claimed.assignee == "read-only-reviewer"


@pytest.mark.parametrize(
    "action",
    [
        "check_mergeability",
        "review_mergeability",
        "inspect_merge_conflicts",
        "read_credit_metadata",
    ],
)
def test_read_action_objects_do_not_become_write_actions(kanban_home, action):
    _write_profile(kanban_home, "read-only-reviewer", "read_only")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Inspect pull request 17",
            body=_pr_body(action=action),
            assignee="read-only-reviewer",
        )

        assert kb.get_task(conn, task_id).assignee == "read-only-reviewer"


@pytest.mark.parametrize("action", ["review_and_approve", "check_and_comment"])
def test_mixed_actions_remain_write_classified(kanban_home, action):
    _write_profile(kanban_home, "read-only-reviewer", "read_only")

    with kb.connect() as conn, pytest.raises(ValueError, match="read-only profile"):
        kb.create_task(
            conn,
            title="Review pull request 17",
            body=_pr_body(action=action),
            assignee="read-only-reviewer",
        )


def test_ordinary_task_does_not_require_profile_authority_metadata(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Fix a local formatting issue",
            body="No pull-request identity is attached.",
            assignee="ordinary-worker",
        )

        assert kb.get_task(conn, task_id).assignee == "ordinary-worker"
