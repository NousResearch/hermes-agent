"""Regression tests for review-required successor routing.

Blocked source cards whose latest worker handoff starts with ``review-required:``
need a separate runnable reviewer/triage card.  The successor must not be a
parent-linked child of the blocked source, because parent links wait for the
source to become terminal and deadlock review execution.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _latest_run_id(conn, task_id: str) -> int:
    row = conn.execute(
        "SELECT id FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    assert row is not None
    return int(row["id"])


def _block_review_required(
    conn,
    task_id: str,
    *,
    reason: str = "review-required: inspect the handoff",
    metadata: dict | None = None,
) -> int:
    assert kb.claim_task(conn, task_id) is not None
    assert kb.block_task(
        conn,
        task_id,
        reason=reason,
        expected_run_id=kb.get_task(conn, task_id).current_run_id,
    )
    run_id = _latest_run_id(conn, task_id)
    if metadata is not None:
        conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = ?",
            (json.dumps(metadata), run_id),
        )
        conn.commit()
    return run_id


def _only_successor(conn, source_id: str):
    rows = conn.execute(
        "SELECT id, title, body, assignee, status, created_by, idempotency_key "
        "FROM tasks WHERE id != ?",
        (source_id,),
    ).fetchall()
    assert len(rows) == 1
    return rows[0]


def _created_successors(conn, source_id: str):
    rows = conn.execute(
        "SELECT id, title, body, assignee, status, created_by, idempotency_key "
        "FROM tasks WHERE idempotency_key LIKE ? ORDER BY created_at ASC",
        (f"review-required:{source_id}:%",),
    ).fetchall()
    return rows


def test_review_required_needed_profile_creates_independent_reviewer_card(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        source = kb.create_task(
            conn,
            title="Patch router",
            body="Code diff is ready; needs independent review.",
            assignee="builder",
            workspace_kind="dir",
            workspace_path="/tmp/project",
            tenant="review-required-router",
        )
        run_id = _block_review_required(
            conn,
            source,
            metadata={
                "needed_profile": "researcher",
                "changed_files": ["router.py"],
                "tests_run": 5,
            },
        )

        result = kb.scan_liveness(conn)

        assert result.created_review_required_successors == [source]
        successor = _only_successor(conn, source)
        assert successor["title"] == "Review required: Patch router"
        assert successor["assignee"] == "brennan"
        assert successor["status"] == "ready"
        assert successor["created_by"] == "kanban-liveness"
        assert successor["idempotency_key"] == f"review-required:{source}:{run_id}"
        assert source in successor["body"]
        assert f"Source run: {run_id}" in successor["body"]
        assert "changed_files" in successor["body"]
        assert kb.get_task(conn, source).status == "blocked"

        # Critical deadlock guard: the review successor is not a child of the
        # blocked source.  Parent-linked children cannot run until the source is
        # done, which is exactly what the review is supposed to decide.
        assert conn.execute(
            "SELECT COUNT(*) FROM task_links WHERE parent_id = ? AND child_id = ?",
            (source, successor["id"]),
        ).fetchone()[0] == 0


def test_review_required_missing_owner_routes_to_default_triage_not_researcher_guess(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        source = kb.create_task(
            conn,
            title="Ambiguous human review",
            body="Needs human review but no reviewer/needed_profile is declared.",
            assignee="builder",
        )
        _block_review_required(conn, source, metadata={"tests_run": 1})

        result = kb.scan_liveness(conn)

        assert result.created_review_required_successors == [source]
        successor = _only_successor(conn, source)
        assert successor["title"] == "Route review-required: Ambiguous human review"
        assert successor["assignee"] == "default"
        assert successor["status"] == "triage"
        assert "No explicit review owner was inferable" in successor["body"]
        assert "Selected reviewer assignee: default (router-triage)" in successor["body"]


def test_ai_influencer_visual_output_without_owner_routes_to_taylor_not_default_triage(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        source = kb.create_task(
            conn,
            title="Build real identity-control workflow for American cheer/gymnast Nia reset",
            body=(
                "AI Influencer / Nia ComfyUI InstantID identity-control smoke: "
                "review the one generated image before accepting the reset."
            ),
            assignee="builder",
            tenant="molly-v2-working-system",
        )
        _block_review_required(
            conn,
            source,
            reason=(
                "review-required: one-image InstantID smoke completed on Jarvis; "
                "output is adult blonde/light-brown collegiate fitness lane but "
                "soft/low-res, needs human decision before accepting identity reset."
            ),
            metadata={
                "output_image": "/Volumes/Jarvis/Molly/content/ai-influencer/03-generations/nia.png",
                "artifacts": [
                    "/Volumes/Jarvis/Molly/content/ai-influencer/03-generations/nia.png"
                ],
                "accepted_scope": "identity-control path smoke only, not final avatar",
            },
        )

        result = kb.scan_liveness(conn)

        assert result.created_review_required_successors == [source]
        successor = _only_successor(conn, source)
        assert successor["title"] == "Review required: Build real identity-control workflow for American cheer/gymnast Nia reset"
        assert successor["assignee"] == "taylor"
        assert successor["status"] == "ready"
        assert "No explicit review owner was inferable" not in successor["body"]
        assert "Selected reviewer assignee: taylor (visual-output-review)" in successor["body"]


def test_explicit_studio_review_owner_canonicalizes_to_taylor(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        source = kb.create_task(conn, title="Studio-routed image review", assignee="builder")
        _block_review_required(conn, source, metadata={"review_assignee": "studio"})

        result = kb.scan_liveness(conn)

        assert result.created_review_required_successors == [source]
        successor = _only_successor(conn, source)
        assert successor["assignee"] == "taylor"
        assert "Selected reviewer assignee: taylor (explicit-metadata)" in successor["body"]


def test_review_required_successor_creation_is_idempotent(kanban_home: Path) -> None:
    with kb.connect() as conn:
        source = kb.create_task(conn, title="Idempotent review", assignee="builder")
        _block_review_required(
            conn,
            source,
            metadata={"review_assignee": "researcher"},
        )

        first = kb.scan_liveness(conn)
        second = kb.scan_liveness(conn)

        assert first.created_review_required_successors == [source]
        assert second.created_review_required_successors == []
        assert conn.execute("SELECT COUNT(*) FROM tasks WHERE id != ?", (source,)).fetchone()[0] == 1


def test_review_required_ignores_unrelated_parent_linked_child(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        source = kb.create_task(conn, title="Patch with unrelated child", assignee="builder")
        unrelated_child = kb.create_task(
            conn,
            title="Unrelated child work",
            assignee="builder",
            parents=[source],
        )
        run_id = _block_review_required(
            conn,
            source,
            metadata={"needed_profile": "researcher"},
        )

        result = kb.scan_liveness(conn)

        assert result.created_review_required_successors == [source]
        successors = _created_successors(conn, source)
        assert len(successors) == 1
        successor = successors[0]
        assert successor["assignee"] == "brennan"
        assert successor["idempotency_key"] == f"review-required:{source}:{run_id}"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_links WHERE parent_id = ? AND child_id = ?",
            (source, successor["id"]),
        ).fetchone()[0] == 0
        assert kb.parent_ids(conn, unrelated_child) == [source]


def test_review_required_idempotency_key_collision_with_parent_linked_child_creates_independent_successor(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        source = kb.create_task(conn, title="Patch with colliding child", assignee="builder")
        run_id = _block_review_required(
            conn,
            source,
            metadata={"needed_profile": "researcher"},
        )
        idem = f"review-required:{source}:{run_id}"
        colliding_child = kb.create_task(
            conn,
            title="Deadlocked child with review idempotency key",
            assignee="builder",
            parents=[source],
            idempotency_key=idem,
        )

        result = kb.scan_liveness(conn)

        assert result.created_review_required_successors == [source]
        successors = _created_successors(conn, source)
        independent_successors = [
            row
            for row in successors
            if conn.execute(
                "SELECT COUNT(*) FROM task_links WHERE parent_id = ? AND child_id = ?",
                (source, row["id"]),
            ).fetchone()[0]
            == 0
        ]
        assert len(independent_successors) == 1
        successor = independent_successors[0]
        assert successor["id"] != colliding_child
        assert successor["assignee"] == "brennan"
        assert successor["status"] == "ready"
        assert successor["idempotency_key"] == idem
        assert kb.parent_ids(conn, colliding_child) == [source]

        second = kb.scan_liveness(conn)
        assert second.created_review_required_successors == []
        assert _created_successors(conn, source) == successors


def test_review_required_owner_can_come_from_latest_handoff_comment(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        source = kb.create_task(conn, title="Comment-routed review", assignee="builder")
        kb.add_comment(
            conn,
            source,
            "builder",
            "review-required handoff:\n"
            + json.dumps(
                {
                    "needed_review_owner": "researcher",
                    "changed_files": ["hermes_cli/kanban_db.py"],
                }
            ),
        )
        _block_review_required(conn, source)

        result = kb.scan_liveness(conn)

        assert result.created_review_required_successors == [source]
        successor = _only_successor(conn, source)
        assert successor["title"] == "Review required: Comment-routed review"
        assert successor["assignee"] == "brennan"
        assert "Selected reviewer assignee: brennan (explicit-comment)" in successor["body"]
        assert "changed_files" in successor["body"]


def test_builder_code_change_receipts_without_owner_route_to_researcher(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        source = kb.create_task(
            conn,
            title="Builder code handoff",
            body="Code patch is ready; no explicit reviewer owner was set.",
            assignee="builder",
        )
        _block_review_required(
            conn,
            source,
            metadata={
                "changed_files": [
                    "hermes_cli/kanban_db.py",
                    "tests/hermes_cli/test_kanban_review_required_successor.py",
                ],
                "tests_run": [
                    {
                        "command": "python -m pytest tests/hermes_cli/test_kanban_review_required_successor.py",
                        "exit_code": 0,
                    }
                ],
            },
        )

        result = kb.scan_liveness(conn)

        assert result.created_review_required_successors == [source]
        successor = _only_successor(conn, source)
        assert successor["title"] == "Review required: Builder code handoff"
        assert successor["assignee"] == "brennan"
        assert successor["status"] == "ready"
        assert (
            "Selected reviewer assignee: brennan (code-review-receipts)"
            in successor["body"]
        )
        assert "RED-GATED: do not execute" not in successor["body"]


def test_declared_independent_review_successor_suppresses_router_default_triage(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        source = kb.create_task(
            conn,
            title="Already has reviewer",
            assignee="builder",
        )
        reviewer = kb.create_task(
            conn,
            title="Manual reviewer successor",
            assignee="researcher",
        )
        _block_review_required(
            conn,
            source,
            metadata={"successor_task_ids": [reviewer], "tests_run": 1},
        )

        result = kb.scan_liveness(conn)

        assert result.created_review_required_successors == []
        rows = conn.execute("SELECT id FROM tasks ORDER BY created_at ASC").fetchall()
        assert [row["id"] for row in rows] == [source, reviewer]


@pytest.mark.parametrize(
    "reason_fragment",
    [
        "no red gate remains",
        "not a red gate",
        "without red gates",
        "non-red-gated",
    ],
)
def test_review_required_negated_red_gate_prose_with_code_receipts_routes_to_researcher(
    kanban_home: Path,
    reason_fragment: str,
) -> None:
    with kb.connect() as conn:
        source = kb.create_task(
            conn,
            title="Negated red prose code patch",
            body="Code patch is ready; the handoff states no Red boundary remains.",
            assignee="builder",
        )
        _block_review_required(
            conn,
            source,
            reason=(
                f"review-required: code review needed; {reason_fragment} "
                "and no STOP/HUMAN boundary remains"
            ),
            metadata={
                "changed_files": [
                    "hermes_cli/kanban_db.py",
                    "tests/hermes_cli/test_kanban_review_required_successor.py",
                ],
                "tests_run": [
                    {
                        "command": "python -m pytest tests/hermes_cli/test_kanban_review_required_successor.py",
                        "exit_code": 0,
                    }
                ],
            },
        )

        result = kb.scan_liveness(conn)

        assert result.created_review_required_successors == [source]
        successor = _only_successor(conn, source)
        assert successor["title"] == "Review required: Negated red prose code patch"
        assert successor["assignee"] == "brennan"
        assert successor["status"] == "ready"
        assert (
            "Selected reviewer assignee: brennan (code-review-receipts)"
            in successor["body"]
        )
        assert "RED-GATED: do not execute" not in successor["body"]


@pytest.mark.parametrize(
    "reason_fragment",
    [
        "no red gate remains",
        "not a red gate",
        "without red gates",
        "non-red-gated",
    ],
)
def test_review_required_negated_red_gate_prose_preserves_explicit_owner(
    kanban_home: Path,
    reason_fragment: str,
) -> None:
    with kb.connect() as conn:
        source = kb.create_task(
            conn,
            title="Typed non-red owner",
            body="Metadata declares a typed non-Red reviewer owner.",
            assignee="builder",
        )
        _block_review_required(
            conn,
            source,
            reason=(
                f"review-required: {reason_fragment}; "
                "no STOP/HUMAN boundary remains; peer review only"
            ),
            metadata={
                "review_assignee": "researcher",
                "status_class": "yellow",
                "red_gate": False,
                "changed_files": ["hermes_cli/kanban_db.py"],
                "tests_run": 1,
            },
        )

        result = kb.scan_liveness(conn)

        assert result.created_review_required_successors == [source]
        successor = _only_successor(conn, source)
        assert successor["title"] == "Review required: Typed non-red owner"
        assert successor["assignee"] == "brennan"
        assert successor["status"] == "ready"
        assert (
            "Selected reviewer assignee: brennan (explicit-metadata)"
            in successor["body"]
        )
        assert "RED-GATED: do not execute" not in successor["body"]


@pytest.mark.parametrize(
    "reason_fragment",
    [
        "no red gate remains",
        "not a red gate",
        "without red gates",
        "non-red-gated",
    ],
)
def test_review_required_block_reason_owner_hint_preserves_typed_non_red_owner(
    kanban_home: Path,
    reason_fragment: str,
) -> None:
    with kb.connect() as conn:
        source = kb.create_task(
            conn,
            title="Canonical typed owner hint",
            body="Metadata declares the reviewer in block_reason.owner_hint.",
            assignee="builder",
        )
        _block_review_required(
            conn,
            source,
            reason=(
                f"review-required: {reason_fragment}; "
                "no STOP/HUMAN boundary remains; peer review only"
            ),
            metadata={
                "block_reason": {
                    "owner_hint": {
                        "profile": "researcher",
                        "status_class": "yellow",
                        "red_gate": False,
                    }
                },
                "changed_files": ["hermes_cli/kanban_db.py"],
                "tests_run": 1,
            },
        )

        result = kb.scan_liveness(conn)

        assert result.created_review_required_successors == [source]
        successor = _only_successor(conn, source)
        assert successor["title"] == "Review required: Canonical typed owner hint"
        assert successor["assignee"] == "brennan"
        assert successor["status"] == "ready"
        assert "Selected reviewer assignee: brennan (explicit-metadata)" in successor["body"]
        assert "RED-GATED: do not execute" not in successor["body"]


def test_review_required_block_reason_owner_hint_red_gate_blocks_default(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        source = kb.create_task(
            conn,
            title="Typed red owner hint",
            body="Metadata owner hint says human approval is still required.",
            assignee="builder",
        )
        _block_review_required(
            conn,
            source,
            reason="review-required: typed routing packet awaits review",
            metadata={
                "block_reason": {
                    "owner_hint": {
                        "profile": "researcher",
                        "status_class": "red",
                        "red_gate": True,
                    }
                },
                "changed_files": ["hermes_cli/kanban_db.py"],
                "tests_run": 1,
            },
        )

        result = kb.scan_liveness(conn)

        assert result.created_review_required_successors == [source]
        successor = _only_successor(conn, source)
        assert successor["title"] == "RED-GATED routing needed: Review required: Typed red owner hint"
        assert successor["assignee"] == "default"
        assert successor["status"] == "blocked"
        assert "RED-GATED: do not execute" in successor["body"]


def test_review_required_stop_human_boundary_prose_blocks_default_with_code_receipts(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        source = kb.create_task(
            conn,
            title="Concrete human boundary",
            body="Code patch is ready but a STOP/HUMAN boundary remains.",
            assignee="builder",
        )
        _block_review_required(
            conn,
            source,
            reason="review-required: STOP/HUMAN boundary remains before production deploy",
            metadata={
                "changed_files": ["hermes_cli/kanban_db.py"],
                "tests_run": [
                    {
                        "command": "python -m pytest tests/hermes_cli/test_kanban_review_required_successor.py",
                        "exit_code": 0,
                    }
                ],
            },
        )

        result = kb.scan_liveness(conn)

        assert result.created_review_required_successors == [source]
        successor = _only_successor(conn, source)
        assert (
            successor["title"]
            == "RED-GATED routing needed: Review required: Concrete human boundary"
        )
        assert successor["assignee"] == "default"
        assert successor["status"] == "blocked"
        assert "RED-GATED: do not execute" in successor["body"]


@pytest.mark.parametrize(
    "reason",
    [
        "review-required: RED gate / human approval before production deploy",
        "review-required: STOP/HUMAN boundary remains before production deploy",
    ],
)
def test_review_required_red_or_human_gate_creates_blocked_default_packet(
    kanban_home: Path,
    reason: str,
) -> None:
    with kb.connect() as conn:
        source = kb.create_task(conn, title="Production apply", assignee="builder")
        _block_review_required(
            conn,
            source,
            reason=reason,
            metadata={
                "needed_profile": "researcher",
                "red_gates": ["production deploy requires Brian approval"],
            },
        )

        result = kb.scan_liveness(conn)

        assert result.created_review_required_successors == [source]
        successor = _only_successor(conn, source)
        assert successor["title"] == "RED-GATED routing needed: Review required: Production apply"
        assert successor["assignee"] == "default"
        assert successor["status"] == "blocked"
        assert "RED-GATED: do not execute" in successor["body"]
        assert "production deploy requires Brian approval" in successor["body"]
        assert kb.get_task(conn, source).status == "blocked"


def test_review_required_comment_receipts_without_owner_route_code_review(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        source = kb.create_task(
            conn,
            title="Comment-only code receipt",
            assignee="builder",
        )
        kb.add_comment(
            conn,
            source,
            "builder",
            json.dumps(
                {
                    "changed_files": ["hermes_cli/kanban_db.py"],
                    "tests_run": [
                        {
                            "command": "python -m pytest tests/hermes_cli/test_kanban_review_required_successor.py",
                            "exit_code": 0,
                        }
                    ],
                }
            ),
        )
        _block_review_required(conn, source)

        result = kb.scan_liveness(conn)

        assert result.created_review_required_successors == [source]
        successor = _only_successor(conn, source)
        assert successor["title"] == "Review required: Comment-only code receipt"
        assert successor["assignee"] == "brennan"
        assert "Selected reviewer assignee: brennan (code-review-receipts)" in successor["body"]


def test_review_required_new_block_run_after_old_review_gets_new_successor(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        source = kb.create_task(conn, title="Repeat review cycle", assignee="builder")
        first_run = _block_review_required(conn, source, metadata={"needed_profile": "researcher"})
        first = kb.scan_liveness(conn)
        assert first.created_review_required_successors == [source]
        first_successor = _created_successors(conn, source)[0]
        assert kb.claim_task(conn, first_successor["id"]) is not None
        assert kb.complete_task(
            conn,
            first_successor["id"],
            summary="APPROVED: first cycle done",
            metadata={"review_of": source, "verdict": "APPROVED"},
            expected_run_id=kb.get_task(conn, first_successor["id"]).current_run_id,
        )
        verdicts = kb.scan_liveness(conn)
        assert verdicts.consumed_review_verdicts == [source]
        assert kb.get_task(conn, source).status == "ready"

        second_run = _block_review_required(conn, source, metadata={"needed_profile": "researcher"})
        second = kb.scan_liveness(conn)

        assert second_run != first_run
        assert second.created_review_required_successors == [source]
        successors = _created_successors(conn, source)
        assert {row["idempotency_key"] for row in successors} == {
            f"review-required:{source}:{first_run}",
            f"review-required:{source}:{second_run}",
        }
        assert len(successors) == 2


def test_dispatch_once_dry_run_does_not_create_review_required_successor(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _profile: True)
    with kb.connect() as conn:
        source = kb.create_task(conn, title="Dry run review", assignee="builder")
        _block_review_required(conn, source, metadata={"needed_profile": "researcher"})

        result = kb.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: None, max_spawn=1, dry_run=True)

        assert result.created_review_required_successors == []
        assert conn.execute("SELECT COUNT(*) FROM tasks WHERE id != ?", (source,)).fetchone()[0] == 0


def test_dispatch_once_runs_review_required_router_before_spawning(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned: list[tuple[str, str]] = []

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _profile: True)

    def fake_spawn(task, workspace, board=None):
        spawned.append((task.id, task.assignee or ""))
        return None

    with kb.connect() as conn:
        source = kb.create_task(conn, title="Dispatch-integrated review", assignee="builder")
        _block_review_required(conn, source, metadata={"needed_profile": "researcher"})

        result = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn=1)

        successor = _only_successor(conn, source)
        assert result.created_review_required_successors == [source]
        assert spawned == [(successor["id"], "brennan")]
        assert kb.get_task(conn, source).status == "blocked"
