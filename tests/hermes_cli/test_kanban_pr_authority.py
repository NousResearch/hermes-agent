"""Authority boundaries for exact-head pull-request automation tasks."""

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


def _write_profile(home: Path, name: str, description: str) -> None:
    profile_dir = home / "profiles" / name
    profile_dir.mkdir(parents=True)
    (profile_dir / "profile.yaml").write_text(
        f"name: {name}\ndescription: {description!r}\n",
        encoding="utf-8",
    )


def _repair_body() -> str:
    return (
        '{"repository":"mrkillbob/luna-bot","pr_number":132,'
        '"expected_head_sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"action":"repair_and_push"}'
    )


def test_create_rejects_read_only_owner_for_atomic_pr_repair(kanban_home):
    _write_profile(
        kanban_home,
        "review-verification-steward",
        "Read-only verifier; never edits, pushes, replies, refreshes, or merges.",
    )

    with kb.connect() as conn, pytest.raises(ValueError, match="read-only profile"):
        kb.create_task(
            conn,
            title="Repair and push ExampleApp PR #132",
            body=_repair_body(),
            assignee="review-verification-steward",
            idempotency_key="github-pr-feedback:repair:132:abc",
        )


def test_reassign_rejects_read_only_owner_and_preserves_current_owner(kanban_home):
    _write_profile(
        kanban_home,
        "review-verification-steward",
        "Read-only verifier; never edits, pushes, replies, refreshes, or merges.",
    )
    _write_profile(
        kanban_home,
        "pr-repair-steward",
        "Repairs pull requests, pushes exact-head fixes, and posts factual replies.",
    )

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="Resolve merge conflict and push PR #132",
            body=_repair_body(),
            assignee="pr-repair-steward",
            idempotency_key="github-pr-feedback:repair:132:abc",
        )
        with pytest.raises(ValueError, match="read-only profile"):
            kb.reassign_task(conn, tid, "review-verification-steward")
        assert kb.get_task(conn, tid).assignee == "pr-repair-steward"


def test_read_only_profile_may_own_exact_head_verification(kanban_home):
    _write_profile(
        kanban_home,
        "review-verification-steward",
        "Read-only verifier; never edits, pushes, replies, refreshes, or merges.",
    )

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="Review exact-head CI evidence for PR #132",
            body=(
                '{"repository":"mrkillbob/luna-bot","pr_number":132,'
                '"expected_head_sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
                '"action":"verify_ci_receipt"}'
            ),
            assignee="review-verification-steward",
            idempotency_key="github-pr-feedback:review:132:abc",
        )
        assert kb.get_task(conn, tid).assignee == "review-verification-steward"


def test_blocked_intent_review_negative_contract_may_use_read_only_owner(
    kanban_home,
):
    _write_profile(
        kanban_home,
        "intent-review-readonly-test",
        "Read-only intent reviewer; never edits, pushes, replies, or merges.",
    )
    body = (
        '{"repository":"mrkillbob/luna-bot","pr_number":132,'
        '"expected_head_sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}'
        "\nDo not edit, push, reply, approve, or merge. "
        "Record only the operator intent decision."
    )

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="Intent review for PR #132",
            body=body,
            assignee="intent-review-readonly-test",
            idempotency_key="github-pr-feedback:intent-review:repo:132:abc",
            initial_status="blocked",
        )
        assert kb.get_task(conn, tid).status == "blocked"


def test_intent_review_exception_does_not_allow_runnable_write_task(kanban_home):
    _write_profile(
        kanban_home,
        "intent-review-readonly-test",
        "Read-only intent reviewer; never edits, pushes, replies, or merges.",
    )
    with kb.connect() as conn, pytest.raises(ValueError, match="read-only profile"):
        kb.create_task(
            conn,
            title="Intent review for PR #132",
            body=_repair_body() + "\nRepair and push the exact head.",
            assignee="intent-review-readonly-test",
            idempotency_key="github-pr-feedback:intent-review:repo:132:abc",
            initial_status="running",
        )
