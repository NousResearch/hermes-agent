"""Tests for the read-only kanban duplicate child guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_duplicate_child_guard import (
    build_duplicate_child_receipt,
    main,
)


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "kanban_duplicate_child_guard_issue155.json"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _review_body(issue: int, scope: str, *, pr: int = 160, head: str = "abc123") -> str:
    return f"""
GitHub issue: https://github.com/GTZhou/TianGongKaiWu/issues/{issue}
PR: https://github.com/GTZhou/TianGongKaiWu/pull/{pr}
head_sha: {head}
review_scope_key: {scope}
audit_scope: {scope}
""".strip()


def _make_root(conn, issue: int, scope: str = "155") -> str:
    root = kb.create_task(
        conn,
        title=f"【#{issue}】root implementation",
        body=_review_body(issue, scope),
        assignee="executor",
        created_by="shumishi",
    )
    assert kb.complete_task(conn, root, summary="root done")
    return root


def _make_review_child(conn, root: str, issue: int, scope: str, *, assignee="reviewer") -> str:
    return kb.create_task(
        conn,
        title=f"审计｜#{issue} scope {scope}",
        body=_review_body(issue, scope),
        assignee=assignee,
        created_by="shumishi",
        parents=[root],
    )


def _make_issue155_fixture(conn) -> tuple[str, list[str], dict]:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    root_spec = fixture["root"]
    root = kb.create_task(
        conn,
        title=root_spec["title"],
        body=root_spec["body"],
        assignee=root_spec["assignee"],
        created_by="shumishi",
    )
    assert kb.complete_task(conn, root, summary="fixture root done")
    children = [
        kb.create_task(
            conn,
            title=child["title"],
            body=child["body"],
            assignee=child["assignee"],
            created_by="shumishi",
            parents=[root],
        )
        for child in fixture["children"]
    ]
    return root, children, fixture


def _make_plain_root(conn) -> str:
    root = kb.create_task(
        conn,
        title="dispatch root without child markers",
        body="root card intentionally omits issue and scope markers",
        assignee="executor",
        created_by="shumishi",
    )
    assert kb.complete_task(conn, root, summary="plain root done")
    return root


def test_detector_flags_unrun_duplicate_and_emits_safe_dry_run_plan(kanban_home):
    with kb.connect() as conn:
        root, children, fixture = _make_issue155_fixture(conn)
        first, second = children

        receipt = build_duplicate_child_receipt(conn, root_task_ids=[root], dry_run=True)

    assert receipt["schema"] == "kanban-duplicate-child-guard:receipt:v1"
    assert receipt["mode"] == "dry_run"
    assert receipt["public_safety"]["no_mutations_performed"] is True
    assert receipt["graph"]["root_task_ids"] == [root]

    groups = receipt["detector"]["duplicate_groups"]
    assert len(groups) == 1
    group = groups[0]
    assert group["group_state"] == fixture["expected"]["group_state"]
    assert group["group_key"]["issues"] == fixture["expected"]["group_key"]["issues"]
    assert group["group_key"]["review_scope_key"] == fixture["expected"]["group_key"]["review_scope_key"]
    assert [member["task_id"] for member in group["members"]] == [first, second]

    plan = receipt["dry_run_plan"]
    assert plan["apply_enabled"] is False
    assert [action["action"] for action in plan["actions"]] == fixture["expected"]["dry_run_actions"]
    assert plan["actions"][0]["action"] == "keep_canonical_child"
    assert plan["actions"][0]["target_task_id"] == first
    assert plan["actions"][1]["action"] == "safe_converge_unrun_duplicate"
    assert plan["actions"][1]["target_task_id"] == second
    assert plan["actions"][1]["destructive"] is False


def test_detector_records_completed_history_duplicates_without_back_editing(kanban_home):
    with kb.connect() as conn:
        root = _make_root(conn, 155)
        first = _make_review_child(conn, root, 155, "155")
        second = _make_review_child(conn, root, 155, "155")
        assert kb.complete_task(conn, first, summary="historical duplicate A")
        assert kb.complete_task(conn, second, summary="historical duplicate B")

        receipt = build_duplicate_child_receipt(conn, root_task_ids=[root], dry_run=True)

    group = receipt["detector"]["duplicate_groups"][0]
    assert group["group_state"] == "historical_duplicate"
    actions = receipt["dry_run_plan"]["actions"]
    assert [a["action"] for a in actions] == ["record_only_historical_duplicate"]
    assert actions[0]["target_task_ids"] == [first, second]
    assert actions[0]["destructive"] is False
    assert receipt["public_safety"]["write_targets"] == []


def test_same_issue_distinct_phase_scope_is_not_duplicate(kanban_home):
    with kb.connect() as conn:
        root = _make_root(conn, 159, "159")
        child_a = _make_review_child(conn, root, 159, "159A")
        child_b = _make_review_child(conn, root, 159, "159B")

        receipt = build_duplicate_child_receipt(conn, root_task_ids=[root], dry_run=True)

    assert receipt["detector"]["duplicate_groups"] == []
    non_duplicates = receipt["detector"]["non_duplicate_review_children"]
    assert {item["task_id"] for item in non_duplicates} == {child_a, child_b}
    assert {item["group_key"]["review_scope_key"] for item in non_duplicates} == {"159A", "159B"}


def test_receipt_can_resolve_root_from_issue_filter_and_includes_profile_smoke(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "jiangzuodajiang")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_executor")
    with kb.connect() as conn:
        root = _make_root(conn, 155)
        _make_review_child(conn, root, 155, "155")
        _make_review_child(conn, root, 155, "155")

        receipt = build_duplicate_child_receipt(
            conn,
            issue="#155",
            actor_profile="jiangzuodajiang",
            dry_run=True,
        )

    assert receipt["input"]["issue"] == "#155"
    assert receipt["graph"]["root_task_ids"] == [root]
    smoke = receipt["runtime_profile_smoke"]
    assert smoke["actual_profile"] == "jiangzuodajiang"
    assert smoke["actor_profile_match"] is True
    assert smoke["current_kanban_task_id"] == "t_executor"


def test_cli_detect_writes_json_receipt_file(kanban_home, tmp_path):
    with kb.connect() as conn:
        root = _make_root(conn, 155)
        _make_review_child(conn, root, 155, "155")
        _make_review_child(conn, root, 155, "155")

    receipt_file = tmp_path / "receipt.json"
    exit_code = main([
        "detect",
        "--root-task",
        root,
        "--receipt-file",
        str(receipt_file),
        "--json",
    ])

    assert exit_code == 0
    data = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert data["schema"] == "kanban-duplicate-child-guard:receipt:v1"
    assert data["detector"]["duplicate_groups"]


def test_receipt_omits_user_controlled_titles_from_public_payload(kanban_home):
    with kb.connect() as conn:
        root = _make_root(conn, 155)
        kb.create_task(
            conn,
            title="审计｜VERY_SENSITIVE_TITLE_SENTINEL A",
            body=_review_body(155, "155"),
            assignee="reviewer",
            created_by="shumishi",
            parents=[root],
        )
        kb.create_task(
            conn,
            title="审计｜VERY_SENSITIVE_TITLE_SENTINEL B",
            body=_review_body(155, "155"),
            assignee="reviewer",
            created_by="shumishi",
            parents=[root],
        )

        receipt = build_duplicate_child_receipt(conn, root_task_ids=[root], dry_run=True)

    encoded = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    assert "VERY_SENSITIVE_TITLE_SENTINEL" not in encoded
    assert receipt["public_safety"]["user_controlled_titles_included"] is False


def test_issue_filter_promotes_matching_children_to_ancestor_root(kanban_home):
    with kb.connect() as conn:
        root = _make_plain_root(conn)
        first = _make_review_child(conn, root, 155, "155")
        second = _make_review_child(conn, root, 155, "155")

        receipt = build_duplicate_child_receipt(conn, issue="#155", dry_run=True)

    assert receipt["graph"]["root_task_ids"] == [root]
    group = receipt["detector"]["duplicate_groups"][0]
    assert [member["task_id"] for member in group["members"]] == [first, second]


def test_issue_url_metadata_normalizes_to_issue_number_marker(kanban_home):
    with kb.connect() as conn:
        root = _make_plain_root(conn)
        first = kb.create_task(
            conn,
            title="审计 metadata issue number",
            body="review_scope_key: 155\naudit_scope: 155",
            assignee="reviewer",
            created_by="shumishi",
            parents=[root],
        )
        second = kb.create_task(
            conn,
            title="审计 metadata issue url",
            body="review_scope_key: 155\naudit_scope: 155",
            assignee="reviewer",
            created_by="shumishi",
            parents=[root],
        )
        assert kb.complete_task(conn, first, summary="done", metadata={"issue_number": 155})
        assert kb.complete_task(
            conn,
            second,
            summary="done",
            metadata={"issue_url": "https://github.com/GTZhou/TianGongKaiWu/issues/155"},
        )

        receipt = build_duplicate_child_receipt(conn, root_task_ids=[root], dry_run=True)

    group = receipt["detector"]["duplicate_groups"][0]
    assert group["group_key"]["issues"] == ["#155"]
    assert [member["task_id"] for member in group["members"]] == [first, second]


def test_assignee_reviewer_alone_does_not_classify_plain_child_as_review(kanban_home):
    with kb.connect() as conn:
        root = _make_plain_root(conn)
        first = kb.create_task(
            conn,
            title="plain implementation child A",
            body="issue: #155\nscope: implementation",
            assignee="reviewer",
            created_by="shumishi",
            parents=[root],
        )
        second = kb.create_task(
            conn,
            title="plain implementation child B",
            body="issue: #155\nscope: implementation",
            assignee="reviewer",
            created_by="shumishi",
            parents=[root],
        )

        receipt = build_duplicate_child_receipt(conn, root_task_ids=[root], dry_run=True)

    assert receipt["detector"]["duplicate_groups"] == []
    assert receipt["graph"]["review_children_scanned"] == 0
    assert {item["task_id"] for item in receipt["detector"]["non_review_children"]} == {first, second}


def test_unmarked_review_children_require_manual_marker_completion(kanban_home):
    with kb.connect() as conn:
        root = _make_plain_root(conn)
        first = kb.create_task(
            conn,
            title="审计 missing marker A",
            body="needs marker completion before duplicate grouping",
            assignee="reviewer",
            created_by="shumishi",
            parents=[root],
        )
        second = kb.create_task(
            conn,
            title="审计 missing marker B",
            body="needs marker completion before duplicate grouping",
            assignee="reviewer",
            created_by="shumishi",
            parents=[root],
        )

        receipt = build_duplicate_child_receipt(conn, root_task_ids=[root], dry_run=True)

    assert receipt["detector"]["duplicate_groups"] == []
    assert [
        item["task_id"] for item in receipt["detector"]["insufficiently_marked_review_children"]
    ] == [first, second]
    assert receipt["dry_run_plan"]["actions"] == []
