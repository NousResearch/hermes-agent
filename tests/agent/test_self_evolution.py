from __future__ import annotations

from agent.self_evolution import (
    build_self_evolution_context,
    evolution_stats,
    export_context,
    list_lessons,
    recall_lessons,
    record_lesson,
    resolve_lesson,
)


def test_record_lesson_creates_and_deduplicates(tmp_path):
    path = tmp_path / "lessons.jsonl"

    first = record_lesson(
        mistake="Claiming tests passed before running them",
        lesson="Run the relevant tests and report failures honestly",
        trigger="code changes",
        fix="pytest tests/unit",
        tags=["tests", "verification"],
        evidence="user correction",
        severity="high",
        ledger_path=path,
    )
    second = record_lesson(
        mistake="Claiming tests passed before running them",
        lesson="Run the relevant tests and report failures honestly",
        tags=["pytest"],
        evidence=["failed review"],
        ledger_path=path,
    )

    assert first["success"] is True
    assert second["action"] == "updated"
    lessons = list_lessons(ledger_path=path)["lessons"]
    assert len(lessons) == 1
    assert lessons[0]["occurrences"] == 2
    assert lessons[0]["severity"] == "high"
    assert set(lessons[0]["tags"]) == {"pytest", "tests", "verification"}


def test_recall_lessons_scores_query_and_tags(tmp_path):
    path = tmp_path / "lessons.jsonl"
    record_lesson(
        mistake="Skipping browser screenshot verification",
        lesson="Use browser screenshots after frontend changes",
        tags=["frontend", "browser"],
        ledger_path=path,
    )
    record_lesson(
        mistake="Forgetting absolute dates in time-sensitive answers",
        lesson="Use concrete dates when user says today or latest",
        tags=["dates"],
        ledger_path=path,
    )

    recalled = recall_lessons(query="frontend browser verification", ledger_path=path)

    assert recalled["count"] >= 1
    assert recalled["lessons"][0]["tags"] == ["browser", "frontend"]


def test_export_context_and_resolve(tmp_path):
    path = tmp_path / "lessons.jsonl"
    created = record_lesson(
        mistake="Editing files before reading local style",
        lesson="Inspect neighboring code and tests before patching",
        trigger="repo edits",
        fix="rg first, then apply_patch",
        tags=["coding"],
        ledger_path=path,
    )["lesson"]

    context = export_context(query="repo edits", ledger_path=path)["context"]
    assert "Avoid:" in context
    assert "Do instead:" in context

    resolved = resolve_lesson(
        lesson_id=created["id"],
        outcome="covered by a stronger skill",
        ledger_path=path,
    )
    assert resolved["success"] is True
    assert recall_lessons(query="repo edits", ledger_path=path)["count"] == 0
    assert evolution_stats(ledger_path=path)["resolved"] == 1


def test_build_self_evolution_context_wraps_recalled_lessons(tmp_path):
    path = tmp_path / "lessons.jsonl"
    record_lesson(
        mistake="Skipping visual verification",
        lesson="Open browser and inspect the final UI",
        tags=["frontend"],
        ledger_path=path,
    )

    context = build_self_evolution_context("frontend verification", ledger_path=path)

    assert context.startswith("<self-evolution-context>")
    assert "prior mistakes" in context
    assert "Skipping visual verification" in context
