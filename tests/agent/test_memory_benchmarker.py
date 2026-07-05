"""Tests for memory benchmark and lint helpers."""

from __future__ import annotations

from pathlib import Path

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _with_home(home: Path, fn):
    token = set_hermes_home_override(home)
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


def test_memory_lint_flags_duplicates_stale_and_skill_like_entries(tmp_path):
    home = tmp_path / ".hermes"
    mem = home / "memories"
    mem.mkdir(parents=True)
    (mem / "MEMORY.md").write_text(
        "Project uses pytest with xdist\n"
        "§\n"
        "Project uses pytest xdist\n"
        "§\n"
        "Submitted PR 123 yesterday\n"
        "§\n"
        "Run pytest -q then commit the code\n",
        encoding="utf-8",
    )

    def run():
        from agent.memory_benchmarker import lint_memory_entries

        return lint_memory_entries()

    findings = _with_home(home, run)
    issue_sets = {f["text"]: set(f["issues"]) for f in findings}
    assert "near_duplicate" in issue_sets["Project uses pytest xdist"]
    assert "stale_task_progress" in issue_sets["Submitted PR 123 yesterday"]
    assert "belongs_in_skill" in issue_sets["Run pytest -q then commit the code"]


def test_memory_benchmark_reports_precision_and_recall_at_k(tmp_path):
    home = tmp_path / ".hermes"
    mem = home / "memories"
    mem.mkdir(parents=True)
    (mem / "USER.md").write_text(
        "User prefers concise responses\n§\nUser works on autonomous agents",
        encoding="utf-8",
    )

    def run():
        from agent.memory_benchmarker import benchmark_memory_retrieval

        cases = [
            {"query": "concise responses user preference", "expected": ["concise responses"]},
            {"query": "autonomous agents user work", "expected": ["autonomous agents"]},
        ]
        return benchmark_memory_retrieval(cases, k=1)

    report = _with_home(home, run)
    assert report["cases"] == 2
    assert report["precision_at_k"] == 1.0
    assert report["recall_at_k"] == 1.0


def test_memory_benchmark_does_not_use_expected_substrings_for_ranking(tmp_path):
    home = tmp_path / ".hermes"
    mem = home / "memories"
    mem.mkdir(parents=True)
    (mem / "MEMORY.md").write_text(
        "Alpha banana\n§\nExpected phrase with no query overlap",
        encoding="utf-8",
    )

    def run():
        from agent.memory_benchmarker import benchmark_memory_retrieval

        return benchmark_memory_retrieval(
            [{"query": "alpha", "expected": ["Expected phrase"]}],
            k=1,
        )

    report = _with_home(home, run)
    assert report["precision_at_k"] == 0.0
    assert report["case_reports"][0]["retrieved"][0]["text"] == "Alpha banana"
