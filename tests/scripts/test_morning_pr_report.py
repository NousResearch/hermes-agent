from scripts.morning_pr_report import (
    ChangedFile,
    GitSnapshot,
    format_report,
    parse_porcelain_status,
    should_silence,
)


def test_parse_porcelain_status_summarizes_index_and_worktree_states():
    status = """
 M scripts/morning_pr_report.py
A  tests/scripts/test_morning_pr_report.py
R  old/name.py -> new/name.py
?? docs/plans/morning-pr-report.md
"""

    assert parse_porcelain_status(status) == [
        ChangedFile(status="modified", path="scripts/morning_pr_report.py"),
        ChangedFile(status="added", path="tests/scripts/test_morning_pr_report.py"),
        ChangedFile(status="renamed", path="new/name.py", old_path="old/name.py"),
        ChangedFile(status="untracked", path="docs/plans/morning-pr-report.md"),
    ]


def test_format_report_returns_traditional_chinese_morning_summary():
    snapshot = GitSnapshot(
        branch="joe/nightly-personal-ops-brief",
        base_branch="main",
        commit_subjects=["feat: add morning PR report helper"],
        changed_files=[
            ChangedFile(status="added", path="scripts/morning_pr_report.py"),
            ChangedFile(status="added", path="tests/scripts/test_morning_pr_report.py"),
        ],
        pr_url="https://github.com/NousResearch/hermes-agent/pull/123",
    )

    report = format_report(
        snapshot,
        title="早晨 PR 報告產生器",
        why="讓 nightly build 的交付內容更穩定、可複製，減少早上讀報告的認知負擔。",
        verification=["scripts/run_tests.sh tests/scripts/test_morning_pr_report.py"],
        blockers=["無。"],
    )

    assert report.startswith("## TL;DR")
    assert "早晨 PR 報告產生器" in report
    assert "https://github.com/NousResearch/hermes-agent/pull/123" in report
    assert "joe/nightly-personal-ops-brief → main" in report
    assert "scripts/morning_pr_report.py" in report
    assert "tests/scripts/test_morning_pr_report.py" in report
    assert "scripts/run_tests.sh tests/scripts/test_morning_pr_report.py" in report
    assert "事實 / 已驗證" in report
    assert "無。" in report


def test_should_silence_when_no_work_or_pr_exists():
    empty = GitSnapshot(branch="main", base_branch="main")
    not_empty = GitSnapshot(
        branch="main",
        base_branch="main",
        changed_files=[ChangedFile(status="modified", path="README.md")],
    )

    assert should_silence(empty) is True
    assert should_silence(not_empty) is False
