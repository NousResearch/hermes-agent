"""Focused tests for the github-pr-workflow Dependabot hygiene helper."""
from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "github" / "github-pr-workflow"
SCRIPT_PATH = SKILL_DIR / "scripts" / "dependabot_pr_hygiene.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("dependabot_pr_hygiene", SCRIPT_PATH)
    assert spec and spec.loader, f"failed to load module spec for {SCRIPT_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()
NOW = datetime(2026, 5, 26, tzinfo=timezone.utc)


def _pr(
    number: int,
    title: str,
    *,
    created_at: str,
    updated_at: str,
    mergeable: bool | None,
    mergeable_state: str,
    login: str = "dependabot[bot]",
    head_ref: str = "dependabot/pip/example-1.2.3",
    head_sha: str | None = None,
    draft: bool = False,
):
    return {
        "number": number,
        "title": title,
        "html_url": f"https://github.com/org/repo/pull/{number}",
        "created_at": created_at,
        "updated_at": updated_at,
        "mergeable": mergeable,
        "mergeable_state": mergeable_state,
        "draft": draft,
        "user": {"login": login},
        "head": {"ref": head_ref, "sha": head_sha or f"sha-{number}"},
        "base": {"ref": "main"},
    }


def _issue(*labels: str):
    return {"labels": [{"name": label} for label in labels]}


def _checks(summary: str = "success"):
    commit_state = summary if summary in {"success", "pending", "failure"} else "success"
    if summary == "pending":
        check_runs = [{"name": "ci", "status": "in_progress", "conclusion": None}]
    elif summary == "failure":
        check_runs = [{"name": "ci", "status": "completed", "conclusion": "failure"}]
    elif summary == "none":
        check_runs = []
        commit_state = ""
    else:
        check_runs = [{"name": "ci", "status": "completed", "conclusion": "success"}]
    return {
        "commit_status": {"state": commit_state},
        "check_runs": check_runs,
    }


def test_skill_mentions_dependabot_hygiene() -> None:
    src = (SKILL_DIR / "SKILL.md").read_text()
    assert "Dependabot" in src
    assert "update_branch" in src
    assert "close_superseded" in src
    assert "recreate_or_manual_rebase" in src


def test_script_exists() -> None:
    assert SCRIPT_PATH.is_file()


def test_detects_dependabot_by_author_or_branch() -> None:
    authored = _pr(
        1,
        "Bump urllib3 from 2.1.0 to 2.2.0",
        created_at="2026-05-01T00:00:00Z",
        updated_at="2026-05-01T00:00:00Z",
        mergeable=True,
        mergeable_state="clean",
    )
    branched = _pr(
        2,
        "Regular PR title",
        created_at="2026-05-01T00:00:00Z",
        updated_at="2026-05-01T00:00:00Z",
        mergeable=True,
        mergeable_state="clean",
        login="some-bot",
        head_ref="dependabot/github_actions/actions/checkout-5.0.0",
    )
    assert MODULE.is_dependabot_pr(authored) is True
    assert MODULE.is_dependabot_pr(branched) is True


def test_extracts_dependency_from_branch_name() -> None:
    pr = _pr(
        3,
        "Update action",
        created_at="2026-05-01T00:00:00Z",
        updated_at="2026-05-01T00:00:00Z",
        mergeable=True,
        mergeable_state="clean",
        login="automation-bot",
        head_ref="dependabot/github_actions/actions/checkout-5.0.0",
    )
    assert MODULE.dependency_key_from_pr(pr) == "actions/checkout"


def test_classifies_stale_behind_pr_for_branch_update() -> None:
    pr = _pr(
        10,
        "Bump urllib3 from 2.1.0 to 2.2.0",
        created_at="2026-05-01T00:00:00Z",
        updated_at="2026-05-10T00:00:00Z",
        mergeable=True,
        mergeable_state="behind",
        head_ref="dependabot/pip/urllib3-2.2.0",
    )
    records = MODULE.build_audit([pr], {10: _issue()}, {10: _checks("success")}, 7, NOW)
    assert records[0]["recommendation"] == "update_branch"
    assert records[0]["outdated_base_branch"] is True
    assert records[0]["checks_outdated"] is True


def test_marks_older_duplicate_as_superseded() -> None:
    newer = _pr(
        20,
        "Bump requests from 2.31.0 to 2.32.0",
        created_at="2026-05-20T00:00:00Z",
        updated_at="2026-05-20T00:00:00Z",
        mergeable=True,
        mergeable_state="clean",
        head_ref="dependabot/pip/requests-2.32.0",
    )
    older = _pr(
        19,
        "Bump requests from 2.30.0 to 2.31.0",
        created_at="2026-05-01T00:00:00Z",
        updated_at="2026-05-02T00:00:00Z",
        mergeable=True,
        mergeable_state="clean",
        head_ref="dependabot/pip/requests-2.31.0",
    )
    records = MODULE.build_audit(
        [older, newer],
        {19: _issue(), 20: _issue()},
        {19: _checks("success"), 20: _checks("success")},
        7,
        NOW,
    )
    older_record = next(record for record in records if record["number"] == 19)
    newer_record = next(record for record in records if record["number"] == 20)
    assert older_record["recommendation"] == "close_superseded"
    assert older_record["superseded_by"] == 20
    assert newer_record["recommendation"] == "ready_to_merge"


def test_marks_old_conflict_for_recreate_or_manual_rebase() -> None:
    pr = _pr(
        30,
        "Bump pydantic from 2.7.0 to 2.8.0",
        created_at="2026-05-01T00:00:00Z",
        updated_at="2026-05-03T00:00:00Z",
        mergeable=False,
        mergeable_state="dirty",
        head_ref="dependabot/pip/pydantic-2.8.0",
    )
    records = MODULE.build_audit(
        [pr],
        {30: _issue("stale")},
        {30: _checks("failure")},
        7,
        NOW,
    )
    assert records[0]["recommendation"] == "recreate_or_manual_rebase"


class _FakeClient:
    def __init__(self) -> None:
        self.closed: list[int] = []
        self.updated: list[int] = []

    def close_pr(self, _repo: str, number: int):
        self.closed.append(number)
        return {"number": number, "state": "closed"}

    def update_branch(self, _repo: str, number: int, _expected_head_sha: str):
        self.updated.append(number)
        return {"number": number, "updated": True}


def test_apply_close_actions_only_closes_superseded_prs() -> None:
    records = [
        {"number": 1, "recommendation": "close_superseded"},
        {"number": 2, "recommendation": "update_branch", "can_update_branch": True, "is_stale": True, "head_sha": "sha-2"},
        {"number": 3, "recommendation": "close_superseded"},
    ]
    client = _FakeClient()
    MODULE.apply_close_actions(client, "org/repo", records, max_closes=1)
    assert client.closed == [1]
    assert records[0]["close_pr"]["success"] is True
    assert records[2]["close_pr"]["skipped"] == "max_closes_reached"


def test_render_watchdog_report_is_silent_when_healthy() -> None:
    report = {
        "repo": "org/repo",
        "prs": [
            {
                "number": 11,
                "title": "Bump urllib3 from 2.1.0 to 2.2.0",
                "recommendation": "ready_to_merge",
            }
        ],
    }
    assert MODULE.render_watchdog_report(report) == ""


def test_render_watchdog_report_surfaces_actions_and_manual_backlog() -> None:
    report = {
        "repo": "org/repo",
        "prs": [
            {
                "number": 12,
                "title": "Bump requests from 2.31.0 to 2.32.0",
                "recommendation": "close_superseded",
                "close_pr": {"attempted": True, "success": True},
            },
            {
                "number": 13,
                "title": "Bump pydantic from 2.7.0 to 2.8.0",
                "recommendation": "recreate_or_manual_rebase",
            },
        ],
    }
    text = MODULE.render_watchdog_report(report)
    assert "closed superseded PRs: 1/1" in text
    assert "#13 recreate_or_manual_rebase" in text