import datetime as dt
import io
import sys

import scripts.morning_pr_brief as morning_pr_brief
from scripts.morning_pr_brief import build_brief, normalize_pull_requests


def test_normalize_pull_requests_scores_review_priority():
    now = dt.datetime(2026, 6, 21, 0, 0, tzinfo=dt.timezone.utc)
    raw = [
        {
            "number": 10,
            "title": "fix gateway restart",
            "url": "https://example.test/pull/10",
            "headRefName": "joe/fix-gateway",
            "author": {"login": "alice"},
            "updatedAt": "2026-06-20T22:00:00Z",
            "reviewDecision": "REVIEW_REQUIRED",
            "isDraft": False,
            "mergeStateStatus": "CLEAN",
        },
        {
            "number": 11,
            "title": "draft large refactor",
            "url": "https://example.test/pull/11",
            "headRefName": "joe/refactor",
            "author": {"login": "bob"},
            "updatedAt": "2026-06-10T00:00:00Z",
            "reviewDecision": "APPROVED",
            "isDraft": True,
            "mergeStateStatus": "DIRTY",
        },
    ]

    prs = normalize_pull_requests(raw, now=now)

    assert [pr.number for pr in prs] == [10, 11]
    assert prs[0].priority == "Review now"
    assert prs[0].age_label == "2h"
    assert prs[1].priority == "Stale / blocked"
    assert prs[1].age_label == "11d"


def test_build_brief_groups_prs_and_limits_output():
    now = dt.datetime(2026, 6, 21, 0, 0, tzinfo=dt.timezone.utc)
    raw = [
        {
            "number": 1,
            "title": "ready change",
            "url": "https://example.test/pull/1",
            "headRefName": "joe/ready",
            "author": {"login": "alice"},
            "updatedAt": "2026-06-20T23:30:00Z",
            "reviewDecision": "REVIEW_REQUIRED",
            "isDraft": False,
            "mergeStateStatus": "CLEAN",
        },
        {
            "number": 2,
            "title": "approved change",
            "url": "https://example.test/pull/2",
            "headRefName": "joe/approved",
            "author": {"login": "alice"},
            "updatedAt": "2026-06-19T00:00:00Z",
            "reviewDecision": "APPROVED",
            "isDraft": False,
            "mergeStateStatus": "CLEAN",
        },
        {
            "number": 3,
            "title": "old conflict",
            "url": "https://example.test/pull/3",
            "headRefName": "joe/conflict",
            "author": {"login": "alice"},
            "updatedAt": "2026-06-01T00:00:00Z",
            "reviewDecision": "REVIEW_REQUIRED",
            "isDraft": False,
            "mergeStateStatus": "DIRTY",
        },
    ]

    brief = build_brief(raw, now=now, limit=2)

    assert "# Morning PR review brief" in brief
    assert "## Review now" in brief
    assert "#1 ready change" in brief
    assert "## Stale / blocked" in brief
    assert "#3 old conflict" in brief
    assert "#2 approved change" not in brief
    assert "1 additional PR omitted" in brief


def test_load_json_falls_back_to_gh_when_automation_stdin_is_empty(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(
        morning_pr_brief,
        "_fetch_open_pr_json",
        lambda: '[{"number": 99, "title": "from gh"}]',
    )

    loaded = morning_pr_brief._load_json(None)

    assert loaded == [{"number": 99, "title": "from gh"}]


def test_build_brief_returns_silent_marker_when_empty():
    now = dt.datetime(2026, 6, 21, 0, 0, tzinfo=dt.timezone.utc)

    assert build_brief([], now=now) == "[SILENT]"
