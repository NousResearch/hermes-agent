"""Tests for scripts/review_debt_brief.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "review_debt_brief.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_review_debt_brief_under_test", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_pull_requests_accepts_list_and_common_wrappers(tmp_path):
    module = _load_module()
    prs = [{"number": 1, "title": "fix one", "url": "https://example.test/1"}]

    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps(prs), encoding="utf-8")
    wrapped_path = tmp_path / "wrapped.json"
    wrapped_path.write_text(json.dumps({"pullRequests": prs}), encoding="utf-8")

    assert module.load_pull_requests(list_path)[0].number == 1
    assert module.load_pull_requests(wrapped_path)[0].title == "fix one"


def test_actionable_items_filter_closed_merged_and_draft_prs():
    module = _load_module()
    raw_prs = [
        {"number": 1, "title": "open", "url": "https://example.test/1", "state": "OPEN"},
        {"number": 2, "title": "closed", "url": "https://example.test/2", "state": "CLOSED"},
        {"number": 3, "title": "merged", "url": "https://example.test/3", "mergedAt": "2026-06-01T00:00:00Z"},
        {"number": 4, "title": "draft", "url": "https://example.test/4", "isDraft": True},
    ]

    actionable = module.actionable_items([module.PullRequest.from_raw(pr) for pr in raw_prs])

    assert [item.number for item in actionable] == [1]


def test_priority_order_prefers_joe_requested_failed_ci_and_stale_age():
    module = _load_module()
    raw_prs = [
        {
            "number": 10,
            "title": "fresh low priority",
            "url": "https://example.test/10",
            "state": "OPEN",
            "createdAt": "2026-06-04T00:00:00Z",
            "updatedAt": "2026-06-04T00:00:00Z",
        },
        {
            "number": 11,
            "title": "joe requested and failing",
            "url": "https://example.test/11",
            "state": "OPEN",
            "createdAt": "2026-05-25T00:00:00Z",
            "reviewRequests": [{"login": "joe102084"}],
            "reviewDecision": "REVIEW_REQUIRED",
            "statusCheckRollup": [{"state": "FAILURE", "name": "tests"}],
        },
        {
            "number": 12,
            "title": "stale but no direct ask",
            "url": "https://example.test/12",
            "state": "OPEN",
            "createdAt": "2026-05-20T00:00:00Z",
        },
    ]

    actionable = module.actionable_items(
        [module.PullRequest.from_raw(pr) for pr in raw_prs],
        joe_logins={"joe102084"},
        now=module.parse_datetime("2026-06-05T00:00:00Z"),
    )

    assert [item.number for item in actionable] == [11, 12, 10]
    assert actionable[0].priority == "P0"
    assert "review request includes Joe" in actionable[0].evidence
    assert "failing checks" in actionable[0].evidence


def test_render_markdown_uses_joe_style_traditional_chinese_sections():
    module = _load_module()
    pr = module.PullRequest.from_raw(
        {
            "number": 11,
            "title": "joe requested and failing",
            "url": "https://example.test/11",
            "state": "OPEN",
            "createdAt": "2026-05-25T00:00:00Z",
            "author": {"login": "alice"},
            "reviewRequests": [{"login": "joe102084"}],
            "statusCheckRollup": [{"state": "FAILURE", "name": "tests"}],
        }
    )

    output = module.render_markdown(
        module.actionable_items(
            [pr],
            joe_logins={"joe102084"},
            now=module.parse_datetime("2026-06-05T00:00:00Z"),
        )
    )

    assert output.startswith("## TL;DR")
    assert "## Fact / verified" in output
    assert "## Hypothesis" in output
    assert "## Action for Joe" in output
    assert "#11" in output
    assert "P0" in output
    assert "https://example.test/11" in output


def test_render_markdown_can_be_exactly_silent_when_empty():
    module = _load_module()

    assert module.render_markdown([], silent_if_empty=True) == "[SILENT]"


def test_cli_reads_json_and_prints_brief(tmp_path):
    data_path = tmp_path / "prs.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "number": 21,
                    "title": "needs Joe review",
                    "url": "https://example.test/21",
                    "state": "OPEN",
                    "createdAt": "2026-05-25T00:00:00Z",
                    "reviewRequests": [{"login": "joe102084"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(data_path),
            "--joe-login",
            "joe102084",
            "--now",
            "2026-06-05T00:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "## TL;DR" in result.stdout
    assert "#21" in result.stdout


def test_cli_silent_if_empty_is_exact(tmp_path):
    data_path = tmp_path / "prs.json"
    data_path.write_text("[]", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(data_path), "--silent-if-empty"],
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout == "[SILENT]\n"
