from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "pr_review_brief.py"


def load_module():
    spec = importlib.util.spec_from_file_location("pr_review_brief", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_prs_accepts_gh_list_and_filters_closed_drafts_and_merged(tmp_path):
    data_path = tmp_path / "prs.json"
    data_path.write_text(
        json.dumps(
            [
                {"number": 1, "title": "Open", "url": "https://example.test/1", "state": "OPEN"},
                {"number": 2, "title": "Draft", "url": "https://example.test/2", "state": "OPEN", "isDraft": True},
                {"number": 3, "title": "Closed", "url": "https://example.test/3", "state": "CLOSED"},
                {"number": 4, "title": "Merged", "url": "https://example.test/4", "state": "MERGED", "mergedAt": "2026-05-01T00:00:00Z"},
            ]
        ),
        encoding="utf-8",
    )

    module = load_module()
    prs = module.load_prs(data_path, today="2026-06-01")

    assert [pr.number for pr in prs] == [1]
    assert prs[0].title == "Open"


def test_load_prs_accepts_object_wrapper_and_normalizes_review_requests(tmp_path):
    data_path = tmp_path / "prs.json"
    data_path.write_text(
        json.dumps(
            {
                "pullRequests": [
                    {
                        "number": 42,
                        "title": "Needs Joe",
                        "url": "https://example.test/42",
                        "createdAt": "2026-05-20T08:00:00Z",
                        "reviewRequests": [{"login": "joe102084"}],
                        "author": {"login": "alice"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    module = load_module()
    prs = module.load_prs(data_path, today="2026-06-01")

    assert len(prs) == 1
    assert prs[0].review_requests == ("joe102084",)
    assert prs[0].age_days == 12
    assert prs[0].author == "alice"


def test_prioritize_puts_joe_review_and_ci_failure_before_routine_items(tmp_path):
    data_path = tmp_path / "prs.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "number": 10,
                    "title": "Routine",
                    "url": "https://example.test/10",
                    "createdAt": "2026-05-31T00:00:00Z",
                },
                {
                    "number": 11,
                    "title": "CI failed",
                    "url": "https://example.test/11",
                    "createdAt": "2026-05-29T00:00:00Z",
                    "statusCheckRollup": [{"conclusion": "FAILURE"}],
                },
                {
                    "number": 12,
                    "title": "Awaiting Joe",
                    "url": "https://example.test/12",
                    "createdAt": "2026-05-25T00:00:00Z",
                    "reviewRequests": [{"login": "joe102084"}],
                },
            ]
        ),
        encoding="utf-8",
    )

    module = load_module()
    ranked = module.prioritize(module.load_prs(data_path, today="2026-06-01"), reviewer="joe102084")

    assert [pr.number for pr in ranked] == [12, 11, 10]
    assert ranked[0].priority_label == "需要 Joe Review"
    assert ranked[1].priority_label == "CI 失敗"


def test_render_brief_uses_joe_style_traditional_chinese_sections(tmp_path):
    data_path = tmp_path / "prs.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "number": 99,
                    "title": "Add local helper",
                    "url": "https://github.com/example/repo/pull/99",
                    "createdAt": "2026-05-20T00:00:00Z",
                    "author": {"login": "bot"},
                    "reviewRequests": [{"login": "joe102084"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    module = load_module()
    brief = module.render_brief(module.load_prs(data_path, today="2026-06-01"), reviewer="joe102084", max_items=3)

    assert brief.startswith("## TL;DR")
    assert "Fact / verified" in brief
    assert "Hypothesis" in brief
    assert "Action for Joe" in brief
    assert "#99" in brief
    assert "需要 Joe Review" in brief
    assert "https://github.com/example/repo/pull/99" in brief


def test_render_brief_returns_exact_silent_when_no_actionable_prs():
    module = load_module()

    assert module.render_brief([], reviewer="joe102084", silent_if_empty=True) == "[SILENT]"


def test_cli_prints_brief(tmp_path):
    data_path = tmp_path / "prs.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "number": 7,
                    "title": "Review queue",
                    "url": "https://example.test/7",
                    "createdAt": "2026-05-29T00:00:00Z",
                    "reviewDecision": "REVIEW_REQUIRED",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(data_path), "--today", "2026-06-01"],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "## TL;DR" in result.stdout
    assert "#7" in result.stdout


def test_cli_silent_if_empty(tmp_path):
    data_path = tmp_path / "prs.json"
    data_path.write_text(json.dumps({"prs": []}), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(data_path), "--silent-if-empty"],
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == "[SILENT]"
