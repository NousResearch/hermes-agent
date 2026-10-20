from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "pr_review_brief.py"


def load_module():
    spec = importlib.util.spec_from_file_location("pr_review_brief", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_brief_ignores_closed_prs_and_orders_actionable_first():
    mod = load_module()
    prs = [
        {
            "number": 11,
            "title": "docs: already merged",
            "state": "MERGED",
            "url": "https://github.com/acme/app/pull/11",
        },
        {
            "number": 12,
            "title": "feat: ordinary open change",
            "state": "OPEN",
            "url": "https://github.com/acme/app/pull/12",
            "additions": 20,
            "deletions": 5,
            "updatedAt": "2026-06-06T10:00:00Z",
        },
        {
            "number": 13,
            "title": "fix: failing check needs Joe",
            "state": "OPEN",
            "url": "https://github.com/acme/app/pull/13",
            "reviewDecision": "REVIEW_REQUIRED",
            "statusCheckRollup": [{"conclusion": "FAILURE", "name": "tests"}],
            "additions": 8,
            "deletions": 2,
            "updatedAt": "2026-06-07T00:30:00Z",
        },
    ]

    brief = mod.build_brief(prs, now="2026-06-07T01:00:00Z", silent_empty=False)

    assert "# PR Review Brief" in brief
    assert "#13" in brief
    assert "failing check needs Joe" in brief
    assert "failed checks" in brief
    assert "#12" in brief
    assert "already merged" not in brief
    assert brief.index("#13") < brief.index("#12")


def test_build_brief_silent_empty_returns_exact_silent_token():
    mod = load_module()

    brief = mod.build_brief(
        [{"number": 1, "title": "done", "state": "CLOSED"}],
        now="2026-06-07T01:00:00Z",
        silent_empty=True,
    )

    assert brief == "[SILENT]"


def test_cli_reads_gh_json_shape_and_writes_brief(tmp_path):
    payload = [
        {
            "number": 21,
            "title": "refactor: prune morning noise",
            "state": "OPEN",
            "url": "https://github.com/acme/app/pull/21",
            "reviewDecision": "CHANGES_REQUESTED",
            "isDraft": False,
            "additions": 120,
            "deletions": 15,
            "author": {"login": "agent"},
        }
    ]
    input_path = tmp_path / "prs.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(input_path),
            "--now",
            "2026-06-07T01:00:00Z",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "# PR Review Brief" in result.stdout
    assert "#21" in result.stdout
    assert "changes requested" in result.stdout
    assert "135 LOC" in result.stdout
