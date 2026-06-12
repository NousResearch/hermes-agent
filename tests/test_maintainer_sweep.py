from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import maintainer_sweep


def test_maintainer_sweep_fixture_writes_proposal_reports(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "items.json"
    source.write_text(
        json.dumps(
            {
                "issues": [
                    {
                        "number": 12,
                        "title": "Bug in scheduler",
                        "state": "OPEN",
                        "url": "https://github.com/acme/widgets/issues/12",
                        "labels": [{"name": "bug"}],
                        "author": {"login": "alice"},
                    }
                ],
                "prs": [
                    {
                        "number": 34,
                        "title": "fix: scheduler bug",
                        "state": "OPEN",
                        "url": "https://github.com/acme/widgets/pull/34",
                        "labels": ["ci"],
                        "author": {"login": "bot"},
                        "isDraft": False,
                        "baseRefName": "main",
                        "headRefName": "clawley/scheduler-bug",
                        "mergeable": "MERGEABLE",
                        "reviewDecision": "APPROVED",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    state_dir = tmp_path / "state"
    rc = maintainer_sweep.main(
        ["--repo", "acme/widgets", "--source-file", str(source), "--state-dir", str(state_dir), "--json"]
    )

    assert rc == 0
    repo_dir = state_dir / "repos" / "acme__widgets"
    summary = json.loads((repo_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["records"] == 2
    assert summary["mutation_allowed"] is False
    assert summary["action_state"] == "proposal"

    issue_report = (repo_dir / "items" / "issue-12.md").read_text(encoding="utf-8")
    pr_report = (repo_dir / "items" / "pr-34.md").read_text(encoding="utf-8")
    dashboard = (repo_dir / "dashboard.md").read_text(encoding="utf-8")
    ledger_lines = (repo_dir / "ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()

    assert "recommendation: \"needs_human\"" in issue_report
    assert "mutation_allowed: false" in issue_report
    assert "Raw snapshot" in issue_report
    assert "Base: `main`" in pr_report
    assert "Do not mutate GitHub from this dashboard" in dashboard
    assert len(ledger_lines) == 2
    assert all(json.loads(line)["mutation_allowed"] is False for line in ledger_lines)


def test_snapshot_hash_is_stable_for_same_item() -> None:
    item = {"number": 1, "title": "x", "labels": ["a", "b"]}
    assert maintainer_sweep.snapshot_hash(item) == maintainer_sweep.snapshot_hash(dict(reversed(item.items())))
