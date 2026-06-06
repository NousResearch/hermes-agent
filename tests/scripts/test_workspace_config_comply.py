from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "workspace_config_comply.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_workspace_config_comply", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_issue_file(path: Path, *, done: int, remaining: int, status: str = "complete", evidence: str = "tests passed") -> None:
    path.write_text(
        "\n".join(
            [
                "# Phase",
                "",
                "| issue_id | phase | goal | done_when | verify_commands | localhost_check | vps_check | status | done_percent | remaining_percent | evidence |",
                "|---|---|---|---|---|---|---|---|---:|---:|---|",
                f"| WCG-P0-I01 | P0 | safe | done | pytest | not_applicable | not_applicable | {status} | {done} | {remaining} | {evidence} |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_complete_issue_file_reports_100_percent(tmp_path):
    module = _load_module()
    issue_file = tmp_path / "issue.md"
    _write_issue_file(issue_file, done=100, remaining=0)

    report = module.inspect_issue_file(issue_file)

    assert report["ok"] is True
    assert report["summary"]["done_percent"] == 100.0
    assert report["summary"]["remaining_percent"] == 0.0


def test_non_wcg_issue_prefix_is_parsed(tmp_path):
    module = _load_module()
    issue_file = tmp_path / "issue.md"
    _write_issue_file(issue_file, done=100, remaining=0)
    issue_file.write_text(issue_file.read_text(encoding="utf-8").replace("WCG-P0-I01", "WCR-P0-I01"), encoding="utf-8")

    report = module.inspect_issue_file(issue_file)

    assert report["summary"]["issues"] == 1
    assert report["issues"][0]["issue_id"] == "WCR-P0-I01"
    assert report["ok"] is True


def test_incomplete_issue_file_fails_closed(tmp_path):
    module = _load_module()
    issue_file = tmp_path / "issue.md"
    _write_issue_file(issue_file, done=80, remaining=20, status="in_progress")

    report = module.inspect_issue_file(issue_file)

    assert report["ok"] is False
    assert report["summary"]["done_percent"] == 80.0
    assert report["issues"][0]["ok"] is False


def test_pending_evidence_fails_even_with_100_percent(tmp_path):
    module = _load_module()
    issue_file = tmp_path / "issue.md"
    _write_issue_file(issue_file, done=100, remaining=0, evidence="pending")

    report = module.inspect_issue_file(issue_file)

    assert report["ok"] is False
    assert report["issues"][0]["ok"] is False


def test_text_report_contains_numeric_compliance(tmp_path):
    module = _load_module()
    issue_file = tmp_path / "issue.md"
    _write_issue_file(issue_file, done=100, remaining=0)

    rendered = module.render_report(module.inspect_issue_file(issue_file))

    assert "| Phase | Issue | รายละเอียด | ทำได้ % | เหลือ % | หลักฐานตรวจ | สถานะ |" in rendered
    assert "| P0 | WCG-P0-I01 | safe | 100 | 0 | tests passed | complete |" in rendered
