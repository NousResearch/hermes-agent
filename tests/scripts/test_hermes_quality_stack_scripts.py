from __future__ import annotations

import json
from pathlib import Path

from scripts.hermes_quality_stack_comply import _ISSUES, build_compliance_report
from scripts.hermes_security_gate import check_security_gate
from scripts.hermes_stack_pilot import build_pilot_report


def test_security_gate_detects_required_files(tmp_path: Path) -> None:
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "osv-scanner.yml").write_text("name: OSV\n", encoding="utf-8")
    (tmp_path / ".github" / "workflows" / "supply-chain-audit.yml").write_text("name: Audit\n", encoding="utf-8")
    (tmp_path / ".semgrep").mkdir()
    (tmp_path / ".semgrep" / "hermes-security.yml").write_text("rules: []\n", encoding="utf-8")
    (tmp_path / "renovate.json").write_text('{"automerge": false}\n', encoding="utf-8")

    report = check_security_gate(tmp_path)

    assert report["score"] == 100
    assert all(item["ok"] for item in report["checks"])


def test_pilot_report_has_four_ranked_candidates() -> None:
    report = build_pilot_report()

    assert report["score"] == 100
    assert {item["id"] for item in report["candidates"]} == {
        "langgraph",
        "pydantic-ai",
        "graphrag",
        "openai-agents-sdk",
    }
    assert report["recommendation"]["first_pilot"] == "trend-discovery-v2"


def test_compliance_report_maps_all_phase_issues(tmp_path: Path) -> None:
    repo = tmp_path
    for _, _, paths in _ISSUES:
        for rel in paths:
            path = repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")

    report = build_compliance_report(repo)

    assert report["overall"]["done"] == 100
    assert report["overall"]["remaining"] == 0
    assert len(report["issues"]) >= 30
    assert all(issue["done"] == 100 for issue in report["issues"])
