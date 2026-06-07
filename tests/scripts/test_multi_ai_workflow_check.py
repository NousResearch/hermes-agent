from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "multi_ai_workflow_check.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_multi_ai_workflow_check", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_ready_project(root: Path) -> None:
    (root / "AGENTS.md").write_text(
        "# AGENTS\nPrompt Shortcut Registry\nCloseout Protocol\nverification\nrecommended next step\n",
        encoding="utf-8",
    )
    (root / "CLAUDE.md").write_text("@AGENTS.md\n", encoding="utf-8")
    (root / "QWEN.md").write_text("Read AGENTS.md before work.\n", encoding="utf-8")
    (root / "GEMINI.md").write_text("@AGENTS.md\n", encoding="utf-8")

    cursor_rules = root / ".cursor" / "rules"
    cursor_rules.mkdir(parents=True)
    (cursor_rules / "multi-ai.mdc").write_text(
        "---\nalwaysApply: true\n---\nRead AGENTS.md.\n",
        encoding="utf-8",
    )

    hermes = root / ".hermes"
    issues = hermes / "issues"
    ai_pair = hermes / "ai-pair"
    issues.mkdir(parents=True)
    ai_pair.mkdir(parents=True)
    (hermes / "context.md").write_text("# Context\n", encoding="utf-8")
    (hermes / "active.md").write_text("# Active\n", encoding="utf-8")
    (hermes / "decisions.md").write_text("# Decisions\n", encoding="utf-8")
    (hermes / "handoff.md").write_text("# Handoff\n", encoding="utf-8")
    (issues / "README.md").write_text("# Issues\n", encoding="utf-8")
    (ai_pair / "README.md").write_text("# AI Pair Jobs\n", encoding="utf-8")

    templates = root / "docs" / "multi-ai-workflow" / "templates"
    templates.mkdir(parents=True)
    (templates / "issue.md").write_text(
        "\n".join(
            [
                "issue_id:",
                "phase:",
                "owner_role:",
                "assigned_ai:",
                "worktree_path:",
                "branch:",
                "goal:",
                "scope:",
                "out_of_scope:",
                "done_when:",
                "verify_commands:",
                "localhost_check:",
                "vps_check:",
                "status:",
                "done_percent:",
                "remaining_percent:",
                "evidence:",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (templates / "handoff.md").write_text(
        "\n".join(
            [
                "task:",
                "latest_state:",
                "next_agent:",
                "next_step:",
                "verification_run:",
                "remaining_risk:",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ai_pair_templates = templates / "ai-pair"
    ai_pair_templates.mkdir(parents=True)
    (ai_pair_templates / "coder-brief.md").write_text(
        "review_focus:\ncommands_run:\n",
        encoding="utf-8",
    )
    (ai_pair_templates / "review-result.md").write_text(
        "decision:\nrequired_changes:\n",
        encoding="utf-8",
    )


def test_ready_project_reports_ok(tmp_path):
    module = _load_module()
    _write_ready_project(tmp_path)

    report = module.inspect_project(tmp_path)

    assert report["ok"] is True
    assert report["summary"]["passed"] == report["summary"]["total"]
    assert report["summary"]["failed"] == 0


def test_missing_adapter_reports_failures(tmp_path):
    module = _load_module()
    _write_ready_project(tmp_path)
    (tmp_path / "QWEN.md").unlink()
    for path in (tmp_path / ".cursor" / "rules").glob("*"):
        path.unlink()

    report = module.inspect_project(tmp_path)

    assert report["ok"] is False
    failed_codes = {check["code"] for check in report["checks"] if not check["ok"]}
    assert "qwen-adapter-present" in failed_codes
    assert "cursor-adapter-present" in failed_codes


def test_issue_template_requires_verification_and_percent_fields(tmp_path):
    module = _load_module()
    _write_ready_project(tmp_path)
    issue_template = tmp_path / "docs" / "multi-ai-workflow" / "templates" / "issue.md"
    issue_template.write_text("issue_id:\nphase:\nstatus:\n", encoding="utf-8")

    report = module.inspect_project(tmp_path)

    assert report["ok"] is False
    failed_codes = {check["code"] for check in report["checks"] if not check["ok"]}
    assert "issue-template-fields" in failed_codes


def test_agents_adapter_requires_closeout_protocol(tmp_path):
    module = _load_module()
    _write_ready_project(tmp_path)
    (tmp_path / "AGENTS.md").write_text(
        "# AGENTS\nPrompt Shortcut Registry\n",
        encoding="utf-8",
    )

    report = module.inspect_project(tmp_path)

    assert report["ok"] is False
    failed_codes = {check["code"] for check in report["checks"] if not check["ok"]}
    assert "agents-closeout-protocol" in failed_codes


def test_json_report_is_machine_readable(tmp_path):
    module = _load_module()
    _write_ready_project(tmp_path)

    rendered = module.render_report(module.inspect_project(tmp_path), "json")
    parsed = json.loads(rendered)

    assert parsed["project"] == str(tmp_path)
    assert parsed["ok"] is True


def test_ai_pair_templates_are_required_for_readiness(tmp_path):
    module = _load_module()
    _write_ready_project(tmp_path)

    report = module.inspect_project(tmp_path)

    assert report["ok"] is True
    codes = {check["code"] for check in report["checks"]}
    assert "ai-pair-registry-present" in codes
    assert "ai-pair-coder-template-fields" in codes
    assert "ai-pair-review-template-fields" in codes


def test_missing_ai_pair_template_reports_failure(tmp_path):
    module = _load_module()
    _write_ready_project(tmp_path)
    (tmp_path / "docs" / "multi-ai-workflow" / "templates" / "ai-pair" / "coder-brief.md").write_text(
        "review_focus:\n",
        encoding="utf-8",
    )

    report = module.inspect_project(tmp_path)

    assert report["ok"] is False
    failed_codes = {check["code"] for check in report["checks"] if not check["ok"]}
    assert "ai-pair-coder-template-fields" in failed_codes
