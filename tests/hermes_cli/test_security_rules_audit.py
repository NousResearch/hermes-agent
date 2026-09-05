"""Tests for hermes_cli.security_rules_audit — security rule coverage audit.

Validates the arXiv:2608.23550 mapping of natural-language rules (in memories,
context files, skills) to deterministic controls (approvals.deny, HARDLINE,
DANGEROUS_PATTERNS, sensitive file write guards).
"""

import json
from pathlib import Path
import pytest

from hermes_cli.security_rules_audit import (
    SecurityRule,
    SecurityRulesAuditReport,
    _classify_rule,
    _discover_scan_files,
    _extract_rules_from_file,
    audit_security_rules,
    run_security_rules_audit_cli,
)


class TestRuleExtraction:
    def test_extract_imperative_negative_rules(self, tmp_path: Path):
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text(
            "# Guidelines\n"
            "- Never push directly to main branch.\n"
            "- Do not touch the production database without approval.\n"
            "- Must not delete .env files.\n"
            "- Don't execute sudo commands.\n"
            "- Always write clean code.\n"
            "```bash\n"
            "never run this inside code block\n"
            "```\n"
            "- Strictly forbidden to drop database tables.\n",
            encoding="utf-8",
        )

        extracted = _extract_rules_from_file(agents_md)
        assert len(extracted) == 5
        lines = [item[1] for item in extracted]
        assert "Never push directly to main branch." in lines
        assert "Do not touch the production database without approval." in lines
        assert "Must not delete .env files." in lines
        assert "Don't execute sudo commands." in lines
        assert "Strictly forbidden to drop database tables." in lines

    def test_code_blocks_and_headings_ignored(self, tmp_path: Path):
        file_p = tmp_path / "CLAUDE.md"
        file_p.write_text(
            "# Do not panic\n"
            "```python\n"
            "# Never do this\n"
            "do_not_call_this()\n"
            "```\n",
            encoding="utf-8",
        )
        assert len(_extract_rules_from_file(file_p)) == 0


class TestRuleClassification:
    def test_hardline_enforced(self):
        cat, mech, glob, cmd = _classify_rule(
            "Never rm -rf / or delete root",
            "never rm -rf /",
            active_deny_globs=[],
        )
        assert cat == "enforced"
        assert "HARDLINE" in (mech or "")

    def test_sensitive_write_target_enforced(self):
        cat, mech, glob, cmd = _classify_rule(
            "Do not modify or edit .env credentials",
            "do not modify .env",
            active_deny_globs=[],
        )
        assert cat == "enforced"
        assert "file/env protection" in (mech or "")

    def test_dangerous_pattern_db_drop_enforced(self):
        cat, mech, glob, cmd = _classify_rule(
            "Never drop database in staging or production",
            "never drop database",
            active_deny_globs=[],
        )
        assert cat == "enforced"
        assert "DANGEROUS_PATTERNS" in (mech or "")

    def test_active_deny_glob_enforced(self):
        cat, mech, glob, cmd = _classify_rule(
            "Never run terraform apply in prod",
            "never run terraform apply",
            active_deny_globs=["terraform apply*"],
        )
        assert cat == "enforced"
        assert "approvals.deny" in (mech or "")

    def test_enforceable_rule_suggests_glob_and_command(self):
        cat, mech, glob, cmd = _classify_rule(
            "Do not push directly to origin main",
            "do not push directly to origin main",
            active_deny_globs=[],
        )
        assert cat == "enforceable"
        assert glob == "git push *main*"
        assert cmd is not None
        assert "hermes config set approvals.deny" in cmd
        assert "git push *main*" in cmd

    def test_enforceable_terraform_apply(self):
        cat, mech, glob, cmd = _classify_rule(
            "Never run terraform apply in staging",
            "never run terraform apply",
            active_deny_globs=[],
        )
        assert cat == "enforceable"
        assert glob == "terraform apply*"

    def test_advisory_only_rule(self):
        cat, mech, glob, cmd = _classify_rule(
            "Never assume user intent without asking",
            "never assume user intent",
            active_deny_globs=[],
        )
        assert cat == "advisory"
        assert "Advisory" in (mech or "")
        assert glob is None
        assert cmd is None


class TestAuditSecurityRulesWorkflow:
    def test_audit_across_mock_workspace(self, tmp_path: Path):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        memories = hermes_home / "memories"
        memories.mkdir()
        skills = hermes_home / "skills" / "deploy"
        skills.mkdir(parents=True)

        # 1. Project AGENTS.md
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text(
            "- Never push to main branch.\n"
            "- Do not touch .env files.\n",
            encoding="utf-8",
        )

        # 2. Memory file
        memory_md = memories / "MEMORY.md"
        memory_md.write_text(
            "- Never drop table users.\n"
            "- Always write unit tests.\n",
            encoding="utf-8",
        )

        # 3. Skill file
        skill_md = skills / "SKILL.md"
        skill_md.write_text(
            "- Do not run npm publish without tag.\n"
            "- Never assume environment is clean.\n",
            encoding="utf-8",
        )

        report = audit_security_rules(hermes_home=hermes_home, cwd=tmp_path)
        assert len(report.rules) >= 5
        assert len(report.enforced) >= 2  # .env protection + drop table gate
        assert len(report.enforceable) >= 2  # git push main, npm publish
        assert len(report.advisory) >= 1  # never assume environment

        report_dict = report.to_dict()
        assert report_dict["summary"]["total_rules"] == len(report.rules)
        assert report_dict["summary"]["enforceable_count"] == len(report.enforceable)

    def test_run_security_rules_audit_cli(self, tmp_path: Path, capsys):
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text(
            "- Never push to main.\n"
            "- Do not edit .env.\n",
            encoding="utf-8",
        )
        run_security_rules_audit_cli(hermes_home=tmp_path / ".hermes", cwd=tmp_path)
        captured = capsys.readouterr().out
        assert "Hermes Security-Rule Coverage Audit" in captured
        assert "enforced by deterministic controls" in captured
        assert "enforceable" in captured
