"""Static sandbox/approval policy contract tests.

Batch 002 is deliberately docs + contract tests only. These tests keep the
existing security posture explicit without changing runtime terminal or approval
behavior.
"""

from __future__ import annotations

import pathlib

from hermes_cli.config import DEFAULT_CONFIG
from tools import path_security
from tools.approval import DANGEROUS_PATTERNS, HARDLINE_PATTERNS_COMPILED

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
POLICY_DOC = REPO_ROOT / "docs" / "security" / "sandbox-approval-policy.md"
CONFIG_DOC = REPO_ROOT / "website" / "docs" / "user-guide" / "configuration.md"
SECURITY_DOC = REPO_ROOT / "SECURITY.md"


def test_default_approval_config_contract():
    approvals = DEFAULT_CONFIG.get("approvals", {})
    assert approvals["mode"] == "manual"
    assert approvals["cron_mode"] == "deny"
    assert approvals["destructive_slash_confirm"] is True
    assert approvals["mcp_reload_confirm"] is True
    assert isinstance(DEFAULT_CONFIG.get("command_allowlist"), list)


def test_approval_pattern_contract_keeps_hardline_floor_and_dangerous_gate():
    descriptions = {description for _pattern, description in DANGEROUS_PATTERNS}
    hardline_descriptions = {description for _pattern, description in HARDLINE_PATTERNS_COMPILED}

    assert "recursive delete" in descriptions
    assert "pipe remote content to shell" in descriptions
    assert "in-place edit of Hermes config/env" in descriptions
    assert "git reset --hard (destroys uncommitted changes)" in descriptions
    assert "recursive delete of root filesystem" in hardline_descriptions
    assert "system shutdown/reboot" in hardline_descriptions


def test_path_security_rejects_traversal_and_escape(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    inside = root / "nested" / "file.txt"
    outside = tmp_path / "outside.txt"

    assert path_security.validate_within_dir(inside, root) is None
    assert path_security.validate_within_dir(outside, root)
    assert path_security.has_traversal_component("nested/../escape") is True
    assert path_security.has_traversal_component("nested/safe") is False


def test_sandbox_approval_policy_doc_exists_and_names_runtime_boundaries():
    assert POLICY_DOC.exists(), f"Missing sandbox/approval policy doc: {POLICY_DOC}"
    content = POLICY_DOC.read_text(encoding="utf-8")
    required_phrases = [
        "OS-level isolation",
        "approval gate is a heuristic",
        "terminal-backend isolation",
        "whole-process wrapping",
        "config.yaml",
        "command_allowlist",
        "docker_mount_cwd_to_workspace",
        "no runtime behavior changes",
    ]
    for phrase in required_phrases:
        assert phrase in content, f"policy doc missing required phrase: {phrase}"


def test_security_and_configuration_docs_link_sandbox_policy():
    security = SECURITY_DOC.read_text(encoding="utf-8")
    config = CONFIG_DOC.read_text(encoding="utf-8")
    assert "docs/security/sandbox-approval-policy.md" in security
    assert "sandbox-approval-policy.md" in config
