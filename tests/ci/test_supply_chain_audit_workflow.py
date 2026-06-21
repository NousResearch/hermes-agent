from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "supply-chain-audit.yml"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_install_hook_review_label_is_narrowly_scoped_to_setup_hits():
    text = _workflow_text()

    assert "install-hook-reviewed" in text
    assert "SETUP_HITS=$(git diff --name-only" in text
    assert "grep -Fxq 'install-hook-reviewed'" in text

    setup_start = text.index("SETUP_HITS=$(git diff --name-only")
    setup_block = text[setup_start:text.index("if [ -n \"$FINDINGS\" ]", setup_start)]

    assert "LABELS=$(gh pr view" in setup_block
    assert "Reviewed install-hook change allowed by install-hook-reviewed label" in setup_block
    assert "Install-hook file added or modified" in setup_block


def test_install_hook_label_does_not_bypass_other_critical_patterns():
    text = _workflow_text()

    setup_start = text.index("SETUP_HITS=$(git diff --name-only")
    pre_setup = text[:setup_start]

    # The maintainer-review label must not appear in the .pth, base64+exec/eval,
    # or obfuscated subprocess checks. Those remain unconditional critical
    # findings when they match.
    assert "install-hook-reviewed" not in pre_setup
    assert "PTH_FILES=$(git diff --name-only" in pre_setup
    assert "B64_EXEC_HITS=$(echo \"$DIFF\"" in pre_setup
    assert "PROC_HITS=$(echo \"$DIFF\"" in pre_setup


def test_mcp_catalog_review_label_gate_is_unchanged():
    text = _workflow_text()

    assert "mcp-catalog-reviewed" in text
    assert "MCP catalog security review" in text
    assert "install-hook-reviewed" != "mcp-catalog-reviewed"
