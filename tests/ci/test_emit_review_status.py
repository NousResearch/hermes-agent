"""Tests for scripts/ci/emit_review_status.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "emit_review_status.py"
_spec = importlib.util.spec_from_file_location("emit_review_status", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load emit_review_status.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["emit_review_status"] = _mod
_spec.loader.exec_module(_mod)


def test_ci_review_status_links_to_each_sensitive_file_at_pr_head():
    results = _mod.build_results(
        ci_review=True,
        mcp_catalog=False,
        supply_chain=False,
        label_present=False,
        ci_review_files='[".github/workflows/ci.yml", "apps/desktop/eslint.config.mjs"]',
        repo_url="https://github.com/nousresearch/hermes-agent",
        head_sha="abc123",
    )

    assert results[0]["detail"] == (
        "**Sensitive files:**\n"
        "- [`.github/workflows/ci.yml`](https://github.com/nousresearch/hermes-agent/blob/abc123/.github/workflows/ci.yml)\n"
        "- [`apps/desktop/eslint.config.mjs`](https://github.com/nousresearch/hermes-agent/blob/abc123/apps/desktop/eslint.config.mjs)"
    )