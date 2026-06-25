"""Static network egress isolation contract tests.

Batch 003 is deliberately docs + contract tests only. These tests keep the
existing network egress isolation documentation explicit without changing
runtime behavior.
"""

from __future__ import annotations

import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NETWORK_DOC = REPO_ROOT / "docs" / "security" / "network-egress-isolation.md"
SANDBOX_DOC = REPO_ROOT / "docs" / "security" / "sandbox-approval-policy.md"
SECURITY_DOC = REPO_ROOT / "SECURITY.md"
PLAN_DOC = REPO_ROOT / "docs" / "plans" / "2026-06-20-003-network-egress-isolation-contract.md"


def test_network_egress_isolation_doc_exists_and_names_key_concepts():
    assert NETWORK_DOC.exists(), f"Missing network egress doc: {NETWORK_DOC}"
    content = NETWORK_DOC.read_text(encoding="utf-8")
    required_phrases = [
        "internal (no internet)",
        "egress (internet-capable)",
        "HTTP_PROXY=http://egress-proxy:3128",
        'network_mode: ""',
        "127.0.0.1:9119:9119",
        "Not a substitute for sandbox backends",
    ]
    for phrase in required_phrases:
        assert phrase in content, f"network doc missing required phrase: {phrase}"


def test_sandbox_approval_policy_links_to_network_egress():
    content = SANDBOX_DOC.read_text(encoding="utf-8")
    assert "network-egress-isolation.md" in content


def test_security_md_cross_links_network_egress_isolation():
    content = SECURITY_DOC.read_text(encoding="utf-8")
    assert "network-egress-isolation.md" in content


def test_plan_doc_exists_for_batch_003():
    assert PLAN_DOC.exists(), f"Missing plan doc: {PLAN_DOC}"
    content = PLAN_DOC.read_text(encoding="utf-8")
    assert "network-egress-isolation-contract" in content
    assert "Batch 003" in content or "batch 003" in content
