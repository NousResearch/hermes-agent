"""Tests for provider-agnostic billing recovery links (agent/billing_links.py).

Behavior/invariant tests — no snapshotting of the exact URL strings beyond the
few that are the whole point of the mapping (the host they must land on).
"""

from __future__ import annotations

from agent.billing_links import (
    build_billing_block,
)

def test_known_provider_by_slug_resolves_label_and_url():
    block = build_billing_block(provider="openai", base_url="", model="gpt-5")
    assert block.provider_label == "OpenAI"
    assert block.billing_url is not None
    assert "openai.com" in block.billing_url
