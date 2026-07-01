# Copyright 2025 Nous Research (Licensed under the Apache License, Version 2.0)
"""Tests for credential_pool_matches_agent provider gating."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.agent_runtime_helpers import credential_pool_matches_agent


def test_matches_same_provider():
    pool = SimpleNamespace(provider="openai-codex")
    agent = SimpleNamespace(provider="openai-codex", base_url="https://example.com/v1")
    assert credential_pool_matches_agent(pool, agent) is True


def test_rejects_cross_provider_mismatch():
    pool = SimpleNamespace(provider="deepseek")
    agent = SimpleNamespace(
        provider="custom",
        base_url="https://primary-host.example.com/v1",
    )
    assert credential_pool_matches_agent(pool, agent) is False


def test_accepts_custom_agent_with_matching_custom_pool_key():
    pool = SimpleNamespace(provider="custom:together")
    agent = SimpleNamespace(provider="custom", base_url="https://api.together.xyz/v1")

    with patch(
        "agent.credential_pool.get_custom_provider_pool_key",
        return_value="custom:together",
    ):
        assert credential_pool_matches_agent(pool, agent) is True
