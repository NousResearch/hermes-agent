"""A stale Codex cooldown must be revalidated before discarding a fallback."""
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from unittest.mock import MagicMock
from agent.chat_completion_helpers import _candidate_pool_exhausted, try_activate_fallback
from agent.credential_pool import CredentialPool, PooledCredential


def test_fallback_candidate_reopens_after_upstream_quota_recovers():
    entry = PooledCredential(
        provider="openai-codex", id="fixture", label="fixture", auth_type="oauth",
        priority=0, source="manual", access_token="fixture-token",
        last_status="exhausted", last_status_at=time.time(), last_error_code=429,
        last_error_reason="usage_limit_reached",
        last_error_reset_at=time.time() + 3 * 86400,
    )
    pool = CredentialPool("openai-codex", [entry])
    agent = SimpleNamespace(_credential_pool=pool)
    with patch.object(pool, "_resync_stale_entry", side_effect=lambda e: e), \
            patch.object(pool, "_persist"), \
            patch.object(pool, "_codex_quota_restored_upstream", return_value=True) as probe:
        assert not _candidate_pool_exhausted(agent, "openai-codex", "gpt-test")
        probe.assert_called_once()
        assert pool.has_available(model="gpt-test")


@pytest.mark.parametrize("restored", [False, None])
def test_unavailable_quota_does_not_lift_cooldown(restored):
    pool = MagicMock(provider="openai-codex")
    pool.has_available.return_value = False
    pool.next_available_at.return_value = time.time() + 86400
    pool.lift_reopened_cooldowns.return_value = restored
    assert _candidate_pool_exhausted(SimpleNamespace(_credential_pool=pool), "openai-codex", "m")
    pool.lift_reopened_cooldowns.assert_called_once_with(model="m")


def test_probe_exception_preserves_exhaustion():
    pool = MagicMock(provider="openai-codex")
    pool.has_available.return_value = False
    pool.next_available_at.return_value = time.time() + 86400
    pool.lift_reopened_cooldowns.side_effect = RuntimeError("fixture failure")
    assert _candidate_pool_exhausted(SimpleNamespace(_credential_pool=pool), "openai-codex", "m")


@pytest.mark.parametrize("provider,available,wait", [
    ("openai-codex", True, 86400),
    ("openai-codex", False, 30),
    ("openrouter", False, 86400),
])
def test_available_short_throttle_and_other_providers_do_not_probe(provider, available, wait):
    pool = MagicMock(provider=provider)
    pool.has_available.return_value = available
    pool.next_available_at.return_value = time.time() + wait
    result = _candidate_pool_exhausted(SimpleNamespace(_credential_pool=pool), provider, "m")
    assert result is (not available and wait > 600)
    pool.lift_reopened_cooldowns.assert_not_called()


def test_real_fallback_activation_rebinds_after_quota_recovers():
    from tests.agent.test_fallback_credential_isolation import _make_agent
    agent = _make_agent(provider="anthropic", model="claude", base_url="https://api.anthropic.com",
                        api_mode="chat_completions")
    agent._fallback_chain = [{"provider": "openai-codex", "model": "gpt-test"}]
    agent._is_azure_openai_url.return_value = False
    agent._is_direct_openai_url.return_value = False
    agent._provider_model_requires_responses_api.return_value = False
    agent._anthropic_prompt_cache_policy.return_value = (False, False)
    agent.context_compressor = None
    pool = MagicMock(provider="openai-codex")
    pool.has_available.return_value = False
    pool.next_available_at.return_value = time.time() + 86400
    pool.lift_reopened_cooldowns.return_value = True
    client = SimpleNamespace(api_key="fixture", base_url="https://chatgpt.com/backend-api/codex",
                             _custom_headers={})
    with patch("agent.credential_pool.load_pool", return_value=pool), \
            patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, "gpt-test")):
        assert try_activate_fallback(agent)
    assert agent.provider == "openai-codex"
    assert agent.model == "gpt-test"
    assert agent._credential_pool is pool
    pool.lift_reopened_cooldowns.assert_called_once_with(model="gpt-test")
