"""Reset-aware primary restore — stay on fallback until the primary's
rate-limit window actually resets.

``restore_primary_runtime`` retries the primary at the top of every turn
once the 60s ``_rate_limited_until`` cooldown clears.  For transient 429s
that is correct, but subscription-window limits (Claude Pro/Max 5-hour
windows, ChatGPT weekly caps) report reset times hours or days away.  The
credential pool already knows that timestamp (``last_error_reset_at``),
and until it elapses every restore attempt is a guaranteed failure that
invalidates the prompt cache twice per turn (primary → fail → fallback).

The gate must FAIL OPEN: no pool, no reset info, or any error in the gate
falls through to the existing per-turn retry, so recovery can never happen
later than it does today.
"""

import time
from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from agent.credential_pool import (
    STATUS_DEAD,
    STATUS_EXHAUSTED,
    STATUS_OK,
    CredentialPool,
    PooledCredential,
)


# =============================================================================
# Helpers
# =============================================================================

def _entry(
    provider="openrouter",
    id="cred-1",
    status=STATUS_OK,
    reset_at=None,
    status_at=None,
    error_code=None,
):
    return PooledCredential(
        provider=provider,
        id=id,
        label=f"label-{id}",
        auth_type="api_key",
        priority=1,
        source="manual",
        access_token="sk-test-1234567890",
        last_status=status,
        last_status_at=status_at,
        last_error_code=error_code,
        last_error_reset_at=reset_at,
    )


class _FakePool:
    """Minimal stand-in for CredentialPool in agent-level tests."""

    def __init__(self, provider, next_at=None, available=False, raise_on_next=False):
        self.provider = provider
        self._next_at = next_at
        self._available = available
        self._raise = raise_on_next
        self.next_available_calls = 0

    def next_available_at(self):
        self.next_available_calls += 1
        if self._raise:
            raise RuntimeError("boom")
        return self._next_at

    def has_credentials(self):
        return True

    def has_available(self):
        return self._available

    def select(self):
        return None


def _make_tool_defs(*names):
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _make_agent(fallback_model=None):
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-12345678",
            base_url="https://my-llm.example.com/v1",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _activate_fallback(agent):
    mock_client = MagicMock()
    mock_client.api_key = "fallback-key-1234"
    mock_client.base_url = "https://openrouter.ai/api/v1"
    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(mock_client, None),
    ):
        assert agent._try_activate_fallback() is True
    assert agent._fallback_activated is True


# =============================================================================
# CredentialPool.next_available_at()
# =============================================================================

class TestNextAvailableAt:
    def test_all_exhausted_returns_earliest_reset(self):
        now = time.time()
        pool = CredentialPool(
            "openrouter",
            [
                _entry(id="a", status=STATUS_EXHAUSTED, reset_at=now + 7200, error_code=429),
                _entry(id="b", status=STATUS_EXHAUSTED, reset_at=now + 3600, error_code=429),
            ],
        )
        assert pool.next_available_at() == now + 3600

    def test_available_entry_returns_none(self):
        now = time.time()
        pool = CredentialPool(
            "openrouter",
            [
                _entry(id="a", status=STATUS_OK),
                _entry(id="b", status=STATUS_EXHAUSTED, reset_at=now + 3600, error_code=429),
            ],
        )
        assert pool.next_available_at() is None

    def test_elapsed_cooldown_counts_as_available(self):
        """An exhausted entry whose reset time has passed re-enters rotation,
        so the pool reports available (None) even without clear_expired."""
        now = time.time()
        pool = CredentialPool(
            "openrouter",
            [_entry(id="a", status=STATUS_EXHAUSTED, reset_at=now - 10, error_code=429)],
        )
        assert pool.next_available_at() is None

    def test_exhausted_without_timestamps_returns_none(self):
        """No reset info at all -> None (fail open), not a guess."""
        pool = CredentialPool(
            "openrouter",
            [_entry(id="a", status=STATUS_EXHAUSTED, reset_at=None, status_at=None)],
        )
        assert pool.next_available_at() is None

    def test_exhausted_with_status_at_uses_ttl(self):
        """Without an explicit reset_at, last_status_at + TTL is the estimate."""
        now = time.time()
        pool = CredentialPool(
            "openrouter",
            [_entry(id="a", status=STATUS_EXHAUSTED, status_at=now, error_code=429)],
        )
        result = pool.next_available_at()
        assert result is not None
        assert result > now

    def test_dead_only_returns_none(self):
        """DEAD entries never re-enter via TTL; report no wait info."""
        pool = CredentialPool(
            "openrouter",
            [_entry(id="a", status=STATUS_DEAD, status_at=time.time())],
        )
        assert pool.next_available_at() is None

    def test_empty_pool_returns_none(self):
        pool = CredentialPool("openrouter", [])
        assert pool.next_available_at() is None


# =============================================================================
# restore_primary_runtime() gate
# =============================================================================

class TestResetAwareRestoreGate:
    FB = {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}

    def test_stays_on_fallback_until_reset(self):
        agent = _make_agent(fallback_model=self.FB)
        _activate_fallback(agent)
        agent._rate_limited_until = 0  # 60s transient cooldown already cleared

        # Attached pool matches the primary provider and says: nobody can
        # serve until an hour from now.
        agent._credential_pool = _FakePool("custom", next_at=time.time() + 3600)

        assert agent._restore_primary_runtime() is False
        assert agent._fallback_activated is True
        assert agent.provider == "openrouter"
        assert agent.model == "anthropic/claude-sonnet-4"

    def test_restores_once_reset_elapsed(self):
        agent = _make_agent(fallback_model=self.FB)
        original_model = agent.model
        _activate_fallback(agent)
        agent._rate_limited_until = 0

        agent._credential_pool = _FakePool("custom", next_at=None)

        with patch("run_agent.OpenAI", return_value=MagicMock()):
            assert agent._restore_primary_runtime() is True
        assert agent._fallback_activated is False
        assert agent.model == original_model
        assert agent.provider == "custom"

    def test_past_reset_time_does_not_block(self):
        agent = _make_agent(fallback_model=self.FB)
        _activate_fallback(agent)
        agent._rate_limited_until = 0

        agent._credential_pool = _FakePool("custom", next_at=time.time() - 5)

        with patch("run_agent.OpenAI", return_value=MagicMock()):
            assert agent._restore_primary_runtime() is True
        assert agent._fallback_activated is False

    def test_fails_open_on_pool_error(self):
        """Any exception inside the gate must not break restore."""
        agent = _make_agent(fallback_model=self.FB)
        _activate_fallback(agent)
        agent._rate_limited_until = 0

        agent._credential_pool = _FakePool("custom", raise_on_next=True)

        with patch("run_agent.OpenAI", return_value=MagicMock()):
            assert agent._restore_primary_runtime() is True
        assert agent._fallback_activated is False

    def test_cross_provider_fallback_loads_primary_pool(self):
        """After a cross-provider fallback the attached pool belongs to the
        fallback provider; the gate must consult the PRIMARY's pool."""
        agent = _make_agent(fallback_model=self.FB)
        _activate_fallback(agent)
        agent._rate_limited_until = 0

        # Attached pool is the fallback provider's (mismatch with "custom").
        agent._credential_pool = _FakePool("openrouter", next_at=None)
        primary_pool = _FakePool("custom", next_at=time.time() + 3600)

        with patch("agent.credential_pool.load_pool", return_value=primary_pool) as lp:
            assert agent._restore_primary_runtime() is False
        assert any(c.args == ("custom",) for c in lp.call_args_list)
        assert primary_pool.next_available_calls == 1
        assert agent._fallback_activated is True

    def test_no_pool_info_falls_through(self):
        """Pool present but no reset info -> existing per-turn retry."""
        agent = _make_agent(fallback_model=self.FB)
        _activate_fallback(agent)
        agent._rate_limited_until = 0

        agent._credential_pool = _FakePool("custom", next_at=None)

        with patch("run_agent.OpenAI", return_value=MagicMock()):
            assert agent._restore_primary_runtime() is True

    def test_logs_wait_only_once(self, caplog):
        import logging

        agent = _make_agent(fallback_model=self.FB)
        _activate_fallback(agent)
        agent._rate_limited_until = 0
        agent._credential_pool = _FakePool("custom", next_at=time.time() + 3600)

        with caplog.at_level(logging.INFO, logger="agent.agent_runtime_helpers"):
            assert agent._restore_primary_runtime() is False
            assert agent._restore_primary_runtime() is False
        waits = [r for r in caplog.records if "staying on fallback" in r.getMessage()]
        assert len(waits) == 1

    def test_transient_cooldown_still_respected(self):
        """The existing 60s monotonic gate fires before the reset-aware one."""
        agent = _make_agent(fallback_model=self.FB)
        _activate_fallback(agent)
        agent._rate_limited_until = time.monotonic() + 60
        pool = _FakePool("custom", next_at=None)
        agent._credential_pool = pool

        assert agent._restore_primary_runtime() is False
        # Reset-aware gate never consulted — short-circuited by the 60s gate.
        assert pool.next_available_calls == 0
