"""Tests for gateway /usage command — agent cache lookup and output fields."""

import threading
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_agent(**overrides):
    """Create a mock AIAgent with realistic session counters."""
    agent = MagicMock()
    defaults = {
        "model": "anthropic/claude-sonnet-4.6",
        "provider": "openrouter",
        "base_url": None,
        "session_total_tokens": 50_000,
        "session_api_calls": 5,
        "session_prompt_tokens": 40_000,
        "session_completion_tokens": 10_000,
        "session_input_tokens": 35_000,
        "session_output_tokens": 10_000,
        "session_cache_read_tokens": 5_000,
        "session_cache_write_tokens": 2_000,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(agent, k, v)

    rl = MagicMock()
    rl.has_data = True
    agent.get_rate_limit_state.return_value = rl

    ctx = MagicMock()
    ctx.last_prompt_tokens = 30_000
    ctx.context_length = 200_000
    ctx.compression_count = 1
    agent.context_compressor = ctx

    return agent


def _make_runner(session_key, agent=None, cached_agent=None):
    """Build a bare GatewayRunner with just the fields _handle_usage_command needs."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner.session_store = MagicMock()

    if agent is not None:
        runner._running_agents[session_key] = agent

    if cached_agent is not None:
        runner._agent_cache[session_key] = (cached_agent, "sig")

    runner._session_key_for_source = MagicMock(return_value=session_key)

    return runner


SK = "agent:main:telegram:private:12345"


class TestUsageCachedAgent:
    """The main fix: /usage should find agents in _agent_cache between turns."""

    @pytest.mark.asyncio
    async def test_cached_agent_shows_detailed_usage(self):
        agent = _make_mock_agent()
        runner = _make_runner(SK, cached_agent=agent)
        event = MagicMock()

        with patch("agent.rate_limit_tracker.format_rate_limit_compact", return_value="RPM: 50/60"), \
             patch("gateway.run.fetch_all_relevant_providers", return_value=[]), \
             patch("agent.usage_pricing.estimate_usage_cost") as mock_cost:
            mock_cost.return_value = MagicMock(amount_usd=0.1234, status="estimated")
            result = await runner._handle_usage_command(event)

        assert "```" in result
        assert "#" * 79 in result
        assert "claude-sonnet-4.6" in result
        assert "35,000" in result
        assert "10,000" in result
        assert "5,000" in result
        assert "2,000" in result
        assert "50,000" in result
        assert "$0.1234" in result
        assert "30,000" in result
        assert "compressions 1" in result

    @pytest.mark.asyncio
    async def test_running_agent_preferred_over_cache(self):
        """When agent is in both dicts, the running one wins."""
        running = _make_mock_agent(session_api_calls=10, session_total_tokens=80_000)
        cached = _make_mock_agent(session_api_calls=5, session_total_tokens=50_000)
        runner = _make_runner(SK, agent=running, cached_agent=cached)
        event = MagicMock()

        with patch("agent.rate_limit_tracker.format_rate_limit_compact", return_value="RPM: 50/60"), \
             patch("gateway.run.fetch_all_relevant_providers", return_value=[]), \
             patch("agent.usage_pricing.estimate_usage_cost") as mock_cost:
            mock_cost.return_value = MagicMock(amount_usd=None, status="unknown")
            result = await runner._handle_usage_command(event)

        assert "80,000" in result
        assert "calls" in result and "10" in result

    @pytest.mark.asyncio
    async def test_sentinel_skipped_uses_cache(self):
        """PENDING sentinel in _running_agents should fall through to cache."""
        from gateway.run import _AGENT_PENDING_SENTINEL

        cached = _make_mock_agent()
        runner = _make_runner(SK, cached_agent=cached)
        runner._running_agents[SK] = _AGENT_PENDING_SENTINEL
        event = MagicMock()

        with patch("agent.rate_limit_tracker.format_rate_limit_compact", return_value="RPM: 50/60"), \
             patch("gateway.run.fetch_all_relevant_providers", return_value=[]), \
             patch("agent.usage_pricing.estimate_usage_cost") as mock_cost:
            mock_cost.return_value = MagicMock(amount_usd=None, status="unknown")
            result = await runner._handle_usage_command(event)

        assert "claude-sonnet-4.6" in result
        assert "#" * 79 in result

    @pytest.mark.asyncio
    async def test_no_agent_anywhere_falls_to_history(self):
        """No running or cached agent → rough estimate from transcript."""
        runner = _make_runner(SK)
        event = MagicMock()

        session_entry = MagicMock()
        session_entry.session_id = "sess123"
        runner.session_store.get_or_create_session.return_value = session_entry
        runner.session_store.load_transcript.return_value = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        with patch("agent.model_metadata.estimate_messages_tokens_rough", return_value=500):
            result = await runner._handle_usage_command(event)

        assert "Session Info" in result
        assert "Messages: 2" in result
        assert "~500" in result

    @pytest.mark.asyncio
    async def test_cache_read_write_hidden_when_zero(self):
        """Cache token lines should be omitted when zero."""
        agent = _make_mock_agent(session_cache_read_tokens=0, session_cache_write_tokens=0)
        runner = _make_runner(SK, cached_agent=agent)
        event = MagicMock()

        with patch("agent.rate_limit_tracker.format_rate_limit_compact", return_value="RPM: 50/60"), \
             patch("gateway.run.fetch_all_relevant_providers", return_value=[]), \
             patch("agent.usage_pricing.estimate_usage_cost") as mock_cost:
            mock_cost.return_value = MagicMock(amount_usd=None, status="unknown")
            result = await runner._handle_usage_command(event)

        assert "cache-r" not in result
        assert "cache-w" not in result

    @pytest.mark.asyncio
    async def test_cost_included_status(self):
        """Subscription-included providers show 'included' instead of dollar amount."""
        agent = _make_mock_agent(provider="openai-codex")
        runner = _make_runner(SK, cached_agent=agent)
        event = MagicMock()

        with patch("agent.rate_limit_tracker.format_rate_limit_compact", return_value="RPM: 50/60"), \
             patch("gateway.run.fetch_all_relevant_providers", return_value=[]), \
             patch("agent.usage_pricing.estimate_usage_cost") as mock_cost:
            mock_cost.return_value = MagicMock(amount_usd=None, status="included")
            result = await runner._handle_usage_command(event)

        assert "cost" in result
        assert "included" in result


class TestUsageAccountSection:
    """Provider balance and quota sections appended to compact /usage output."""

    @pytest.mark.asyncio
    async def test_usage_command_includes_account_section(self, monkeypatch):
        agent = _make_mock_agent(provider="openai-codex")
        agent.base_url = "https://chatgpt.com/backend-api/codex"
        agent.api_key = "unused"
        runner = _make_runner(SK, cached_agent=agent)
        event = MagicMock()

        monkeypatch.setattr(
            "gateway.run.fetch_all_relevant_providers",
            lambda provider, base_url=None, api_key=None: [],
        )
        monkeypatch.setattr(
            "gateway.run.render_multi_provider_hash",
            lambda snapshots: [("openai-codex", "Provider: openai-codex (Pro)")],
        )
        with patch("agent.rate_limit_tracker.format_rate_limit_compact", return_value="RPM: 50/60"), \
             patch("agent.usage_pricing.estimate_usage_cost") as mock_cost:
            mock_cost.return_value = MagicMock(amount_usd=None, status="included")
            result = await runner._handle_usage_command(event)

        assert "#" * 79 in result
        assert "BALANCES" in result
        assert "Provider: openai-codex (Pro)" in result

    @pytest.mark.asyncio
    async def test_usage_command_uses_persisted_provider_when_agent_not_running(self, monkeypatch):
        runner = _make_runner(SK)
        runner._session_db = MagicMock()
        runner._session_db.get_session.return_value = {
            "billing_provider": "openai-codex",
            "billing_base_url": "https://chatgpt.com/backend-api/codex",
        }
        session_entry = MagicMock()
        session_entry.session_id = "sess-1"
        runner.session_store.get_or_create_session.return_value = session_entry
        runner.session_store.load_transcript.return_value = [
            {"role": "user", "content": "earlier"},
        ]

        calls = {}

        async def _fake_to_thread(fn, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs
            return fn(*args, **kwargs)

        monkeypatch.setattr("gateway.run.asyncio.to_thread", _fake_to_thread)
        monkeypatch.setattr(
            "gateway.run.fetch_all_relevant_providers",
            lambda provider, base_url=None, api_key=None: [],
        )
        monkeypatch.setattr(
            "gateway.run.render_multi_provider_hash",
            lambda snapshots: [("openai-codex", "Provider: openai-codex (Pro)")],
        )

        event = MagicMock()
        result = await runner._handle_usage_command(event)

        assert calls["args"] == ("openai-codex",)
        assert calls["kwargs"]["base_url"] == "https://chatgpt.com/backend-api/codex"
        assert "#" * 79 in result
        assert "BALANCES" in result

    @pytest.mark.asyncio
    async def test_usage_command_renders_quota_sections_in_compact_table(self, monkeypatch):
        from datetime import datetime
        from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow

        agent = _make_mock_agent(provider="openrouter")
        runner = _make_runner(SK, cached_agent=agent)
        event = MagicMock()

        monkeypatch.setattr(
            "gateway.run.fetch_all_relevant_providers",
            lambda provider, base_url=None, api_key=None: [
                AccountUsageSnapshot(
                    provider="openrouter",
                    source="credits_api",
                    fetched_at=datetime.now(),
                    details=("Credits balance: $44.48",),
                ),
                AccountUsageSnapshot(
                    provider="anthropic",
                    source="oauth_usage_api",
                    fetched_at=datetime.now(),
                    windows=(
                        AccountUsageWindow(
                            label="Current session",
                            used_percent=55.0,
                            detail="in 3h 10m",
                        ),
                    ),
                ),
                AccountUsageSnapshot(
                    provider="openai-codex",
                    source="usage_api",
                    fetched_at=datetime.now(),
                    windows=(
                        AccountUsageWindow(
                            label="5h limit",
                            used_percent=0.0,
                            detail="in 5h 26m",
                        ),
                    ),
                ),
            ],
        )

        with patch("agent.rate_limit_tracker.format_rate_limit_compact", return_value="RPM: 50/60"), \
             patch("agent.usage_pricing.estimate_usage_cost") as mock_cost:
            mock_cost.return_value = MagicMock(amount_usd=0.1234, status="estimated")
            result = await runner._handle_usage_command(event)

        assert "claude code" in result
        assert "codex / openai" in result
        assert "| bal $44.48 |" in result
        assert "[███████" in result or "[████████" in result
