from hermes_state import AsyncSessionDB
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

    # Rate limit state
    rl = MagicMock()
    rl.has_data = True
    agent.get_rate_limit_state.return_value = rl

    # Context compressor
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

    # Wire helper
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

        with patch("agent.rate_limit_tracker.format_rate_limit_compact", return_value="RPM: 50/60"):
            result = await runner._handle_usage_command(event)

        assert "claude-sonnet-4.6" in result
        assert "35,000" in result  # input tokens
        assert "10,000" in result  # output tokens
        assert "50,000" in result  # total
        assert "30,000" in result  # context
        assert "Compressions: 1" in result
        # Cost reporting stays omitted; nonzero cache buckets are visible.
        assert "$" not in result
        assert "Cache read tokens: 5,000" in result
        assert "Cache write tokens: 2,000" in result
        assert "Cost" not in result

    @pytest.mark.asyncio
    async def test_running_agent_preferred_over_cache(self):
        """When agent is in both dicts, the running one wins."""
        running = _make_mock_agent(session_api_calls=10, session_total_tokens=80_000)
        cached = _make_mock_agent(session_api_calls=5, session_total_tokens=50_000)
        runner = _make_runner(SK, agent=running, cached_agent=cached)
        event = MagicMock()

        with patch("agent.rate_limit_tracker.format_rate_limit_compact", return_value="RPM: 50/60"), \
             patch("agent.usage_pricing.estimate_usage_cost") as mock_cost:
            mock_cost.return_value = MagicMock(amount_usd=None, status="unknown")
            result = await runner._handle_usage_command(event)

        assert "80,000" in result   # running agent's total
        assert "API calls: 10" in result

    @pytest.mark.asyncio
    async def test_sentinel_skipped_uses_cache(self):
        """PENDING sentinel in _running_agents should fall through to cache."""
        from gateway.run import _AGENT_PENDING_SENTINEL

        cached = _make_mock_agent()
        runner = _make_runner(SK, cached_agent=cached)
        runner._running_agents[SK] = _AGENT_PENDING_SENTINEL
        event = MagicMock()

        with patch("agent.rate_limit_tracker.format_rate_limit_compact", return_value="RPM: 50/60"), \
             patch("agent.usage_pricing.estimate_usage_cost") as mock_cost:
            mock_cost.return_value = MagicMock(amount_usd=None, status="unknown")
            result = await runner._handle_usage_command(event)

        assert "claude-sonnet-4.6" in result
        assert "Session Token Usage" in result

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
             patch("agent.usage_pricing.estimate_usage_cost") as mock_cost:
            mock_cost.return_value = MagicMock(amount_usd=None, status="unknown")
            result = await runner._handle_usage_command(event)

        assert "Cache read" not in result
        assert "Cache write" not in result


class TestUsageAccountSection:
    """Account-limits section appended to /usage output (PR #2486)."""

    @pytest.mark.asyncio
    async def test_usage_command_includes_account_section(self, monkeypatch):
        agent = _make_mock_agent(provider="openai-codex")
        agent.base_url = "https://chatgpt.com/backend-api/codex"
        agent.api_key = "unused"
        runner = _make_runner(SK, cached_agent=agent)
        event = MagicMock()

        monkeypatch.setattr(
            "gateway.slash_commands.fetch_account_usage",
            lambda provider, base_url=None, api_key=None: object(),
        )
        monkeypatch.setattr(
            "gateway.slash_commands.render_account_usage_lines",
            lambda snapshot, markdown=False: [
                "📈 **Account limits**",
                "Provider: openai-codex (Pro)",
                "Session: 85% remaining (15% used)",
            ],
        )
        with patch("agent.rate_limit_tracker.format_rate_limit_compact", return_value="RPM: 50/60"), \
             patch("agent.usage_pricing.estimate_usage_cost") as mock_cost:
            mock_cost.return_value = MagicMock(amount_usd=None, status="included")
            result = await runner._handle_usage_command(event)

        assert "📊 **Session Token Usage**" in result
        assert "📈 **Account limits**" in result
        assert "Provider: openai-codex (Pro)" in result

    @pytest.mark.asyncio
    async def test_usage_command_uses_persisted_provider_when_agent_not_running(self, monkeypatch):
        runner = _make_runner(SK)
        runner._session_db = AsyncSessionDB(MagicMock())
        runner._session_db._db.get_session.return_value = {
            "billing_provider": "openai-codex",
            "billing_base_url": "https://chatgpt.com/backend-api/codex",
        }
        session_entry = MagicMock()
        session_entry.session_id = "sess-1"
        runner.session_store.get_or_create_session.return_value = session_entry
        runner.session_store.load_transcript.return_value = [
            {"role": "user", "content": "earlier"},
        ]

        calls = []

        async def _fake_to_thread(fn, *args, **kwargs):
            # /usage dispatches BOTH the account fetch (fetch_account_usage, called
            # with the provider positionally) and the Nous credits fetch
            # (nous_credits_lines, markdown-only) through to_thread — record every
            # call rather than last-wins so we can pick out the account fetch.
            calls.append({"args": args, "kwargs": kwargs})
            return fn(*args, **kwargs)

        monkeypatch.setattr("gateway.run.asyncio.to_thread", _fake_to_thread)
        monkeypatch.setattr(
            "gateway.slash_commands.fetch_account_usage",
            lambda provider, base_url=None, api_key=None: object(),
        )
        monkeypatch.setattr(
            "gateway.slash_commands.render_account_usage_lines",
            lambda snapshot, markdown=False: [
                "📈 **Account limits**",
                "Provider: openai-codex (Pro)",
            ],
        )
        # The credits block routes through the shared nous_credits_lines() helper;
        # stub it so this account-section test stays hermetic (no portal/auth lookup).
        monkeypatch.setattr("agent.account_usage.nous_credits_lines", lambda markdown=False: [])

        event = MagicMock()
        result = await runner._handle_usage_command(event)

        account_call = next(c for c in calls if c["args"] == ("openai-codex",))
        assert account_call["kwargs"]["base_url"] == "https://chatgpt.com/backend-api/codex"
        assert "📊 **Session Info**" in result
        assert "📈 **Account limits**" in result


class TestUsageContextBreakdown:
    """The /usage output includes the per-category context breakdown."""

    @pytest.mark.asyncio
    async def test_breakdown_lines_rendered_for_live_agent(self):
        agent = _make_mock_agent()
        runner = _make_runner(SK, cached_agent=agent)
        session_entry = MagicMock()
        session_entry.session_id = "sess-bd"
        runner.session_store.get_or_create_session.return_value = session_entry
        runner.session_store.load_transcript.return_value = [
            {"role": "user", "content": "hi"},
        ]
        event = MagicMock()

        fake_payload = {
            "categories": [
                {"id": "system_prompt", "label": "System prompt", "tokens": 4000, "color": "x"},
                {"id": "tool_definitions", "label": "Tool definitions", "tokens": 6000, "color": "x"},
                {"id": "conversation", "label": "Conversation", "tokens": 0, "color": "x"},
            ],
            "estimated_total": 10000,
            "context_max": 200000,
            "context_percent": 5,
            "context_used": 30000,
            "model": "anthropic/claude-sonnet-4.6",
        }

        with patch("agent.rate_limit_tracker.format_rate_limit_compact", return_value="RPM: 50/60"), \
             patch("agent.context_breakdown.compute_session_context_breakdown", return_value=fake_payload):
            result = await runner._handle_usage_command(event)

        # Localized header + at least the two non-zero category labels appear,
        # each labelled as a percentage of the estimated total.
        assert "Context breakdown" in result
        assert "System prompt" in result
        assert "Tool definitions" in result
        assert "4,000" in result   # system prompt tokens, comma-formatted
        assert "40%" in result     # 4000 / 10000
        assert "60%" in result     # 6000 / 10000
        # Zero-token category is dropped, not rendered.
        assert "Conversation" not in result

    @pytest.mark.asyncio
    async def test_breakdown_failure_is_non_fatal(self):
        """A breakdown engine error must not break the rest of /usage."""
        agent = _make_mock_agent()
        runner = _make_runner(SK, cached_agent=agent)
        runner.session_store.get_or_create_session.side_effect = RuntimeError("boom")
        event = MagicMock()

        with patch("agent.rate_limit_tracker.format_rate_limit_compact", return_value="RPM: 50/60"), \
             patch("agent.context_breakdown.compute_session_context_breakdown",
                   side_effect=RuntimeError("engine down")):
            result = await runner._handle_usage_command(event)

        # Core usage lines still render; no breakdown header.
        assert "📊 **Session Token Usage**" in result
        assert "50,000" in result  # total tokens
        assert "Context breakdown" not in result
