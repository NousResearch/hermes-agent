"""Tests for gateway /usage command — agent cache lookup and output fields."""

import asyncio
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
    from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL

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

        with patch("agent.rate_limit_tracker.format_rate_limit_compact", return_value="RPM: 50/60"), \
             patch("agent.usage_pricing.estimate_usage_cost") as mock_cost:
            mock_cost.return_value = MagicMock(amount_usd=0.1234, status="estimated")
            result = await runner._handle_usage_command(event)

        assert "claude-sonnet-4.6" in result
        assert "35,000" in result  # input tokens
        assert "10,000" in result  # output tokens
        assert "5,000" in result   # cache read
        assert "2,000" in result   # cache write
        assert "50,000" in result  # total
        assert "$0.1234" in result
        assert "30,000" in result  # context
        assert "Compressions: 1" in result

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

    @pytest.mark.asyncio
    async def test_cost_included_status(self):
        """Subscription-included providers show 'included' instead of dollar amount."""
        agent = _make_mock_agent(provider="openai-codex")
        runner = _make_runner(SK, cached_agent=agent)
        event = MagicMock()

        with patch("agent.rate_limit_tracker.format_rate_limit_compact", return_value="RPM: 50/60"), \
             patch("agent.usage_pricing.estimate_usage_cost") as mock_cost:
            mock_cost.return_value = MagicMock(amount_usd=None, status="included")
            result = await runner._handle_usage_command(event)

        assert "Cost: included" in result


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
            "gateway.run.fetch_account_usage",
            lambda provider, base_url=None, api_key=None: object(),
        )
        monkeypatch.setattr(
            "gateway.run.render_account_usage_lines",
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
            "gateway.run.fetch_account_usage",
            lambda provider, base_url=None, api_key=None: object(),
        )
        monkeypatch.setattr(
            "gateway.run.render_account_usage_lines",
            lambda snapshot, markdown=False: [
                "📈 **Account limits**",
                "Provider: openai-codex (Pro)",
            ],
        )

        event = MagicMock()
        result = await runner._handle_usage_command(event)

        assert calls["args"] == ("openai-codex",)
        assert calls["kwargs"]["base_url"] == "https://chatgpt.com/backend-api/codex"
        assert "📊 **Session Info**" in result
        assert "📈 **Account limits**" in result

    @pytest.mark.asyncio
    async def test_usage_command_probes_credential_pool_when_no_agent_and_no_billing_row(
        self, monkeypatch,
    ):
        """Regression guard for #15167: a fresh-after-login user who fires
        ``/usage`` before sending a turn has (a) no agent in cache and
        (b) no ``billing_provider`` row on the session DB.  Before the fix,
        the handler would emit the 'Session Info' stub and never attempt
        to read account usage — even though an OAuth credential existed
        on disk in ``auth.json -> credential_pool``.  The fallback probe
        must now pick that credential up and resolve ``provider`` from it.
        """
        runner = _make_runner(SK)
        runner._session_db = MagicMock()
        # No billing row on the session — simulates a fresh session.
        runner._session_db.get_session.return_value = {}
        session_entry = MagicMock()
        session_entry.session_id = "sess-empty"
        runner.session_store.get_or_create_session.return_value = session_entry
        runner.session_store.load_transcript.return_value = []

        calls = {}

        async def _fake_to_thread(fn, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs
            return fn(*args, **kwargs)

        monkeypatch.setattr("gateway.run.asyncio.to_thread", _fake_to_thread)
        monkeypatch.setattr(
            "gateway.run.fetch_account_usage",
            lambda provider, base_url=None, api_key=None: object(),
        )
        monkeypatch.setattr(
            "gateway.run.render_account_usage_lines",
            lambda snapshot, markdown=False: [
                "📈 **Account limits**",
                "Provider: openai-codex (Plus)",
            ],
        )

        # Simulate an OAuth credential stored in the pool but no legacy
        # providers row and no billing row — the state `hermes auth add
        # openai-codex --type oauth` leaves on disk before the first turn.
        def _fake_read_pool(provider_id=None):
            if provider_id == "openai-codex":
                return [{
                    "auth_type": "oauth",
                    "access_token": "pool-at",
                    "refresh_token": "pool-rt",
                }]
            return []

        monkeypatch.setattr(
            "hermes_cli.auth.read_credential_pool", _fake_read_pool,
        )

        event = MagicMock()
        result = await runner._handle_usage_command(event)

        assert calls.get("args") == ("openai-codex",), (
            "gateway must probe the auth credential_pool when no agent "
            "and no billing row resolve a provider — otherwise /usage "
            "silently skips account-usage fetch (#15167)"
        )
        assert "📈 **Account limits**" in result

    @pytest.mark.asyncio
    async def test_usage_command_stops_probing_pool_on_first_hit(self, monkeypatch):
        """The probe iterates known providers in order (currently
        openai-codex, anthropic).  First one with a stored credential
        wins — we don't scan the whole pool and pick arbitrarily."""
        runner = _make_runner(SK)
        runner._session_db = MagicMock()
        runner._session_db.get_session.return_value = {}
        session_entry = MagicMock()
        session_entry.session_id = "sess-e"
        runner.session_store.get_or_create_session.return_value = session_entry
        runner.session_store.load_transcript.return_value = []

        calls = {}

        async def _fake_to_thread(fn, *args, **kwargs):
            calls["args"] = args
            return fn(*args, **kwargs)

        monkeypatch.setattr("gateway.run.asyncio.to_thread", _fake_to_thread)
        monkeypatch.setattr(
            "gateway.run.fetch_account_usage",
            lambda provider, base_url=None, api_key=None: object(),
        )
        monkeypatch.setattr(
            "gateway.run.render_account_usage_lines",
            lambda snapshot, markdown=False: ["📈 Account limits"],
        )

        probed = []

        def _fake_read_pool(provider_id=None):
            probed.append(provider_id)
            # Both providers have credentials — the first probed wins.
            return [{"auth_type": "oauth", "access_token": "x", "refresh_token": "y"}]

        monkeypatch.setattr(
            "hermes_cli.auth.read_credential_pool", _fake_read_pool,
        )

        event = MagicMock()
        await runner._handle_usage_command(event)

        assert probed == ["openai-codex"], (
            f"expected to stop after first hit, but probed {probed}"
        )
        assert calls["args"] == ("openai-codex",)

    @pytest.mark.asyncio
    async def test_usage_command_pool_probe_errors_fall_through_gracefully(
        self, monkeypatch,
    ):
        """The pool probe is strictly best-effort.  If ``read_credential_pool``
        raises (corrupted auth.json, permissions issue, etc.) the handler
        must fall through to the no-provider path rather than bubbling the
        exception up and breaking /usage entirely."""
        runner = _make_runner(SK)
        runner._session_db = MagicMock()
        runner._session_db.get_session.return_value = {}
        session_entry = MagicMock()
        session_entry.session_id = "sess-e"
        runner.session_store.get_or_create_session.return_value = session_entry
        runner.session_store.load_transcript.return_value = []

        def _boom(provider_id=None):
            raise RuntimeError("simulated auth.json corruption")

        monkeypatch.setattr("hermes_cli.auth.read_credential_pool", _boom)

        # fetch_account_usage should NOT be called, because provider is still
        # unresolved.  Make it raise if it IS called so a silent regression
        # fails the test loudly.
        monkeypatch.setattr(
            "gateway.run.fetch_account_usage",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("provider should not have been resolved")
            ),
        )

        event = MagicMock()
        # Must not raise — the handler swallows the pool-probe exception
        # and falls through to its normal no-provider path.  The exact
        # message ("Session Info" stub vs "No usage data available") depends
        # on whether the session has transcript history; either is fine,
        # the invariant is that the handler returned SOMETHING rather than
        # propagating the RuntimeError.
        result = await runner._handle_usage_command(event)
        assert isinstance(result, str) and result, (
            "handler must return a string even when pool probe raises"
        )
