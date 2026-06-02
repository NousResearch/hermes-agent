"""Tests for the /system_prompt command, prompt-size renderer, and last-turn /usage.

Covers:
  * Command registry: /system_prompt + aliases, gateway-visible.
  * Renderer: no raw SOUL/MEMORY/USER/system-prompt text leaks into output.
  * Gateway /system_prompt: resolves a cached agent without a network call.
  * Gateway /usage: last-turn cached/uncached breakdown.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Command registration
# ---------------------------------------------------------------------------

class TestSystemPromptCommandRegistration:
    def test_command_registered(self):
        from hermes_cli.commands import resolve_command
        cmd = resolve_command("system_prompt")
        assert cmd is not None
        assert cmd.name == "system_prompt"
        assert cmd.category == "Info"

    def test_aliases_resolve(self):
        from hermes_cli.commands import resolve_command
        for alias in ("system-prompt", "prompt"):
            cmd = resolve_command(alias)
            assert cmd is not None
            assert cmd.name == "system_prompt"

    def test_gateway_visible(self):
        """Unlike CLI-only commands, /system_prompt should be reachable from the gateway."""
        from hermes_cli.commands import resolve_command, GATEWAY_KNOWN_COMMANDS
        cmd = resolve_command("system_prompt")
        assert cmd is not None
        assert cmd.cli_only is False
        assert "system_prompt" in GATEWAY_KNOWN_COMMANDS


# ---------------------------------------------------------------------------
# Renderer — privacy + structure
# ---------------------------------------------------------------------------

def _fake_breakdown():
    return {
        "platform": "telegram",
        "resident": True,
        "model": "claude-opus-4-8",
        "provider": "claude-api-proxy",
        "system_prompt": {"chars": 40_000, "bytes": 41_000},
        "tiers": [
            {"label": "stable", "description": "identity / guidance / skills catalog", "chars": 30_000, "bytes": 30_500},
            {"label": "context", "description": "cwd context files / caller system message", "chars": 2_000, "bytes": 2_050},
            {"label": "volatile", "description": "memory / user profile / timestamp", "chars": 8_000, "bytes": 8_450},
        ],
        "major_blocks": {
            "skills_catalog": {"label": "skills catalog (<available_skills>)", "chars": 20_000, "bytes": 20_100},
            "memory": {"label": "MEMORY.md snapshot", "chars": 2_800, "bytes": 2_847},
            "user_profile": {"label": "USER.md profile", "chars": 1_360, "bytes": 1_360},
        },
        "tools": {
            "count": 3,
            "chars": 12_000,
            "bytes": 12_010,
            "toolsets": [
                {"toolset": "web", "tools": 2, "chars": 8_000, "bytes": 8_010},
                {"toolset": "file", "tools": 1, "chars": 4_000, "bytes": 4_000},
            ],
        },
        "cache_note": "Hermes caches the whole system prompt as one provider prefix block.",
    }


class TestRendererPrivacyAndStructure:
    def test_render_contains_sizes_not_raw_content(self):
        from hermes_cli.prompt_size import render_system_prompt_breakdown
        out = render_system_prompt_breakdown(_fake_breakdown(), markdown=True)
        # Structural labels present
        assert "System prompt breakdown" in out
        assert "skills catalog" in out
        assert "MEMORY.md snapshot" in out
        assert "USER.md profile" in out
        assert "Tool schemas by toolset" in out
        assert "web" in out and "file" in out
        # Privacy note present
        assert "raw" in out.lower()

    def test_render_never_emits_known_secret_markers(self):
        """Renderer only receives sizes, never raw text — guard against regressions
        that might start passing raw blocks through."""
        from hermes_cli.prompt_size import render_system_prompt_breakdown
        out = render_system_prompt_breakdown(_fake_breakdown(), markdown=False)
        # These would only appear if raw SOUL/MEMORY content were threaded in.
        assert "OP_SERVICE_ACCOUNT_TOKEN" not in out
        assert "Universal Homelab Password" not in out

    def test_legacy_breakdown_shape_preserved(self):
        """compute_prompt_breakdown keeps its historical keys for `hermes prompt-size`."""
        from hermes_cli import prompt_size

        fake = {
            "platform": "cli",
            "resident": False,
            "model": "m",
            "provider": "p",
            "system_prompt": {"chars": 10, "bytes": 11},
            "tiers": [
                {"label": "stable", "description": "d", "chars": 5, "bytes": 5},
                {"label": "context", "description": "d", "chars": 2, "bytes": 2},
                {"label": "volatile", "description": "d", "chars": 3, "bytes": 4},
            ],
            "major_blocks": {
                "skills_catalog": {"label": "s", "chars": 4, "bytes": 4},
                "memory": {"label": "m", "chars": 2, "bytes": 2},
                "user_profile": {"label": "u", "chars": 1, "bytes": 1},
            },
            "tools": {"count": 1, "chars": 6, "bytes": 6, "toolsets": []},
            "cache_note": "n",
        }
        with patch.object(prompt_size, "compute_system_prompt_breakdown", return_value=fake), \
             patch.object(prompt_size, "_build_inspection_agent", return_value=MagicMock()):
            data = prompt_size.compute_prompt_breakdown("cli")
        assert set(data) >= {"platform", "model", "system_prompt", "skills_index", "memory", "user_profile", "tools", "sections"}
        assert data["skills_index"]["chars"] == 4
        assert data["tools"]["json_bytes"] == 6
        assert data["tools"]["count"] == 1
        assert len(data["sections"]) == 3


# ---------------------------------------------------------------------------
# Gateway /system_prompt — agent lookup, offline
# ---------------------------------------------------------------------------

def _make_runner(session_key, agent=None, cached_agent=None):
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner.session_store = MagicMock()
    if agent is not None:
        runner._running_agents[session_key] = agent
    if cached_agent is not None:
        runner._agent_cache[session_key] = (cached_agent, "sig")
    runner._session_key_for_source = MagicMock(return_value=session_key)
    return runner


SK = "agent:aegis:telegram:private:571820863"


class TestGatewaySystemPromptCommand:
    @pytest.mark.asyncio
    async def test_uses_cached_agent_and_renders(self):
        agent = MagicMock()
        runner = _make_runner(SK, cached_agent=agent)
        event = MagicMock()
        event.source = MagicMock()

        fake = _fake_breakdown()
        with patch("hermes_cli.prompt_size.compute_system_prompt_breakdown", return_value=fake) as mock_compute:
            result = await runner._handle_system_prompt_command(event)

        # Resident agent passed through (read-only inspection)
        assert mock_compute.call_count == 1
        _, kwargs = mock_compute.call_args
        assert kwargs.get("agent") is agent
        assert kwargs.get("resident") is True
        assert "System prompt breakdown" in result

    @pytest.mark.asyncio
    async def test_no_agent_falls_back_to_offline_inspection(self):
        runner = _make_runner(SK)  # no running, no cached
        event = MagicMock()
        event.source = MagicMock()

        fake = _fake_breakdown()
        fake["resident"] = False
        with patch("hermes_cli.prompt_size.compute_system_prompt_breakdown", return_value=fake) as mock_compute:
            result = await runner._handle_system_prompt_command(event)

        _, kwargs = mock_compute.call_args
        assert kwargs.get("agent") is None
        assert kwargs.get("resident") is False
        assert "System prompt breakdown" in result


# ---------------------------------------------------------------------------
# Gateway /usage — last-turn breakdown
# ---------------------------------------------------------------------------

def _make_usage_agent(last_turn=None):
    agent = MagicMock()
    agent.model = "claude-opus-4-8"
    agent.provider = "claude-api-proxy"
    agent.base_url = None
    agent.session_total_tokens = 50_000
    agent.session_api_calls = 3
    agent.session_input_tokens = 35_000
    agent.session_output_tokens = 10_000
    agent.session_cache_read_tokens = 5_000
    agent.session_cache_write_tokens = 0
    agent.session_reasoning_tokens = 0
    agent.last_turn_usage = last_turn
    rl = MagicMock()
    rl.has_data = False
    agent.get_rate_limit_state.return_value = rl
    ctx = MagicMock()
    ctx.last_prompt_tokens = 30_000
    ctx.context_length = 200_000
    ctx.compression_count = 0
    agent.context_compressor = ctx
    return agent


class TestGatewayUsageLastTurn:
    @pytest.mark.asyncio
    async def test_last_turn_section_shown(self):
        from agent.usage_pricing import CanonicalUsage
        last = CanonicalUsage(input_tokens=2_000, output_tokens=800, cache_read_tokens=28_000, cache_write_tokens=0)
        agent = _make_usage_agent(last_turn=last)
        runner = _make_runner(SK, cached_agent=agent)
        event = MagicMock()
        event.source = MagicMock()

        with patch("agent.usage_pricing.estimate_usage_cost") as mock_cost:
            mock_cost.return_value = MagicMock(amount_usd=None, status="included")
            result = await runner._handle_usage_command(event)

        assert "Last turn" in result
        assert "Cached: 28,000" in result
        # prompt = 2000 + 28000 = 30000; uncached = 30000 - 28000 = 2000
        assert "Uncached: 2,000" in result

    @pytest.mark.asyncio
    async def test_last_turn_section_absent_when_no_usage(self):
        agent = _make_usage_agent(last_turn=None)
        runner = _make_runner(SK, cached_agent=agent)
        event = MagicMock()
        event.source = MagicMock()

        with patch("agent.usage_pricing.estimate_usage_cost") as mock_cost:
            mock_cost.return_value = MagicMock(amount_usd=None, status="included")
            result = await runner._handle_usage_command(event)

        assert "Last turn" not in result
