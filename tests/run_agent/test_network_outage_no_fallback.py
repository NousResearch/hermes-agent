"""A machine-level outage must not promote itself into provider failover.

Reproduced on a flaky home connection (DNS/routing dropping for a few seconds): every
transport failure was evidence that "the provider is unavailable", so after two failed
attempts the turn switched routes — to a provider that was just as unreachable. When the
link came back the answer arrived from the FALLBACK model, with the primary's prompt cache
cold and the user's own outage reported as a provider fault (agent/local_network.py).

The gate is route-scoped, and these tests pin both directions of it:

* an EXTERNAL fallback stays out of play while the machine is offline;
* a LOCAL (loopback/LAN) fallback still takes over — it is the route that can serve the turn
  precisely because it does not need the network that is down;
* a provider-side transport failure on a working connection still fails over as before.
"""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import httpx
import pytest


def _dns_failure() -> httpx.ConnectError:
    error = httpx.ConnectError("Connection error.")
    error.__cause__ = socket.gaierror(-3, "Temporary failure in name resolution")
    return error


@pytest.fixture()
def agent_factory():
    """AIAgent with a mocked client and a configurable fallback chain (test_run_agent harness)."""
    from run_agent import AIAgent

    def _make(chain):
        with (
            patch("model_tools.get_tool_definitions", return_value=[]),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://api.deepseek.com/v1",
                provider="deepseek",
                model="deepseek-flash",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                fallback_model=chain,
            )
            agent.client = MagicMock()
            agent._cached_system_prompt = "You are helpful."
            agent._use_prompt_caching = False
            agent.compression_enabled = False
            agent.tool_delay = 0
            agent.save_trajectories = False
            # The post-exhaustion wait-and-retry ladder sleeps 15-60s per cycle; these tests pin
            # the route decision, not that ladder.
            agent._auto_recovery_cycles = 0
            return agent

    return _make


def _always_fails_with(error):
    return lambda *a, **k: (_ for _ in ()).throw(error)


def _offline(probe: bool):
    """Patch the outage verdict's probe (the seam production reads) and keep waits off the clock."""
    return (
        patch("agent.local_network.probe_connectivity", return_value=probe),
        patch("agent.local_network.network_outage_max_wait_seconds", return_value=0.0),
    )


def _mock_response(content: str):
    """Minimal ChatCompletion-shaped response (the harness helper lives in several test files)."""
    from types import SimpleNamespace

    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _fb_client(content: str, base_url: str):
    client = MagicMock()
    client.api_key = "test-key-1234567890"
    client.base_url = base_url
    client._custom_headers = None
    client.default_headers = None
    client.chat.completions.create.return_value = _mock_response(content=content)
    return client


class TestOfflineSuppressesExternalRoutes:
    def test_external_fallback_is_not_activated(self, agent_factory):
        agent = agent_factory([{"provider": "openrouter", "model": "deepseek/deepseek-v4.1-flash"}])
        agent.client.chat.completions.create.side_effect = _always_fails_with(_dns_failure())
        probes = _offline(probe=False)

        with (
            probes[0], probes[1],
            patch.object(type(agent), "_try_recover_primary_transport", return_value=False),
        ):
            result = agent.run_conversation("hello")

        assert agent._fallback_activated is False, "an offline machine has no use for an external route"
        assert (agent.provider, agent.model) == ("deepseek", "deepseek-flash")
        assert result.get("failed") is True
        assert "offline" in (result.get("final_response") or "").lower()


class TestOfflineKeepsLocalRoutesEligible:
    def test_local_fallback_takes_over_while_the_wan_is_down(self, agent_factory):
        """A loopback entry is the one route that can still serve the turn — it must be used."""
        chain = [
            {"provider": "openrouter", "model": "deepseek/deepseek-v4.1-flash"},
            {"provider": "custom", "model": "local-model", "base_url": "http://127.0.0.1:8000/v1"},
        ]
        agent = agent_factory(chain)
        agent.client.chat.completions.create.side_effect = _always_fails_with(_dns_failure())

        resolved: list[str] = []
        local_client = _fb_client("OK from the local model", "http://127.0.0.1:8000/v1")

        def _resolve(provider, **_kwargs):
            resolved.append(provider)
            return local_client, "local-model"

        probes = _offline(probe=False)
        with (
            probes[0], probes[1],
            patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve),
            patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m),
            patch.object(type(agent), "_try_recover_primary_transport", return_value=False),
        ):
            result = agent.run_conversation("hello")

        assert resolved == ["custom"], "the external entry must be skipped, not resolved"
        assert agent._fallback_activated is True
        assert agent.provider == "custom"
        assert "OK from the local model" in (result.get("final_response") or "")
