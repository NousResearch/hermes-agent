"""Secondary-profile MCP servers must register on the first routed turn.

Gateway startup calls ``discover_mcp_tools()`` once under the default
profile's home. Multiplexed inbound turns swap ``HERMES_HOME`` via
``_profile_runtime_scope`` but historically never re-ran discovery, so a
secondary profile's ``mcp_servers`` (e.g. an owner ``kb-writer``) never
entered the process-global registry. Cron already rediscovers inside
profile scope before constructing the agent; the gateway inbound path
must mirror that, once per profile home.
"""
import asyncio
from pathlib import Path
from unittest import mock

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
import gateway.run as gateway_run


def _make_runner(multiplex: bool) -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=multiplex)
    return runner


def _clear_discovery_cache():
    cache = getattr(gateway_run, "_mcp_discovered_profile_homes", None)
    if cache is not None:
        cache.clear()


class TestMultiplexInboundMcpDiscovery:
    """``_run_agent`` lazily discovers MCP tools inside the profile scope."""

    def test_first_routed_turn_discovers_under_profile_home(self, tmp_path, monkeypatch):
        _clear_discovery_cache()
        owner_home = tmp_path / "profiles" / "owner"
        owner_home.mkdir(parents=True)

        homes_seen = []

        def fake_discover():
            from hermes_constants import get_hermes_home
            homes_seen.append(Path(get_hermes_home()))
            return []

        monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", fake_discover)

        runner = _make_runner(multiplex=True)
        inner = mock.AsyncMock(return_value={"final_response": "ok"})
        runner._run_agent_inner = inner
        monkeypatch.setattr(
            GatewayRunner,
            "_resolve_profile_home_for_source",
            lambda self, source: owner_home,
        )

        source = mock.MagicMock()
        asyncio.run(
            runner._run_agent(
                message="hi",
                context_prompt="",
                history=[],
                source=source,
                session_id="s1",
            )
        )

        assert [Path(p).resolve() for p in homes_seen] == [owner_home.resolve()]
        inner.assert_awaited_once()

    def test_discovery_runs_once_per_profile_home(self, tmp_path, monkeypatch):
        _clear_discovery_cache()
        owner_home = tmp_path / "profiles" / "owner"
        owner_home.mkdir(parents=True)

        calls = {"n": 0}

        def fake_discover():
            calls["n"] += 1
            return []

        monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", fake_discover)

        runner = _make_runner(multiplex=True)
        runner._run_agent_inner = mock.AsyncMock(return_value={"final_response": "ok"})
        monkeypatch.setattr(
            GatewayRunner,
            "_resolve_profile_home_for_source",
            lambda self, source: owner_home,
        )

        source = mock.MagicMock()

        async def _two_turns():
            kwargs = dict(
                message="hi",
                context_prompt="",
                history=[],
                source=source,
                session_id="s1",
            )
            await runner._run_agent(**kwargs)
            await runner._run_agent(**kwargs)

        asyncio.run(_two_turns())
        assert calls["n"] == 1

    def test_distinct_profile_homes_each_discover_once(self, tmp_path, monkeypatch):
        _clear_discovery_cache()
        owner_home = tmp_path / "profiles" / "owner"
        customer_home = tmp_path / "profiles" / "default"
        owner_home.mkdir(parents=True)
        customer_home.mkdir(parents=True)

        homes_seen = []

        def fake_discover():
            from hermes_constants import get_hermes_home
            homes_seen.append(Path(get_hermes_home()).resolve())
            return []

        monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", fake_discover)

        runner = _make_runner(multiplex=True)
        runner._run_agent_inner = mock.AsyncMock(return_value={"final_response": "ok"})

        homes = {"owner": owner_home, "default": customer_home}

        def _resolve(self, source):
            return homes[source.profile]

        monkeypatch.setattr(GatewayRunner, "_resolve_profile_home_for_source", _resolve)

        async def _turns():
            owner = mock.MagicMock()
            owner.profile = "owner"
            customer = mock.MagicMock()
            customer.profile = "default"
            kwargs = dict(message="hi", context_prompt="", history=[], session_id="s1")
            await runner._run_agent(source=owner, **kwargs)
            await runner._run_agent(source=customer, **kwargs)
            await runner._run_agent(source=owner, **kwargs)

        asyncio.run(_turns())
        assert [Path(p).resolve() for p in homes_seen] == [
            owner_home.resolve(),
            customer_home.resolve(),
        ]

    def test_unmultiplexed_inbound_does_not_rediscover(self, monkeypatch):
        _clear_discovery_cache()
        fake_discover = mock.Mock(return_value=[])
        monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", fake_discover)

        runner = _make_runner(multiplex=False)
        runner._run_agent_inner = mock.AsyncMock(return_value={"final_response": "ok"})

        asyncio.run(
            runner._run_agent(
                message="hi",
                context_prompt="",
                history=[],
                source=mock.MagicMock(),
                session_id="s1",
            )
        )

        fake_discover.assert_not_called()
