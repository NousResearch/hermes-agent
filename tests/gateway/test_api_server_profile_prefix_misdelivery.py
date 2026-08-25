"""A ``/p/<profile>/`` prefix on a non-multiplexed gateway must not misdeliver.

The prefix is an address: the caller is naming WHICH agent the request is
for. The old behavior ignored it whenever ``gateway.multiplex_profiles`` was
off and answered as the process's own (single) profile — so a request
explicitly addressed to one agent was silently answered by a different one.
Observed live (Aug 2026): ``hermes peer dm mini/researcher`` was answered by
the mini's *default* agent, with no error on either side, because the mini
runs one LaunchDaemon per profile and only the default daemon hosted an
api_server.

The contract now: with multiplexing off, a prefix naming this process's own
profile is honored (peers address single-profile daemons this way without
knowing the host's topology); any other name fails closed with the existing
404, because a wrong-agent answer is strictly worse than an error.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server import _PROFILE_REJECTED, APIServerAdapter
from gateway.config import PlatformConfig


@pytest.fixture()
def adapter():
    # No gateway_runner configured -> multiplex_profiles is falsy, the exact
    # shape of a standalone single-profile daemon.
    return APIServerAdapter(PlatformConfig(enabled=True))


def _request(profile: str | None):
    return SimpleNamespace(match_info={} if profile is None else {"profile": profile})


class TestResolverWithMultiplexOff:
    def test_no_prefix_is_untouched(self, adapter):
        assert adapter._resolve_request_profile(_request(None)) is None

    def test_prefix_naming_own_profile_is_honored(self, adapter, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.profiles.profile_matches_home",
            lambda name, home=None: name == "researcher",
        )
        assert adapter._resolve_request_profile(_request("researcher")) is None

    def test_prefix_naming_default_on_default_home_is_honored(self, adapter, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.profiles.profile_matches_home",
            lambda name, home=None: name == "default",
        )
        assert adapter._resolve_request_profile(_request("default")) is None

    def test_prefix_naming_another_agent_fails_closed(self, adapter, monkeypatch):
        """The misdelivery case: addressed to researcher, running as default."""
        monkeypatch.setattr(
            "hermes_cli.profiles.profile_matches_home",
            lambda name, home=None: name == "default",
        )
        assert adapter._resolve_request_profile(_request("researcher")) is _PROFILE_REJECTED

    def test_unresolvable_own_identity_fails_closed(self, adapter, monkeypatch):
        """If the process cannot prove who it is, it must not answer as anyone."""

        def boom(name, home=None):
            raise RuntimeError("no home")

        monkeypatch.setattr("hermes_cli.profiles.profile_matches_home", boom)
        assert adapter._resolve_request_profile(_request("researcher")) is _PROFILE_REJECTED


class TestMultiplexOnUnchanged:
    def test_served_profile_resolves(self, adapter, monkeypatch):
        adapter.gateway_runner = SimpleNamespace(
            config=SimpleNamespace(
                multiplex_profiles=True, multiplex_profile_allowlist=None
            )
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex, profile_allowlist: [("worker", object())],
        )
        assert adapter._resolve_request_profile(_request("worker")) == "worker"
        assert adapter._resolve_request_profile(_request("ghost")) is _PROFILE_REJECTED


@pytest.mark.asyncio
async def test_http_request_addressed_to_another_agent_is_404_not_answered(
    adapter, monkeypatch
):
    """End to end through the real middleware: the wrong-agent request must
    404 instead of being served — a body would mean the misdelivery is back."""
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: "default"
    )

    async def handler(request):
        return web.json_response({"served_by": "default"})

    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_get("/p/{profile}/v1/test", handler)
    app.router.add_get("/v1/test", handler)

    async with TestClient(TestServer(app)) as cli:
        # Addressed to a different agent: refused.
        resp = await cli.get("/p/researcher/v1/test")
        assert resp.status == 404
        body = await resp.json()
        assert "profile" in str(body.get("error", "")).lower()

        # Addressed to this agent, and unaddressed: both served.
        assert (await cli.get("/p/default/v1/test")).status == 200
        assert (await cli.get("/v1/test")).status == 200
