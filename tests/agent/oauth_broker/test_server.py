"""Broker server: loopback-only binding, redacted health endpoints, and
aiohttp dependency readiness. Synthetic values and loopback sockets only."""

from __future__ import annotations

import ast
import asyncio
import json
import tomllib
from pathlib import Path

import aiohttp
from aiohttp import web
import pytest

from agent.oauth_broker.server import (
    BrokerDependencyError,
    create_server_app,
    run_broker,
    validate_bind_host,
)
from tests.agent.oauth_broker.test_proxy import (
    AUTH,
    LOCAL_KEY,
    SSE_CHUNKS,
    UPSTREAM_ACCESS_A,
    UPSTREAM_REFRESH_A,
    FakeUpstream,
    make_slot,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Bind-host validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_validate_bind_host_accepts_canonical_loopback(host):
    assert validate_bind_host(host) == host


@pytest.mark.parametrize(
    "host",
    [
        "0.0.0.0",
        "::",
        "localhost",  # hostname, not a canonical loopback literal
        "chatgpt.com",
        "192.168.1.5",
        "10.0.0.7",
        "127.0.0.2",  # loopback subnet but not the canonical address
        "127.0.0.1:17880",
        " 127.0.0.1",
        "",
        None,
    ],
)
def test_validate_bind_host_rejects_everything_else(host):
    with pytest.raises(ValueError):
        validate_bind_host(host)


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------


async def _start_server(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    return runner, f"http://{host}:{port}"


def _server_env(tmp_path, **kwargs):
    """Async context helpers shared by the health tests."""

    class _Env:
        async def __aenter__(self):
            self.upstream = FakeUpstream()
            await self.upstream.start()
            self.slot_a, self.calls_a = make_slot(tmp_path, "A")
            self.slot_b, _ = make_slot(
                tmp_path,
                "B",
                access="synthetic-upstream-access-B",
                account_id="acct-synthetic-b",
            )
            app = create_server_app(
                slots={"A": self.slot_a, "B": self.slot_b},
                local_key=LOCAL_KEY,
                upstream_origin=self.upstream.origin,
                allow_test_upstream=True,
                **kwargs,
            )
            self.runner, self.base = await _start_server(app)
            self.session = aiohttp.ClientSession()
            return self

        async def __aexit__(self, *exc):
            await self.session.close()
            await self.runner.cleanup()
            await self.upstream.stop()
            return False

    return _Env()


def test_health_is_ok_without_auth_and_without_account_metadata(tmp_path):
    async def scenario():
        async with _server_env(tmp_path) as env:
            resp = await env.session.get(f"{env.base}/health")
            assert resp.status == 200
            payload = await resp.json()
            assert payload == {"status": "ok"}

    asyncio.run(scenario())


def test_health_detailed_requires_local_key(tmp_path):
    async def scenario():
        async with _server_env(tmp_path) as env:
            resp = await env.session.get(f"{env.base}/health/detailed")
            assert resp.status == 401

    asyncio.run(scenario())


def test_health_detailed_exposes_only_redacted_fields(tmp_path):
    async def scenario():
        async with _server_env(tmp_path) as env:
            # Load both slots so this is the all-ready control case.
            await env.slot_a.get_access_token()
            await env.slot_b.get_access_token()
            resp = await env.session.get(
                f"{env.base}/health/detailed", headers=AUTH
            )
            assert resp.status == 200
            raw = await resp.text()
            payload = json.loads(raw)
            assert payload["status"] == "ok"
            accounts = {entry["alias"]: entry for entry in payload["accounts"]}
            assert set(accounts) == {"A", "B"}
            for entry in payload["accounts"]:
                assert set(entry) == {
                    "alias",
                    "present",
                    "healthy",
                    "expires_at",
                    "last_refresh_result",
                    "persistence_degraded",
                }
            assert accounts["A"]["present"] is True
            assert accounts["A"]["healthy"] is True
            assert accounts["A"]["persistence_degraded"] is False
            assert accounts["B"]["present"] is True
            assert accounts["B"]["healthy"] is True
            assert isinstance(accounts["A"]["expires_at"], float)
            for secret in (
                UPSTREAM_ACCESS_A,
                UPSTREAM_REFRESH_A,
                LOCAL_KEY,
                "synthetic-upstream-access-B",
            ):
                assert secret not in raw

    asyncio.run(scenario())


def test_health_detailed_is_503_when_persistence_is_degraded(tmp_path):
    async def scenario():
        async with _server_env(tmp_path) as env:
            await env.slot_a.get_access_token()
            await env.slot_b.get_access_token()
            env.slot_a._persistence_degraded = True

            detailed = await env.session.get(
                f"{env.base}/health/detailed", headers=AUTH
            )
            assert detailed.status == 503
            payload = await detailed.json()
            assert payload["status"] == "degraded"
            accounts = {entry["alias"]: entry for entry in payload["accounts"]}
            assert accounts["A"]["healthy"] is False
            assert accounts["A"]["persistence_degraded"] is True

            # Public liveness stays independent from authenticated readiness.
            liveness = await env.session.get(f"{env.base}/health")
            assert liveness.status == 200
            assert await liveness.json() == {"status": "ok"}

    asyncio.run(scenario())


def test_proxy_routes_are_served_by_the_server_app(tmp_path):
    async def scenario():
        async with _server_env(tmp_path) as env:
            resp = await env.session.post(
                f"{env.base}/accounts/A/backend-api/codex/responses",
                data=b"{}",
                headers=AUTH,
            )
            assert resp.status == 200
            assert await resp.read() == b"".join(SSE_CHUNKS)

    asyncio.run(scenario())


def test_concurrency_limit_smoke(tmp_path):
    async def scenario():
        async with _server_env(tmp_path, max_concurrent_requests=1) as env:
            for _ in range(2):
                resp = await env.session.get(f"{env.base}/health")
                assert resp.status == 200

    asyncio.run(scenario())


def test_health_bypasses_saturated_stream_concurrency_limit(tmp_path):
    async def scenario():
        async with _server_env(tmp_path, max_concurrent_requests=1) as env:
            env.upstream.responses_script.append(("sse-infinite",))
            streaming = await env.session.post(
                f"{env.base}/accounts/A/backend-api/codex/responses",
                data=b"{}",
                headers=AUTH,
            )
            await asyncio.wait_for(env.upstream.streaming_started.wait(), timeout=2)

            liveness = await asyncio.wait_for(
                env.session.get(f"{env.base}/health"), timeout=0.5
            )
            assert liveness.status == 200
            assert await liveness.json() == {"status": "ok"}

            streaming.close()
            await asyncio.wait_for(env.upstream.client_disconnected.wait(), timeout=2)

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Dependency readiness
# ---------------------------------------------------------------------------


def test_missing_aiohttp_gives_actionable_error_without_auto_install(
    tmp_path, monkeypatch
):
    slot, _ = make_slot(tmp_path)
    monkeypatch.setitem(__import__("sys").modules, "aiohttp", None)
    with pytest.raises(BrokerDependencyError) as excinfo:
        create_server_app(slots={"A": slot}, local_key=LOCAL_KEY)
    message = str(excinfo.value)
    assert "hermes-agent[oauth-broker]" in message
    assert "pip install" in message


def test_run_broker_validates_bind_before_dependency_and_binding(
    tmp_path, monkeypatch
):
    slot, _ = make_slot(tmp_path)
    monkeypatch.setitem(__import__("sys").modules, "aiohttp", None)
    # Non-loopback bind fails first, even with the dependency missing.
    with pytest.raises(ValueError):
        run_broker(host="0.0.0.0", port=0, slots={"A": slot}, local_key=LOCAL_KEY)
    # Loopback bind with the dependency missing exits before binding.
    with pytest.raises(BrokerDependencyError):
        run_broker(
            host="127.0.0.1", port=0, slots={"A": slot}, local_key=LOCAL_KEY
        )


def test_server_module_has_no_top_level_aiohttp_import():
    import agent.oauth_broker.server as server_mod

    tree = ast.parse(Path(server_mod.__file__).read_text(encoding="utf-8"))
    for node in tree.body:  # module top level only
        if isinstance(node, ast.Import):
            assert not any(a.name.split(".")[0] == "aiohttp" for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "aiohttp"


def test_pyproject_declares_pinned_oauth_broker_extra():
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        pyproject = tomllib.load(handle)
    extras = pyproject["project"]["optional-dependencies"]
    assert extras["oauth-broker"] == ["aiohttp==3.14.1"]
    # Stays lock-step with the messaging pin so one CVE bump updates both.
    assert "aiohttp==3.14.1" in extras["messaging"]
