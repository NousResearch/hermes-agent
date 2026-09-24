"""Control-plane routes stay available when asyncio's default pool is wedged.

Regression for #95559: the event loop and lightweight async routes remained
healthy, but management routes queued blocking work behind an unavailable
default executor and timed out until the Desktop backend was restarted.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading

import pytest
import hermes_cli.web_server_gateway as _web_server_gateway
import hermes_cli.web_server_lifecycle as _web_server_lifecycle


def _occupy_default_executor(loop: asyncio.AbstractEventLoop):
    release = threading.Event()
    started = threading.Event()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(pool)

    def _block() -> None:
        started.set()
        release.wait()

    future = loop.run_in_executor(None, _block)
    return pool, future, started, release


async def _request_with_wedged_default_executor(path: str, *, warm: bool = False):
    try:
        import httpx
    except ImportError:
        pytest.skip("httpx not installed")

    from hermes_cli import web_server

    if warm:
        warm_transport = httpx.ASGITransport(app=web_server.app)
        async with httpx.AsyncClient(
            transport=warm_transport,
            base_url="http://testserver",
            headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
        ) as client:
            warm_response = await client.get(path)
        assert warm_response.status_code == 200

    loop = asyncio.get_running_loop()
    pool, blocker, started, release = _occupy_default_executor(loop)
    try:
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set(), "default-executor blocker did not start"

        transport = httpx.ASGITransport(app=web_server.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
        ) as client:
            return await asyncio.wait_for(client.get(path), timeout=2.0)
    finally:
        release.set()
        await blocker
        pool.shutdown(wait=True, cancel_futures=True)


def test_profiles_route_survives_default_executor_starvation(monkeypatch):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "list_profiles", lambda: [])

    response = asyncio.run(_request_with_wedged_default_executor("/api/profiles"))

    assert response.status_code == 200
    assert response.json() == {"profiles": []}


def test_toolsets_route_survives_default_executor_starvation(monkeypatch):
    from hermes_cli import platforms, tools_config
    import toolsets

    monkeypatch.setattr(
        tools_config,
        "_get_effective_configurable_toolsets",
        lambda: [("test", "Test", "Test toolset")],
    )
    monkeypatch.setattr(tools_config, "_toolset_configuration_platform", lambda _name: "cli")
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(tools_config, "get_nous_subscription_features", lambda _cfg: {})
    monkeypatch.setattr(tools_config, "_toolset_has_keys", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(tools_config, "gui_toolset_label", lambda label: label)
    monkeypatch.setattr(platforms, "platform_label", lambda _key, fallback: fallback)
    monkeypatch.setattr(toolsets, "resolve_toolset", lambda _name: [])

    response = asyncio.run(
        _request_with_wedged_default_executor("/api/tools/toolsets")
    )

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_status_route_survives_default_executor_starvation(monkeypatch):
    from hermes_cli import web_server

    monkeypatch.setattr(
        _web_server_gateway,
        "_collect_profile_gateway_topology_cached",
        lambda: {
            "profiles": ["default"],
            "gateway_mode": "none",
            "gateways": [],
            "profile_platforms": {},
        },
    )
    monkeypatch.setattr(_web_server_lifecycle, "_resolve_restart_drain_timeout", lambda: 30.0)
    monkeypatch.setattr(web_server, "get_install_id", lambda: None)

    response = asyncio.run(
        _request_with_wedged_default_executor("/api/status", warm=True)
    )

    assert response.status_code == 200
    assert response.json()["version"]
