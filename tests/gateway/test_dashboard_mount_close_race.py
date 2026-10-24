"""A gateway stop that wins the race with the dashboard's lifespan start never leaves it mounted."""
import asyncio
import sys
import types

import pytest

from gateway.run_api import GatewayDashboardMount


def test_close_during_lifespan_entry_unmounts_and_exits_the_lifespan(monkeypatch):
    entered, release, exits = asyncio.Event(), asyncio.Event(), []

    class _Lifespan:
        async def __aenter__(self):
            entered.set()
            await release.wait()

        async def __aexit__(self, *exc):
            exits.append(exc)

    app = types.SimpleNamespace(state=types.SimpleNamespace(gateway_runner=None),
                                router=types.SimpleNamespace(lifespan_context=lambda app: _Lifespan()))
    web = types.SimpleNamespace(app=app, _configure_auth_gate=lambda *a: None)
    monkeypatch.setitem(sys.modules, "hermes_cli.web_server", web)
    runner = types.SimpleNamespace(session_authority=object())
    mount = GatewayDashboardMount(runner, types.SimpleNamespace(bound_host="127.0.0.1", bound_port=1))

    async def scenario():
        warm = asyncio.create_task(mount.ensure())
        await entered.wait()
        assert app.state.gateway_runner is runner
        await mount.close()
        release.set()
        with pytest.raises(RuntimeError, match="stopped"):
            await warm

    asyncio.run(scenario())
    assert not mount.loaded
    assert len(exits) == 1
    assert app.state.gateway_runner is None and app.state.session_authority is None
