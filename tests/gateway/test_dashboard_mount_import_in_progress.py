"""The gateway mounts a complete dashboard module even while another thread is importing it.

``hermes_cli.web_server`` is listed in ``sys.modules`` as soon as any thread in the gateway process
starts importing it (a lazy ``from hermes_cli.web_server import ...`` elsewhere). The mount took
that listing as loaded and bound a module that had no ``app`` yet.
"""
import asyncio
import types

from gateway.run_api import GatewayDashboardMount
from tests.held_import import held_import


class _Lifespan:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return None


def test_mount_waits_for_a_dashboard_import_another_thread_started():
    app = types.SimpleNamespace(state=types.SimpleNamespace(gateway_runner=None),
                                router=types.SimpleNamespace(lifespan_context=lambda app: _Lifespan()))
    runner = types.SimpleNamespace(session_authority=object())
    mount = GatewayDashboardMount(runner, types.SimpleNamespace(bound_host="127.0.0.1", bound_port=1))

    with held_import("hermes_cli.web_server", app=app, _configure_auth_gate=lambda *a: None) as held:
        web = held.run_during_import(lambda: asyncio.run(mount.ensure()))

    assert web.app is app
    assert app.state.gateway_runner is runner
    asyncio.run(mount.close())
    assert app.state.gateway_runner is None
