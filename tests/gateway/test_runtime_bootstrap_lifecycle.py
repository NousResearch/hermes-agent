"""Live runtime HTTP lifecycle boundaries, with no provider inference."""
import asyncio
import json

import pytest
from websockets.asyncio.client import connect


@pytest.mark.asyncio
async def test_runtime_preserves_browser_ticket_gate(tmp_path):
    from hermes_agent.gateway.config import GatewayConfig
    from hermes_agent.gateway.run import GatewayRunner
    from hermes_agent.gateway.run_runtime import (initialize_gateway_runtime, start_gateway_runtime_api,
                                     publish_gateway_runtime_ready)
    from hermes_agent.gateway.runtime_ownership import process_ownership
    from hermes_agent.hermes_constants import get_hermes_home
    from hermes_agent.hermes_cli.dashboard_auth import register_provider, clear_providers
    from hermes_agent.hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    home = get_hermes_home()
    (home / 'config.yaml').write_text('dashboard:\n  public_url: https://gateway.example.test\n')
    register_provider(StubAuthProvider())
    process_ownership.reserve([home])
    runner = GatewayRunner(GatewayConfig())
    try:
        await initialize_gateway_runtime(runner)
        from hermes_agent.gateway.control_socket import GatewayControlServer
        runner.session_control_server = GatewayControlServer()
        assert await runner.session_control_server.start()
        await start_gateway_runtime_api(runner)
        assert await runner.start()
        publish_gateway_runtime_ready(runner)
        ticket = mint_ticket(user_id='browser-owner', provider='stub')
        url = runner.session_api.api_origin.replace('http:', 'ws:') + '/api/ws'
        async with connect(url, origin='https://gateway.example.test', subprotocols=[
                'hermes-gateway-v1', 'hermes-gateway-ticket.' + ticket]) as ws:
            await ws.send(json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'gateway.ping'}))
            while True:
                reply = json.loads(await asyncio.wait_for(ws.recv(), 10))
                if reply.get('id') == 1:
                    break
            assert reply['result']['ok'] is True
    finally:
        await runner.stop()
        await runner.session_control_server.stop()
        process_ownership.close()
        clear_providers()


@pytest.mark.asyncio
async def test_drain_withdraws_native_admission_before_listener_close(tmp_path):
    from hermes_agent.gateway.config import GatewayConfig, Platform, PlatformConfig
    from hermes_agent.gateway.platforms.base import BasePlatformAdapter, SendResult
    from hermes_agent.gateway.platforms.event import MessageEvent
    from hermes_agent.gateway.run import GatewayRunner
    from hermes_agent.gateway.run_runtime import initialize_gateway_runtime, drain_gateway_runtime
    from hermes_agent.gateway.runtime_ownership import process_ownership
    from hermes_agent.gateway.session import SessionSource
    from hermes_agent.hermes_constants import get_hermes_home

    class LocalAdapter(BasePlatformAdapter):
        async def connect(self, *, is_reconnect=False):
            return True

        async def disconnect(self):
            pass

        async def get_chat_info(self, chat_id):
            return {'id': chat_id}

        async def send(self, *args, **kwargs):
            return SendResult(success=True)

    home = get_hermes_home()
    process_ownership.reserve([home])
    runner = GatewayRunner(GatewayConfig())
    adapter = LocalAdapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    handled = asyncio.Event()

    async def handler(event):
        handled.set()
        return 'local control'

    adapter.set_message_handler(handler)
    runner.adapters[Platform.TELEGRAM] = adapter
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='owned', user_id='owned')
    try:
        await initialize_gateway_runtime(runner)
        first = MessageEvent(text='/status', source=source)
        await adapter.handle_message(first)
        await asyncio.wait_for(handled.wait(), 5)
        assert first._gateway_accepted
        await drain_gateway_runtime(runner)
        assert runner.session_runtime_descriptor['state'] == 'draining'
        later = MessageEvent(text='/status', source=SessionSource(
            platform=Platform.TELEGRAM, chat_id='another-owned-route', user_id='owned'))
        await adapter.handle_message(later)
        assert not later._gateway_accepted
    finally:
        await runner.stop()
        process_ownership.close()


@pytest.mark.asyncio
async def test_drain_stops_adapter_inbound_before_withdrawing_handlers(tmp_path):
    """The drain closes the inbound window FIRST: stop_inbound runs while the handler is still
    installed, so an update fetched just before the stop is admitted instead of hitting the
    handler-less drop in the disconnect gap."""
    from hermes_agent.gateway.config import GatewayConfig, Platform, PlatformConfig
    from hermes_agent.gateway.platforms.base import BasePlatformAdapter, SendResult
    from hermes_agent.gateway.run import GatewayRunner
    from hermes_agent.gateway.run_runtime import initialize_gateway_runtime, drain_gateway_runtime
    from hermes_agent.gateway.runtime_ownership import process_ownership
    from hermes_agent.hermes_constants import get_hermes_home

    class SequencedAdapter(BasePlatformAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls: list[str] = []
            self.stop_inbound_error: Exception | None = None

        async def connect(self, *, is_reconnect=False):
            return True

        async def disconnect(self):
            pass

        async def get_chat_info(self, chat_id):
            return {'id': chat_id}

        async def send(self, *args, **kwargs):
            return SendResult(success=True)

        async def stop_inbound(self):
            self.calls.append('stop_inbound')
            if self.stop_inbound_error is not None:
                raise self.stop_inbound_error

        def set_message_handler(self, handler):
            self.calls.append(f'set_message_handler={handler is not None}')
            super().set_message_handler(handler)

    home = get_hermes_home()
    process_ownership.reserve([home])
    runner = GatewayRunner(GatewayConfig())
    primary = SequencedAdapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    secondary = SequencedAdapter(PlatformConfig(enabled=True), Platform.DISCORD)
    try:
        await initialize_gateway_runtime(runner)
        runner.adapters[Platform.TELEGRAM] = primary
        runner._profile_adapters['secondary'] = {Platform.DISCORD: secondary}
        await drain_gateway_runtime(runner)
        assert primary.calls == ['stop_inbound', 'set_message_handler=False']
        assert secondary.calls == ['stop_inbound', 'set_message_handler=False']
        # A failing stop_inbound cannot block the drain: handlers are still withdrawn.
        tertiary = SequencedAdapter(PlatformConfig(enabled=True), Platform.WEIXIN)
        tertiary.stop_inbound_error = RuntimeError('wedge')
        runner.adapters[Platform.WEIXIN] = tertiary
        primary.calls.clear()
        secondary.calls.clear()
        await drain_gateway_runtime(runner)
        assert tertiary.calls == ['stop_inbound', 'set_message_handler=False']
        assert primary.calls == ['stop_inbound', 'set_message_handler=False']
        assert secondary.calls == ['stop_inbound', 'set_message_handler=False']
        # Wedged fetchers share one bound and stop together, leaving the settle its leash.
        import asyncio
        import time
        from unittest.mock import patch

        tertiary.stop_inbound_error = None
        never = asyncio.Event()
        for adapter in (primary, secondary, tertiary):
            adapter.stop_inbound = never.wait
        started = time.monotonic()
        with patch('hermes_agent.gateway.run_runtime.INBOUND_STOP_SECONDS', 0.2):
            await drain_gateway_runtime(runner)
        assert time.monotonic() - started < 2.0
        for adapter in (primary, secondary, tertiary):
            del adapter.stop_inbound
    finally:
        await runner.stop()
        process_ownership.close()
