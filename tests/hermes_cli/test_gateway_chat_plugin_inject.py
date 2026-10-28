"""`ctx.inject_message(text)` with no session_key/origin (plugins.md "Active CLI conversation") reaches
the attached classic `hermes chat` session as a queued prompt, from a plugin's own thread (B2-11).

The classic CLI loads no plugin code on a gateway session (PLG-7): the plugin runs in the gateway, which
picks the one attached classic CLI session of the plugin's profile."""
import pytest

from tests.gateway.test_cli_plugin_command_in_gateway import _Wire
from tests.gateway.test_plugin_keyless_inject import keyless_gateway, landed_once, noted


@pytest.mark.platforms("linux")
def test_no_key_inject_submits_into_the_attached_classic_session(tmp_path):
    with keyless_gateway(tmp_path) as (desc, home, work, log, peer):
        wire = _Wire(home, desc)
        try:
            sid = wire.call('session.create', request_id='b2-11', source='cli', toolsets=[], cwd=str(work))['session_id']
            wire.call('slash.exec', session_id=sid, command='/inj-thread New data arrived')
            accepted = noted(log, 'thread', 'New data arrived')
            target = landed_once(home, peer, 'New data arrived')
        finally:
            wire.close()
    assert accepted is True
    assert target == sid


@pytest.mark.asyncio
async def test_the_classic_chat_view_reads_no_plugin_manager(monkeypatch):
    """No plugin code runs in the classic CLI process on a gateway session, so its chat view has no
    plugin inject slot to install."""
    import asyncio
    import hermes_cli.plugins as plugins_mod
    from hermes_cli.gateway_chat_view import GatewayChatView
    monkeypatch.setattr(plugins_mod, 'get_plugin_manager', lambda: pytest.fail('the chat view read the plugin manager'))

    class Peer:
        events = asyncio.Queue()

        async def rpc(self, method, **params):
            return {}

    async def read_line(prompt, symbol):
        return '/quit'

    view = GatewayChatView(Peer(), {'stored_session_id': 'sid'})
    view._read_line = read_line
    assert await view.run() == 0
