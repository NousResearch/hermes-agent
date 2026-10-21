"""A provider redelivery of a committed media message is the same admission, whatever staging name
its re-download got; changed bytes under the same provider id still conflict."""
import asyncio
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import pytest

PNG = b'\x89PNG\r\n\x1a\n' + b'\x00' * 24


class _Media(BaseHTTPRequestHandler):
    body = PNG

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Content-Length', str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args):
        pass


@pytest.mark.asyncio
async def test_redelivered_media_message_returns_its_receipt_and_changed_bytes_conflict(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.relay.media import RelayMediaClient
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_ingress_context import native_callback
    from gateway.session_ingress_media import _media_root, release_admission_media
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import (RuntimeStoreError, claim_session_input, list_session_admissions,
                                      settle_session_input)
    from plugins.platforms.discord.adapter import DiscordAdapter
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    server = ThreadingHTTPServer(('127.0.0.1', 0), _Media)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{server.server_port}/photo.png'
    client = RelayMediaClient(f'http://127.0.0.1:{server.server_port}', None, None)
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')

    async def deliver():
        # Every delivery re-downloads the attachment, as the Relay adapter does before admission.
        staged = await client.download(url)
        # The Relay adapter pins the timestamp: the connector sends no provider time.
        event = MessageEvent(text='look', source=source, message_id='provider-1', message_type=MessageType.PHOTO,
                             media_urls=[staged], media_types=['image/png'],
                             timestamp=datetime.fromtimestamp(0, UTC))
        with native_callback(runner, event, get_hermes_home()):
            return staged, await authority.admit_native(event)

    try:
        first_staged, first = await deliver()
        second_staged, second = await deliver()  # its ACK was lost: same provider id, same bytes
        assert Path(first_staged).name != Path(second_staged).name
        assert second.admission_id == first.admission_id and second.status == 'queued'
        sid = first.ref.session_id
        claim = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
        settle_session_input(authority.db, epoch=authority.epoch, admission_id=first.admission_id,
                             generation=claim['generation'], outcome='completed')
        assert release_admission_media(authority.db, first.admission_id) == 1
        _, late = await deliver()  # redelivered after the turn ran and its bytes were released
        assert late.admission_id == first.admission_id and late.status == 'terminal'
        assert not list(_media_root().rglob('*.png')), 'a redelivery copy no admission owns stayed retained'
        rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
        assert [row['request_id'] for row in rows] == ['provider-1'], 'the message must execute once'
        _Media.body = PNG + b'changed'
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await deliver()
    finally:
        _Media.body = PNG
        server.shutdown()
        server.server_close()
        await asyncio.to_thread(authority.db.close)
