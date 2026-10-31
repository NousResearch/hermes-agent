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
    from contextlib import contextmanager

    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')

    @contextmanager
    def home_override(path):
        tok = set_hermes_home_override(path)
        try:
            yield
        finally:
            reset_hermes_home_override(tok)

    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')

    server = ThreadingHTTPServer(('127.0.0.1', 0), _Media)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{server.server_port}/photo.png'
    client = RelayMediaClient(f'http://127.0.0.1:{server.server_port}', None, None)

    authority = None

    async def deliver():
        nonlocal authority
        with home_override(tmp_path):
            authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
            staged = await client.download(url)
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


@pytest.mark.asyncio
async def test_text_redelivery_resolves_to_its_receipt_when_timestamp_mismatches(tmp_path, monkeypatch):
    """A text-only message redelivered with a different arrival-time timestamp resolves to the stored
    receipt instead of being refused as admission_conflict. The provider message time is not message
    identity, so a delivered text must not climb the admission queue on every redelivery."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_ingress_context import native_callback
    from contextlib import contextmanager
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
    from hermes_state_runtime import claim_session_input, list_session_admissions, settle_session_input
    from plugins.platforms.discord.adapter import DiscordAdapter

    @contextmanager
    def home_override(path):
        tok = set_hermes_home_override(path)
        try:
            yield
        finally:
            reset_hermes_home_override(tok)

    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')

    async def deliver(*, timestamp, authority):
        event = MessageEvent(text='hello', source=source, message_id='provider-1', message_type=MessageType.TEXT,
                             timestamp=timestamp)
        with home_override(tmp_path):
            with native_callback(runner, event, get_hermes_home()):
                return await authority.admit_native(event)

    authority = None
    try:
        async with home_override(tmp_path):
            authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
        first = await deliver(timestamp=datetime.fromtimestamp(0, UTC), authority=authority)
        # ACK lost: same message id, only the delivery's arrival timestamp changed.
        second = await deliver(timestamp=datetime.fromtimestamp(60, UTC), authority=authority)
        assert second.admission_id == first.admission_id
        assert second.status == 'queued'
        sid = first.ref.session_id
        claim = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
        settle_session_input(authority.db, epoch=authority.epoch, admission_id=first.admission_id,
                             generation=claim['generation'], outcome='completed')
        _late = await deliver(timestamp=datetime.fromtimestamp(120, UTC), authority=authority)
        assert _late.admission_id == first.admission_id and _late.status == 'terminal'
        assert 1 == len(list_session_admissions(authority.db, session_id=sid, pending_only=False)), \
            'the message must execute once'
    finally:
        if authority is not None:
            await asyncio.to_thread(authority.db.close())
