"""Regression: a ``pre_gateway_dispatch`` rewrite keeps the event's native provenance (#139045).

``_hm_pre_gateway_dispatch_hook`` used ``dataclasses.replace`` to apply a rewritten text, so the
inbound event changed identity between the adapter's ``native_callback`` binding and authority
admission. ``capture_provenance`` matches that callback by event identity; a rebuilt event carries
none, and under ``gateway.multiplex_profiles`` (or a stamped transport home) ``_validate_native``
refuses it with ``invalid_params`` — the exact refusal #139045 reports for every rewritten plain
message. The rewrite now mutates ``text`` in place; this suite drives the real adapter → handler →
native admission chain on a real multiplexed registry and asserts the rewritten turn is admitted.
"""
import asyncio
import secrets

import pytest

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


def _adapter_cls():
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    class Adapter(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token='tok-' + secrets.token_hex(8)),
                             Platform.TELEGRAM)
            self.deliveries = []

        async def connect(self, *, is_reconnect=False):
            return True

        async def disconnect(self):
            pass

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.deliveries.append(str(content))
            return SendResult(success=True, message_id='r')

        async def edit_message(self, chat_id, message_id, content, *, finalize=False):
            return SendResult(success=True, message_id=message_id)

        async def send_typing(self, chat_id, metadata=None):
            pass

        async def get_chat_info(self, chat_id):
            return {'id': chat_id}

    return Adapter


def _admissions(runner):
    """(request_id, payload) rows across every served authority's durable FIFO."""
    rows = []
    for authority in runner.session_authorities:
        with authority.db._read_ctx() as conn:
            rows.extend(conn.execute(
                'SELECT request_id, payload_json FROM session_admissions ORDER BY seq').fetchall())
    return rows


@pytest.mark.asyncio
async def test_rewritten_plain_message_keeps_native_provenance(tmp_path, monkeypatch):
    from agent import secret_scope
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership

    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('beta',))
    (root / '.env').write_text('TELEGRAM_ALLOWED_USERS=u1\n')
    process_ownership.reserve([home for _, home in homes])
    prev = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    runner = _runner(root, homes)
    await initialize_gateway_runtime(runner)
    Adapter = _adapter_cls()
    adapter = Adapter()
    adapter.gateway_runner = runner
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._wire_adapter_handlers(adapter)

    async def _rewrite(name, **kwargs):
        return [{'action': 'rewrite', 'text': kwargs['event'].text + ' (filtered)'}]

    monkeypatch.setattr('hermes_cli.plugins.ainvoke_hook', _rewrite)
    try:
        source = adapter.build_source(chat_id='c1', chat_type='dm', user_id='u1')
        event = MessageEvent(text='PLAIN', source=source, message_id='m1')
        try:
            await adapter.handle_message(event)
        except Exception:
            pass  # turn-side model resolution is out of scope; admission is what is asserted
        await asyncio.sleep(2)
        assert not any('invalid_params' in d for d in adapter.deliveries), adapter.deliveries
        rows = _admissions(runner)
        assert rows and rows[0][0] == 'm1', rows
        # The rewrite was applied to the admitted turn, not silently dropped with the identity.
        assert 'PLAIN (filtered)' in rows[0][1], rows
    finally:
        secret_scope.set_multiplex_active(prev)
