"""Regression: messaging /retry must re-admit natively under session authority.

Background: /retry re-dispatches the last user turn by building a NEW MessageEvent and
calling the gateway's message handler directly, which bypasses the adapter's
native_callback wrapper. admission matches provenance by event identity, so a freshly
built event carries no provenance and native admission refuses it as invalid_params when
gateway.multiplex_profiles is true. The fix carries the /retry event's native callback to
the re-dispatched turn (gateway/session_ingress_context.carry_callback), which is built on
the same authenticated source object.

This test reproduces the refusal on a real multiplexed authority and asserts the fix makes
the re-dispatched turn's envelope prepare with its provenance resolved. Without the fix the
re-dispatch raises RuntimeStoreError('invalid_params').
"""
import weakref
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_runtime import initialize_gateway_runtime
from gateway.runtime_ownership import process_ownership
from gateway.session import SessionSource
from gateway.session_envelope import prepare_native
from gateway.session_ingress_context import (
    capture_provenance, native_callback, register_transport_home)

from hermes_state_runtime import RuntimeStoreError


class _FakeAdapter:
    """Weakref-able stand-in for a live platform adapter (SimpleNamespace cannot weakref).

    Deliberately attribute-free: the code under test only ever identity-compares the adapter
    (``_binding`` / ``_owning_profile``), never reads transport state off it.
    """


def _reserve_homes(tmp_path, monkeypatch, names=('beta',)):
    root = tmp_path / '.hermes'
    root.mkdir(mode=0o700)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    import hermes_state
    monkeypatch.setattr(
        hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
    homes = [('default', root)]
    for name in names:
        home = root / 'profiles' / name
        home.mkdir(parents=True, mode=0o700)
        (home / 'config.yaml').write_text('{}')
        homes.append((name, home))
    return root, homes


def _runner(root, homes):
    config = GatewayConfig()
    config.multiplex_profiles = True
    config._runtime_profile_homes = tuple(homes)
    return GatewayRunner(config)


@pytest.mark.asyncio
async def test_retry_redispatch_admits_natively_under_session_authority(
        tmp_path, monkeypatch):
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    runner = _runner(root, homes)
    process_ownership.reserve([home for _, home in homes])
    try:
        await initialize_gateway_runtime(runner)
        register_transport_home(runner, None, root)
        adapter = _FakeAdapter()
        runner.adapters = {Platform.DISCORD: adapter}
        runner._is_user_authorized_for_source = lambda *a, **k: True

        source = SessionSource(
            platform=Platform.DISCORD, chat_id='probe-chat', chat_type='dm',
            user_id='probe-user', user_name='probe')
        # build_source stamps this on every inbound event; hand-built sources need it too,
        # otherwise _transport_owner fail-closes to None (not_found).
        source._transport_adapter_ref = weakref.ref(adapter)
        entry = await runner.async_session_store.get_or_create_session(source)
        sid = entry.session_id
        runner.session_store.append_to_transcript(
            sid, {'role': 'user', 'content': 'probe ask'})
        runner.session_store.append_to_transcript(
            sid, {'role': 'assistant', 'content': 'probe answer'})

        seen = []

        async def spy_handle_message(event):
            prov = capture_provenance(runner, event)
            try:
                payload = await prepare_native(runner, event)
                seen.append(('prepared',
                             payload['native_text_v1'].get('provenance') is not None,
                             prov is not None))
            except RuntimeStoreError as exc:
                seen.append(('refused', exc.reason, prov is not None))
            return 'spy-ok'

        runner._handle_message = spy_handle_message

        retry_event = MessageEvent(
            text='/retry', message_type=MessageType.TEXT, source=source,
            message_id='probe-msg')
        with native_callback(runner, retry_event, root, None):
            await runner._handle_retry_command(retry_event)

        assert seen, 'the re-dispatched turn was never admitted'
        assert seen[0][0] == 'prepared', f'retry was refused: {seen}'
        assert seen[0][1] is True, 'provenance not resolved on re-dispatch'
        assert seen[0][2] is True, 'capture_provenance returned None inside rebind'
    finally:
        for authority in list(getattr(runner, 'session_authorities', []) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)
