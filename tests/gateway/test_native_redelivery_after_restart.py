"""The first inbound after a gateway restart is not refused when it is the provider's redelivery.

A messaging provider redelivers an event whose ACK was lost (Slack retries an event the stalled
gateway never acknowledged; a watchdog restart drops the in-memory dedupe). The redelivery carries the
same provider message id, so it is the same admission, but a native adapter stamps it with its own
local arrival time (``MessageEvent.timestamp`` defaults to now) and the first reply per thread in a
fresh process re-hydrates the thread history (``channel_context``). Both are in the admission's
payload, so main refused the redelivery ``admission_conflict`` (ADMITTED_REQUEST_CONFLICT): the chat got
a generic error (or, with warnings suppressed, nothing) instead of its answer or the pause notice.

The committed row stays the one exactly-once identity: a redelivery that differs only in those
gateway-derived fields resolves to it (as ``reconcile_native_retry`` already does for re-downloaded
media), and changed user content under the same id still conflicts, now with a visible reply.
Real GatewayRunner + SessionAuthority + SQLite, two owner epochs; the transport is stubbed.
"""
import asyncio
from datetime import datetime, timedelta

import pytest

MESSAGE_ID = '1730000000.000100'


async def _gateway(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.run import GatewayRunner
    from gateway.session_ingress_context import register_transport_home
    from tests.gateway.test_command_bypass_active_session import _StubAdapter
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', '42')
    monkeypatch.setattr('gateway.run._hermes_home', tmp_path)
    monkeypatch.setattr('hermes_state.DEFAULT_DB_PATH', tmp_path / 'state.db')
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=False, sessions_dir=tmp_path / 'sessions'))
    adapter = _StubAdapter(PlatformConfig(enabled=True, token='fixture', typing_indicator=False), Platform.TELEGRAM)
    sent, calls = [], []

    async def send(chat_id, content, **kwargs):
        sent.append(content)
        return SendResult(success=True, message_id=str(len(sent)))
    monkeypatch.setattr(adapter, 'send', send)

    async def run_agent(**kwargs):
        calls.append(kwargs.get('message'))
        return {'final_response': 'the answer', 'messages': [], 'tools': [], 'history_offset': 0,
                'last_prompt_tokens': 0}
    runner._run_agent = run_agent
    adapter.gateway_runner = runner
    runner.adapters = {Platform.TELEGRAM: adapter}
    register_transport_home(runner, None, tmp_path)
    return runner, adapter, sent, calls


def _wire(runner, adapter):
    # As at startup: the authority exists before the adapters are wired, so the primary handler is
    # the shared runtime's (``_make_default_profile_message_handler`` binds the native callback).
    runner._wire_adapter_handlers(adapter)


async def _settled(adapter, sent, calls, *, want):
    for _ in range(200):
        await asyncio.sleep(0.05)  # health: allow HX013 -- polls the adapter's background turn to quiesce
        if not adapter._background_tasks and not adapter._session_tasks and want():
            break
    await asyncio.sleep(0.2)  # health: allow HX013 -- a late duplicate send would land here


async def _admitted_before_crash(runner, adapter, *, started, **event_fields):
    """Owner 1 commits the delivery (and, when ``started``, claims it) and dies before answering."""
    from gateway.platforms.event import MessageEvent
    from gateway.session_authority import initialize_session_authority
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import claim_session_input
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42', thread_id='T1')
    first = await initialize_session_authority(runner, profile_id=str(get_hermes_home()), instance_id='before-crash')
    first._schedule = lambda ref: None  # the crash: its drain never ran
    original = MessageEvent(text='hello', source=source, message_id=MESSAGE_ID,
                            timestamp=datetime.now() - timedelta(seconds=40), **event_fields)
    with native_callback(runner, original, get_hermes_home()):
        receipt = await first.admit_native(original)
    if started:
        claim_session_input(first.db, epoch=first.epoch, session_id=receipt.ref.session_id)
    return source, receipt


async def _restart(runner):
    from gateway.run_runtime import recover_profile_native_sessions
    from gateway.session_authority import initialize_session_authority
    from hermes_constants import get_hermes_home
    second = await initialize_session_authority(runner, profile_id=str(get_hermes_home()), instance_id='after-restart')
    await recover_profile_native_sessions(runner, second)
    _wire(runner, runner.adapters[next(iter(runner.adapters))])
    return second


@pytest.mark.asyncio
@pytest.mark.parametrize('rehydrated', [False, True], ids=['arrival-time', 'arrival-time-and-thread-context'])
async def test_redelivery_after_restart_is_answered_once_not_refused(tmp_path, monkeypatch, rehydrated):
    from gateway.platforms.event import MessageEvent
    from hermes_state_runtime import list_session_admissions
    runner, adapter, sent, calls = await _gateway(tmp_path, monkeypatch)
    source, receipt = await _admitted_before_crash(runner, adapter, started=False)
    try:
        second = await _restart(runner)
        # The provider's redelivery: same message id and text, this process's own arrival time and,
        # for the first reply of a thread in a fresh process, the re-fetched thread history.
        await adapter.handle_message(MessageEvent(
            text='hello', source=source, message_id=MESSAGE_ID,
            channel_context='[thread history fetched after the restart]' if rehydrated else None))
        await _settled(adapter, sent, calls, want=lambda: 'the answer' in sent)
        rows = list_session_admissions(second.db, session_id=receipt.ref.session_id, pending_only=False)
    finally:
        runner.session_store.close_all_db_handles()
    replies = [text for text in sent if 'home channel' not in text]  # the first-contact /sethome tip is not a reply
    assert replies == ['the answer'], 'the redelivery must be answered once, with no refusal beside it'
    assert calls == ['hello'], 'the message must run exactly once'
    assert [(row['request_id'], row['status']) for row in rows] == [(MESSAGE_ID, 'terminal')]


@pytest.mark.asyncio
async def test_redelivery_of_a_turn_lost_mid_run_tells_the_chat_it_is_paused(tmp_path, monkeypatch):
    """Killed mid-turn: the turn is ``unknown`` and never replayed (GW-10). The redelivery is that
    admission, so the chat gets the pause notice naming the way out, not a generic refusal."""
    from gateway.platforms.event import MessageEvent
    from hermes_state_runtime import list_session_admissions
    runner, adapter, sent, calls = await _gateway(tmp_path, monkeypatch)
    source, receipt = await _admitted_before_crash(runner, adapter, started=True)
    try:
        second = await _restart(runner)
        await adapter.handle_message(MessageEvent(text='hello', source=source, message_id=MESSAGE_ID))
        await _settled(adapter, sent, calls, want=lambda: bool(sent))
        rows = list_session_admissions(second.db, session_id=receipt.ref.session_id, pending_only=False)
    finally:
        runner.session_store.close_all_db_handles()
    assert calls == [], 'a turn lost mid-run is never replayed'
    assert [row['status'] for row in rows] == ['unknown']
    assert len(sent) == 1 and 'paused' in sent[0] and '/discard' in sent[0], sent


@pytest.mark.asyncio
async def test_changed_content_under_an_admitted_id_still_conflicts_and_says_so(tmp_path, monkeypatch):
    """Exactly-once is unchanged: other user content under the same provider id is not this
    admission. It is still refused, and the chat is told why instead of a generic turn error."""
    from gateway.platforms.event import MessageEvent
    from hermes_state_runtime import list_session_admissions
    runner, adapter, sent, calls = await _gateway(tmp_path, monkeypatch)
    source, receipt = await _admitted_before_crash(runner, adapter, started=False)
    try:
        second = await _restart(runner)
        await _settled(adapter, sent, calls, want=lambda: 'the answer' in sent)  # the committed row runs
        await adapter.handle_message(MessageEvent(text='a different message', source=source, message_id=MESSAGE_ID))
        await _settled(adapter, sent, calls, want=lambda: len(sent) >= 2)
        rows = list_session_admissions(second.db, session_id=receipt.ref.session_id, pending_only=False)
    finally:
        runner.session_store.close_all_db_handles()
    assert calls == ['hello']
    assert [row['payload']['text'] for row in rows] == ['hello']
    refusal = [text for text in sent if text != 'the answer' and 'home channel' not in text]
    assert len(refusal) == 1 and 'already received' in refusal[0] and 'RuntimeStoreError' not in refusal[0], sent
