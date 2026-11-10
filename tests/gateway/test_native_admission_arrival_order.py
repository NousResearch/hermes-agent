"""A chat's inputs reach its FIFO in arrival order, whatever off-loop media capture each needs.

The capture (copy + SHA-256 + fsyncs) runs in a worker thread before the ordered admission write,
so a text sent right after a photo or video must wait for it instead of taking the earlier slot
and running first without the media. Other chats never wait.
"""
import asyncio
import threading

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from hermes_state_runtime import list_session_admissions
from tests.gateway.test_shared_busy_receive_loop import Adapter
from tests.waiting import async_wait_until

X, Y = '15551234567@s.whatsapp.net', '15557654321@s.whatsapp.net'


def _order(authority):
    rows = (row for sid in authority.sessions
            for row in list_session_admissions(authority.db, session_id=sid, pending_only=False))
    return [row['request_id'] for row in sorted(rows, key=lambda row: row['seq'])]


def _hold_capture(monkeypatch):
    """The real capture, held on an Event: a large video on a slow disk."""
    import gateway.session_envelope as envelope
    import gateway.session_ingress_media as media
    entered, release = threading.Event(), threading.Event()
    real_capture = media.capture_native_media

    def held(paths):
        entered.set()
        release.wait(15.0)  # outlasts every bound below: an input done within one never waited on it
        return real_capture(paths)
    monkeypatch.setattr(envelope, 'capture_native_media', held)
    return entered, release


def _video(source, mid, clip):
    return MessageEvent(text='', message_type=MessageType.VIDEO, source=source, message_id=mid,
                        media_urls=[str(clip)], media_types=['video/mp4'])


@pytest.mark.asyncio
@pytest.mark.parametrize('busy', [False, True], ids=['idle', 'busy'])
async def test_media_then_quick_text_keep_arrival_order(tmp_path, monkeypatch, busy):
    """The adapter path: an idle video then a busy text, or both while a turn runs."""
    import gateway.session_finite as finite
    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', 'fixture-user')
    gate = asyncio.Event()

    async def produce(authority, ref, row):
        await gate.wait()
        return 'ok'
    monkeypatch.setattr(finite, 'execute_finite_admission', produce)
    entered, release = _hold_capture(monkeypatch)
    runner = GatewayRunner(GatewayConfig())
    adapter = Adapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='order')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='order-chat', chat_type='dm', user_id='fixture-user')
    clip = tmp_path / 'clip.mp4'
    clip.write_bytes(b'video-bytes' * 1000)
    expected = ['v1', 't1']
    try:
        if busy:
            await adapter.handle_message(MessageEvent(text='hello', source=source, message_id='h0'))
            await async_wait_until(lambda: _order(authority) == ['h0'], what='the running turn admitted')
            expected.insert(0, 'h0')
        # One task per delivery, as Discord and Slack dispatch them.
        first = asyncio.create_task(adapter.handle_message(_video(source, 'v1', clip)))
        await asyncio.to_thread(entered.wait, 5)
        text = MessageEvent(text='what is this?', source=source, message_id='t1')
        second = asyncio.create_task(adapter.handle_message(text))
        # A serial reader (Signal, IRC) awaits this: it returns once the text queues behind the
        # video, without acknowledging it (main commits the text here, ahead of the video).
        assert second in (await asyncio.wait({second}, timeout=2.0))[0], 'the reader waited for the capture'
        assert 't1' not in _order(authority) and text._gateway_accepted is False
        release.set()
        await asyncio.gather(first, second)
        await async_wait_until(lambda: len(_order(authority)) == len(expected), what='both admitted')
        assert _order(authority) == expected
    finally:
        release.set()
        gate.set()
        await async_wait_until(lambda: not adapter._background_tasks, what='adapter tasks done')
        authority.db.close()


async def _held_owner(tmp_path, monkeypatch):
    """Real authority over SQLite, WhatsApp sources, the capture held on an Event; no drain."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567,15557654321')
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    entered, release = _hold_capture(monkeypatch)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    runner.adapters = {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='held')
    clip = tmp_path / 'clip.mp4'
    clip.write_bytes(b'video-bytes' * 1000)

    def admit(chat, mid, media=False, ingress=False):
        """``ingress``: through the messaging ingress (``admit_message``), else ``admit_native``."""
        from gateway.session_ingress import admit_message
        source = adapter.build_source(chat_id=chat, user_id=chat, chat_type='dm')
        event = _video(source, mid, clip) if media else MessageEvent(text='text ' + mid, source=source,
                                                                       message_id=mid)
        return asyncio.create_task(admit_message(authority, event) if ingress else authority.admit_native(event))
    route = lambda chat: runner._session_key_for_source(adapter.build_source(chat_id=chat, user_id=chat, chat_type='dm'))
    return authority, entered, release, admit, route


@pytest.mark.asyncio
async def test_a_cancelled_middle_input_does_not_let_the_next_overtake(tmp_path, monkeypatch):
    authority, entered, release, admit, _ = await _held_owner(tmp_path, monkeypatch)
    try:
        first = admit(X, 'x1-video', media=True)
        await asyncio.to_thread(entered.wait, 5)
        second = admit(X, 'x2-text')
        await asyncio.sleep(0)
        third = admit(X, 'x3-text')
        await asyncio.sleep(0)
        second.cancel()  # e.g. the busy caller's task is cancelled
        await asyncio.gather(second, return_exceptions=True)
        await asyncio.wait({third}, timeout=2.0)  # a negative window (>= 2 s): main lets x3 overtake here
        release.set()
        await asyncio.gather(first, third)
        assert _order(authority) == ['x1-video', 'x3-text']
        assert authority._route_turns == {}  # the per-chat entries go once idle
    finally:
        release.set()
        authority.db.close()


@pytest.mark.asyncio
async def test_a_held_capture_never_delays_another_chat_nor_outlives_its_cancel(tmp_path, monkeypatch):
    authority, entered, release, admit, _ = await _held_owner(tmp_path, monkeypatch)
    try:
        first = admit(X, 'x1-video', media=True)
        await asyncio.to_thread(entered.wait, 5)
        other = admit(Y, 'y-text')
        # X's capture is held past this bound, so chat Y finishing within it proves it never queued on X.
        await async_wait_until(other.done, timeout=5.0, what='chat Y admitted while chat X still captures')
        assert not first.done(), 'chat Y waited on chat X'
        follower = admit(X, 'x2-text')
        await asyncio.sleep(0)
        first.cancel()  # the earlier input is abandoned mid-capture: its successor goes on
        await asyncio.gather(first, return_exceptions=True)
        await async_wait_until(follower.done, timeout=5.0, what='the next input after a cancelled capture')
    finally:
        release.set()
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('chat', ['existing', 'first-message', 'paused'])
async def test_stop_fences_inputs_still_capturing_or_waiting(tmp_path, monkeypatch, chat):
    """/stop (and /new, /reset: the same hook) reaches the chat's inputs that arrived before it but
    have not committed yet, also when the chat has no session yet. A FIFO paused on a lost turn
    keeps its queued followers through /stop and /new, and so these too."""
    authority, entered, release, admit, route = await _held_owner(tmp_path, monkeypatch)
    queued = lambda: [row['request_id'] for sid in authority.sessions for row in
                      list_session_admissions(authority.db, session_id=sid) if row['status'] == 'queued']
    first = second = None
    try:
        if chat != 'first-message':
            await admit(X, 'x0-text')
        if chat == 'paused':  # x0 was running when the previous owner died
            authority.db._execute_write(lambda conn: conn.execute(
                "UPDATE session_admissions SET status='unknown' WHERE request_id='x0-text'"))
        first = admit(X, 'x1-video', media=True, ingress=True)
        await asyncio.to_thread(entered.wait, 5)
        second = admit(X, 'x2-text', ingress=True)
        await asyncio.wait({second}, timeout=2.0)  # a negative window (>= 2 s): main commits it here
        await authority.runner._hm_stop_owner_turn(route(X))
        release.set()
        if chat == 'paused':
            await async_wait_until(lambda: len(queued()) == 2, timeout=3.0, what='both kept')
            assert queued() == ['x1-video', 'x2-text']
            return
        done, _ = await asyncio.wait({first, second}, timeout=3.0)
        assert queued() == [], queued()
        assert done == {first, second} and first.result() is second.result() is None  # dropped silently
        assert authority._route_turns == {}
    finally:
        release.set()
        for task in (first, second):
            if task is not None:
                task.cancel()
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('chat', ['existing', 'first-message', 'paused'])
async def test_one_stop_epoch_per_route_fences_inputs_and_command_turns(tmp_path, monkeypatch, chat):
    """/stop and /new bump ONE per-route epoch on the authority. A slash command's turn (``_stop_epoch``)
    and an input still capturing read the same counter, so they are fenced under the same conditions:
    both on a live or first-message chat, neither on a FIFO paused on a lost turn (it keeps its
    followers). The counter outlives the route's FIFO entry, and the runner keeps no second one."""
    authority, _entered, release, admit, route = await _held_owner(tmp_path, monkeypatch)
    release.set()
    runner = authority.runner
    try:
        if chat != 'first-message':
            await admit(X, 'x0-text')
        if chat == 'paused':
            authority.db._execute_write(lambda conn: conn.execute(
                "UPDATE session_admissions SET status='unknown' WHERE request_id='x0-text'"))
        before = runner._route_stop_epoch(route(X))
        assert before == authority.route_epoch(route(X))
        await runner._hm_stop_owner_turn(route(X))
        after = runner._route_stop_epoch(route(X))
        assert after == authority.route_epoch(route(X)) == before + (chat != 'paused')
        assert authority._route_turns == {} and runner._route_stop_epoch(route(Y)) == 0
        assert not hasattr(runner, '_route_stop_epochs')
    finally:
        authority.db.close()
