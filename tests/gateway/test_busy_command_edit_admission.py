"""An edit of a busy ``/queue`` or ``/steer`` command is its own input: the queued turn it becomes keeps
the edit's revision, so native admission keys it ``<message_id>:edit:<revision>`` instead of
refusing it as a changed replay of the original command (``admission_conflict``)."""
import pytest

from tests.gateway.test_messaging_slash_command_fifo import _rows, _start_a
from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/queue', '/steer'])  # _Agent cannot steer: the queue fallback
async def test_edited_busy_command_is_queued_as_its_own_turn(monkeypatch, command):
    from gateway.platforms.event import MessageEvent, MessageType

    rig = await _start_a(monkeypatch, 'queue')
    for text, revision in ((f'{command} C', None), (f'{command} C edited', 'u101')):
        await rig.adapter.handle_message(MessageEvent(text=text, message_type=MessageType.COMMAND,
                                                      source=rig.source, message_id='C', edit_revision=revision))
    await async_wait_until(lambda: len(_rows(rig)) >= 3 or any('admission_conflict' in s for s in rig.sent),
                           what='the edited command to be admitted')
    rig.finish.set()
    await async_wait_until(lambda: [m for m, _ in rig.order].count('C') >= 2, timeout=15, what='both turns to run')
    assert not [s for s in rig.sent if 'admission_conflict' in s or 'RuntimeStoreError' in s], rig.sent
    assert _rows(rig) == ['A', 'C', 'C:edit:u101'], _rows(rig)
