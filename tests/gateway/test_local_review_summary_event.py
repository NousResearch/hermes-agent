"""A local viewer sees the background review's summary as a ``review.summary`` event.

LocalSessionAdapter.send publishes nothing (TurnRunner owns the canonical stream), so the
"💾 Self-improvement review: …" line handed to the status adapter reached no Desktop/TUI
viewer once every local turn ran through the session authority. The in-process tui_gateway
emitted it as ``review.summary``; the authority route must publish the same frame.
"""
from __future__ import annotations

from types import SimpleNamespace

from gateway.config import Platform
from gateway.run_turn_runner import TurnRunner


def _runner(platform, published):
    turn = object.__new__(TurnRunner)
    adapter_sends = []

    async def send(chat_id, text, metadata=None):
        adapter_sends.append(text)

    turn._ctx = SimpleNamespace(
        source=SimpleNamespace(platform=platform),
        _status_adapter=SimpleNamespace(send=send),
        _status_chat_id='chat',
        _status_thread_metadata=None,
        _run_still_current=lambda: True,
    )
    live = SimpleNamespace(event_stream=SimpleNamespace(
        execution={},
        publish=lambda sid, payload, event_type: published.append((sid, event_type, payload)) or True))
    authority = SimpleNamespace(sessions={'s': live})
    turn._approval_owner = (authority, 's', 3)
    turn._schedule = lambda coro, _msg: coro.close()
    return turn, adapter_sends


def test_local_review_summary_follows_the_turn_settlement_and_is_never_fenced():
    import threading
    published = []
    turn, adapter_sends = _runner(Platform.LOCAL, published)
    stream = turn._approval_owner[0].sessions['s'].event_stream
    stream.execution = {'execution_generation': 3}  # the drain is still settling turn 3
    turn._ctx._run_still_current = lambda: False  # and the review fork outlives the turn
    send, _release = turn._make_bg_review_callbacks()
    worker = threading.Thread(target=send, args=('💾 Self-improvement review: memory updated',))
    worker.start()
    worker.join(0.3)
    # Not before the turn's message.complete: the drain clears the stamp only after it.
    assert worker.is_alive() and published == []
    stream.execution = {}
    worker.join(5)
    assert published == [('s', 'review.summary', {'text': '💾 Self-improvement review: memory updated'})]
    assert adapter_sends == []


def test_platform_review_summary_still_waits_for_the_post_delivery_release():
    published = []
    turn, _ = _runner(Platform.TELEGRAM, published)
    sent = []
    turn._send_status_text = lambda text, metadata, msg: sent.append(text)
    send, release = turn._make_bg_review_callbacks()
    send('💾 Memory updated')
    assert sent == []
    release()
    assert sent == ['💾 Memory updated'] and published == []
