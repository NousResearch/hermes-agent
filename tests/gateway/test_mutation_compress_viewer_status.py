"""A manual /compress on the canonical path shows its viewers that it is working.

Main's in-process /compress pinned the pair itself (``tui_gateway/methods_session.py::
_compress_live``: ``compressing`` → ``ready``); the canonical ``session.mutate(compress)`` published
nothing, and the in-process path is unreachable once the Ink TUI/dashboard attach to the shared
owner — so a viewer's status bar sat on "ready" for the whole summarization. Regression for the
unified-gateway cutover (#106742, tracking #134889).
"""
import asyncio
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession

LOGICAL_ID = 'local-logical'
PHYSICAL_ID = 'physical-child'


class _Compressor:
    """The summarizer's LLM call is the one thing a test cannot make; everything else runs for real."""

    result = None
    error = None
    gate: threading.Event | None = None
    done: threading.Event | None = None

    def __init__(self, *args, **kwargs):
        pass

    def compress(self, messages, **kwargs):
        gate = type(self).gate
        error = type(self).error
        try:
            if gate is not None:
                gate.wait(5)
            if error is not None:
                raise error
            return type(self).result
        finally:
            done = type(self).done
            if done is not None:
                done.set()


def _history(count):
    return [{'role': 'user' if index % 2 == 0 else 'assistant', 'content': f'm{index}'}
            for index in range(count)]


def _lane(monkeypatch, messages):
    """``(frames, run)``: a real prepare_compress over a real session event stream."""
    from gateway.session_mutation_compress import prepare_compress
    monkeypatch.setattr('gateway.session_policy.restore_policy',
                        lambda receipt: SimpleNamespace(config=lambda authority: {}))
    monkeypatch.setattr('gateway.session_policy.policy_scope', lambda policy, **kwargs: nullcontext())
    monkeypatch.setattr('agent.context_compressor.ContextCompressor', _Compressor)
    live = LiveSession(SimpleNamespace(platform=None, user_id='human'), 'route')
    frames = []
    live.event_stream.observers.add(frames.append)
    authority = SimpleNamespace(
        db=SimpleNamespace(get_messages_as_conversation=lambda target: messages),
        runner=SimpleNamespace(_resolve_session_agent_runtime=lambda **kwargs: ('model-x', {})))
    prepared = {'snapshot': {'receipt': {'policy': {}, 'session_id': LOGICAL_ID},
                             'target': PHYSICAL_ID, 'target_revision': 1}}

    def run(payload=None, compressed=None, error=None, cancel=False):
        _Compressor.result, _Compressor.error = compressed, error
        call = payload or {'focus': ''}
        if not cancel:
            return asyncio.run(prepare_compress(authority, live, call, prepared))
        # A real cancellation: the task awaits nothing before the summarizer call, so cancelling it
        # lands on that await. The gate holds the worker back, and waiting for it here keeps the
        # outcome the same whichever of the thread and the loop gets there first.
        gate = threading.Event()
        done = threading.Event()
        _Compressor.gate, _Compressor.done = gate, done

        async def drive():
            task = asyncio.create_task(prepare_compress(authority, live, call, prepared))
            await asyncio.sleep(0)
            task.cancel()
            gate.set()
            done.wait(5)
            return await task

        try:
            return asyncio.run(drive())
        finally:
            _Compressor.gate = _Compressor.done = None

    return frames, run


def _published(frames):
    return [(frame['params']['type'], frame['params']['payload']['kind'], frame['params']['payload']['text'])
            for frame in frames]


def test_manual_compress_publishes_the_viewer_lifecycle(monkeypatch):
    messages = _history(6)
    frames, run = _lane(monkeypatch, messages)
    result = run(compressed=[{'_compressed_summary': 'sum'}, *messages[4:]])

    published = _published(frames)
    assert [entry[:2] for entry in published] == [('status.update', 'compacting'),
                                                  ('status.update', 'compacted')]
    assert published[0][2].startswith('⠋ compressing 6 messages (~') and published[0][2].endswith('…')
    assert published[1][2] == '✓ Context compression complete'
    assert result['summary']['headline']
    # The viewer's own (logical) id: a non-in-place rotation moves the transcript, not the viewer,
    # and the clients drop a frame whose session_id is not the session they are showing.
    assert {frame['params']['session_id'] for frame in frames} == {LOGICAL_ID}


def test_short_session_compress_stays_silent(monkeypatch):
    """Main pinned only from four messages up; a short session must not flash a status."""
    frames, run = _lane(monkeypatch, _history(3))
    run(compressed=[{'_compressed_summary': 'sum'}])
    assert frames == []


def test_preview_publishes_nothing(monkeypatch):
    """``--preview`` is a read-only report: no summarization, so nothing to report in flight."""
    frames, run = _lane(monkeypatch, _history(6))
    result = run(payload={'focus': '--preview'})
    assert result['status'] == 'preview'
    assert frames == []


def test_failed_summary_retires_without_claiming_success(monkeypatch):
    """A raise must retire the status with the failure wording, not a success or a no-change one."""
    frames, run = _lane(monkeypatch, _history(6))
    with pytest.raises(RuntimeError, match='summary provider down'):
        run(error=RuntimeError('summary provider down'))
    published = _published(frames)
    assert [entry[1] for entry in published] == ['compacting', 'compacted']
    assert '✓' not in published[1][2] and 'did not complete' in published[1][2]


def test_a_cancelled_compress_retires_with_the_failure_wording(monkeypatch):
    """Cancelling the owner task (loop teardown) must not leave a success line behind.

    A cancellation is not an ``Exception``: it arrives as ``CancelledError`` at the summarizer await,
    so the status has to be retired from a ``finally`` that survives a ``BaseException``.
    """
    frames, run = _lane(monkeypatch, _history(6))
    with pytest.raises(asyncio.CancelledError):
        run(compressed=[{'_compressed_summary': 'sum'}], cancel=True)
    published = _published(frames)
    assert [entry[1] for entry in published] == ['compacting', 'compacted']
    assert '✓' not in published[1][2] and 'did not complete' in published[1][2]


def test_a_compress_that_removes_nothing_retires_without_claiming_success(monkeypatch):
    """The summarizer's no-op is reported as such: not a success, and not the failure wording either."""
    messages = _history(6)
    frames, run = _lane(monkeypatch, messages)
    result = run(compressed=messages)

    published = _published(frames)
    assert result['status'] == 'noop'
    assert [entry[1] for entry in published] == ['compacting', 'compacted']
    assert '✓' not in published[1][2] and 'no changes' in published[1][2]


def test_the_four_message_boundary_is_inside_the_guard(monkeypatch):
    """Main pinned from four messages up: the boundary itself announces (see the silence test for three)."""
    frames, run = _lane(monkeypatch, _history(4))
    run(compressed=[{'_compressed_summary': 'sum'}, *_history(4)[-2:]])
    assert [entry[1] for entry in _published(frames)] == ['compacting', 'compacted']
