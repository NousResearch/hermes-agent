"""Only outstanding prompts cross a worker ownership handoff."""

import io
import os
import socket
import sys
import threading
from types import SimpleNamespace

import pytest

from agent.managed_worker import WorkerControls, encode_frame, read_frame
from agent.managed_worker_detach.channel import DetachableChannel
from gateway.session_managed_worker import _prompt_frame
from tools import approval
from tools.approval_gateway_wait import _await_gateway_decision


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('persisted', [False, True])
def test_detachable_controls_observe_owner_eof_until_finish_persisted(monkeypatch, persisted):
    """Acknowledgment is not persistence; a completed finish no longer needs an owner."""
    import agent.managed_worker as worker

    read_fd, write_fd = os.pipe()
    observed = threading.Event()

    class Channel:
        def await_owner(self, stream):
            observed.set()

    monkeypatch.setattr(worker, '_drain_background_delegations', lambda budget: None)
    monkeypatch.setattr(worker, '_die_with_owner', lambda: None)
    with os.fdopen(read_fd, 'rb') as reader, os.fdopen(write_fd, 'wb', buffering=0) as writer:
        monkeypatch.setattr(sys, 'stdin', SimpleNamespace(buffer=reader))
        controls = WorkerControls(Channel(), 'fixture')
        try:
            writer.write(encode_frame({'type': 'finish'}))
            assert controls.finish.wait(3)
            if persisted:
                controls.finish_persisted.set()
        finally:
            writer.close()
            controls.reader.join(5)
        assert not controls.reader.is_alive()
        assert observed.is_set() is not persisted


class _ReplayWriter(io.BytesIO):
    def __init__(self, fail=False):
        super().__init__()
        self.flushes = 0
        self.fail = fail

    def flush(self):
        self.flushes += 1
        if self.fail and self.flushes == 2:
            raise BrokenPipeError('fixture: owner left during replay')
        super().flush()


class _Peer:
    def __init__(self, sock, epoch, *, fail=False):
        self.sock = sock
        self.reader = io.BytesIO(encode_frame({'type': 'reattached', 'epoch': epoch}))
        self.writer = _ReplayWriter(fail)

    def getsockopt(self, *args):
        return self.sock.getsockopt(*args)

    def settimeout(self, timeout):
        self.sock.settimeout(timeout)

    def makefile(self, mode):
        return self.reader if mode == 'rb' else self.writer

    def frames(self):
        self.writer.seek(0)
        frames = []
        while self.writer.tell() < len(self.writer.getbuffer()):
            frames.append(read_frame(self.writer))
        return frames


def _orphaned_channel():
    channel = DetachableChannel(io.BytesIO(), 'unused', {'pid': os.getpid()})
    # Enter the orphaned state without redirecting the pytest process's stdio or starting its watchdog.
    channel.orphaned = True
    channel.timer = SimpleNamespace(cancel=lambda: None)
    return channel


def _viewer():
    controls = SimpleNamespace(remote_responders={}, pending={})
    live = SimpleNamespace(controls=controls, route='fixture', event_stream=SimpleNamespace(publish=lambda *a, **kw: None))
    owner = SimpleNamespace(sessions={'s': live})
    owner.register_approval = lambda sid, gen, route, data: controls.pending.update(
        {data['request_id']: (None, {'kind': 'approval'})})
    return owner, controls


@pytest.mark.platforms('linux')
def test_only_open_approval_replays_after_resolved_waits(monkeypatch):
    monkeypatch.setattr('tools.approval_context._get_approval_timeout', lambda: 4)
    channel = _orphaned_channel()
    worker = WorkerControls.__new__(WorkerControls)
    worker.route, worker.channel, worker.finish = 'detached-test', channel, threading.Event()

    def settle(data):
        worker.approval(data)
        answer = {'type': 'approval', 'prompt_id': data['request_id'], 'value': 'deny'}
        worker._read_controls(io.BytesIO(encode_frame(answer) + encode_frame({'type': 'finish'})))

    for i in range(18):
        outcome = _await_gateway_decision(worker.route, settle, {'command': f'fixture {i}'})
        assert outcome['choice'] == 'deny'

    opened = []
    ready = threading.Event()

    def notify_open(data):
        worker.approval(data)
        opened.append(data['request_id'])
        ready.set()

    finished = []
    waiter = threading.Thread(target=lambda: finished.append(_await_gateway_decision(
        worker.route, notify_open, {'command': 'another fixture'})))
    waiter.start()
    try:
        assert ready.wait(3)
        left, right = socket.socketpair()
        with left, right:
            peer = _Peer(left, 2)
            assert channel._reattach(peer)
            owner, controls = _viewer()
            for frame in peer.frames()[1:]:
                _prompt_frame(owner, SimpleNamespace(session_id='s'), {'generation': 1},
                              SimpleNamespace(respond=lambda *a: None), frame)
            assert list(controls.pending) == opened
            assert [f['data']['request_id'] for f in peer.frames()[1:] if f['type'] == 'approval'] == opened
    finally:
        approval.resolve_gateway_approval(worker.route, 'deny', resolve_all=True)
        waiter.join(5)
    assert not waiter.is_alive() and finished[0]['choice'] == 'deny'


@pytest.mark.platforms('linux')
def test_failed_replay_retains_buffer_and_late_old_eof_keeps_new_owner(monkeypatch):
    channel = _orphaned_channel()
    old_stream = object()
    monkeypatch.setattr(channel, '_orphan_locked', lambda: pytest.fail('old EOF detached the new owner'))
    for text in ('first', 'second'):
        channel.send('delta', text=text)
    original_size = channel.buffered
    for fail, epoch in ((True, 2), (False, 3)):
        left, right = socket.socketpair()
        with left, right:
            peer = _Peer(left, epoch, fail=fail)
            assert channel._reattach(peer) is (not fail)
            if fail:
                assert channel.orphaned and channel.attachments == 0
                assert channel.buffered == original_size
            else:
                assert peer.frames()[1:] == [{'type': 'delta', 'text': text} for text in ('first', 'second')]
                assert channel.await_owner(old_stream) is peer.reader
                assert not channel.buffer and channel.buffered == 0
