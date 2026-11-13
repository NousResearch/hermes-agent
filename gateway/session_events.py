"""Authority-local subscription ordering over the existing bounded transports/replay."""
from copy import deepcopy
import logging
import threading
import time
import uuid
import weakref

from hermes_agent.tui_gateway import event_replay
from hermes_agent.tui_gateway.transport import FanoutTransport


class SessionEvents:
    def __init__(self):
        self.lock = threading.RLock()
        self.fanout = FanoutTransport()
        # A private lifetime key prevents two profiles (or legacy TUI owners)
        # with the same persisted ID from reading each other's replay.
        self._key = uuid.uuid4().hex
        # Nothing but this object can name the key, so its ring and counters go when it does (a
        # retired or replaced LiveSession), instead of waiting for the shared cache's FIFO cap.
        weakref.finalize(self, event_replay.release_key, self._key, blocking=False).atexit = False
        self.epoch = uuid.uuid4().hex
        self.sequence = 0
        self._public_id = None
        self.execution = {}
        # Synchronous same-thread recipients (API run projections): unlike fanout peers they
        # cannot lose a frame to a detach that races the writer thread.
        self.observers = set()
        # Set by the authority that attaches transports: retires the subscription of a
        # peer whose bounded backlog overflowed, so it cannot linger looking subscribed.
        self.on_overflow = None
        # The running execution's open tool calls and its last frame time; a hosted room's info
        # poll reports them so the room shows which tool a member is in (``progress_for``).
        self._progress = None

    def note_progress(self, generation, event_type, payload):
        """Record one execution frame (the caller holds ``lock`` and passed the generation fence)."""
        progress = self._progress
        if progress is None or progress['generation'] != generation:
            progress = self._progress = {'generation': generation, 'tools': {}}
        payload = payload if isinstance(payload, dict) else {}
        tool_id = str(payload.get('tool_id') or '')
        if event_type == 'tool.start':
            progress['tools'][tool_id] = str(payload.get('name') or '') or 'tool'
        elif event_type == 'tool.complete':
            progress['tools'].pop(tool_id, None)
        progress['at'] = time.time()

    def progress_for(self, generation):
        """``{"tool", "progress_at"}`` for *generation*'s execution: the newest open tool call
        (absent between tools) and its last frame time; ``{}`` before its first frame."""
        with self.lock:
            progress = self._progress
            if progress is None or progress['generation'] != generation:
                return {}
            tools = list(progress['tools'].values())
            return {'progress_at': progress['at'], **({'tool': tools[-1]} if tools else {})}

    def release(self):
        """The session ended with no viewer left (it may stay resumable): free its replay frames
        now. A later publish or attach finds them gone and starts a new epoch, as after eviction."""
        with self.lock:
            event_replay.release_key(self._key)

    def watermark(self):
        with self.lock:
            if self.sequence and not event_replay.latest_seq(self._key):
                # The shared bounded cache evicted this session. Never reuse a
                # sequence in the old epoch, even if publication resumes later.
                self.epoch = uuid.uuid4().hex
                self.sequence = 0
            return self.epoch, self.sequence

    def publish(self, session_id, payload, *, event_type="message.complete", execution=None):
        # ``execution`` stamps this frame alone; the shared stamp is read without the lock elsewhere.
        with self.lock:
            self.watermark()
            frame = {'jsonrpc': '2.0', 'method': 'event', 'params': {
                **(self.execution if execution is None else execution), 'type': event_type, 'session_id': self._key,
                'payload': deepcopy(payload), 'replay_epoch': self.epoch}}
            event_replay._stamp_event(frame)
            if frame['params']['seq'] <= self.sequence:
                # Another publisher may evict us after watermark's lookup.
                self.epoch = uuid.uuid4().hex
                frame['params']['replay_epoch'] = self.epoch
            # The ring froze the frame under its private lookup key; live recipients see the
            # canonical ID here and ``since`` restores it on the thawed replay copies.
            frame['params']['session_id'] = session_id
            self._public_id = session_id
            self.sequence = frame['params']['seq']

            def overflow(transport):
                # The peer's socket is healthy; only its event subscription ended. Same
                # shape as the client's own synthetic gap so one consumer handles both.
                if self.on_overflow is not None:
                    self.on_overflow(transport)
                return {'jsonrpc': '2.0', 'method': 'event', 'params': {
                    'type': 'session.replay_gap', 'session_id': session_id,
                    'payload': {'replay_epoch': self.epoch, 'latest_seq': self.sequence}}}
            delivered = self.fanout.write(frame, overflow=overflow)
            for observer in tuple(self.observers):
                try:
                    observer(frame)
                    delivered = True
                except Exception:
                    # A request-owned projection cannot undo committed publication or starve peers.
                    logging.getLogger(__name__).warning('Session event observer failed for %s', session_id,
                                                        exc_info=True)
            # True when a live viewer or observer took the frame; the replay ring alone is not
            # delivery (a detached finite viewer reads nothing until it reattaches).
            return delivered

    def catch_up(self, transport, epoch, sequence):
        """Caller holds ``lock``, before *transport* joins ``fanout``: write it the frames after the
        cursor its connection's replay answered. A reconnect asks ``session.events.since`` and then
        ``session.resume``; whatever the session published in between went to no live peer of this
        socket, and the replay was already over, so a reply's tail and ``message.complete`` were lost
        and the view stayed on its first chunk. Under the lock nothing can publish ahead of them."""
        replay = self.since(epoch, sequence)
        if replay['snapshot_required']:
            return
        for frame in replay['events']:
            transport.write({'jsonrpc': '2.0', 'method': 'event', 'params': frame})

    def since(self, epoch, sequence):
        with self.lock:
            current_epoch, latest = self.watermark()
            frames = event_replay.events_since(self._key, sequence)
            missing = (epoch != current_epoch or sequence > latest
                       or event_replay.is_truncated(self._key, sequence)
                       or event_replay.latest_seq(self._key) != latest)
            events = [] if missing else [
                {**frame, 'session_id': self._public_id} if frame.get('session_id') == self._key else frame
                for frame in frames]
            return {'events': events, 'latest_seq': latest, 'last_sequence': latest,
                    'epoch': current_epoch, 'replay_epoch': current_epoch,
                    'truncated': missing, 'snapshot_required': missing, 'count': len(events)}
