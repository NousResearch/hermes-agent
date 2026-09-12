"""Authority-local subscription ordering over the existing bounded transports/replay."""
from copy import deepcopy
import threading
import uuid

from tui_gateway import event_replay
from tui_gateway.transport import FanoutTransport


class SessionEvents:
    def __init__(self):
        self.lock = threading.RLock()
        self.fanout = FanoutTransport()
        # A private lifetime key prevents two profiles (or legacy TUI owners)
        # with the same persisted ID from reading each other's replay.
        self._key = uuid.uuid4().hex
        self.epoch = uuid.uuid4().hex
        self.sequence = 0
        self.execution = {}
        # Synchronous same-thread recipients (API run projections): unlike fanout peers they
        # cannot lose a frame to a detach that races the writer thread.
        self.observers = set()

    def watermark(self):
        with self.lock:
            if self.sequence and not event_replay.latest_seq(self._key):
                # The shared bounded cache evicted this session. Never reuse a
                # sequence in the old epoch, even if publication resumes later.
                self.epoch = uuid.uuid4().hex
                self.sequence = 0
            return self.epoch, self.sequence

    def publish(self, session_id, payload, *, event_type="message.complete"):
        with self.lock:
            self.watermark()
            frame = {'jsonrpc': '2.0', 'method': 'event', 'params': {
                **self.execution, 'type': event_type, 'session_id': self._key,
                'payload': deepcopy(payload), 'replay_epoch': self.epoch}}
            event_replay._stamp_event(frame)
            if frame['params']['seq'] <= self.sequence:
                # Another publisher may evict us after watermark's lookup.
                self.epoch = uuid.uuid4().hex
                frame['params']['replay_epoch'] = self.epoch
            # The ring owns this same event object; its lookup key remains
            # private while both replay and live recipients see canonical IDs.
            frame['params']['session_id'] = session_id
            self.sequence = frame['params']['seq']
            self.fanout.write(frame)
            for observer in tuple(self.observers):
                observer(frame)

    def since(self, epoch, sequence):
        with self.lock:
            current_epoch, latest = self.watermark()
            frames = event_replay.events_since(self._key, sequence)
            missing = (epoch != current_epoch or sequence > latest
                       or event_replay.is_truncated(self._key, sequence)
                       or event_replay.latest_seq(self._key) != latest)
            events = [] if missing else deepcopy(frames)
            return {'events': events, 'latest_seq': latest, 'last_sequence': latest,
                    'epoch': current_epoch, 'replay_epoch': current_epoch,
                    'truncated': missing, 'snapshot_required': missing, 'count': len(events)}
