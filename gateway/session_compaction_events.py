"""Execution-owned compaction lifecycle, without internal summary or error text."""
import logging
import threading
import uuid

logger = logging.getLogger(__name__)
STATUSES = {'in_progress', 'completed', 'failed', 'cancelled'}


def valid_update(update):
    return (isinstance(update, dict) and set(update) == {'compactionId', 'status'}
            and isinstance(update['compactionId'], str) and 0 < len(update['compactionId']) <= 64
            and isinstance(update['status'], str) and update['status'] in STATUSES)


class CompactionProgress:
    def __init__(self, publish):
        self.publish = publish
        self.active = None
        self.closed = False
        self.lock = threading.RLock()

    def send(self, status):
        with self.lock:
            if self.closed or (status != 'in_progress' and self.active is None):
                return
            identity = self.active or str(uuid.uuid4())
            self.active = identity if status == 'in_progress' else None
            try:
                self.publish({'compactionId': identity, 'status': status})
            except Exception:
                logger.debug('Compaction lifecycle publication failed', exc_info=True)

    def status(self, kind, text):
        from agent.conversation_compression import COMPACTION_HEARTBEAT_STATUS, is_compaction_progress_status
        from gateway.warning_notifications import is_warning_status
        with self.lock:
            if (kind in {'lifecycle', 'compacting'} and not is_warning_status(kind, text)
                    and is_compaction_progress_status(text)
                    and (text != COMPACTION_HEARTBEAT_STATUS or self.active is not None)):
                self.send('in_progress')

    def install(self, agent):
        previous_status = getattr(agent, 'status_callback', None)
        previous_terminal = getattr(agent, 'compaction_callback', None)

        def status(kind, text=None):
            self.status(kind, text)
            if callable(previous_status):
                previous_status(kind, text)

        def terminal(value):
            self.send(value)
            if callable(previous_terminal):
                previous_terminal(value)

        agent.status_callback = status
        agent.compaction_callback = terminal

    def close(self, agent):
        with self.lock:
            self.send('cancelled' if getattr(agent, '_interrupt_requested', False) is True else 'failed')
            self.closed = True


def publish_worker_progress(authority, session_id, generation, frame):
    from gateway.run_turn_progress import publish_worker_tool_event
    if frame.get('type') == 'compaction' and set(frame) == {'type', 'update'} and valid_update(frame['update']):
        authority.publish_execution(session_id, generation, 'context.compaction', frame['update'])
        return True
    return publish_worker_tool_event(authority, session_id, generation, frame)
