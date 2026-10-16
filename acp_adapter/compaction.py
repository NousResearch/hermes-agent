"""ACP v1 compaction entities; no internal summary or prompt framing is exposed."""

from contextlib import contextmanager, asynccontextmanager
import logging
import threading
import uuid

from agent.async_utils import safe_schedule_threadsafe
from agent.conversation_compression import COMPACTION_HEARTBEAT_STATUS, is_compaction_progress_status

logger = logging.getLogger(__name__)


class CompactionEvents:
    def __init__(self, transport, session_id, cancel_event, *, defer_terminal=False):
        self.transport, self.session_id, self.cancel_event = transport, session_id, cancel_event
        self.defer_terminal = defer_terminal
        self.deferred_outcome = None
        self.active_id = None
        self.closed = False
        self.lock = threading.RLock()

    def _send(self, status):
        if not self.transport.supports_session("compaction") or self.transport.sender is None:
            return
        with self.lock:
            if self.closed:
                return
            compaction_id = self.active_id or str(uuid.uuid4())
            update = {"sessionUpdate": "compaction_update", "compactionId": compaction_id, "status": status}
            future = safe_schedule_threadsafe(self.transport.send(self.session_id, update), self.transport.loop,
                                               logger=logger)
            if future is not None:
                future.result(timeout=5)
                self.active_id = compaction_id if status == "in_progress" else None

    def status(self, kind, message):
        from gateway.warning_notifications import is_warning_status

        if is_warning_status(kind, message):
            return
        if kind in {"lifecycle", "compacting"} and is_compaction_progress_status(message):
            with self.lock:
                if self.deferred_outcome is not None:
                    return
                if message == COMPACTION_HEARTBEAT_STATUS and self.active_id is None:
                    return  # A late heartbeat must not invent another compaction.
                self._send("in_progress")

    def terminal(self, status):
        if status in {"completed", "failed", "cancelled"}:
            with self.lock:
                if self.closed:
                    return
                if self.defer_terminal:
                    self.deferred_outcome = status
                else:
                    self._send(status)

    @contextmanager
    def bind(self, agent):
        if not self.transport.supports_session("compaction"):
            yield
            return
        previous_status = getattr(agent, "status_callback", None)
        previous_terminal = getattr(agent, "compaction_callback", None)

        def status(kind, message):
            try:
                self.status(kind, message)
            except Exception:
                logger.debug("Could not send compaction progress", exc_info=True)
            if callable(previous_status):
                try:
                    previous_status(kind, message)
                except Exception:
                    logger.debug("Previous status observer failed", exc_info=True)

        def terminal(value):
            try:
                self.terminal(value)
            except Exception:
                logger.debug("Could not send compaction completion", exc_info=True)
            if callable(previous_terminal):
                try:
                    previous_terminal(value)
                except Exception:
                    logger.debug("Previous compaction observer failed", exc_info=True)

        agent.status_callback, agent.compaction_callback = status, terminal
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            # Never leave an announced operation spinning after its turn/command ends.
            with self.lock:
                try:
                    if self.active_id or self.deferred_outcome is not None:
                        cancelled = bool(self.cancel_event and self.cancel_event.is_set())
                        cancelled = cancelled or getattr(agent, "_interrupt_requested", False) is True
                        outcome = self.deferred_outcome if succeeded else None
                        self._send(outcome or ("cancelled" if cancelled else "failed"))
                except Exception:
                    logger.debug("Could not close compaction entity", exc_info=True)
                finally:
                    self.closed = True
                    agent.status_callback, agent.compaction_callback = previous_status, previous_terminal


@asynccontextmanager
async def manual_compaction(agent, session_id, *, enabled):
    """A gateway mutation completes only after its durable commit and viewer refresh."""
    transport = agent.session_updates
    if not enabled or not transport.supports_session('compaction'):
        yield
        return
    identity = str(uuid.uuid4())

    async def send(status):
        try:
            await transport.send(agent._editor_id(session_id), {
                'sessionUpdate': 'compaction_update', 'compactionId': identity, 'status': status})
        except Exception:
            logger.debug('Could not send manual compaction lifecycle', exc_info=True)

    await send('in_progress')
    outcome = {'status': 'completed'}
    try:
        yield outcome
    except BaseException as exc:
        import asyncio
        await send('cancelled' if isinstance(exc, asyncio.CancelledError) else 'failed')
        raise
    else:
        await send(outcome['status'])


async def project_compaction(agent, kind, payload, editor_id):
    from gateway.session_compaction_events import valid_update
    if kind != 'context.compaction':
        return False
    if agent.session_updates.supports_session('compaction') and valid_update(payload):
        await agent.session_updates.send(editor_id, {'sessionUpdate': 'compaction_update', **payload})
    return True
