"""Bind authenticated child completion input to one in-process parent's persistence boundary."""
from __future__ import annotations

from copy import deepcopy
from functools import partial

from gateway.completion_delivery.storage import (
    completion_ids, eligible_rows, expose, held, hold_after_stop, result_block, resume_on_submission,
)
from gateway.config import Platform
from gateway.session_contract import SessionRef
from gateway.session_policy import STORED_AS_LOCAL, policy_for_source
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError
from gateway.session_producer_lifecycle import drop_producer_turn
from gateway.session_runtime_workers import tracked_write


class CompletionInbox:
    """Writes a fresh tool result and the child-result receipts in the same owner transaction."""

    def __init__(self, authority, session_id, generation):
        self.authority = authority
        self.session_id = session_id
        self.generation = generation
        task = authority.sessions[session_id].task
        self.loop = task.get_loop() if task is not None else None

    def append_messages(self, agent, messages, **kwargs):
        db = agent._session_db
        target = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get('role') == 'tool'
                       and not messages[i].get('api_content')), None)
        if target is None or getattr(agent, '_interrupt_requested', False):
            db.append_messages_batch(session_id=agent.session_id, messages=messages, **kwargs)
            return {}
        original = deepcopy(messages[target])
        exposed = []
        blocks = {}

        def write(conn):
            # SQLite retries must not append the same excerpt twice to the caller-owned row.
            messages[target].clear()
            messages[target].update(deepcopy(original))
            exposed.clear()
            blocks.clear()
            rows = [] if getattr(agent, '_interrupt_requested', False) else eligible_rows(
                conn, self.session_id, self.authority.epoch, self.generation, agent.session_id)
            if rows:
                message = messages[target]
                block = result_block(rows)
                blocks[target] = block
                content = message.get('content')
                message['content'] = ([*content, {'type': 'text', 'text': block}] if isinstance(content, list)
                                      else str(content or '') + block)
                message['display_metadata'] = {**(message.get('display_metadata') or {}),
                                               'completion_admissions': [row['admission_id'] for row in rows]}
            inserted = db._append_messages_in_transaction(conn, agent.session_id, messages, **kwargs)
            expose(conn, rows, messages[target], self.session_id, self.authority.epoch, self.generation)
            exposed.extend(row['admission_id'] for row in rows)
            return inserted

        try:
            db._execute_transcript_write(write, messages, patience_s=db._TRANSCRIPT_WRITE_PATIENCE_S)
        except BaseException:
            messages[target].clear()
            messages[target].update(original)
            raise
        if exposed and self.loop is not None and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._publish, tuple(exposed))
        return blocks

    def _publish(self, admission_ids):
        authority = self.authority
        if self.session_id not in authority.sessions:
            return
        live = authority.sessions[self.session_id]
        for admission_id in admission_ids:
            # Frame-local identity: unlocked readers must keep seeing the running parent's generation.
            live.event_stream.publish(self.session_id, {
                'admission_id': admission_id, 'text': '', 'content': '', 'status': 'complete',
                'outcome': 'completed', 'delivery_only': True},
                execution={'authority_epoch': authority.epoch, 'admission_id': admission_id})
        for admission_id in admission_ids:
            authority.native_waiters.discard(admission_id)
            drop_producer_turn(authority, admission_id)
            waiter = authority.waiters.pop(admission_id, None)
            if waiter is not None and not waiter.done():
                waiter.set_result('')
        authority._publish_pending(SessionRef(authority.profile_id, self.session_id))


def interactive_owner(authority, session_id):
    live = authority.sessions.get(session_id)
    if live is None or live.source is None or live.source.platform != Platform.LOCAL:
        return False
    try:
        policy = policy_for_source(authority.runner, live.source)
    except RuntimeStoreError:
        return False  # Missing launch proof must not turn a transport label into producer authority.
    return policy is not None and policy.source in STORED_AS_LOCAL


def bind_inbox(authority, session_id, generation, agent):
    """Only a live LOCAL owner with the same physical state store gets this private channel."""
    db = getattr(agent, '_session_db', None)
    # Path equality also holds for AsyncSessionDB/remote facades; their methods are not this sync writer.
    same_store = isinstance(db, SessionDB) and db.db_path == authority.db.db_path
    if interactive_owner(authority, session_id) and same_store and not getattr(agent, '_persist_disabled', False):
        agent._completion_inbox = CompletionInbox(authority, session_id, generation)
        agent._completion_result_blocks = {}
    elif '_completion_inbox' in getattr(agent, '__dict__', {}):
        agent._completion_inbox = None


async def pause_completions(authority, ref, generation):
    if interactive_owner(authority, ref.session_id):
        await tracked_write(authority, partial(hold_after_stop, authority.db, authority.epoch, ref.session_id, generation))


def completion_is_held(authority, row):
    if row is None or not completion_ids(row):
        return False
    with authority.db._read_ctx() as conn:
        return held(conn, row['target_session_id'])


def viewer_admission_guard(authority, ref, guard):
    """Resume only on an authorized public submission, atomically with its admission."""
    def authorize(conn):
        guard(conn)
        resume_on_submission(conn, ref.session_id)
    return authorize
