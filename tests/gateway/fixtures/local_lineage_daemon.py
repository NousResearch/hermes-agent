"""Pause only after real rotation/reset publication and durable queued admission."""
import copy
import os
import runpy
import signal
from types import SimpleNamespace

from agent.conversation_compression import _publish_rotated_compaction
from gateway.session_authority import SessionAuthority
from hermes_state_runtime import list_session_admissions

schedule = SessionAuthority._schedule


def boundary(self, ref):
    pending = list_session_admissions(self.db, session_id=ref.session_id)
    text = pending[-1]['payload'].get('text') if pending else None
    if text in {'RECOVER_RESET', 'FOREIGN_QUEUE'}:
        return
    if text == 'RECOVER_SAFE':
        for logical_id, live in self.sessions.items():
            agent = self.agent(SimpleNamespace(session_id=logical_id))
            if agent is None:
                continue
            if agent.model == 'frozen-reset':
                self.runner.session_store.reset_session(live.route)
                continue
            old = agent.session_id
            holder = 'lineage-fixture'
            assert self.db.try_acquire_compression_lock(old, holder)
            try:
                history = self.db.get_messages_as_conversation(old)
                _publish_rotated_compaction(agent, history, copy.deepcopy(history),
                    new_system_prompt=agent._cached_system_prompt or '',
                    lease=SimpleNamespace(holder=holder, ttl=300, watermark=None),
                    old_session_id=old, compressed_user_turn_outcome='already_present')
            finally:
                self.db.release_compression_lock(old, holder)
        os.kill(os.getpid(), signal.SIGSTOP)
    return schedule(self, ref)


SessionAuthority._schedule = boundary
runpy.run_module('gateway.run', run_name='__main__')
