"""Test-only committed-before-schedule crash barrier around the ordinary daemon."""
import os
import runpy
import signal

from gateway.session_authority import SessionAuthority
from hermes_state_runtime import list_session_admissions

schedule = SessionAuthority._schedule


def crash_barrier(self, ref):
    pending = list_session_admissions(self.db, session_id=ref.session_id)
    text = pending[-1]['payload'].get('text') if pending else None
    if text in {'FOREIGN_QUEUE', 'MISSING_QUEUE', 'CORRUPT_QUEUE'}:
        return
    if text == 'RECOVER_QUEUED':
        os.kill(os.getpid(), signal.SIGSTOP)
    return schedule(self, ref)


SessionAuthority._schedule = crash_barrier
# submit's ordinary path must use the same production scheduler for this barrier.
runpy.run_module('gateway.run', run_name='__main__')
