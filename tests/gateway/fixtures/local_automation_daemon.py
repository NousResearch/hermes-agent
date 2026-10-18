"""Crash barrier after real automation commit, before producer ACK/first claim."""
import os
import runpy
import signal

from gateway.session_authority import SessionAuthority
from hermes_state_runtime import list_session_admissions

original = SessionAuthority._schedule


def schedule(self, ref):
    original(self, ref)
    rows = list_session_admissions(self.db, session_id=ref.session_id)
    if any(r['principal_id'].startswith('automation:') for r in rows):
        os.kill(os.getpid(), signal.SIGSTOP)


SessionAuthority._schedule = schedule
runpy.run_module('gateway.run', run_name='__main__')
