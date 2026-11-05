"""Crash barrier after real automation commit, before producer ACK/first claim."""
import os
import runpy
import signal

from gateway.session_authority import SessionAuthority, admission_row

original = SessionAuthority._schedule


def schedule(self, ref):
    original(self, ref)
    rows = [admission_row(record) for record in self.persistence.admissions.list(session_id=ref.session_id)]
    if any(r['principal_id'].startswith('automation:') for r in rows):
        os.kill(os.getpid(), signal.SIGSTOP)


SessionAuthority._schedule = schedule
runpy.run_module('gateway.run', run_name='__main__')
