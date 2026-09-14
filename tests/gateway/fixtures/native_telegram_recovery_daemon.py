"""First-owner-only scheduler barrier; every recovery boot is ordinary."""
import runpy

from gateway.session_authority import SessionAuthority
from hermes_state_runtime import list_session_admissions

schedule = SessionAuthority._schedule


def hold_queued(self, ref):
    pending = list_session_admissions(self.db, session_id=ref.session_id)
    if any(row['payload'].get('text', '').startswith(('SAFE_', 'REVOKED_')) for row in pending):
        return
    return schedule(self, ref)


SessionAuthority._schedule = hold_queued
runpy.run_module('gateway.run', run_name='__main__')
