"""Hold only execution scheduling after a real API commit and HTTP ACK."""
import runpy
from gateway.session_authority import SessionAuthority
from hermes_state_runtime import list_session_admissions

schedule = SessionAuthority._schedule


def hold_safe(self, ref):
    pending = list_session_admissions(self.db, session_id=ref.session_id)
    if any(row['payload'].get('text') == 'SAFE_QUEUE' for row in pending):
        return
    return schedule(self, ref)


SessionAuthority._schedule = hold_safe
runpy.run_module('gateway.run', run_name='__main__')
