"""Hold only execution scheduling after a real API commit and HTTP ACK."""
import runpy
from gateway.session_authority import SessionAuthority, admission_row

schedule = SessionAuthority._schedule


def hold_safe(self, ref):
    pending = [admission_row(record) for record in self.persistence.admissions.list(session_id=ref.session_id)]
    if any(row['payload'].get('text') == 'SAFE_QUEUE' for row in pending):
        return
    return schedule(self, ref)


SessionAuthority._schedule = hold_safe
runpy.run_module('gateway.run', run_name='__main__')
