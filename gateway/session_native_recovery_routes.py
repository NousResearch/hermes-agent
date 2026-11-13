"""Rebuild queued native routes from currently verified connector provenance."""
from typing import Any

from gateway.session_envelope import restore_native_source
from hermes_state_runtime import RuntimeStoreError, list_session_admissions


def recovered_native_source(runner: Any, authority: Any, entry: Any, owner: str) -> Any:
    """Old routing entries may lack a receiving bot; a verified admission names it.

    Never infer a primary transport from an unverified stored profile. The regular
    recovery preflight still checks every queued sender before scheduling any work.
    """
    rows = list_session_admissions(authority.db, session_id=owner, pending_only=False)
    native = [row for row in rows if 'native_text_v1' in row['payload']]
    if native:
        try:
            return restore_native_source(native[-1]['payload'], runner)
        except RuntimeStoreError:
            pass  # the authority reports the precise refusal without consuming the queue
    return runner._restored_source(entry)
