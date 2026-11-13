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


def verified_restored_relay_source(runner: Any, entry: Any) -> Any:
    """Verify a settled Relay lane too, without trusting an old primary-bot guess."""
    from gateway.session_authorities import serving_authorities

    for authority in serving_authorities(runner):
        owner = authority.logical_owner(entry.session_id)
        rows = list_session_admissions(authority.db, session_id=owner, pending_only=False)
        native = [row['payload'] for row in rows if 'native_text_v1' in row['payload']]
        if not native:
            continue
        envelope = native[-1]['native_text_v1']
        if envelope.get('route') != entry.session_key or 'relay' not in envelope.get('provenance', {}):
            continue
        try:
            return restore_native_source(native[-1], runner)
        except RuntimeStoreError:
            continue  # no current verified connector: keep the ordinary fail-closed selector
    return None
