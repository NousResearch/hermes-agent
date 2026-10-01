"""Private input-copy collection at the gateway owner's existing maintenance points."""
from pathlib import Path
import logging
import sqlite3

from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)


def _owns_store(authority):
    from gateway.runtime_ownership import process_ownership

    home = Path(authority.profile_id)
    return (home.is_absolute() and home.resolve() == home
            and Path(authority.db.db_path).resolve().parent == home
            and process_ownership.owns(home))


def collect_gateway_input_copies(runner):
    """Called by the existing registered housekeeping writer, never by a viewer."""
    from gateway.hosted_room_input_reclamation import collect_working_copies
    from gateway.session_authorities import all_authorities, owner_scope

    for authority in all_authorities(runner):
        if runner._draining:
            break
        if not _owns_store(authority):
            continue
        try:
            with owner_scope(authority):
                collect_working_copies(authority.db, epoch=authority.epoch, limit=64)
        except (RuntimeStoreError, OSError, sqlite3.Error):
            # One unavailable profile must not starve the other owned stores.
            logger.debug('Working-copy collection deferred for an unavailable store')


def collect_native_inputs_before_ingress(runner, authority):
    """Startup only, before the owner is published: release unheld image reservations.

    Runs only while the whole gateway is provably still starting; a runner whose
    state is unknown is skipped. Deferred, never fatal: startup never depends on it.
    """
    from gateway.hosted_room_input_reclamation import collect_native_inputs

    descriptor = getattr(runner, 'session_runtime_descriptor', None)
    if not (isinstance(descriptor, dict) and descriptor.get('state') == 'starting'
            and getattr(runner, '_running', None) is False
            and getattr(runner, '_draining', None) is False
            and getattr(runner, 'session_api', None) is None
            and getattr(runner, 'session_control_server', None) is None
            and not getattr(runner, 'adapters', None)
            and not any((getattr(runner, '_profile_adapters', None) or {}).values())
            and _owns_store(authority)):
        return
    try:
        collect_native_inputs(authority.db, epoch=authority.epoch, limit=64)
    except (RuntimeStoreError, OSError, sqlite3.Error) as exc:
        logger.warning('Image input collection deferred for %s: %s', authority.profile_id, exc)
