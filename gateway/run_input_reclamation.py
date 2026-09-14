"""Working-copy collection at the gateway owner's existing maintenance points."""
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


def collect_legacy_copies_before_ingress(runner, authority):
    """Only the pre-listener initialization path may drain shared legacy aliases."""
    from gateway.hosted_room_input_reclamation import initialize_working_copies, collect_legacy_input_aliases

    descriptor = getattr(runner, 'session_runtime_descriptor', {})
    if (descriptor.get('state') != 'starting'
            or runner._running or runner._draining
            or getattr(runner, 'session_api', None) is not None
            or getattr(runner, 'session_control_server', None) is not None
            or runner.adapters
            or any(getattr(runner, '_profile_adapters', {}).values())
            or not _owns_store(authority)):
        return
    initialize_working_copies(authority.db, epoch=authority.epoch)
    collect_legacy_input_aliases(authority.db, epoch=authority.epoch, limit=64)
