"""``session.prune``: the owner applies `hermes sessions prune` and the dashboard's bulk deletes.

The client computes the selection and shows the preview from a read-only view; the delete itself
is this owner's write, through the same ``SessionDB`` paths the offline commands use (busy sessions
and the logical owner of a retained reset child are skipped). A CLI or a standalone dashboard
writing the store under a live gateway would be a second writer, so neither does.
"""
import asyncio
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError

_SCALARS = (type(None), bool, int, float, str)
#: Same cap as the dashboard's ``POST /api/sessions/bulk-delete``: one selection never holds the writer long.
_MAX_IDS = 500


def _filters(raw):
    from hermes_state_maintenance import _PRUNE_FILTER_NAMES
    if (not isinstance(raw, dict) or set(raw) - (_PRUNE_FILTER_NAMES | {'older_than_days'})
            or not all(isinstance(value, _SCALARS) for value in raw.values())):
        raise RuntimeStoreError('invalid_params')
    return dict(raw)


# Exactly one selector per call: `hermes sessions prune` filters, `--never-active` days, the
# dashboard's "Delete empty" sweep, or the dashboard's hand-picked ids.
_SELECTORS = {
    'filters': lambda value: isinstance(value, dict),
    'never_active_days': lambda value: (not isinstance(value, bool) and isinstance(value, (int, float))
                                        and value >= 0),
    'empty': lambda value: value is True,
    'ids': lambda value: (isinstance(value, list) and len(value) <= _MAX_IDS
                          and all(isinstance(sid, str) for sid in value)),
}


def _selector(params):
    present = [key for key in _SELECTORS if params.get(key) is not None]
    if len(present) != 1 or not _SELECTORS[present[0]](params[present[0]]):
        raise RuntimeStoreError('invalid_params')
    key = present[0]
    return key, _filters(params[key]) if key == 'filters' else params[key]


async def prune(connection, ref, params):
    authority, actor = connection.authority, connection.actor
    # Store-wide deletion across every principal's rows: the local operator only (a ticket from
    # this host's private control socket), never a remote or browser grant.
    if not connection.native_owner or 'session:operator' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    selector, value = _selector(params)
    authority._require_admission_open()
    from gateway.session_runtime_workers import track_mutation
    task = track_mutation(authority, _apply(authority, selector, value))
    return await asyncio.shield(task)


def _by_filters(db, value, sessions_dir, deleted_ids):
    return {'deleted': db.prune_sessions(sessions_dir=sessions_dir, exclude_active_write_guards=True,
                                         deleted_ids=deleted_ids, **value)}


def _by_never_active(db, value, sessions_dir, deleted_ids):
    deleted, routing, skipped = db.prune_never_active_keyed_sessions(
        older_than_days=float(value), sessions_dir=sessions_dir, deleted_ids=deleted_ids)
    return {'deleted': deleted, 'routing_deleted': routing, 'skipped': skipped}


def _by_empty(db, _value, sessions_dir, deleted_ids):
    return {'deleted': db.delete_empty_sessions(sessions_dir=sessions_dir, deleted_ids=deleted_ids)}


def _by_ids(db, value, sessions_dir, deleted_ids):
    skipped = []  # rows a live turn/compression still owns; the dashboard keeps them listed
    deleted = db.delete_sessions(value, sessions_dir=sessions_dir, exclude_active_write_guards=True,
                                 skipped_ids=skipped, include_compression_chain=True, deleted_ids=deleted_ids)
    return {'deleted': deleted, 'skipped_ids': skipped}


_APPLY = {'filters': _by_filters, 'never_active_days': _by_never_active, 'empty': _by_empty, 'ids': _by_ids}


async def _apply(authority, selector, value):
    from hermes_constants import get_hermes_home
    from gateway.session_mutations import retire_live_sessions
    # Same transcript dir the CLI's local prune unlinks under (dispatch runs in the owner's scope).
    sessions_dir = Path(get_hermes_home()) / 'sessions'
    deleted_ids = []
    result = await asyncio.to_thread(_APPLY[selector], authority.db, value, sessions_dir, deleted_ids)
    retire_live_sessions(authority, deleted_ids)
    return result
