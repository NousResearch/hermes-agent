"""``session.prune``: the owner runs `hermes sessions prune` for its own store.

The CLI computes the filters and shows the preview from a read-only view; the delete itself is
this owner's write, through the same ``prune_sessions`` / ``retire_prunable`` path housekeeping
uses (busy sessions and the logical owner of a retained reset child are skipped). A CLI writing
the store under a live gateway would be a second writer, so it never does.
"""
import asyncio
from pathlib import Path

from persistence.core.errors import AuthorityRefusal

_SCALARS = (type(None), bool, int, float, str)


def _filters(raw):
    from persistence.housekeeping.maintenance import PRUNE_FILTER_NAMES
    if (not isinstance(raw, dict) or set(raw) - (PRUNE_FILTER_NAMES | {'older_than_days'})
            or not all(isinstance(value, _SCALARS) for value in raw.values())):
        raise AuthorityRefusal('invalid_params')
    return dict(raw)


async def prune(connection, ref, params):
    authority, actor = connection.authority, connection.actor
    # Store-wide deletion across every principal's rows: the local operator only (a ticket from
    # this host's private control socket), never a remote or browser grant.
    if not connection.native_owner or 'session:operator' not in actor.capabilities:
        raise AuthorityRefusal('permission_denied')
    if actor.profile_id != authority.profile_id:
        raise AuthorityRefusal('profile_mismatch')
    never_active = params.get('never_active_days')
    if ('filters' in params) == (never_active is not None) or (
            never_active is not None and (isinstance(never_active, bool) or not isinstance(never_active, (int, float))
                                          or never_active < 0)):
        raise AuthorityRefusal('invalid_params')
    filters = _filters(params['filters']) if 'filters' in params else None
    authority._require_admission_open()
    from gateway.session_runtime_workers import track_mutation
    task = track_mutation(authority, _apply(authority, filters, never_active))
    return await asyncio.shield(task)


async def _apply(authority, filters, never_active):
    from hermes_constants import get_hermes_home
    from gateway.session_ingress_media import native_inputs_root
    from gateway.session_mutations import retire_live_sessions
    from gateway.session_persistence import authority_persistence
    from persistence.housekeeping.maintenance import PruneCriteria
    from session_retention import after_sessions_removed
    store = authority_persistence(authority)
    # Same transcript dir the CLI's local prune unlinks under (dispatch runs in the owner's scope).
    sessions_dir = Path(get_hermes_home()) / 'sessions'
    media_root = str(native_inputs_root())

    def prune_never_active():
        outcome = store.routing.prune_never_active(older_than_days=float(never_active), media_root=media_root)
        if outcome.deleted_ids or outcome.sessions_skipped:  # the delete ran: its post-commit work follows
            single = len(outcome.deleted_ids) == 1 and not outcome.sessions_skipped
            after_sessions_removed(store, store.scope.profile_dir, outcome.removed_ids, reason='delete',
                                   primary=outcome.deleted_ids[0] if single else None, sessions_dir=sessions_dir)
            store.routing.sweep_routing_entries_for_sessions(outcome.deleted_ids)
        return outcome

    def prune_sessions():
        bounds = {key: filters[key] for key in ('older_than_days', 'source') if key in filters}
        criteria = PruneCriteria(**{key: value for key, value in filters.items() if key not in bounds})
        outcome = store.maintenance.prune_sessions(
            **bounds, criteria=criteria, exclude_active_write_guards=True, media_root=media_root)
        after_sessions_removed(store, store.scope.profile_dir, outcome.deleted_ids, reason='prune',
                               sessions_dir=sessions_dir)
        return outcome

    if filters is None:
        outcome = await asyncio.to_thread(prune_never_active)
        result = {'deleted': outcome.sessions_deleted, 'routing_deleted': outcome.routing_entries_deleted,
                  'skipped': outcome.sessions_skipped}
    else:
        outcome = await asyncio.to_thread(prune_sessions)
        result = {'deleted': outcome.count}
    retire_live_sessions(authority, list(outcome.deleted_ids))
    return result
