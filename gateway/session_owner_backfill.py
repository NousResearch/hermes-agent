"""``session.owner_backfill``: the live owner stamps its own legacy NULL-profile rows.

Every Desktop/dashboard boot issues ``POST /api/sessions/owner-backfill``. While a standalone
gateway owns the profile it holds the owner reservation for its whole lifetime, so the dashboard
can never take the exclusive-maintenance write and answered 409 on every boot (#148531) — and a
dashboard that answers "deferred" without stamping leaves the client fail-closed on those rows for
good. The owner is the only writer there, so the stamp is its own write, on its own store, with the
same profile name its bring-up derives (``gateway/session_authority.py::
backfill_legacy_session_owners``). Unlike that bring-up path this one never swallows a failure into
a silent ``0``: the caller is a request that must see the truth.
"""
import asyncio


async def backfill(connection, ref, params):
    from hermes_state_runtime import RuntimeStoreError

    authority, actor = connection.authority, connection.actor
    # Store-wide ownership metadata across every principal's rows: the local operator only (a
    # ticket from this host's private control socket), never a remote or browser grant.
    if not connection.native_owner or 'session:operator' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if params:
        raise RuntimeStoreError('invalid_params')
    stamp = authority.db._own_profile_name()
    if not stamp:
        # Outside the profile tree there is no derivable owner: NULL beats a fabricated one.
        raise RuntimeStoreError('profile_mismatch', field='profile',
                                detail='store is outside the profile tree, so it has no owner to stamp')
    stamped = await asyncio.to_thread(authority.db.backfill_null_session_profiles, stamp)
    return {'stamped': stamped, 'profile': stamp}
