"""``session.bulk_delete``: the owner runs a dashboard bulk delete / empty sweep for its own store.

A standalone ``hermes dashboard`` / ``hermes serve`` process is not the store owner while a
gateway serves the same home, so its own exclusive-maintenance reservation can never win and the
Sessions-page bulk delete and "Delete empty" actions die on a bare 409 (#138915). The owner already
exposes the single-row ``session.mutate`` verb; this adds the bulk family over the same control
socket, through the same store methods the offline route uses
(:meth:`SessionRepository.delete_many` / :meth:`SessionRepository.delete_empty`), so selection,
compression-chain expansion and live-lease guards behave identically on both paths.

Store-wide deletion across every principal's rows is a local-operator capability only (a ticket
from this host's private control socket), never a remote or browser grant — the same gate
``session.prune`` applies. Cross-login scoping for forwarded dashboard logins is a separate
hardening pass and tracked on #138915.
"""

import asyncio

from persistence.core.errors import AuthorityRefusal

# A bulk selection is capped on the HTTP route too; keep the owner-side gate identical so a caller
# cannot bypass the 500-row writer lock limit by talking to the control socket directly.
_MAX_IDS = 500


async def bulk_delete(connection, ref, params):
    authority, actor = connection.authority, connection.actor
    # Same local-operator gate as session.prune: never a remote or browser grant.
    if not connection.native_owner or "session:operator" not in actor.capabilities:
        raise AuthorityRefusal("permission_denied")
    if actor.profile_id != authority.profile_id:
        raise AuthorityRefusal("profile_mismatch")

    empty_sweep = bool(params.get("empty_sweep"))
    raw_ids = params.get("ids")
    if empty_sweep:
        if raw_ids:
            raise AuthorityRefusal("invalid_params")
    else:
        if not isinstance(raw_ids, list) or not all(
            isinstance(sid, str) and sid for sid in raw_ids
        ):
            raise AuthorityRefusal("invalid_params")
        # De-duplicate while keeping the caller's selection; reject the runaway-selection case.
        raw_ids = list(dict.fromkeys(raw_ids))
        if len(raw_ids) > _MAX_IDS:
            raise AuthorityRefusal("invalid_params")
        if not raw_ids:
            return {"deleted": 0, "skipped_active": []}

    authority._require_admission_open()
    from gateway.session_runtime_workers import track_mutation

    task = track_mutation(
        authority, _apply(authority, raw_ids if not empty_sweep else None, empty_sweep)
    )
    return await asyncio.shield(task)


async def _apply(authority, ids, empty_sweep):
    from gateway.session_ingress_media import native_inputs_root
    from gateway.session_mutations import retire_live_sessions
    from gateway.session_persistence import authority_persistence
    from session_retention import after_sessions_removed

    store = authority_persistence(authority)
    media_root = str(native_inputs_root())

    def delete():
        if empty_sweep:
            # Same sweep the offline DELETE /api/sessions/empty runs: ended, unarchived, no-message
            # rows only; active sessions are structurally excluded inside delete_empty.
            return store.sessions.delete_empty(media_root=media_root)
        # Mirror the offline bulk route: guarded rows are skipped (not a hard error), compression
        # chains are removed whole, and the toast counts requested rows only.
        outcome = store.sessions.delete_many(
            ids,
            skip_guarded=True,
            include_compression_chain=True,
            media_root=media_root,
        )
        return outcome

    if empty_sweep:
        removed = await asyncio.to_thread(delete)
        after_sessions_removed(
            store, store.scope.profile_dir, removed, reason="empty_sweep"
        )
        retire_live_sessions(authority, list(removed))
        return {"deleted": len(removed), "skipped_active": []}

    outcome = await asyncio.to_thread(delete)
    after_sessions_removed(
        store,
        store.scope.profile_dir,
        outcome.removed_ids,
        reason="delete",
        primary=ids[0] if len(ids) == 1 else None,
    )
    retire_live_sessions(authority, list(outcome.removed_ids))
    return {"deleted": outcome.deleted, "skipped_active": list(outcome.skipped_guarded)}
