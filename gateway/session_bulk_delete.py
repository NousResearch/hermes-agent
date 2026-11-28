"""``session.bulk_delete``: the owner runs a dashboard bulk delete / empty sweep for its own store.

A standalone ``hermes dashboard`` / ``hermes serve`` process is not the store owner while a
gateway serves the same home, so its own exclusive-maintenance reservation can never win and the
Sessions-page bulk delete and "Delete empty" actions die on a bare 409 (#138915). The owner already
exposes the single-row ``session.mutate`` verb; this adds the bulk family over the same control
socket, through the same store methods the offline route uses
(:meth:`SessionRepository.delete_many` / :meth:`SessionRepository.delete_empty`), so selection,
compression-chain expansion and live-lease guards behave identically on both paths.

Two caller kinds, mirroring the single-row ``session.mutate`` authorization boundary:

* a **native local operator** (the private bootstrap ticket; its subject is not an ``auth:v1:``
  login) keeps store-wide reach — housekeeping, cron and the CLI delete every principal's rows,
  exactly as ``session.prune`` does;
* a **forwarded dashboard login** (an ``auth:v1:[provider, issuer, user]`` subject) may only remove
  rows its single-row delete could remove. Each candidate is checked with the same
  ``authorize_history`` rule a single delete runs (without claiming); a row another login owns is
  skipped and reported back, never silently dropped, so the dashboard keeps it listed.

Store-wide deletion still requires the local-operator capability gate (native bootstrap +
``session:operator``); a remote or browser grant without it is refused outright.
"""
import asyncio
from typing import TYPE_CHECKING

from persistence.core.errors import AuthorityRefusal

if TYPE_CHECKING:
    from persistence.conversation.sessions import DeleteManyOutcome

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
        authority, _apply(authority, actor, raw_ids if not empty_sweep else None, empty_sweep)
    )
    return await asyncio.shield(task)


def _forwarded_login(actor):
    """The (provider, issuer, user) tuple for a forwarded dashboard login subject, else None.

    A native local-operator subject (CLI/worker receipt) is not an ``auth:v1:`` login, so the parser
    returns None and the caller keeps full store-wide reach.
    """
    from gateway.session_identity import parse_authenticated_subject

    return parse_authenticated_subject(actor.subject)


def _owner_rule():
    from gateway.session_persistence import owner_rule

    return owner_rule()


def _rows_actor_may_delete(store, actor, session_ids):
    """Partition *session_ids* into (allowed, foreign_skipped) for a forwarded login.

    A row passes when the same history rule a single-row delete runs
    (``bindings.authorize_history``, no claim) admits this subject; a row it refuses or cannot see is
    returned in ``foreign_skipped`` so the dashboard keeps it listed. An unbound, owner-less local
    row is admissible for any same-profile login, matching the single-delete local rule.
    """
    allowed, foreign_skipped = [], []
    for sid in session_ids:
        try:
            ok = store.bindings.authorize_history(
                session_id=sid,
                profile_id=actor.profile_id,
                subject=actor.subject,
                may_create="session:create" in actor.capabilities,
                owner=_owner_rule(),
            )
        except AuthorityRefusal:
            ok = False
        (allowed if ok else foreign_skipped).append(sid)
    return allowed, foreign_skipped


def _empty_candidate_ids(store):
    """The current empty, ended, non-archived session ids (the DELETE /empty selection), read-only."""
    from persistence.sqlite.conversation._store.sessions import _EMPTY_SESSION_WHERE

    holder = store.sessions._store.holder()
    rows = holder._read_all(
        f"SELECT id FROM sessions WHERE {_EMPTY_SESSION_WHERE}"
    )
    return [row["id"] for row in rows]


async def _apply(authority, actor, ids, empty_sweep):
    from gateway.session_ingress_media import native_inputs_root
    from gateway.session_mutations import retire_live_sessions
    from gateway.session_persistence import authority_persistence
    from session_retention import after_sessions_removed

    store = authority_persistence(authority)
    media_root = str(native_inputs_root())
    is_forwarded_login = _forwarded_login(actor) is not None

    def run_empty_sweep() -> tuple[tuple[str, ...], tuple[str, ...]]:
        if not is_forwarded_login:
            # Native operator: the same whole-store sweep the offline route runs; active
            # sessions are structurally excluded inside delete_empty.
            return store.sessions.delete_empty(media_root=media_root), ()
        # Forwarded login: resolve the empty candidates first, authorize each, then delete only
        # the allowed ones through the guarded bulk path, so a foreign empty row is never
        # removed (deleting it first and filtering after would already have lost it).
        candidates = _empty_candidate_ids(store)
        allowed, foreign_skipped = _rows_actor_may_delete(store, actor, candidates)
        if not allowed:
            return (), tuple(foreign_skipped)
        outcome = store.sessions.delete_many(
            allowed,
            skip_guarded=True,
            include_compression_chain=False,
            media_root=media_root,
        )
        return tuple(outcome.removed_ids), tuple(
            [*foreign_skipped, *outcome.skipped_guarded]
        )

    def run_bulk_delete() -> tuple["DeleteManyOutcome | None", tuple[str, ...]]:
        if is_forwarded_login:
            # Authorize BEFORE deleting: a foreign row is reported as skipped, never removed. The
            # store's own guarded pass below still skips live-lease rows, so both namespaces come
            # back through skipped_active.
            ids_to_delete, foreign_skipped = _rows_actor_may_delete(store, actor, ids)
        else:
            ids_to_delete, foreign_skipped = list(ids), []

        if not ids_to_delete:
            return None, tuple(foreign_skipped)

        # Mirror the offline bulk route: guarded rows are skipped (not a hard error), compression
        # chains are removed whole, and the toast counts requested rows only.
        outcome = store.sessions.delete_many(
            ids_to_delete,
            skip_guarded=True,
            include_compression_chain=True,
            media_root=media_root,
        )
        return outcome, tuple([*foreign_skipped, *outcome.skipped_guarded])

    if empty_sweep:
        removed, skipped = await asyncio.to_thread(run_empty_sweep)
        after_sessions_removed(
            store, store.scope.profile_dir, removed, reason="empty_sweep"
        )
        retire_live_sessions(authority, list(removed))
        return {"deleted": len(removed), "skipped_active": list(skipped)}

    outcome, skipped = await asyncio.to_thread(run_bulk_delete)
    if outcome is None:
        return {"deleted": 0, "skipped_active": list(skipped)}

    after_sessions_removed(
        store,
        store.scope.profile_dir,
        outcome.removed_ids,
        reason="delete",
        primary=ids[0] if len(ids) == 1 else None,
    )
    retire_live_sessions(authority, list(outcome.removed_ids))
    return {"deleted": outcome.deleted, "skipped_active": list(skipped)}
