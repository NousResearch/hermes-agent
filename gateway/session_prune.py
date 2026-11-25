"""``session.prune``: the owner runs `hermes sessions prune` for its own store.

The CLI computes the filters and shows the preview from a read-only view; the delete itself is
this owner's write, through the same ``prune_sessions`` / ``retire_prunable`` path housekeeping
uses (busy sessions and the logical owner of a retained reset child are skipped). A CLI writing
the store under a live gateway would be a second writer, so it never does.

Two caller kinds, mirroring the ``session.bulk_delete`` authorization boundary:

* a **native local operator** (the private bootstrap ticket; its subject is not an ``auth:v1:``
  login) keeps store-wide reach — housekeeping, cron and the CLI prune every principal's rows;
* a **forwarded dashboard login** (an ``auth:v1:[provider, issuer, user]`` subject, forwarded by
  the dashboard's prune route while a gateway owns the store) may only prune rows its single-row
  delete could remove. Each candidate is checked with the same ``authorize_history`` rule a single
  delete runs (without claiming); a row another login owns is skipped and survives, so a dashboard
  login can never prune another login's sessions (#138915 review F1).
"""

import asyncio
from pathlib import Path

from persistence.core.errors import AuthorityRefusal

_SCALARS = (type(None), bool, int, float, str)


def _filters(raw):
    from persistence.housekeeping.maintenance import PRUNE_FILTER_NAMES

    if (
        not isinstance(raw, dict)
        or set(raw) - (PRUNE_FILTER_NAMES | {"older_than_days"})
        or not all(isinstance(value, _SCALARS) for value in raw.values())
    ):
        raise AuthorityRefusal("invalid_params")
    return dict(raw)


async def prune(connection, ref, params):
    authority, actor = connection.authority, connection.actor
    # Store-wide deletion across every principal's rows: the local operator only (a ticket from
    # this host's private control socket), never a remote or browser grant.
    if not connection.native_owner or "session:operator" not in actor.capabilities:
        raise AuthorityRefusal("permission_denied")
    if actor.profile_id != authority.profile_id:
        raise AuthorityRefusal("profile_mismatch")
    never_active = params.get("never_active_days")
    if ("filters" in params) == (never_active is not None) or (
        never_active is not None
        and (
            isinstance(never_active, bool)
            or not isinstance(never_active, (int, float))
            or never_active < 0
        )
    ):
        raise AuthorityRefusal("invalid_params")
    filters = _filters(params["filters"]) if "filters" in params else None
    authority._require_admission_open()
    from gateway.session_runtime_workers import track_mutation

    # A forwarded dashboard login carries an auth:v1: subject through the native ticket; the native
    # local operator's subject is not that form and keeps full store-wide reach.
    from gateway.session_bulk_delete import _forwarded_login

    forwarded_login = _forwarded_login(actor)
    task = track_mutation(
        authority, _apply(authority, actor, filters, never_active, forwarded_login)
    )
    return await asyncio.shield(task)


async def _apply(authority, actor, filters, never_active, forwarded_login):
    from hermes_constants import get_hermes_home
    from gateway.session_ingress_media import native_inputs_root
    from gateway.session_mutations import retire_live_sessions
    from gateway.session_persistence import authority_persistence
    from persistence.housekeeping.maintenance import PruneCriteria
    from session_retention import after_sessions_removed

    store = authority_persistence(authority)
    # Same transcript dir the CLI's local prune unlinks under (dispatch runs in the owner's scope).
    sessions_dir = Path(get_hermes_home()) / "sessions"
    media_root = str(native_inputs_root())

    def prune_never_active():
        # The never-active selector only matches keyed (routed) gateway rows, whose ownership is the
        # route/lane principal, not an import history binding. This verb is only reachable from the
        # native CLI (`hermes sessions prune --never-active`), whose ticket carries no auth:v1:
        # subject; the dashboard prune route forwards the filters path only. A forwarded login can
        # therefore never reach here, and a store-wide never-active sweep under one would be an
        # unauthorized cross-login delete — refuse it explicitly rather than silently pruning.
        if forwarded_login is not None:
            raise AuthorityRefusal("permission_denied")
        outcome = store.routing.prune_never_active(
            older_than_days=float(never_active), media_root=media_root
        )
        if (
            outcome.deleted_ids or outcome.sessions_skipped
        ):  # the delete ran: its post-commit work follows
            single = len(outcome.deleted_ids) == 1 and not outcome.sessions_skipped
            after_sessions_removed(
                store,
                store.scope.profile_dir,
                outcome.removed_ids,
                reason="delete",
                primary=outcome.deleted_ids[0] if single else None,
                sessions_dir=sessions_dir,
            )
            store.routing.sweep_routing_entries_for_sessions(outcome.deleted_ids)
        return outcome

    def prune_sessions():
        bounds = {
            key: filters[key] for key in ("older_than_days", "source") if key in filters
        }
        criteria = PruneCriteria(
            **{key: value for key, value in filters.items() if key not in bounds}
        )
        if forwarded_login is None:
            outcome = store.maintenance.prune_sessions(
                **bounds,
                criteria=criteria,
                exclude_active_write_guards=True,
                media_root=media_root,
            )
        else:
            # Forwarded login: take the same whole-lineage selection the offline prune would run,
            # but admit only the rows this login's single-row delete could remove. A foreign row is
            # skipped (retained), never deleted.
            from gateway.session_bulk_delete import _rows_actor_may_delete

            candidates = [
                str(c.session_id)
                for c in store.maintenance.list_prune_candidates(
                    **bounds, criteria=criteria, whole_lineages=True
                )
            ]
            allowed, _foreign_skipped = _rows_actor_may_delete(store, actor, candidates)
            outcome = _prune_via_delete_many(store, allowed, media_root=media_root)
        after_sessions_removed(
            store,
            store.scope.profile_dir,
            outcome.deleted_ids,
            reason="prune",
            sessions_dir=sessions_dir,
        )
        return outcome

    if filters is None:
        outcome = await asyncio.to_thread(prune_never_active)
        result = {
            "deleted": outcome.sessions_deleted,
            "routing_deleted": outcome.routing_entries_deleted,
            "skipped": outcome.sessions_skipped,
        }
    else:
        outcome = await asyncio.to_thread(prune_sessions)
        result = {"deleted": outcome.count}
    retire_live_sessions(authority, list(outcome.deleted_ids))
    return result


def _prune_via_delete_many(store, allowed_ids, *, media_root):
    """Run a forwarded login's admissible prune selection through the guarded bulk-delete verb.

    The offline ``prune_sessions`` writes whole lineages in one transaction; for a forwarded login
    we have already reduced the selection to the rows the login owns, so the repository's own
    guarded bulk delete (which skips live-lease/compression rows and reports them) is the safe
    write. Foreign rows never reach it. The returned object mirrors the ``PruneResult`` shape the
    native path and the post-commit work read (``count`` / ``deleted_ids``).
    """
    from persistence.housekeeping.maintenance import PruneResult

    if not allowed_ids:
        return PruneResult(count=0, deleted_ids=())
    outcome = store.sessions.delete_many(
        allowed_ids,
        skip_guarded=True,
        include_compression_chain=True,
        media_root=media_root,
    )
    return PruneResult(
        count=outcome.deleted,
        deleted_ids=tuple(outcome.removed_ids),
    )
