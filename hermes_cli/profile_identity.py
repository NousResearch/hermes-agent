"""Rekey a renamed profile's session/routing identity (#111926), and purge a deleted profile's.

``rename_profile`` moves ``profiles/<old>/`` to ``profiles/<new>/`` so row DATA travels with the
directory, but the profile name is also baked into keys and values the move never touches:
``agent:<old>:*`` session-key namespaces (routing index + the profile's own ``sessions`` rows),
``sessions.profile_name``, ``gateway_heartbeats.profile`` and ``delivery_obligations``. Left alone,
every inbound event on a chat keyed to the old name logs ``Profile 'old' does not exist`` and
falls back to the global home, and renamed sessions drop out of the Desktop sidebar.

The same identity has to be *purged* when a profile is deleted — the mirror image of the rekey,
and ``delete_profile`` calls :func:`purge_profile_identity` for it.

Ownership decides who rewrites: a live multiplexer holds the routing index in memory
(``SessionStore._entries``) and writes it back periodically, so a CLI-side DB rewrite would be
clobbered on its next save — the CLI delegates the rename to the ``migrate-profile-identity``
control verb and the delete to the delete-only ``purge-profile-identity`` verb. Neither runs inside
``_unserve_profile()``: that hook fires for every name leaving the served set, and a rename's old
name leaves it exactly like a deleted one, so identity must survive it for the rekey that follows.
With no live multiplexer nothing else holds the store and the durable rewrite is safe here.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path


def migrate_profile_identity(old_name: str, new_name: str) -> bool:
    """Retry the session/routing identity migration of a rename that already completed.

    ``rename_profile`` runs the migration itself; this is the standalone retry behind
    ``hermes profile migrate-identity <old> <new>`` for when that attempt failed. The rename
    cannot simply be repeated — ``profiles/<old>`` is gone — and the identity to migrate is read
    from the DB rows that still name *old*, so only the new profile has to exist here.

    A live multiplexer holds the routing index in memory and therefore stays the owner of the
    migration (the CLI delegates to its control verb); with no live multiplexer the durable
    rewrite is safe because nothing else holds the store. Idempotent: re-running a completed
    migration succeeds with nothing left to rekey. Returns True when the identity was migrated,
    False when a live gateway would not do it — the caller reports that as a failure.
    """
    from hermes_cli.profiles import _canon_valid, _live_default_multiplexer, _unknown_profile_error, get_profile_dir
    old_canon = _canon_valid(old_name)
    new_canon = _canon_valid(new_name)
    if "default" in (old_canon, new_canon):
        raise ValueError("Identity migration applies to named profiles only.")
    if not get_profile_dir(new_canon).is_dir():
        raise _unknown_profile_error(new_canon)
    return _migrate_profile_identity(old_canon, new_canon, _live_default_multiplexer())


def purge_profile_identity(profile: str) -> bool:
    """Purge a deleted profile's session/routing identity from the durable stores (#111926).

    The mirror of :func:`migrate_profile_identity`: a rename must rekey a profile's identity, a
    delete must purge it. ``agent:<name>:*`` routing keys, ``gateway_heartbeats.profile`` and
    ``delivery_obligations`` rows are bookkeeping for a profile that no longer exists; left behind,
    every inbound event on a chat keyed to the deleted name enters the routing index, resolves a
    profile whose directory is gone, and logs ``Profile '<name>' does not exist`` on each event.

    A live multiplexer holds that index in memory and writes it back periodically, so it owns the
    purge exactly as it owns the rename migration: the delete path asks it through the
    ``purge-profile-identity`` control verb and treats the answer as the settlement — a gateway that
    times out, predates the verb, or fails the purge is reported with a retry command instead of
    being silently assumed. With no live multiplexer nothing else holds the store and the durable
    delete happens here. Session history rows are not part of this purge — it settles identity only.
    Refuses a name that is a live profile again: the purge keys off the name alone, so a same-name
    profile created after the delete (the very case a failed settlement leaves behind) would
    otherwise have the new incarnation's routing/heartbeat identity deleted from under it. The
    delete path tombstones the directory before calling this, so the guard never blocks it.
    Idempotent: purging an already-purged profile succeeds. Returns False only when the identity was
    not settled — by this process or by the live gateway.
    """
    from hermes_cli.profiles import _canon_valid, _live_default_multiplexer, profile_exists
    canon = _canon_valid(profile)
    if canon == "default":
        raise ValueError("Identity purge applies to named profiles only.")
    if profile_exists(canon):
        raise ValueError(
            f"Profile '{canon}' exists; purge-identity only settles the identity of a delete that "
            "has already completed.")
    return _purge_profile_identity(canon, _live_default_multiplexer())


def _purge_profile_identity(canon: str, live_mux: bool) -> bool:
    """Purge deleted-profile identity without racing a live gateway's in-memory routing index.

    Returns True when the identity was purged — by the gateway's control verb, or by this process's
    durable delete when no gateway holds the store — and False when a live gateway did not accept
    it. Never fatal to the delete, which has already happened by this point.
    """
    if live_mux:
        from hermes_constants import get_default_hermes_root
        root = get_default_hermes_root()
        try:
            from gateway.control_socket import purge_gateway_profile_identity
            answer = purge_gateway_profile_identity(root, canon)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
        else:
            if isinstance(answer, dict) and answer.get("ok") is True:
                return True
            reason = _control_answer_failure(answer)
            if answer is None and _gateway_accepts_profile_identity_verb(root):
                reason += (" — the gateway is running but does not implement "
                           "'purge-profile-identity' (an older process than this CLI)")
        print(
            "⚠ Profile was deleted, but the live gateway could not purge its session identity"
            f" ({reason}). Restart the gateway, then run:\n"
            f"    hermes profile purge-identity {canon}",
            file=sys.stderr)
        return False

    from hermes_cli.profiles import get_profile_dir
    from hermes_constants import get_default_hermes_root
    from hermes_state_registry import acquire, release_or_close
    root = get_default_hermes_root()
    purged = True
    for db_path in (root / "state.db", get_profile_dir(canon) / "state.db"):
        if not db_path.exists():
            continue
        db = None
        try:
            db = acquire(db_path)
            db.purge_profile_state(canon)
        except Exception as exc:
            purged = False
            print(
                f"⚠ Profile was deleted, but identity purge failed for {db_path}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr)
        finally:
            if db is not None:
                with contextlib.suppress(Exception):
                    release_or_close(db)
    return purged


def _control_answer_failure(answer) -> str:
    """Why a control-socket answer is not a success. Keeps the raw answer when the payload carries
    no reason field, so a malformed or old-gateway response stays diagnosable instead of
    collapsing into a generic warning."""
    if isinstance(answer, dict):
        failure = answer.get("error") or answer.get("message") or answer.get("detail")
        return str(failure) if failure else repr(answer)
    if answer is not None:
        return repr(answer)
    return "no response from gateway control socket"


def _gateway_accepts_profile_identity_verb(root: Path) -> bool:
    """True when the gateway at *root* answers a verb it has always had. Distinguishes a failed
    migration verb caused by an older gateway process from one caused by no gateway at all."""
    try:
        from gateway.control_socket import identify_gateway
        return identify_gateway(root) is not None
    except Exception:
        return False


def _migrate_profile_identity(old_canon: str, new_canon: str, live_mux: bool) -> bool:
    """Rekey renamed-profile identity without racing a live gateway's in-memory routing index.

    Returns True when the identity was migrated — by the gateway's control verb, or by this
    process's durable rewrite when no gateway holds the store — and False when a live gateway did
    not accept it. Never fatal to the rename, which has already happened by this point.
    """
    if live_mux:
        from hermes_constants import get_default_hermes_root
        root = get_default_hermes_root()
        try:
            from gateway.control_socket import migrate_gateway_profile_identity
            answer = migrate_gateway_profile_identity(root, old_canon, new_canon)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
        else:
            if isinstance(answer, dict) and answer.get("ok") is True:
                return True
            reason = _control_answer_failure(answer)
            if answer is None and _gateway_accepts_profile_identity_verb(root):
                reason += (" — the gateway is running but does not implement "
                           "'migrate-profile-identity' (an older process than this CLI)")
        print(
            "⚠ Profile was renamed, but the live gateway could not migrate session identity"
            f" ({reason}). Restart the gateway, then run:\n"
            f"    hermes profile migrate-identity {old_canon} {new_canon}",
            file=sys.stderr)
        return False

    from hermes_cli.profiles import get_profile_dir
    from hermes_state_registry import acquire, release_or_close
    from hermes_constants import get_default_hermes_root
    root = get_default_hermes_root()
    migrated = True
    for db_path in (root / "state.db", get_profile_dir(new_canon) / "state.db"):
        if not db_path.exists():
            continue
        db = None
        try:
            db = acquire(db_path)
            db.rekey_profile_state(old_canon, new_canon)
        except Exception as exc:
            migrated = False
            print(
                f"⚠ Profile was renamed, but identity migration failed for {db_path}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr)
        finally:
            if db is not None:
                with contextlib.suppress(Exception):
                    release_or_close(db)
    return migrated
