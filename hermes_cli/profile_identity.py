"""Rekey a renamed profile's session/routing identity (#111926).

``rename_profile`` moves ``profiles/<old>/`` to ``profiles/<new>/`` so row DATA travels with the
directory, but the profile name is also baked into keys and values the move never touches:
``agent:<old>:*`` session-key namespaces (routing index + the profile's own ``sessions`` rows),
``sessions.profile_name``, ``gateway_heartbeats.profile`` and ``delivery_obligations``. Left alone,
every inbound event on a chat keyed to the old name logs ``Profile 'old' does not exist`` and
falls back to the global home, and renamed sessions drop out of the Desktop sidebar.

Ownership decides who rewrites: a live multiplexer holds the routing index in memory
(``SessionStore._entries``) and writes it back periodically, so a CLI-side DB rewrite would be
clobbered on its next save — the CLI delegates to the ``migrate-profile-identity`` control verb.
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
