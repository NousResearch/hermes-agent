"""Rekey a renamed profile's durable identity (#111926).

``rename_profile`` moves ``profiles/<old>/`` to ``profiles/<new>/`` so row DATA travels with the
directory, but several identities are derived from the old name or absolute path and do not:
``agent:<old>:*`` session-key namespaces (routing index + the profile's own ``sessions`` rows),
``sessions.profile_name``, ``gateway_heartbeats.profile``, ``delivery_obligations``, and checkpoint
project refs/metadata/ledgers keyed by the absolute workdir path. Left alone, routing resolves to a
profile that no longer exists and profile-local `/rollback` history disappears under the new path.

Ownership decides who rewrites session/routing state: a live multiplexer holds the routing index in
memory (``SessionStore._entries``) and writes it back periodically, so a CLI-side DB rewrite would
be clobbered on its next save — the CLI delegates to the ``migrate-profile-identity`` control verb.
Checkpoint project state has no corresponding live in-memory owner and is rekeyed locally after the
directory move. With no live multiplexer the durable session rewrite is safe here too.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path


def migrate_profile_identity(old_name: str, new_name: str) -> bool:
    """Retry the durable identity migration of a rename that already completed.

    ``rename_profile`` runs the migration itself; this is the standalone retry behind
    ``hermes profile migrate-identity <old> <new>`` for when that attempt failed. The rename
    cannot simply be repeated — ``profiles/<old>`` is gone — and the identity to migrate is read
    from rows/metadata that still name the old profile or old absolute workdir, so only the new
    profile has to exist here.

    A live multiplexer holds the routing index in memory and therefore stays the owner of that
    portion of the migration (the CLI delegates to its control verb); checkpoint identity and,
    with no live multiplexer, session identity are durable-only and safe to rewrite here.
    Idempotent: re-running a completed migration succeeds with nothing left to rekey. Returns True
    when all applicable identity was migrated, False when any durable migration failed or a live
    gateway would not accept the routing migration.
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


def _migrate_checkpoint_identity(old_canon: str, new_canon: str) -> bool:
    """Rekey checkpoint projects whose absolute workdirs moved with the profile directory."""
    from hermes_cli.profiles import get_profile_dir
    from tools.checkpoint_profile_migration import migrate_profile_checkpoint_projects

    old_dir = get_profile_dir(old_canon)
    new_dir = get_profile_dir(new_canon)
    result = migrate_profile_checkpoint_projects(old_dir, new_dir)
    if result["migrated"]:
        print(f"✓ Checkpoint identity updated: {result['migrated']} project(s)")
    if not result["errors"]:
        return True
    print(
        "⚠ Profile was renamed, but checkpoint identity migration failed for "
        f"{result['errors']} project(s). Retry with:\n"
        f"    hermes profile migrate-identity {old_canon} {new_canon}",
        file=sys.stderr,
    )
    return False


def _migrate_profile_identity(old_canon: str, new_canon: str, live_mux: bool) -> bool:
    """Rekey renamed-profile identity without racing a live gateway's in-memory routing index.

    Returns True when every applicable identity was migrated. Session/routing identity is handled
    by the gateway's control verb when it owns the live store, otherwise by this process; checkpoint
    project identity is always durable-only. Never fatal to the rename, which has already happened
    by this point.
    """
    checkpoint_migrated = _migrate_checkpoint_identity(old_canon, new_canon)

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
                return checkpoint_migrated
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
    migrated = checkpoint_migrated
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
