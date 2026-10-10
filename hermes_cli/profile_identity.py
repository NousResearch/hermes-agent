"""Rekey a renamed profile's durable identity (#111926), and purge a deleted profile's (#112727).

``rename_profile`` moves ``profiles/<old>/`` to ``profiles/<new>/`` so row DATA travels with the
directory, but several identities are derived from the old name or absolute path and do not:
``agent:<old>:*`` session-key namespaces (routing index + the profile's own ``sessions`` rows),
``sessions.profile_name``, ``gateway_heartbeats.profile``, ``delivery_obligations``, and checkpoint
project refs/metadata/ledgers keyed by the absolute workdir path. Left alone, routing resolves to a
profile that no longer exists and profile-local `/rollback` history disappears under the new path.

The same identity has to be *purged* when a profile is deleted — the mirror image of the rekey,
and ``delete_profile`` calls :func:`purge_profile_identity` for it.

Ownership decides who rewrites: a live multiplexer holds the routing index in memory
(``SessionStore._entries``) and writes it back periodically, so a CLI-side DB rewrite would be
clobbered on its next save — the CLI delegates the rename to the ``migrate-profile-identity``
control verb and the delete to the delete-only ``purge-profile-identity`` verb. Neither runs inside
``_unserve_profile()``: that hook fires for every name leaving the served set, and a rename's old
name leaves it exactly like a deleted one, so identity must survive it for the rekey that follows.
With no live multiplexer nothing else holds the store and the durable rewrite is safe here.
Checkpoint project state has no live in-memory owner and is rekeyed locally after the directory move.
"""
from __future__ import annotations

import contextlib
import json
import re
import sys
from pathlib import Path


# Keys whose whole value is a profile name. Only an exact match is rewritten.
_PROFILE_NAME_KEYS = frozenset({"profile", "transport_profile", "target_profile", "profile_name"})
# Characters that continue a profile name; every rule below refuses a match followed by one, so
# renaming ``bot`` never touches ``bot2`` / ``bot-dev``.
_NAME_TAIL = r"(?![A-Za-z0-9_-])"


def _profile_reference_rules(old_dir: Path, new_dir: Path):
    """Anchored (pattern, replacement) pairs for references embedded inside a string value:
    the absolute profile home (and paths under it), ``agent:<name>:`` session keys, and
    ``bot-chat:<name>`` delivery tokens (also inside comma-separated ``deliver`` lists)."""
    old_name, new_name = old_dir.name, new_dir.name
    # Replacements are callables so a Windows path's backslashes are never read as group escapes.
    return (
        (re.compile(re.escape(str(old_dir)) + _NAME_TAIL), lambda m: str(new_dir)),
        (re.compile(r"(?<![A-Za-z0-9_-])agent:" + re.escape(old_name) + ":"), lambda m: f"agent:{new_name}:"),
        (re.compile(r"(?<![A-Za-z0-9_-])(bot-chat:)" + re.escape(old_name) + _NAME_TAIL, re.IGNORECASE),
         lambda m: m.group(1) + new_name),
    )


def _rewrite_reference_text(value: str, rules) -> str:
    for pattern, replacement in rules:
        value = pattern.sub(replacement, value)
    return value


def _rewrite_json_references(node, old_name: str, new_name: str, rules):
    """Rewrite identity fields (exact profile-name values) and embedded references in string
    values; free text such as a cron ``command`` or a prompt is never searched for the bare name."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key in _PROFILE_NAME_KEYS and value == old_name:
                out[key] = new_name
            else:
                out[key] = _rewrite_json_references(value, old_name, new_name, rules)
        return out
    if isinstance(node, list):
        return [_rewrite_json_references(item, old_name, new_name, rules) for item in node]
    if isinstance(node, str):
        return _rewrite_reference_text(node, rules)
    return node


def _rewrite_yaml_references(text: str, old_name: str, new_name: str, rules) -> str:
    """Line-level YAML rewrite that keeps comments and layout: ``<identity key>: <old>`` values
    (optionally quoted, optionally a list item) plus the anchored embedded-reference rules."""
    keys = "|".join(sorted(_PROFILE_NAME_KEYS))
    identity = re.compile(
        rf"^(\s*(?:-\s+)?(?:{keys})\s*:\s*)([\"']?){re.escape(old_name)}\2(?=\s*(?:#.*)?$)",
        re.MULTILINE)
    text = identity.sub(lambda m: f"{m.group(1)}{m.group(2)}{new_name}{m.group(2)}", text)
    return _rewrite_reference_text(text, rules)


def _durable_reference_files(new_dir: Path) -> list:
    """Every file that may durably name the renamed profile: its own cron jobs, config and plugin
    JSON state; the shared root's config and cron jobs (default-profile jobs can deliver to
    ``bot-chat:<name>``); and sibling profiles' cron jobs, which can target it the same way."""
    profiles_root = new_dir.parent
    hermes_root = profiles_root.parent
    candidates = [
        new_dir / "cron" / "jobs.json",
        new_dir / "config.yaml",
        hermes_root / "config.yaml",
        hermes_root / "cron" / "jobs.json",
    ]
    if profiles_root.is_dir():
        candidates.extend(sorted(profiles_root.glob("*/cron/jobs.json")))
    plugins = new_dir / "plugins"
    if plugins.is_dir():
        candidates.extend(sorted(plugins.rglob("*.json")))
    return list(dict.fromkeys(candidates))


def migrate_profile_durable_references(old_dir: Path, new_dir: Path) -> bool:
    """Rekey durable profile names and absolute homes that the directory move did not carry.

    The directory rename carries files but not values embedded in cron targets, plugin state, or
    routes. Keep the inventory here rather than growing one-off rename hooks. Only identity
    fields and anchored references are rewritten (never the bare name in free text), JSON is
    rewritten structurally and YAML line by line, and writes are atomic, so retries are harmless.
    """
    from utils import atomic_write_text

    old_name, new_name = old_dir.name, new_dir.name
    rules = _profile_reference_rules(old_dir, new_dir)
    ok = True
    for path in _durable_reference_files(new_dir):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
            if path.suffix == ".json":
                # Malformed plugin state raises here and is retained untouched.
                data = json.loads(text)
                migrated_data = _rewrite_json_references(data, old_name, new_name, rules)
                if migrated_data == data:
                    continue
                migrated = json.dumps(migrated_data, indent=2, ensure_ascii=False) + "\n"
            else:
                migrated = _rewrite_yaml_references(text, old_name, new_name, rules)
                if migrated == text:
                    continue
            atomic_write_text(path, migrated)
        except Exception as exc:
            ok = False
            print(f"⚠ Profile reference migration failed for {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return ok


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


def _migrate_checkpoint_identity(old_canon: str, new_canon: str) -> bool:
    """Rekey checkpoint projects whose absolute workdirs moved with the profile directory."""
    from hermes_cli.profiles import get_profile_dir
    from tools.checkpoint_manager_profile_rename import migrate_profile_checkpoint_projects

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
    from hermes_cli.profiles import get_profile_dir
    references_migrated = migrate_profile_durable_references(
        get_profile_dir(old_canon), get_profile_dir(new_canon))
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
                return checkpoint_migrated and references_migrated
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

    from hermes_state_registry import acquire, release_or_close
    from hermes_constants import get_default_hermes_root
    root = get_default_hermes_root()
    migrated = checkpoint_migrated and references_migrated
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
