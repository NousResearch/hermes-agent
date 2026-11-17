"""The one-time import of a profile's local secrets into the secrets plane.

Every boot of a profile runs it under one exclusive per-profile import lock (:data:`LOCK`, beside
the profile's ``.env``), taken before anything is read and held until the import is done: a
concurrent boot of the same profile, in another process or thread, waits for it and then starts
from a fresh read (normally finding nothing left to import). Under the lock (:func:`prepare`):

1. the process home's ``.env`` is normalised by the dotenv loader's own sanitiser, as every load of
   it does (a satellite's never is); then ``.env``, ``auth.json`` and an unfinished staging file are
   each read once (:class:`Sources`);
2. every kept-local ``.env`` line must be literal (:func:`non_literal_kept_names`): one the loader
   would expand stops the boot (``SecretStoreUnavailable``, names only) before anything is staged,
   refreshed, written or sent;
3. every entry to be sent (the staging's, then the files' entries outside the bootstrap set, then
   both together) must meet the plane's name rules (:func:`invalid_names`, contract §2) and its
   limits (:mod:`.limits`: value and record size, entries per profile, request size): a violation
   stops the boot (``SecretStoreUnavailable``, names shown escaped, never a value) before anything
   is backed up, staged, refreshed, written or sent; for a leftover staging file the stop names
   that file;
4. the exact bytes read of ``.env`` and ``auth.json`` are appended to their append-only backups
   (:mod:`.pre_import`: ``.env.pre-import`` / ``auth.pre-import.json``, 0600) unless the last
   entry already holds them; and every later rewrite of either file by the import appends the
   exact bytes it replaces first. Nothing here rewrites or removes a backup. The plane stores what
   the local store serves, which for some files is not every byte the user wrote (a satellite
   serves only the first line of a quoted multiline value, or of one holding U+2028 / NEL); the
   backups keep every version replaced recoverable;
5. the entries outside the bootstrap set (``.env`` values exactly as the local store serves them
   for the captured bytes, ``hermes_secret_routing.local_served_env``; ``auth.json`` records from
   the captured bytes) are staged in :data:`STAGING` (0600, merged over an unfinished staging)
   before the credential resolves: a token refresh saves ``auth.json`` with its bootstrap records
   only.

Then, against the fetched view (the backend): a profile the plane has never stored
(``profileVersion == 0``) gets the staged entries in one ``PATCH /self`` with
``expectedProfileVersion: 0``; a profile it already has wins (:func:`finish`): what its profile
level already holds with the same value was imported by an earlier boot and is dropped, the rest
moves to ``.env.unimported`` / ``auth.unimported.json`` (0600) and ``hermes doctor`` warns. The
files are then rewritten to their bootstrap entries, each only while it still holds the bytes read
under the lock (users editing the files do not take the lock): a file changed since is left whole,
and the next boot sets aside what it holds. Each step is idempotent: a crash leaves the staging
file and the next boot redoes the step. Nothing here logs or raises with a value.
"""
from __future__ import annotations

import codecs
import contextlib
import io
import json
import logging
import os
import re
import stat
import threading
from pathlib import Path
from typing import Any, NamedTuple, NoReturn

from dotenv.parser import parse_stream
from agent.secret_scope import invalidate_env_file_cache
from plugins.secret_store.remote import limits, pre_import
from utils import atomic_json_write, atomic_replace, fsync_directory, mkstemp_beside

logger = logging.getLogger(__name__)

LOCK = ".secrets-import.lock"
STAGING = ".secrets-import.json"
UNIMPORTED_ENV = ".env.unimported"
UNIMPORTED_AUTH = "auth.unimported.json"
#: Each import input and its append-only backup (beside it, 0600, never rewritten or removed).
PRE_IMPORT = pre_import.BACKUPS
#: The plane's name rules (contract §2; gateway-gateway refuses others with ``secret_name_invalid``).
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
AUTH_ID = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,63}")
#: How long a boot waits for another boot's import of the same profile before failing closed.
LOCK_SECONDS = 600.0

_LOCK_HOLDERS: dict[str, threading.local] = {}
_LOCK_HOLDERS_GUARD = threading.Lock()


class Sources(NamedTuple):
    """One read of a home's import inputs, under the import lock: the ``.env``, ``auth.json`` and
    staging bytes (None: absent; an unreadable ``.env`` / ``auth.json`` reads as absent here, the
    loader and the auth store report it)."""
    env: bytes | None
    auth: bytes | None
    staging: bytes | None


class Prepared(NamedTuple):
    """What :func:`prepare` read and staged: the bytes and the entries to import."""
    sources: Sources
    env: dict[str, str]
    auth: dict[str, Any]


@contextlib.contextmanager
def import_lock(home: Path, profile: str):
    """Hold *home*'s exclusive import lock (cross-process, reentrant per thread). The outermost
    lock of the import: never taken while holding the ``.env`` / ``auth.json`` / config locks."""
    from hermes_cli import auth
    path = Path(home) / LOCK
    with _LOCK_HOLDERS_GUARD:
        holder = _LOCK_HOLDERS.setdefault(str(path), threading.local())
    message = (f"Remote secrets: another Hermes process is still importing profile {profile!r} into the secrets "
               "plane. Hermes does not start (there is no fallback to the local store); try again.")
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(auth._file_lock(path, holder, LOCK_SECONDS, message))
        except TimeoutError:
            _stop(message)
        yield


def _stop(message: str) -> NoReturn:
    """Fail the boot closed (``SecretStoreUnavailable``) with *message*: names only, never a value."""
    from hermes_secret_routing import SecretStoreUnavailable
    raise SecretStoreUnavailable(message) from None


def capture(home: Path) -> Sources:
    """*home*'s import inputs, each read once. An unreadable staging file raises."""
    def read(name, missing=OSError):
        try:
            return (Path(home) / name).read_bytes()
        except missing:
            return None
    return Sources(read(".env"), read("auth.json"), read(STAGING, FileNotFoundError))


def prepare(home: Path, provider, profile: str) -> Prepared:
    """Under the import lock: normalise the process home's ``.env`` (the loader's sanitiser), read
    the files once, refuse a non-literal kept line, stage the entries to import. Nothing is written
    before the check but what the loader's own sanitiser writes on every load of that home."""
    from agent.secret_scope import _is_process_home
    from hermes_cli.env_loader import _sanitize_env_file_if_needed
    env_path = Path(home) / ".env"
    if _is_process_home(Path(home)):
        _sanitize_env_file_if_needed(env_path)
    sources = capture(home)
    assignments = _assignments(sources.env)
    if names := non_literal_kept_names(provider, assignments):
        _stop(refusal_text(profile, names))
    staged_env, staged_auth = _staged(sources.staging)
    if names := invalid_names(staged_env, staged_auth):
        _stop(staging_refusal_text(profile, Path(home), [f"invalid name {n}" for n in names]))
    if found := limits.problems(staged_env, staged_auth, _shown):
        _stop(staging_refusal_text(profile, Path(home), found))
    env, records = _local_entries(home, provider, sources)
    from_files = (bool(env), bool(records))
    if names := invalid_names(env, records):
        _stop(invalid_text(profile, names))
    if found := limits.problems(env, records, _shown):
        _stop(limit_text(profile, found))
    env, records = {**staged_env, **env}, {**staged_auth, **records}
    if sources.staging is not None and (found := limits.problems(env, records, _shown)):
        _stop(staging_refusal_text(profile, Path(home), found))
    if env or records:
        _keep_originals(Path(home), sources, from_files)
        _write_private_json(Path(home) / STAGING, {"env": env, "auth": records})
    return Prepared(sources, env, records)


def invalid_names(env, records) -> list[str]:
    """The env names and auth record ids the plane would refuse as names (contract §2), shown
    escaped (:func:`_shown`); names only, never a value."""
    bad = [n for n in env if not (isinstance(n, str) and ENV_NAME.fullmatch(n))]
    bad += [r for r in records if not (isinstance(r, str) and AUTH_ID.fullmatch(r))]
    return sorted({_shown(n) for n in bad})


def _shown(name) -> str:
    """*name* for a message: ASCII-escaped and quoted (no control or bidi character reaches a
    terminal), cut at 80 characters."""
    text = ascii(str(name))
    return text if len(text) <= 80 else text[:77] + "..."


def invalid_text(profile: str, names: list[str]) -> str:
    return (f"Remote secrets: profile {profile!r} has .env / auth.json entries the secrets plane cannot store: "
            f"invalid name(s) {', '.join(names)} (env names: letters, digits and _, not starting with a digit; "
            "auth provider ids: lower-case letters, digits and _ . : -). Hermes does not start (nothing was "
            "sent or written; there is no fallback to the local store). Rename or remove those entries (a "
            ".env saved as UTF-16 must be saved as UTF-8) and restart Hermes.")


def limit_text(profile: str, found: list[str]) -> str:
    return (f"Remote secrets: profile {profile!r} has .env / auth.json entries the secrets plane cannot store: "
            f"{'; '.join(found)}. Hermes does not start (nothing was sent or written; there is no fallback to "
            "the local store). Shorten or remove those entries and restart Hermes.")


def staging_refusal_text(profile: str, home: Path, found: list[str]) -> str:
    return (f"Remote secrets: profile {profile!r} has an unfinished import ({home / STAGING}) holding "
            f"what the secrets plane cannot store: {', '.join(found)}. Hermes does not start (nothing was "
            "sent or written; there is no fallback to the local store). That file is JSON (its \"env\" and "
            "\"auth\" entries may be the only copy of a value or auth record): copy out what you need, check "
            f"that {' / '.join(str(home / n) for n in PRE_IMPORT.values())} (if present) and the .env hold "
            f"what you need, then delete {home / STAGING} and restart Hermes.")


def _keep_originals(home: Path, sources: Sources, from_files: tuple[bool, bool]) -> None:
    """Append the captured bytes of each import input that holds entries to import to its backup
    (:mod:`.pre_import`), unless its last entry already holds them."""
    for backup_name, raw, holds in zip(PRE_IMPORT.values(), (sources.env, sources.auth), from_files):
        if raw is not None and holds:
            pre_import.append(home / backup_name, raw, "staged", unless_last=True)


def pre_import_backups(home: Path) -> list[Path]:
    """The pre-import backups present in *home* (``hermes doctor`` names them)."""
    return pre_import.present(home)


def non_literal_kept_names(provider, assignments: list[tuple[str, str | None]]) -> list[str]:
    """The kept-local names (``stays_local``) assigned a value the dotenv loader would expand: one
    holding a ``${NAME}`` / ``${NAME:-default}`` reference (``dotenv.variables.parse_variables``,
    the loader's only expansion; a bare ``$NAME`` is literal to it). Names only, never a value."""
    from dotenv.variables import Variable, parse_variables
    from hermes_secret_routing import stays_local
    return sorted({name for name, value in assignments
                   if value is not None and stays_local(provider, name)
                   and any(isinstance(atom, Variable) for atom in parse_variables(value))})


def refusal_text(profile: str, names: list[str]) -> str:
    """The stop's words for *profile*'s non-literal kept-local *names*, also ``hermes doctor``'s."""
    return (f"Remote secrets: profile {profile!r} keeps the .env line(s) {', '.join(names)} local, and a kept "
            "line must be literal, but each references another variable (${...}). Hermes does not start "
            "(nothing was sent or written; there is no fallback to the local store). Write the value itself "
            "on those lines and restart Hermes.")


def _assignments(raw: bytes | None) -> list[tuple[str, str | None]]:
    from hermes_cli.env_loader import parse_dotenv_bytes
    return [] if raw is None else parse_dotenv_bytes(raw)


def _local_entries(home: Path, provider, sources: Sources
                   ) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """The local entries outside the bootstrap set: what the local store serves from the captured
    ``.env`` (``local_served_env``), stored literally from then on; the captured ``auth.json``'s records."""
    from hermes_secret_routing import auth_records, local_served_env, stays_local
    env = {} if sources.env is None else local_served_env(Path(home), sources.env)
    env = {name: value for name, value in env.items() if not stays_local(provider, name)}
    keep = provider.bootstrap_auth_records()
    records = {rid: r for rid, r in auth_records(_auth_store(sources.auth) or {}).items() if rid not in keep}
    return env, records


def _auth_store(raw: bytes | None) -> dict[str, Any] | None:
    """The captured ``auth.json`` as the auth store parses it; None when absent or unparseable (the
    auth store reports that one, and the cleanup leaves it alone)."""
    if raw is None:
        return None
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _write_private_json(path: Path, data: Any) -> None:
    atomic_json_write(path, data, mode=0o600, fsync_dir=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        logger.warning("Remote secrets: %s is unreadable; leaving it in place", path)
        raise
    return data if isinstance(data, dict) else {}


def _staged(raw: bytes | None) -> tuple[dict[str, str], dict[str, Any]]:
    """The entries of a staging file's *raw* bytes."""
    if raw is None:
        return {}, {}
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except ValueError:
        logger.warning("Remote secrets: %s is unreadable; leaving it in place", STAGING)
        raise
    data = data if isinstance(data, dict) else {}
    return dict(data.get("env") or {}), dict(data.get("auth") or {})


def unimported_paths(home: Path) -> tuple[Path, Path]:
    return Path(home) / UNIMPORTED_ENV, Path(home) / UNIMPORTED_AUTH


def finish(home: Path, profile: str, view, provider, prepared: Prepared, imported: bool) -> None:
    """Still under the import lock, after the plane answered: set aside what the plane did not
    take, rewrite the files to their bootstrap entries (only while they hold the bytes read under
    the lock), then drop the staging. Staging survives a failed cleanup for the next boot."""
    if prepared.env or prepared.auth:
        if not imported:
            _move_unimported(home, profile, view, prepared.env, prepared.auth)
        _keep_bootstrap_only(home, provider, prepared.sources, "cleanup" if imported else "plane-wins")
    _remove(Path(home) / STAGING)
    if imported:
        logger.info("Remote secrets: imported %d env name(s) and %d auth record(s) of profile %r",
                    len(prepared.env), len(prepared.auth), profile)


def _move_unimported(home: Path, profile: str, view, env: dict[str, str], records: dict[str, Any]) -> None:
    """The plane already holds this profile: move what it does not already hold to the
    ``*.unimported`` files."""
    from hermes_cli.config import _quote_env_value
    from hermes_secret_port import thaw_secret
    from hermes_secret_routing import auth_records, auth_store_from_records
    env_left = {k: v for k, v in env.items()
                if not (view.env_meta.get(k, ("",))[0] == "profile" and view.env.get(k) == v)}
    auth_left = {rid: r for rid, r in records.items()
                 if not (view.auth_meta.get(rid, ("",))[0] == "profile" and thaw_secret(view.auth.get(rid)) == r)}
    env_path, auth_path = unimported_paths(home)
    if env_left:
        try:
            earlier = {b.key: b.value for b in _dotenv_file(env_path.read_bytes())[2]
                       if b.key is not None and b.value is not None}
        except FileNotFoundError:
            earlier = {}
        merged = {**earlier, **env_left}
        _write_bytes(env_path, "".join(f"{k}={_quote_env_value(v)}\n" for k, v in merged.items()).encode("utf-8"))
        earlier = merged = None
    if auth_left:
        existing = auth_records(_read_json(auth_path))
        _write_private_json(auth_path, auth_store_from_records({**existing, **auth_left}))
        existing = None
    if env_left or auth_left:
        logger.warning("Remote secrets: the secrets plane already has profile %r; %d local env name(s) and %d "
                       "auth record(s) were moved to %s / %s, not imported", profile, len(env_left),
                       len(auth_left), env_path.name, auth_path.name)


def _dotenv_file(raw: bytes) -> tuple[bytes, str, list]:
    """A dotenv file's *raw* bytes as ``(BOM, codec, bindings)``: python-dotenv's bindings, each
    carrying its exact source text (``original.string``). Invalid UTF-8 decodes as latin-1, as the
    loader does."""
    bom = codecs.BOM_UTF8 if raw.startswith(codecs.BOM_UTF8) else b""
    body = raw[len(bom):]
    try:
        text, codec = body.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        text, codec = body.decode("latin-1"), "latin-1"
    return bom, codec, list(parse_stream(io.StringIO(text)))


def _write_bytes(path: Path, data: bytes, mode: Any = 0o600) -> None:
    """Atomically replace *path* with *data* (0600, or *mode*)."""
    fd, tmp = mkstemp_beside(path, suffix=".tmp", prefix=f"{path.name}_")
    try:
        with open(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        replaced = atomic_replace(tmp, path)
        fsync_directory(Path(replaced).parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _keep_bootstrap_only(home: Path, provider, sources: Sources, reason: str) -> None:
    """Rewrite *home*'s ``.env`` and ``auth.json`` to their bootstrap entries, each only while it
    still holds the bytes in *sources* (compared under its lock), appending those exact bytes to
    its backup first (*reason* in the entry): a file another writer changed since is left whole,
    and the next boot backs up and sets aside what it holds. In ``.env`` each other
    assignment is removed whole (a quoted value with all its lines); every other byte stays. No
    import, credential resolution or logging under the file locks (trap 25)."""
    from hermes_cli import auth
    from hermes_cli.config import _env_write_lock
    from hermes_secret_routing import auth_records, auth_store_from_records, stays_local
    changed = []
    env_path = Path(home) / ".env"
    if sources.env is not None:
        bom, codec, bindings = _dotenv_file(sources.env)
        kept = _kept_text(bindings, lambda name: stays_local(provider, name))
        if kept is not None:
            data = bom + kept.encode(codec)
            with _env_write_lock(env_path):
                if _holds(env_path, sources.env, changed):
                    pre_import.append(pre_import.backup_path(env_path), sources.env, reason)
                    _write_bytes(env_path, data, stat.S_IMODE(env_path.stat().st_mode))
        bindings = kept = data = None
    store = _auth_store(sources.auth)
    auth_path = Path(home) / "auth.json"
    keep = provider.bootstrap_auth_records()
    if store is not None and not set(auth_records(store)) <= keep:    # else nothing to remove: left as is
        local = auth_store_from_records({rid: r for rid, r in auth_records(store).items() if rid in keep})
        local.update((k, store[k]) for k in ("version", "updated_at") if k in store)
        data = json.dumps(local).encode("utf-8")
        with auth._auth_store_lock(target_path=auth_path):
            if _holds(auth_path, sources.auth, changed):
                pre_import.append(pre_import.backup_path(auth_path), sources.auth, reason)
                _write_bytes(auth_path, data)
        store = local = data = None
    invalidate_env_file_cache()
    for name in changed:
        logger.info("Remote secrets: %s changed while its entries were being imported; it is left as it is "
                    "and the next start sets aside what it still holds", name)


def _holds(path: Path, raw: bytes, changed: list) -> bool:
    """Whether *path* still holds *raw*; a changed (or removed) file's name goes to *changed*."""
    try:
        same = path.read_bytes() == raw
    except FileNotFoundError:
        same = False
    if not same:
        changed.append(path.name)
    return same


def _kept_text(bindings: list, keeps) -> str | None:
    """The ``.env`` text with only its kept assignments (every other byte as it was), or None when
    it holds no other assignment."""
    kept = []
    changed = False
    for binding in bindings:
        original = binding.original.string
        if binding.key is None or keeps(binding.key):
            kept.append(original)
        else:
            # The parser attaches leading whitespace to the next assignment, not to a separate
            # blank binding. Preserve it without retaining the assignment.
            kept.append(original[:len(original) - len(original.lstrip())])
            changed = True
    return "".join(kept) if changed else None


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
