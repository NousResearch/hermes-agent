"""The one-time import of a profile's local secrets into the secrets plane.

On boot, the local ``.env`` and ``auth.json`` entries outside the bootstrap set are first staged in
``<home>/.secrets-import.json`` (0600), before anything can rewrite the files (a token refresh saves
``auth.json`` with its bootstrap records only). Then, against the fetched view:

- the plane has never stored this profile (``profileVersion == 0``): the staged entries are sent in
  one ``PATCH /self`` with ``expectedProfileVersion: 0``; on success the files keep their bootstrap
  entries only;
- the plane already has the profile (the PATCH answers 409, or ``profileVersion`` is not 0): the
  plane wins. Staged entries the plane's profile level already holds with the same value were
  imported by an earlier boot that stopped before cleaning up, and are dropped; every other one
  moves to ``.env.unimported`` / ``auth.unimported.json`` (0600, merged with what is there), the
  files keep their bootstrap entries, and ``hermes doctor`` warns.

Each step is idempotent: a crash anywhere leaves the staging file, and the next boot redoes the
step it stopped in. Nothing here logs or raises with a value.
"""
from __future__ import annotations

import codecs
import io
import json
import logging
import os
import stat

from dotenv.parser import parse_stream
from agent.secret_scope import _is_global_env, invalidate_env_file_cache
from utils import atomic_json_write, atomic_replace, fsync_directory, mkstemp_beside
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STAGING = ".secrets-import.json"
UNIMPORTED_ENV = ".env.unimported"
UNIMPORTED_AUTH = "auth.unimported.json"


def unimported_paths(home: Path) -> tuple[Path, Path]:
    return Path(home) / UNIMPORTED_ENV, Path(home) / UNIMPORTED_AUTH


def _dotenv_file(path: Path) -> tuple[bytes, str, list]:
    """*path* as the launch load reads it: ``(BOM, codec, bindings)``, the bindings from
    python-dotenv's parser (``hermes_cli.env_loader._parse_dotenv_file``'s), so a quoted value
    spans its lines and every binding carries its exact source text (``original.string``).
    Invalid UTF-8 decodes as latin-1, as there. Raises ``OSError`` (absent, unreadable)."""
    raw = path.read_bytes()
    bom = codecs.BOM_UTF8 if raw.startswith(codecs.BOM_UTF8) else b""
    body = raw[len(bom):]
    try:
        text, codec = body.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        text, codec = body.decode("latin-1"), "latin-1"
    return bom, codec, list(parse_stream(io.StringIO(text)))


def _dotenv_values(path: Path) -> dict[str, str]:
    """Each name's value as the launch load leaves it for *path*, stored literally from then on:
    the loader's own steps (file sanitizing, parser, resolver with order, duplicates and every
    reference, credential sanitizing). References see the launch environment for the process
    home; for any other home only process-global names, never a launch credential."""
    from agent.secret_scope import _is_process_home
    from hermes_cli.env_loader import (
        _DOTENV_LOCK, _dotenv_lookup_env, _parse_local_dotenv_file, _resolve_dotenv_assignments,
        _sanitize_env_file_if_needed, _sanitize_loaded_credentials,
    )

    _sanitize_env_file_if_needed(path)
    try:
        assignments = _parse_local_dotenv_file(path)
    except OSError:
        return {}
    with _DOTENV_LOCK:
        lookup = _dotenv_lookup_env(None)
    if not _is_process_home(path.parent):
        lookup = {k: v for k, v in lookup.items() if _is_global_env(k)}
    resolved = _resolve_dotenv_assignments(assignments, lookup, override=True)
    values = {name: value for name, value in resolved.items() if value is not None}
    _sanitize_loaded_credentials(values)
    return values


def _local_entries(home: Path, provider) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """The local files' entries outside the bootstrap set, read from the files themselves."""
    from hermes_cli import auth
    from hermes_secret_routing import auth_records, stays_local
    env = {k: v for k, v in _dotenv_values(Path(home) / ".env").items() if not stays_local(provider, k)}
    raw = auth._load_auth_store.__wrapped__(Path(home) / "auth.json")  # the file, not the store's view
    records = {rid: r for rid, r in auth_records(raw).items() if rid not in provider.bootstrap_auth_records()}
    return env, records


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


def stage(home: Path, provider) -> None:
    """Stage the local entries to import (merged over an earlier, unfinished staging)."""
    env, records = _local_entries(home, provider)
    if not env and not records:
        return
    path = Path(home) / STAGING
    staged = _read_json(path)
    merged_env = {**(staged.get("env") or {}), **env}
    merged_auth = {**(staged.get("auth") or {}), **records}
    _write_private_json(path, {"env": merged_env, "auth": merged_auth})
    merged_env = merged_auth = env = records = staged = None


def prepared_entries(home: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """Read staging before the fetch lock: malformed-file diagnostics may read config."""
    staged = _read_json(Path(home) / STAGING)
    return dict(staged.get("env") or {}), dict(staged.get("auth") or {})


def finish(home: Path, profile: str, view, provider, entries, imported: bool) -> None:
    """Publish local cleanup and diagnostics only after releasing the fetch lock.

    Staging survives a failed cleanup, so a later boot can finish it against the plane's view.
    """
    env, records = entries
    if env or records:
        if not imported:
            _move_unimported(home, profile, view, env, records)
        _keep_bootstrap_only(home, provider)
    _remove(Path(home) / STAGING)
    if imported:
        logger.info("Remote secrets: imported %d env name(s) and %d auth record(s) of profile %r",
                    len(env), len(records), profile)


def _move_unimported(home: Path, profile: str, view, env: dict[str, str], records: dict[str, Any]) -> None:
    """The plane already holds this profile: move what it does not already hold to the
    ``*.unimported`` files."""
    env_left = {k: v for k, v in env.items()
                if not (view.env_meta.get(k, ("",))[0] == "profile" and view.env.get(k) == v)}
    from hermes_secret_port import thaw_secret
    auth_left = {rid: r for rid, r in records.items()
                 if not (view.auth_meta.get(rid, ("",))[0] == "profile" and thaw_secret(view.auth.get(rid)) == r)}
    env_path, auth_path = unimported_paths(home)
    if env_left:
        from hermes_cli.config import _quote_env_value
        try:
            earlier = {b.key: b.value for b in _dotenv_file(env_path)[2] if b.key is not None and b.value is not None}
        except FileNotFoundError:
            earlier = {}
        merged = {**earlier, **env_left}
        text = "".join(f"{k}={_quote_env_value(v)}\n" for k, v in merged.items())
        _write_private_text(env_path, text)
        earlier = None
        merged = text = None
    if auth_left:
        from hermes_secret_routing import auth_records, auth_store_from_records
        existing = auth_records(_read_json(auth_path))
        _write_private_json(auth_path, auth_store_from_records({**existing, **auth_left}))
        existing = None
    if env_left or auth_left:
        logger.warning("Remote secrets: the secrets plane already has profile %r; %d local env name(s) and %d "
                       "auth record(s) were moved to %s / %s, not imported", profile, len(env_left),
                       len(auth_left), env_path.name, auth_path.name)


def _write_private_text(path: Path, text: str) -> None:
    _write_bytes(path, text.encode("utf-8"))


def _write_bytes(path: Path, data: bytes, mode: Any = None) -> None:
    """Atomically replace *path* with *data* (0600, or *mode* when given)."""
    fd, tmp = mkstemp_beside(path, suffix=".tmp", prefix=f"{path.name}_")
    try:
        with open(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        replaced = atomic_replace(tmp, path)
        fsync_directory(Path(replaced).parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _keep_bootstrap_only(home: Path, provider) -> None:
    """Rewrite *home*'s ``.env`` and ``auth.json`` to their bootstrap entries. In ``.env`` each
    other assignment is removed whole (a quoted value with all its lines); comments, blank lines,
    lines the parser cannot read and the bootstrap assignments stay byte for byte."""
    from hermes_cli import auth
    from hermes_cli.config import _env_write_lock
    from hermes_secret_routing import auth_records, auth_store_from_records, stays_local
    def keeps(name):
        return stays_local(provider, name)
    env_path = Path(home) / ".env"
    if env_path.exists():
        with _env_write_lock(env_path):
            bom, codec, bindings = _dotenv_file(env_path)
            kept = []
            removed = False
            for binding in bindings:
                original = binding.original.string
                if binding.key is None or keeps(binding.key):
                    kept.append(original)
                else:
                    # The parser attaches leading whitespace to the next assignment, not to a
                    # separate blank binding. Preserve it without retaining the assignment.
                    kept.append(original[:len(original) - len(original.lstrip())])
                    removed = True
            if removed:
                _write_bytes(env_path, bom + "".join(kept).encode(codec), stat.S_IMODE(env_path.stat().st_mode))
            bindings = kept = None
    auth_path = Path(home) / "auth.json"
    if auth_path.exists():
        with auth._auth_store_lock(target_path=auth_path):
            raw = json.loads(auth_path.read_text(encoding="utf-8-sig"))
            keep = provider.bootstrap_auth_records()
            local = auth_store_from_records({rid: r for rid, r in auth_records(raw).items() if rid in keep})
            for k in ("version", "updated_at"):
                if k in raw:
                    local[k] = raw[k]
            _write_bytes(auth_path, json.dumps(local).encode("utf-8"), 0o600)
            raw = local = None
    invalidate_env_file_cache()


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
