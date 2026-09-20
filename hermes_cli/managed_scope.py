"""Managed scope — IT-pushed, user-immutable config & env layer.

DISTINCT from ``hermes_cli.config.is_managed()`` / ``HERMES_MANAGED`` (a coarse package-manager
write-lock that blocks all mutation); this layer injects specific immutable values. The two are
independent and may coexist. v1 enforcement is filesystem permissions only (see
``docs/design/managed-scope.md`` §7); ``get_managed_dir()`` is the single seam for adding
macOS / Windows native locations later.
"""
from __future__ import annotations

import copy
import logging
import os
import stat
import threading
from pathlib import Path
from typing import Dict, Optional

import yaml

# Stale-module bridge: this module binds ``utils.file_signature`` at import time, so a fresh
# import in a post-pull updater process (pre-handoff purge keeps root modules cached) dies
# unless the stale ``utils`` is dropped first. See hermes_cli.stale_modules.
from hermes_cli.stale_modules import drop_stale_root_modules

drop_stale_root_modules()

from utils import file_signature

logger = logging.getLogger(__name__)

# POSIX default. Other-platform locations belong ONLY inside get_managed_dir().
_DEFAULT_MANAGED_DIR = Path("/etc/hermes")

_CACHE_LOCK = threading.Lock()
# path_key -> (*file_signature, parsed)
_CONFIG_CACHE: Dict[str, tuple] = {}
_ENV_CACHE: Dict[str, tuple] = {}


def _under_pytest() -> bool:
    """True inside the test suite: ignore the system ``/etc/hermes`` so a real managed scope on a
    dev/CI box can't leak policy into the suite. An explicit ``HERMES_MANAGED_DIR`` still wins."""
    return "PYTEST_CURRENT_TEST" in os.environ


def get_managed_dir(*, fail_closed: bool = False) -> Optional[Path]:
    """Resolve the managed-scope directory, or None when no scope is present.

    Priority: ``$HERMES_MANAGED_DIR`` (IT-only bootstrap override; never
    persisted to any .env; honored only when non-empty AND the directory
    exists), then ``/etc/hermes`` when it exists. A missing directory normally
    resolves to None. Strict readers reject discovery errors and a missing or
    non-directory explicit override instead of silently dropping policy.
    """
    override = os.environ.get("HERMES_MANAGED_DIR", "").strip()
    if override:
        path = Path(override)
    elif _under_pytest():
        return None
    else:
        path = _DEFAULT_MANAGED_DIR
    if not fail_closed:
        return path if path.is_dir() else None
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        if override:
            raise
        return None
    if not stat.S_ISDIR(mode):
        raise NotADirectoryError("managed scope must be a directory")
    return path


def invalidate_managed_cache() -> None:
    """Drop cached managed config/env. For tests and post-edit reloads."""
    with _CACHE_LOCK:
        _CONFIG_CACHE.clear()
        _ENV_CACHE.clear()


def _cached_read(path: Path, cache: Dict[str, tuple], parse):
    """Shared stat-signature-keyed read; returns a deepcopy of the parsed value.

    ``None`` when the file is absent or fails to parse (fail-open). A parse failure is logged
    LOUDLY — the admin needs to know their policy isn't applied — but never raises, so a malformed
    managed file can't brick startup.
    """
    try:
        st = path.stat()
    except OSError:
        return None  # absent
    key = file_signature(st)
    path_key = str(path)
    with _CACHE_LOCK:
        hit = cache.get(path_key)
        if hit is not None and hit[:len(key)] == key:
            return copy.deepcopy(hit[len(key)])
    try:
        parsed = parse(path)
    except Exception as exc:  # noqa: BLE001 — fail-open, but LOUD
        logger.warning(
            "managed scope: failed to parse %s: %s — IGNORING this managed file. "
            "Admin policy from this file is NOT being applied. Fix and restart.",
            path, exc)
        return None
    with _CACHE_LOCK:
        cache[path_key] = (*key, copy.deepcopy(parsed))
    return parsed


def _load_managed_file(name: str, cache: Dict[str, tuple], parse) -> dict:
    managed_dir = get_managed_dir()
    if managed_dir is None:
        return {}
    parsed = _cached_read(managed_dir / name, cache, parse)
    return parsed if isinstance(parsed, dict) else {}


def load_managed_config(*, fail_closed: bool = False) -> dict:
    """Read managed config; strict policy readers never accept stale data.

    An absent config file or empty YAML is an empty optional layer. Strict reads
    propagate all other I/O and parse failures and reject non-mapping roots.
    Non-strict callers retain cached, fail-open behavior.
    """
    if not fail_closed:
        return _load_managed_file(
            "config.yaml",
            _CONFIG_CACHE,
            lambda path: yaml.safe_load(path.read_text(encoding="utf-8")) or {},
        )
    managed_dir = get_managed_dir(fail_closed=True)
    if managed_dir is None:
        return {}
    path = managed_dir / "config.yaml"
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # Only a genuinely absent leaf is optional, not a vanished directory or
        # a dangling link/read failure on a still-present configured file.
        if not stat.S_ISDIR(managed_dir.stat().st_mode):
            raise NotADirectoryError("managed scope must be a directory")
        try:
            path.lstat()
        except FileNotFoundError:
            return {}
        raise
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("managed config.yaml must contain a mapping")
    return parsed


def load_managed_env() -> Dict[str, str]:
    """Parsed managed .env (KEY=VALUE), or {} when absent (fail-open)."""
    return _load_managed_file(".env", _ENV_CACHE, _parse_managed_env)


def _parse_managed_env(path: Path) -> Dict[str, str]:
    from agent.secret_scope import load_env_file

    path.read_text(encoding="utf-8-sig")  # load_env_file swallows decode errors; an admin file must fail LOUD
    return load_env_file(path)


def apply_managed_overlay(
    config: dict, *, fail_closed: bool = False
) -> dict:
    """Overlay administrator-pinned values on an already-built config.

    ``${VAR}`` refs in managed config expand against the PROCESS env only, so
    a user cannot shadow a managed literal via a ref they control; a bare root
    ``model: x/y`` string is promoted to ``model.default``. Managed values
    deep-merge on top per leaf. Permissive callers fail open; strict callers
    bypass managed caches and propagate every policy error.
    """
    try:
        managed = load_managed_config(fail_closed=fail_closed)
        if not managed:
            return config
        # Imported lazily to avoid an import cycle (config imports managed_scope).
        from hermes_cli.config import (
            _deep_merge,
            _expand_env_vars,
            _normalize_root_model_keys,
        )

        managed_expanded = _normalize_root_model_keys(_expand_env_vars(managed))
        # _normalize_root_model_keys only promotes the string when root
        # provider/base_url keys exist to migrate; handle the bare case here.
        if isinstance(managed_expanded.get("model"), str):
            managed_expanded = dict(managed_expanded)
            managed_expanded["model"] = {"default": managed_expanded["model"]}
        return _deep_merge(config, managed_expanded)
    except Exception:  # noqa: BLE001 — permissive startup retains recovery
        if fail_closed:
            raise
        logger.warning("managed scope: failed to apply config overlay", exc_info=True)
        return config


def _flatten_keys(d: dict, prefix: str = "") -> set:
    keys: set = set()
    for k, v in d.items():
        dotted = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and v:
            keys |= _flatten_keys(v, dotted)
        else:
            keys.add(dotted)
    return keys


def managed_config_keys() -> set:
    """Dotted leaf keys pinned by the managed config (e.g. {'model.default'})."""
    return _flatten_keys(load_managed_config())


def is_key_managed(dotted_key: str) -> bool:
    """True if the exact dotted config key is pinned by the managed layer."""
    return dotted_key in managed_config_keys()


def is_env_managed(name: str) -> bool:
    """True if the env var name is pinned by the managed .env layer."""
    return name in load_managed_env()
