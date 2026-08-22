"""Managed scope — IT-pushed, user-immutable config & env layer.

A system-level directory (default ``/etc/hermes``, root-owned and not
user-writable) supplies ``config.yaml`` and ``.env`` values that WIN over the
user's ``~/.hermes/config.yaml`` and ``~/.hermes/.env`` on a per-leaf-key basis.

This is DISTINCT from ``hermes_cli.config.is_managed()`` / ``HERMES_MANAGED``,
which is a coarse package-manager write-lock (declarative-distro / formula
installs). That lock blocks all mutation; this layer injects specific immutable
values. The two are independent and may coexist.

v1 enforcement is filesystem permissions only — see
``docs/design/managed-scope.md`` §7. v1 is Linux/POSIX-first; ``get_managed_dir()``
is the single seam for adding macOS / Windows native locations later.

Attribution: do not reference any third-party product by name in this file.
"""
from __future__ import annotations

import copy
import logging
import os
import threading
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# POSIX default. Other-platform locations are a deliberate v2 item; when added,
# they belong ONLY inside get_managed_dir().
_DEFAULT_MANAGED_DIR = Path("/etc/hermes")

_CACHE_LOCK = threading.Lock()
# path_key -> (mtime_ns, size, parsed)
_CONFIG_CACHE: Dict[str, tuple] = {}
_ENV_CACHE: Dict[str, tuple] = {}
_CONFIG_FAILURE_CACHE: Dict[str, tuple] = {}


class ManagedConfigResolutionError(RuntimeError):
    """The current managed config exists but cannot be resolved."""


def _under_pytest() -> bool:
    """True when running inside the test suite.

    Used to ignore the system default ``/etc/hermes`` during tests so a real
    managed scope on a developer/CI box can't leak policy into the suite. Tests
    that exercise managed scope set ``HERMES_MANAGED_DIR`` explicitly, which is
    still honored (the override path below runs before this guard takes effect).
    """
    return "PYTEST_CURRENT_TEST" in os.environ


def get_managed_dir() -> Optional[Path]:
    """Resolve the managed-scope directory, or None when no scope is present.

    Resolution (highest priority first):
      1. ``$HERMES_MANAGED_DIR`` — deployment/bootstrap path override (IT-only;
         never persisted to any .env). Honored only when set to a non-empty value
         AND the directory exists.
      2. ``/etc/hermes`` — POSIX default, when it exists. Ignored under pytest so
         a real system managed scope can't leak into the test suite.

    A non-existent directory at either tier resolves to None (no managed scope),
    which is the common case and must be cheap + side-effect-free.
    """
    override = os.environ.get("HERMES_MANAGED_DIR", "").strip()
    if override:
        p = Path(override)
        return p if p.is_dir() else None
    if _under_pytest():
        return None
    return _DEFAULT_MANAGED_DIR if _DEFAULT_MANAGED_DIR.is_dir() else None


def invalidate_managed_cache() -> None:
    """Drop cached managed config/env. For tests and post-edit reloads."""
    with _CACHE_LOCK:
        _CONFIG_CACHE.clear()
        _ENV_CACHE.clear()
        _CONFIG_FAILURE_CACHE.clear()


def _dangling_symlink_key(path: Path) -> Optional[tuple]:
    """Return lstat provenance when *path* is a dangling symlink."""
    if not path.is_symlink():
        return None
    try:
        st = path.lstat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ctime_ns, st.st_mode)


def _cached_read(
    path: Path,
    cache: Dict[str, tuple],
    parse,
    *,
    require_resolved: bool = False,
    failure_cache: Optional[Dict[str, tuple]] = None,
):
    """Shared (mtime_ns, size)-keyed read. Returns a deepcopy of the parsed value.

    Returns ``None`` when the file is absent or fails to parse in the default
    fail-open mode. A parse failure is logged LOUDLY — the admin needs to know
    their policy isn't being applied. Strict callers receive
    :class:`ManagedConfigResolutionError` instead, with failure memoization
    invalidated by content or permission metadata changes.
    """
    path_key = str(path)
    try:
        st = path.stat()
    except FileNotFoundError as exc:
        key = _dangling_symlink_key(path)
        if key is None:
            return None  # genuinely absent
        logger.warning(
            "managed scope: %s is a dangling symlink — IGNORING this managed "
            "file. Admin policy from this file is NOT being applied.",
            path,
        )
        if failure_cache is not None:
            with _CACHE_LOCK:
                failure_cache[path_key] = key
        if require_resolved:
            raise ManagedConfigResolutionError(
                f"Current managed config source at {path} is unresolved"
            ) from exc
        return None
    except OSError as exc:
        logger.warning(
            "managed scope: failed to access %s: %s — IGNORING this managed file. "
            "Admin policy from this file is NOT being applied. Fix and restart.",
            path,
            exc,
        )
        if require_resolved:
            raise ManagedConfigResolutionError(
                f"Current managed config source at {path} is inaccessible"
            ) from exc
        return None
    key = (st.st_mtime_ns, st.st_size, st.st_ctime_ns, st.st_mode)
    with _CACHE_LOCK:
        if (
            require_resolved
            and failure_cache is not None
            and failure_cache.get(path_key) == key
        ):
            raise ManagedConfigResolutionError(
                f"Current managed config source at {path} is unresolved"
            )
        hit = cache.get(path_key)
        if hit is not None and hit[:4] == key:
            return copy.deepcopy(hit[4])
    try:
        with open(path, encoding="utf-8") as f:
            parsed = parse(f)
    except Exception as exc:  # noqa: BLE001 — fail-open, but LOUD
        logger.warning(
            "managed scope: failed to parse %s: %s — IGNORING this managed file. "
            "Admin policy from this file is NOT being applied. Fix and restart.",
            path,
            exc,
        )
        if failure_cache is not None:
            with _CACHE_LOCK:
                failure_cache[path_key] = key
        if require_resolved:
            raise ManagedConfigResolutionError(
                f"Current managed config source at {path} is unresolved"
            ) from exc
        return None
    with _CACHE_LOCK:
        if failure_cache is not None:
            failure_cache.pop(path_key, None)
        cache[path_key] = (*key, copy.deepcopy(parsed))
    return parsed


def _parse_managed_config(f) -> dict:
    parsed = yaml.safe_load(f)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("managed config root must be a mapping")
    return parsed


def load_managed_config() -> dict:
    """Parsed managed config.yaml, or {} when absent/malformed (fail-open)."""
    managed_dir = get_managed_dir()
    if managed_dir is None:
        return {}
    parsed = _cached_read(
        managed_dir / "config.yaml",
        _CONFIG_CACHE,
        _parse_managed_config,
        failure_cache=_CONFIG_FAILURE_CACHE,
    )
    return parsed if isinstance(parsed, dict) else {}


def load_managed_config_strict() -> dict:
    """Parsed managed config, raising while a present source is unresolved.

    Safety-sensitive readers use this to distinguish an absent managed policy
    from one that exists but was ignored because of I/O or parse failure. Parse
    failures are memoized by ``(mtime_ns, size)`` until the admin fixes the file.
    """
    managed_dir = get_managed_dir()
    if managed_dir is None:
        return {}
    parsed = _cached_read(
        managed_dir / "config.yaml",
        _CONFIG_CACHE,
        _parse_managed_config,
        require_resolved=True,
        failure_cache=_CONFIG_FAILURE_CACHE,
    )
    return parsed or {}


def require_managed_config_resolved() -> None:
    """Validate the current managed config without copying it on cache hits."""
    managed_dir = get_managed_dir()
    if managed_dir is None:
        return
    path = managed_dir / "config.yaml"
    try:
        st = path.stat()
    except FileNotFoundError as exc:
        key = _dangling_symlink_key(path)
        if key is None:
            return
        path_key = str(path)
        with _CACHE_LOCK:
            _CONFIG_FAILURE_CACHE[path_key] = key
        raise ManagedConfigResolutionError(
            f"Current managed config source at {path} is unresolved"
        ) from exc
    except OSError as exc:
        raise ManagedConfigResolutionError(
            f"Current managed config source at {path} is inaccessible"
        ) from exc
    key = (st.st_mtime_ns, st.st_size, st.st_ctime_ns, st.st_mode)
    path_key = str(path)
    with _CACHE_LOCK:
        if _CONFIG_FAILURE_CACHE.get(path_key) == key:
            raise ManagedConfigResolutionError(
                f"Current managed config source at {path} is unresolved"
            )
        hit = _CONFIG_CACHE.get(path_key)
        if hit is not None and hit[:4] == key:
            return
    load_managed_config_strict()


def load_managed_env() -> Dict[str, str]:
    """Parsed managed .env (KEY=VALUE), or {} when absent (fail-open)."""
    managed_dir = get_managed_dir()
    if managed_dir is None:
        return {}
    parsed = _cached_read(managed_dir / ".env", _ENV_CACHE, _parse_env)
    return parsed if isinstance(parsed, dict) else {}


def apply_managed_overlay(config: dict) -> dict:
    """Overlay administrator-pinned config values on top of an already-built dict.

    The single, shared way for any config loader that builds its own dict
    (rather than going through hermes_cli.config.load_config) to honor managed
    scope. Mirrors hermes_cli.config._load_config_impl's managed merge exactly:

      * expand the managed config's ``${VAR}`` refs against the PROCESS env only
        (never user-config-defined refs), so a user cannot shadow a managed
        literal via a ${VAR} they control;
      * normalize the managed config's root ``model`` key (a bare ``model: x/y``
        string is promoted to ``model.default``) so it can't clobber the dict
        shape callers expect;
      * leaf-level deep-merge managed ON TOP, so managed wins per-leaf while
        sibling keys stay user-controlled.

    Fail-open: returns ``config`` unchanged if no managed scope is present or on
    any error — managed scope must never break a caller's startup. Mutates and
    returns ``config`` (callers pass a dict they own).
    """
    try:
        managed = load_managed_config()
        if not managed:
            return config
        # Imported lazily to avoid an import cycle (config imports managed_scope).
        from hermes_cli.config import _deep_merge, _expand_env_vars, _normalize_root_model_keys

        managed_expanded = _normalize_root_model_keys(_expand_env_vars(managed))
        # A bare ``model: x/y`` string in the managed file must merge as
        # ``model.default`` — otherwise _deep_merge would replace the caller's
        # ``model`` dict with a string and break every ``cfg["model"]["..."]``
        # read. _normalize_root_model_keys only promotes the string when there
        # are root provider/base_url keys to migrate, so handle the bare case
        # here (matches cli.py's own string-model handling).
        if isinstance(managed_expanded.get("model"), str):
            managed_expanded = dict(managed_expanded)
            managed_expanded["model"] = {"default": managed_expanded["model"]}
        return _deep_merge(config, managed_expanded)
    except Exception:  # noqa: BLE001 — overlay must never break a caller
        logger.warning("managed scope: failed to apply config overlay", exc_info=True)
        return config


def _parse_env(f) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in f:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip("\"'")
    return out


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
