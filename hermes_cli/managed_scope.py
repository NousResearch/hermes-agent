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


def _cached_read(path: Path, cache: Dict[str, tuple], parse):
    """Shared (mtime_ns, size)-keyed read. Returns a deepcopy of the parsed value.

    Returns ``None`` when the file is absent or fails to parse (fail-open). A
    parse failure is logged LOUDLY — the admin needs to know their policy isn't
    being applied — but never raises, so a malformed managed file can't brick
    startup.
    """
    try:
        st = path.stat()
    except OSError:
        return None  # absent
    key = (st.st_mtime_ns, st.st_size)
    path_key = str(path)
    with _CACHE_LOCK:
        hit = cache.get(path_key)
        if hit is not None and hit[:2] == key:
            return copy.deepcopy(hit[2])
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
        return None
    with _CACHE_LOCK:
        cache[path_key] = (key[0], key[1], copy.deepcopy(parsed))
    return parsed


def load_managed_config() -> dict:
    """Parsed managed config.yaml, or {} when absent/malformed (fail-open)."""
    managed_dir = get_managed_dir()
    if managed_dir is None:
        return {}
    parsed = _cached_read(
        managed_dir / "config.yaml",
        _CONFIG_CACHE,
        lambda f: yaml.safe_load(f) or {},
    )
    return parsed if isinstance(parsed, dict) else {}


def load_managed_config_strict() -> tuple:
    """Strict sibling of :func:`load_managed_config` for fail-closed readers.

    Returns ``(provenance_signature, parsed_config)``:

    * no managed scope configured → ``(("absent",), {})`` — the common
      case; nothing is broken, defaults/user config decide;
    * managed config.yaml absent → same ``("absent",)`` signature (an
      existing managed *directory* without a config.yaml pins nothing);
    * managed config.yaml present and valid mapping → a signature over
      the file's full stat identity (mtime_ns/size/ctime_ns/mode/ino/
      dev/uid/gid) plus the parsed dict;
    * present but malformed, unreadable, dangling, or non-mapping →
      raises ``ConfigResolutionError`` (imported lazily to avoid a
      circular import): admin policy is BROKEN, not absent, and a
      fail-closed consumer must not silently fall back to user values.

    The general fail-open ``load_managed_config()`` callers are untouched;
    this reader shares nothing with the fail-open caches, so a fail-open
    prewarm can never satisfy a strict lookup (and vice versa).
    """
    managed_dir = get_managed_dir()
    if managed_dir is None:
        return (("absent",), {})
    path = managed_dir / "config.yaml"
    import stat as _stat

    try:
        lst = path.lstat()
    except FileNotFoundError:
        return (("absent",), {})
    except OSError as exc:
        from hermes_cli.config import ConfigResolutionError

        raise ConfigResolutionError(
            f"Managed config source at {path} is inaccessible: {exc}"
        ) from exc
    if _stat.S_ISLNK(lst.st_mode):
        try:
            stt = path.stat()
        except OSError as exc:
            from hermes_cli.config import ConfigResolutionError

            raise ConfigResolutionError(
                f"Managed config source at {path} is a dangling or "
                f"inaccessible symlink: {exc}"
            ) from exc
        sig = (
            stt.st_mtime_ns, stt.st_size, stt.st_ctime_ns, stt.st_mode,
            stt.st_ino, stt.st_dev, stt.st_uid, stt.st_gid,
        )
    elif _stat.S_ISREG(lst.st_mode):
        sig = (
            lst.st_mtime_ns, lst.st_size, lst.st_ctime_ns, lst.st_mode,
            lst.st_ino, lst.st_dev, lst.st_uid, lst.st_gid,
        )
    else:
        from hermes_cli.config import ConfigResolutionError

        raise ConfigResolutionError(
            f"Managed config source at {path} is not a regular file"
        )
    try:
        with open(path, encoding="utf-8") as f:
            parsed = yaml.safe_load(f)
    except Exception as exc:
        from hermes_cli.config import ConfigResolutionError

        raise ConfigResolutionError(
            f"Managed config at {path} could not be read/parsed: {exc}"
        ) from exc
    if parsed is None:
        return (sig, {})
    if not isinstance(parsed, dict):
        from hermes_cli.config import ConfigResolutionError

        raise ConfigResolutionError(
            f"Managed config at {path} has a non-mapping root "
            f"({type(parsed).__name__})"
        )
    return (sig, parsed)


def cached_managed_config_for(managed_sig: tuple) -> dict:
    """Return the parsed managed config matching ``managed_sig``, or {}.

    Companion to :func:`load_managed_config_strict` for callers that
    already hold the strict provenance signature (so the file is not read
    twice per resolution). The signature is re-verified cheaply: when it
    no longer matches the live source, the strict read runs again.
    """
    sig, parsed = load_managed_config_strict()
    if sig == managed_sig:
        return parsed
    # Provenance drifted between the two reads (racy repair); re-resolve
    # against the live state rather than serving stale policy.
    raise_from_live = load_managed_config_strict()
    return raise_from_live[1]


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
