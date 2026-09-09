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
import threading
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# POSIX default. Other-platform locations belong ONLY inside get_managed_dir().
_DEFAULT_MANAGED_DIR = Path("/etc/hermes")

_CACHE_LOCK = threading.Lock()
# path_key -> (mtime_ns, size, parsed)
_CONFIG_CACHE: Dict[str, tuple] = {}
_ENV_CACHE: Dict[str, tuple] = {}
_MISSING = object()


def _under_pytest() -> bool:
    """True inside the test suite: ignore the system ``/etc/hermes`` so a real managed scope on a
    dev/CI box can't leak policy into the suite. An explicit ``HERMES_MANAGED_DIR`` still wins."""
    return "PYTEST_CURRENT_TEST" in os.environ


def get_managed_dir() -> Optional[Path]:
    """Resolve the managed-scope directory, or None when no scope is present.

    Priority: ``$HERMES_MANAGED_DIR`` (IT-only bootstrap override; never persisted to any .env;
    honored only when non-empty AND the directory exists), then ``/etc/hermes`` when it exists.
    A missing directory resolves to None — the common case, so it must be cheap + side-effect-free.
    """
    override = os.environ.get("HERMES_MANAGED_DIR", "").strip()
    if override:
        p = Path(override)
    elif _under_pytest():
        return None
    else:
        p = _DEFAULT_MANAGED_DIR
    return p if p.is_dir() else None


def invalidate_managed_cache() -> None:
    """Drop cached managed config/env. For tests and post-edit reloads."""
    with _CACHE_LOCK:
        _CONFIG_CACHE.clear()
        _ENV_CACHE.clear()


def _cached_read_with_status(
    path: Path, cache: Dict[str, tuple], parse, *, use_cache: bool = True
):
    """Shared cached read returning ``(value, current_file_loaded)``.

    An absent optional file is a successful no-op. Read or parse failures return
    ``(None, False)`` and are logged, while preserving the managed scope's
    non-throwing startup behavior.
    """
    try:
        st = path.stat()
    except FileNotFoundError:
        return _MISSING, True
    except OSError:
        return None, False
    key = (st.st_mtime_ns, st.st_size)
    path_key = str(path)
    if use_cache:
        with _CACHE_LOCK:
            hit = cache.get(path_key)
            if hit is not None and hit[:2] == key:
                return copy.deepcopy(hit[2]), True
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
        return None, False
    with _CACHE_LOCK:
        cache[path_key] = (key[0], key[1], copy.deepcopy(parsed))
    return parsed, True


def _cached_read(path: Path, cache: Dict[str, tuple], parse):
    """Compatibility wrapper returning only the parsed managed value."""
    parsed, _ = _cached_read_with_status(path, cache, parse)
    return None if parsed is _MISSING else parsed


def _load_managed_file(name: str, cache: Dict[str, tuple], parse) -> dict:
    managed_dir = get_managed_dir()
    if managed_dir is None:
        return {}
    parsed = _cached_read(managed_dir / name, cache, parse)
    return parsed if isinstance(parsed, dict) else {}


def load_managed_config() -> dict:
    """Parsed managed config.yaml, or {} when absent/malformed (fail-open)."""
    config, _ = load_managed_config_with_status()
    return config


def load_managed_config_with_status(*, use_cache: bool = True) -> tuple[dict, bool]:
    """Return managed config and whether an existing file loaded successfully.

    ``use_cache=False`` revalidates the current file for consent/policy callers.
    """
    managed_dir = get_managed_dir()
    if managed_dir is None:
        return {}, True
    parsed, loaded = _cached_read_with_status(
        managed_dir / "config.yaml",
        _CONFIG_CACHE,
        yaml.safe_load,
        use_cache=use_cache,
    )
    if parsed is _MISSING:
        return {}, True
    if not isinstance(parsed, dict):
        return {}, False
    return parsed, loaded


def load_managed_env() -> Dict[str, str]:
    """Parsed managed .env (KEY=VALUE), or {} when absent (fail-open)."""
    return _load_managed_file(".env", _ENV_CACHE, _parse_env)


def apply_managed_overlay(config: dict) -> dict:
    """Overlay administrator-pinned config values on top of an already-built dict.

    ``${VAR}`` refs in the managed config expand against the PROCESS env only, so a user cannot
    shadow a managed literal via a ref they control; a bare root ``model: x/y`` string is promoted
    to ``model.default`` so it can't clobber the dict shape callers expect; managed values
    deep-merge ON TOP per leaf while sibling keys stay user-controlled. Fail-open: returns
    ``config`` unchanged when no scope is present or on any error. Mutates and returns ``config``.
    """
    try:
        managed = load_managed_config()
        if not managed:
            return config
        # Imported lazily to avoid an import cycle (config imports managed_scope).
        from hermes_cli.config import _deep_merge, _expand_env_vars, _normalize_root_model_keys
        managed_expanded = _normalize_root_model_keys(_expand_env_vars(managed))
        # _normalize_root_model_keys only promotes the string when root provider/base_url
        # keys exist to migrate; handle the bare case here (matches cli.py) so _deep_merge
        # never replaces the caller's ``model`` dict with a string.
        if isinstance(managed_expanded.get("model"), str):
            managed_expanded = dict(managed_expanded)
            managed_expanded["model"] = {"default": managed_expanded["model"]}
        return _deep_merge(config, managed_expanded)
    except Exception:  # noqa: BLE001 — overlay must never break a caller
        logger.warning("managed scope: failed to apply config overlay", exc_info=True)
        return config


def _parse_env(f) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in map(str.strip, f):
        if line and not line.startswith("#") and "=" in line:
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
