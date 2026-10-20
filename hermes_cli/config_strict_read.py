"""The fail-closed strict config read that guards every config.yaml write.

``hermes_cli.config.require_readable_config_before_write`` is the facade entry; this sibling owns
the behaviour. Tool hooks (pre/post/transform) resolve their state directory through the guard on
every call, and a gateway serving many sessions re-parsed the whole YAML under the GIL each time
(~31 ms per call on a real profile config, ~9% of GIL-held samples with ~11 concurrent sessions,
`session.resume` RPCs timing out behind it — #141856). A successful parse is now cached on the
file's backend cache version (config_version's token plus the racy content digest), so an
unchanged file costs one signature probe + one deepcopy. Failures are NEVER cached: an unreadable
or unparseable file is re-validated and refused on every call.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

from hermes_cli.config_read_errors import (
    _FIX_PERMS, _FIX_YAML, _backups_dir_display, _refuse_overwrite, _warn_config_parse_failure)

# path -> (*config_cache_version(config_path), parsed mapping) for the last successful strict
# parse. Published as ONE tuple replaced wholesale (same contract as ``config._RAW_CONFIG_CACHE``),
# so the lock-free hit path sees a complete entry; the writers' atomic replace produces a fresh
# inode + mtime and the cache key carries the racy content digest, so no explicit invalidation
# hook is needed (``save_config`` pops alongside ``_RAW_CONFIG_CACHE``).
_STRICT_READ_CACHE: dict[str, tuple[int, ...]] = {}


def _strict_read_cache_hit(path_key: str, cache_key: tuple[Any, ...]) -> Optional[dict[str, Any]]:
    """The cached parse for ``path_key`` if its cache version equals ``cache_key``, else ``None``."""
    cached = _STRICT_READ_CACHE.get(path_key)
    if cached is not None and cached[:len(cache_key)] == cache_key:
        return cached[len(cache_key)]
    return None


def require_readable_strict_config(config_path: Optional[Path]) -> dict[str, Any]:
    """Strict config.yaml read for read-then-write callers: the mapping, or a refusal.

    Fails closed on an unreadable file (permissions / broken mount) and an unparseable or
    non-mapping root — bare-``except`` loaders treat both as ``{}``, and a ``{}`` would let the
    subsequent write replace the recoverable file with only the caller's partial dict. Deliberately
    lock-free: ``save_config`` calls this while holding ``config._CONFIG_LOCK``, and a concurrent
    duplicate parse is no worse than the per-call parse this cache replaces.
    """
    # Late-bind the facade so patches on hermes_cli.config (read_config_doc / cache-version seams
    # in the test suite) keep intercepting the probe and the parse, and no import cycle forms.
    from hermes_cli.config import (
        _worker_config_snapshot, config_cache_version, get_config_path, read_config_doc)

    if config_path is None:
        snapshot = _worker_config_snapshot()
        if snapshot is not None:
            # A bypass worker's config IS its frozen snapshot; it never reads the profile file.
            return copy.deepcopy(snapshot)
        config_path = get_config_path()
    try:
        # One backend-routed probe is both the existence check and the cache key; its errors are
        # config_version's (FileNotFoundError / OSError), matching the inline body this replaces.
        cache_key = config_cache_version(config_path)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise _refuse_overwrite(config_path, "cannot be accessed", exc, _FIX_PERMS) from exc

    path_key = str(config_path)
    hit = _strict_read_cache_hit(path_key, cache_key)
    if hit is not None:
        return copy.deepcopy(hit)  # callers mutate the result before writing it back

    try:
        loaded = read_config_doc(config_path)
    except OSError as exc:
        raise _refuse_overwrite(config_path, "cannot be read", exc, _FIX_PERMS) from exc
    except Exception as exc:
        _warn_config_parse_failure(config_path, exc, fallback="refuse-write")
        raise _refuse_overwrite(
            config_path, "has a formatting error", exc, _FIX_YAML.format(backups=_backups_dir_display())) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        exc = TypeError(f"top-level YAML must be a mapping, got {type(loaded).__name__}")
        _warn_config_parse_failure(config_path, exc, fallback="refuse-write")
        raise _refuse_overwrite(
            config_path, f"must start with settings names, but its top level is a {type(loaded).__name__}",
            exc, _FIX_YAML.format(backups=_backups_dir_display())) from exc
    _STRICT_READ_CACHE[path_key] = (*cache_key, copy.deepcopy(loaded))
    return loaded
