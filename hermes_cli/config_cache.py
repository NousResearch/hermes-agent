"""Cache key + file-signature helpers for ``hermes_cli.config``'s ``_LOAD_CONFIG_CACHE``.

Split out of the ``config.py`` facade (#141857). Both helpers answer the same question — "may this
memoized ``load_config()`` result still be served?" — so they live together here while the cache
dict and its store/hit seams stay in the facade, where every load path reads them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def load_config_cache_key(path_key: str) -> tuple[str, bool, Optional[str]]:
    """Cache key for ``_LOAD_CONFIG_CACHE``: ``(path, in_scope, scope_home)``.

    Scoped and unscoped callers expand ``${VAR}`` against different sources (profile ``.env`` vs
    ``os.environ``), so a single path-keyed entry makes them evict each other and re-parse on every
    crossing (#141857). Keying by scope keeps each side's entry separate.
    """
    try:
        from agent.secret_scope import current_secret_scope, current_secret_scope_home
    except ImportError:
        # No secret-scope support in this install: every caller is unscoped.
        return path_key, False, None
    if current_secret_scope() is None:
        return path_key, False, None
    return path_key, True, current_secret_scope_home()


def load_config_cache_sig(config_path: Path) -> tuple[Optional[tuple[Any, ...]], Optional[tuple[Any, ...]]]:
    """Return ``(user_sig, cache_sig)`` for ``_LOAD_CONFIG_CACHE`` (racy-clean guarded, see ``config_cache_version``).

    The managed config file's signature is folded in (zeros = none) so editing it invalidates
    the merged result. ``cache_sig`` is None only when neither file exists (nothing to cache on)."""
    from hermes_cli import config_backend, managed_scope
    from utils import file_cache_version

    try:
        user_sig: Optional[tuple[Any, ...]] = config_backend.config_cache_version(config_path)
    except FileNotFoundError:
        user_sig = None
    managed_dir = managed_scope.get_managed_dir()
    try:
        managed_sig = file_cache_version(managed_dir / "config.yaml") if managed_dir else (0, 0, 0, 0, "")  # config-reader: ok — managed overlay
    except OSError:
        managed_sig = (0, 0, 0, 0, "")
    if user_sig is None and managed_sig == (0, 0, 0, 0, ""):
        return None, None
    return user_sig, (*(user_sig or (0, 0, 0, 0, "")), *managed_sig)
