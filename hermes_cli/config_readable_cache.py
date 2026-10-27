"""Successful-parse memo for :func:`hermes_cli.config.require_readable_config_before_write`.

That function guards read-then-write callers against wiping a config it could not read, so it
parses ``config.yaml`` on every call. Plugins that resolve their state directory through it call
it from tool hooks (pre/post/transform), so a gateway serving many sessions re-parsed the whole
config several times per tool call while holding the GIL (~18ms per call, ~9% of GIL-held samples
with ~11 concurrent sessions). This module memoizes the successful parse instead.

Two invariants carry the guard's meaning:

* **Keyed on the backend's cache signature.** ``hermes_cli.config_backend.config_version`` changes
  whenever the user layer does, so an edit invalidates the entry and non-file home layers stay
  correct.
* **Failures are never stored.** A malformed or unreadable config is re-validated and refused on
  every call, so a repaired file is honoured immediately.

One entry per path: a config file has a single current state, so a newer signature supersedes the
old one and a long-lived process cannot grow the memo without bound.
"""
from __future__ import annotations

import copy
from typing import Any

from utils import mtime_is_racy

#: ``(str(path), config_version) -> parsed mapping``.
_CACHE: dict[tuple[str, tuple[Any, ...]], dict[str, Any]] = {}


def clear_readable_config_cache() -> None:
    """Drop every memoized parse (tests, and any caller that must force a fresh read)."""
    _CACHE.clear()


def get_readable_config(path: str, version: tuple[Any, ...]) -> dict[str, Any] | None:
    """A copy of the parse memoized for ``(path, version)``, or None when there is no entry.

    The copy matters: most ``require_readable_config_before_write`` call sites mutate the mapping
    they get back, and the memoized dict must survive that untouched.
    """
    cached = _CACHE.get((path, version))
    return None if cached is None else copy.deepcopy(cached)


def store_readable_config(path: str, version: tuple[Any, ...], loaded: dict[str, Any]) -> None:
    """Memoize a successful parse under ``(path, version)``, superseding older versions of *path*."""
    cache_key = (path, version)
    for stale in [key for key in _CACHE if key[0] == path and key != cache_key]:
        del _CACHE[stale]
    _CACHE[cache_key] = loaded


def version_is_racily_fresh(version: tuple[Any, ...]) -> bool:
    """True when *version* leads with an mtime too recent to rule out a same-tick write.

    Filesystems tick coarsely, so a write in the same tick as the version read leaves the signature
    unchanged; such an entry must not be trusted as "nothing changed since". The file backend's
    version leads with ``st_mtime_ns`` (see ``config_backend.FileBackend.version_path``); a version
    that does not lead with an int carries no local mtime to judge, so it is trusted.
    """
    head = version[0] if version else None
    if isinstance(head, bool) or not isinstance(head, int):
        return False
    return mtime_is_racy(head)
