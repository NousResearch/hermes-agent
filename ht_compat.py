"""Backward-compatible brand-identifier aliases (Phase 6 foundation).

The fork's user-visible identity is **HT AI Agent**, but the load-bearing
internal identifiers — environment variables (``HERMES_*``), the ``~/.hermes``
data directory, and ``X-Hermes-*`` HTTP headers — are inter-process /
inter-client contracts. Renaming them outright would break running installs,
subprocess hand-offs, and third-party API clients.

This module introduces the new ``HT_*`` / ``X-HT-*`` identifiers **additively**:
both namespaces resolve to the same value, with the ``HT_*`` (new) name taking
precedence when both are present. Nothing that reads the legacy names breaks.

Design rules:
  * The mapping is a pure *prefix* rule — ``HERMES_FOO`` ⇆ ``HT_FOO`` and
    ``X-Hermes-Foo`` ⇆ ``X-HT-Foo`` — so all ~329 env vars and every header are
    covered without an enumerated table that would drift out of date.
  * Mirroring is *non-clobbering*: a counterpart is only filled in when it is
    absent, so a caller that deliberately sets both keeps both.
  * Import-safe: stdlib only, no project imports, so ``hermes_constants`` and
    other early-boot modules can depend on it without circular-import risk.

The physical ``~/.hermes`` → ``~/.ht-ai-agent`` directory migration and the
Python module-name renames are deliberately **not** in this module — they need
migration tooling and a coordinated pass, and are tracked as follow-up Phase 6
increments (see ``docs/phase-6-internal-identifiers.md``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from typing import Optional

# Canonical (new) prefix and the legacy prefix it aliases.
_ENV_NEW_PREFIX = "HT_"
_ENV_OLD_PREFIX = "HERMES_"

# HTTP header prefixes. Header field names are case-insensitive, so callers
# should not rely on a particular casing; the helpers below compare
# case-insensitively.
_HEADER_NEW_PREFIX = "X-HT-"
_HEADER_OLD_PREFIX = "X-Hermes-"


def new_env_name(name: str) -> str:
    """Return the ``HT_*`` counterpart of a ``HERMES_*`` env var name.

    Idempotent: a name that is already ``HT_*`` (or that does not start with
    the legacy prefix) is returned unchanged.
    """
    if name.startswith(_ENV_OLD_PREFIX):
        return _ENV_NEW_PREFIX + name[len(_ENV_OLD_PREFIX):]
    return name


def old_env_name(name: str) -> str:
    """Return the ``HERMES_*`` counterpart of an ``HT_*`` env var name.

    Idempotent for names that are not ``HT_*``.
    """
    if name.startswith(_ENV_NEW_PREFIX):
        return _ENV_OLD_PREFIX + name[len(_ENV_NEW_PREFIX):]
    return name


def resolve_env(
    name: str,
    default: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Resolve a brand env var honouring both the new and legacy names.

    ``name`` may be given as either the ``HT_*`` or ``HERMES_*`` spelling; the
    new (``HT_*``) value wins when both are set. Returns ``default`` when
    neither is present.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    new = new_env_name(name)
    old = old_env_name(name)
    val = source.get(new)
    if val is not None and val != "":
        return val
    val = source.get(old)
    if val is not None and val != "":
        return val
    # Fall back to a set-but-empty value under either name before the default,
    # so callers can distinguish "explicitly blank" from "unset" if needed.
    if new in source:
        return source[new]
    if old in source:
        return source[old]
    return default


def mirror_brand_env(env: Optional[MutableMapping[str, str]] = None) -> None:
    """Mirror ``HERMES_*`` ⇆ ``HT_*`` in *env* in place (default ``os.environ``).

    After this runs, every brand env var is readable under both names, so code
    that reads the legacy ``HERMES_*`` names and code that reads the new
    ``HT_*`` names both work, and spawned subprocesses inherit both.

    Non-clobbering and idempotent: an already-present counterpart is left
    untouched, and running twice changes nothing. When only one side is set,
    the missing side is filled from it. When both are set (even to different
    values) neither is overwritten.
    """
    target: MutableMapping[str, str] = os.environ if env is None else env

    # Snapshot keys first — we mutate the mapping while iterating.
    old_keys = [k for k in list(target.keys()) if k.startswith(_ENV_OLD_PREFIX)]
    new_keys = [k for k in list(target.keys()) if k.startswith(_ENV_NEW_PREFIX)]

    for old in old_keys:
        new = new_env_name(old)
        if new not in target:
            target[new] = target[old]
    for new in new_keys:
        old = old_env_name(new)
        if old not in target:
            target[old] = target[new]


def mirror_brand_headers(headers: MutableMapping[str, str]) -> MutableMapping[str, str]:
    """Add ``X-HT-*`` mirrors for every ``X-Hermes-*`` header, in place.

    Additive and non-clobbering: existing ``X-Hermes-*`` headers are preserved
    (third-party clients still read them) and an ``X-HT-*`` header that is
    already present is not overwritten. Returns *headers* for chaining.
    """
    old_prefix_lower = _HEADER_OLD_PREFIX.lower()
    # Case-insensitive index of existing keys so we don't add a duplicate that
    # differs only in casing.
    existing_lower = {k.lower() for k in headers.keys()}
    additions: dict[str, str] = {}
    for key, value in list(headers.items()):
        if key.lower().startswith(old_prefix_lower):
            suffix = key[len(_HEADER_OLD_PREFIX):]
            new_key = _HEADER_NEW_PREFIX + suffix
            if new_key.lower() not in existing_lower:
                additions[new_key] = value
    headers.update(additions)
    return headers


def read_brand_header(
    headers: Mapping[str, str],
    suffix: str,
    default: str = "",
) -> str:
    """Read a brand header by its suffix, honouring both name spellings.

    ``suffix`` is the part after the prefix, e.g. ``"Session-Id"`` reads either
    ``X-HT-Session-Id`` (preferred) or ``X-Hermes-Session-Id``. Comparison is
    case-insensitive. Returns ``default`` when neither is present.
    """
    new_key = (_HEADER_NEW_PREFIX + suffix).lower()
    old_key = (_HEADER_OLD_PREFIX + suffix).lower()
    # Fast path for exact-case Mapping.get (the common Starlette/dict case).
    for candidate in (_HEADER_NEW_PREFIX + suffix, _HEADER_OLD_PREFIX + suffix):
        try:
            val = headers.get(candidate)  # type: ignore[union-attr]
        except AttributeError:
            val = None
        if val:
            return val
    # Case-insensitive fallback for plain dicts with unexpected casing.
    for key, value in headers.items():
        kl = key.lower()
        if kl == new_key and value:
            return value
    for key, value in headers.items():
        kl = key.lower()
        if kl == old_key and value:
            return value
    return default
