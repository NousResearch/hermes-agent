"""User-configured deny policy helpers.

This module centralizes the user-editable ``permissions.deny`` namespace.  The
checks are defense-in-depth guardrails for an honest-but-wrong agent, not a
sandbox against malicious local code: terminal access still runs as the user's
OS account.  Keep this code small, deterministic, and import-light so tool
paths can call it before doing any I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import os
from typing import Any, Iterable


@dataclass(frozen=True)
class DenyMatch:
    """A matched deny rule."""

    pattern: str
    source: str


def load_user_config() -> dict[str, Any]:
    """Load user config lazily so importing this module has no config side effects."""
    from hermes_cli.config import load_config

    loaded = load_config()
    return loaded if isinstance(loaded, dict) else {}


def _string_patterns(value: Any) -> list[str]:
    """Return non-empty string patterns from a YAML scalar/list-ish value."""
    if isinstance(value, str):
        candidates: Iterable[Any] = [value]
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        candidates = value
    else:
        return []
    return [item.strip() for item in candidates if isinstance(item, str) and item.strip()]


def _permissions_deny_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the ``permissions.deny`` mapping, or an empty mapping."""
    if config is None:
        config = load_user_config()
    permissions = config.get("permissions") if isinstance(config, dict) else None
    if not isinstance(permissions, dict):
        return {}
    deny = permissions.get("deny")
    return deny if isinstance(deny, dict) else {}


def permissions_deny_commands(config: dict[str, Any] | None = None) -> list[str]:
    """Return command deny globs from ``permissions.deny.commands``.

    ``approvals.deny`` remains the historical terminal-command location.  This
    helper provides a forward-compatible alias in the broader ``permissions``
    namespace without changing the existing config shape.
    """
    try:
        deny = _permissions_deny_config(config)
    except Exception:
        return []
    return _string_patterns(deny.get("commands"))


def permissions_deny_paths(config: dict[str, Any] | None = None) -> list[str]:
    """Return path deny globs from ``permissions.deny.paths``."""
    try:
        deny = _permissions_deny_config(config)
    except Exception:
        return []
    return _string_patterns(deny.get("paths"))


def _normalize_path_for_match(path: str) -> str:
    """Normalize a path string for stable deny matching.

    The policy intentionally matches case-insensitively so the same rule protects
    case-insensitive filesystems (Windows/macOS defaults).  ``realpath`` is used
    for symlink-aware checks where possible; it also normalizes non-existent
    write targets well enough for pre-write policy decisions.
    """
    expanded = os.path.expanduser(str(path))
    try:
        expanded = os.path.realpath(expanded)
    except (OSError, ValueError, RuntimeError):
        expanded = os.path.normpath(expanded)
    normalized = os.path.normpath(expanded).replace("\\", "/")
    # Collapse duplicate slashes except a leading UNC-ish pair; deny globs are
    # easier to reason about in slash form and fnmatch treats '/' as ordinary.
    while "//" in normalized.replace("//", "", 1):
        normalized = normalized.replace("//", "/")
    return normalized.casefold()


def _normalize_pattern_for_match(pattern: str) -> str:
    """Normalize a user-supplied deny glob while preserving glob characters."""
    expanded = os.path.expanduser(pattern.strip()).replace("\\", "/")
    normalized = os.path.normpath(expanded).replace("\\", "/")
    if pattern.rstrip().endswith("/") and not normalized.endswith("/"):
        normalized += "/"
    return normalized.casefold()


def _path_matches_pattern(candidate: str, pattern: str) -> bool:
    """Return True when normalized *candidate* is denied by *pattern*."""
    pat = _normalize_pattern_for_match(pattern)
    if not pat:
        return False
    if fnmatch.fnmatchcase(candidate, pat):
        return True

    # Treat a plain directory pattern as "that directory and everything below".
    has_glob = any(ch in pat for ch in "*?[")
    if not has_glob:
        base = pat.rstrip("/")
        return candidate == base or candidate.startswith(base + "/")

    # Common spelling: /secret/** should also block /secret itself.
    if pat.endswith("/**"):
        base = pat[:-3].rstrip("/")
        return candidate == base or candidate.startswith(base + "/")
    return False


def match_permissions_deny_path(
    path: str,
    *,
    patterns: list[str] | None = None,
    source: str = "permissions.deny.paths",
) -> DenyMatch | None:
    """Return the matching path deny rule for *path*, or ``None``.

    Matching uses case-insensitive fnmatch globs against a realpath-normalized,
    slash-separated candidate.  Empty/non-string patterns are ignored.
    """
    if patterns is None:
        patterns = permissions_deny_paths()
    globs = _string_patterns(patterns)
    if not globs:
        return None
    candidate = _normalize_path_for_match(path)
    for pattern in globs:
        if _path_matches_pattern(candidate, pattern):
            return DenyMatch(pattern=pattern, source=source)
    return None


def path_deny_error(path: str, match: DenyMatch) -> str:
    """Human/model-facing error for a path deny match."""
    return (
        f"BLOCKED: path {path!r} matches the user-defined deny rule "
        f"{match.pattern!r} ({match.source} in config.yaml). It cannot be "
        "accessed via file tools. Do NOT retry or rephrase this file-tool "
        "call; the user has explicitly forbidden this path. "
        "(Defense-in-depth — not a security boundary; terminal access may "
        "still reach the same OS path.)"
    )
