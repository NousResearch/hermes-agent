"""User-configured deny policy helpers.

This module centralizes the user-editable ``permissions.deny`` namespace. The
checks are defense-in-depth guardrails for an honest-but-wrong agent, not a
sandbox against malicious local code: terminal access still runs as the user's
OS account. Keep this code small, deterministic, and import-light so tool
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


class DenyPolicyError(ValueError):
    """Raised when a configured deny policy cannot be evaluated safely."""


def load_user_config() -> dict[str, Any]:
    """Load user config lazily so importing this module has no config side effects."""
    from hermes_cli.config import load_config

    loaded = load_config()
    if not isinstance(loaded, dict):
        raise DenyPolicyError("Hermes config must be a mapping")
    return loaded


def parse_deny_patterns(
    value: Any,
    *,
    field: str,
    require_list: bool = False,
) -> list[str]:
    """Return non-empty string patterns, rejecting unsafe config shapes."""
    if value is None:
        return []
    if require_list and not isinstance(value, (list, tuple)):
        raise DenyPolicyError(f"{field} must be a list of strings")
    if isinstance(value, str):
        candidates: Iterable[Any] = [value]
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        candidates = value
    else:
        raise DenyPolicyError(f"{field} must be a string or list of strings")

    patterns: list[str] = []
    for item in candidates:
        if not isinstance(item, str):
            raise DenyPolicyError(f"{field} entries must be strings")
        if stripped := item.strip():
            patterns.append(stripped)
    return patterns


def _permissions_deny_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return ``permissions.deny`` while rejecting malformed explicit values."""
    if config is None:
        config = load_user_config()
    if not isinstance(config, dict):
        raise DenyPolicyError("Hermes config must be a mapping")

    permissions = config.get("permissions")
    if permissions is None:
        return {}
    if not isinstance(permissions, dict):
        raise DenyPolicyError("permissions must be a mapping")

    deny = permissions.get("deny")
    if deny is None:
        return {}
    if not isinstance(deny, dict):
        raise DenyPolicyError("permissions.deny must be a mapping")
    return deny


def permissions_deny_commands(config: dict[str, Any] | None = None) -> list[str]:
    """Return command deny globs from ``permissions.deny.commands``.

    ``approvals.deny`` remains the historical terminal-command location. This
    helper provides a forward-compatible alias in the broader ``permissions``
    namespace without changing the existing config shape.
    """
    deny = _permissions_deny_config(config)
    return parse_deny_patterns(
        deny.get("commands"),
        field="permissions.deny.commands",
    )


def permissions_deny_paths(config: dict[str, Any] | None = None) -> list[str]:
    """Return path deny globs from ``permissions.deny.paths``."""
    deny = _permissions_deny_config(config)
    return parse_deny_patterns(
        deny.get("paths"),
        field="permissions.deny.paths",
    )


def _expand_tilde(path: str) -> str:
    """Expand ``~`` with the same effective-home policy as file tools."""
    if not path or "~" not in path:
        return path
    from hermes_constants import get_subprocess_home

    home = get_subprocess_home()
    if home and (path == "~" or path.startswith("~/") or path.startswith("~\\")):
        if path == "~":
            return home
        return os.path.join(home, path[2:])
    return os.path.expanduser(path)


def _normalize_slash_path(path: str) -> str:
    """Normalize separators and case without dereferencing filesystem aliases."""
    normalized = os.path.normpath(path).replace("\\", "/")
    # Collapse duplicate slashes except a leading UNC-ish pair; deny globs are
    # easier to reason about in slash form and fnmatch treats '/' as ordinary.
    while "//" in normalized.replace("//", "", 1):
        normalized = normalized.replace("//", "/")
    return normalized.casefold()


def _normalize_path_variants(path: str, *, canonicalize: bool = True) -> tuple[str, ...]:
    """Return lexical and canonical identities for stable deny matching.

    The policy intentionally matches case-insensitively so the same rule protects
    case-insensitive filesystems (Windows/macOS defaults). Lexical identity is
    retained so a wildcard-before-symlink rule cannot be erased by canonical
    resolution; canonical identity catches aliases to an otherwise denied path.
    """
    expanded = _expand_tilde(str(path))
    lexical = _normalize_slash_path(expanded)
    if not canonicalize:
        return (lexical,)
    try:
        canonical = _normalize_slash_path(os.path.realpath(expanded))
    except (OSError, ValueError, RuntimeError):
        canonical = lexical
    return tuple(dict.fromkeys((lexical, canonical)))


def _normalize_pattern_lexical(pattern: str) -> str:
    """Normalize a deny glob without losing case or glob characters."""
    expanded = _expand_tilde(pattern.strip()).replace("\\", "/")
    normalized = os.path.normpath(expanded).replace("\\", "/")
    if pattern.rstrip().endswith("/") and not normalized.endswith("/"):
        normalized += "/"
    return normalized


def _normalize_pattern_for_match(pattern: str) -> str:
    """Normalize and case-fold a user-supplied deny glob for comparison."""
    return _normalize_pattern_lexical(pattern).casefold()


def _normalize_pattern_variants(
    pattern: str,
    *,
    canonicalize: bool = True,
) -> tuple[str, ...]:
    """Return lexical and safely canonicalized identities for a deny glob."""
    lexical_original_case = _normalize_pattern_lexical(pattern)
    lexical = lexical_original_case.casefold()
    if not canonicalize:
        return (lexical,)

    glob_indexes = [
        lexical_original_case.find(ch)
        for ch in "*?["
        if ch in lexical_original_case
    ]
    glob_index = min(glob_indexes) if glob_indexes else len(lexical_original_case)
    literal_prefix = lexical_original_case[:glob_index].rstrip("/")
    if not literal_prefix:
        return (lexical,)
    suffix = lexical_original_case[len(literal_prefix):].casefold()
    try:
        canonical_prefix = _normalize_slash_path(os.path.realpath(literal_prefix))
    except (OSError, ValueError, RuntimeError):
        return (lexical,)
    canonical = canonical_prefix.rstrip("/") + suffix
    return tuple(dict.fromkeys((lexical, canonical)))


def _search_overlap_prefixes(
    pattern: str,
    *,
    canonicalize: bool = True,
) -> tuple[tuple[str, bool], ...]:
    """Return literal prefixes and whether their wildcard starts a new segment."""
    prefixes: list[tuple[str, bool]] = []
    for normalized in _normalize_pattern_variants(pattern, canonicalize=canonicalize):
        glob_indexes = [normalized.find(ch) for ch in "*?[" if ch in normalized]
        if not glob_indexes:
            prefix = normalized.rstrip("/")
            segment_boundary = True
        else:
            raw_prefix = normalized[:min(glob_indexes)]
            segment_boundary = raw_prefix.endswith("/")
            prefix = raw_prefix.rstrip("/")
        prefixes.append((prefix, segment_boundary))
    return tuple(dict.fromkeys(prefixes))


def _path_matches_normalized_pattern(candidate: str, pat: str) -> bool:
    """Return whether normalized *candidate* is denied by normalized *pat*."""
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


def _path_matches_pattern(
    candidate: str,
    pattern: str,
    *,
    canonicalize: bool = True,
) -> bool:
    """Return True when normalized *candidate* is denied by *pattern*."""
    return any(
        _path_matches_normalized_pattern(candidate, pat)
        for pat in _normalize_pattern_variants(pattern, canonicalize=canonicalize)
    )


def match_permissions_deny_path(
    path: str,
    *,
    patterns: list[str] | None = None,
    canonicalize: bool = True,
    source: str = "permissions.deny.paths",
) -> DenyMatch | None:
    """Return the matching path deny rule for *path*, or ``None``.

    Matching uses case-insensitive fnmatch globs against a realpath-normalized,
    slash-separated candidate. Empty patterns are ignored.
    """
    if patterns is None:
        patterns = permissions_deny_paths()
    globs = parse_deny_patterns(patterns, field=source)
    if not globs:
        return None
    candidates = _normalize_path_variants(path, canonicalize=canonicalize)
    for pattern in globs:
        if any(
            _path_matches_pattern(candidate, pattern, canonicalize=canonicalize)
            for candidate in candidates
        ):
            return DenyMatch(pattern=pattern, source=source)
    return None


def match_permissions_deny_search_root(
    path: str,
    *,
    patterns: list[str] | None = None,
    root_is_file: bool = False,
    canonicalize: bool = True,
    source: str = "permissions.deny.paths",
) -> DenyMatch | None:
    """Block a search root that is denied or can contain a denied descendant.

    Recursive search backends may open descendants before returning results, so
    post-result filtering is too late. A deny rule whose literal prefix sits
    beneath the requested root therefore blocks the search before backend I/O.
    """
    if patterns is None:
        patterns = permissions_deny_paths()
    globs = parse_deny_patterns(patterns, field=source)
    if not globs:
        return None

    candidates = tuple(
        candidate.rstrip("/")
        for candidate in _normalize_path_variants(path, canonicalize=canonicalize)
    )
    for pattern in globs:
        if any(
            _path_matches_pattern(candidate, pattern, canonicalize=canonicalize)
            for candidate in candidates
        ):
            return DenyMatch(pattern=pattern, source=source)
        if root_is_file:
            continue
        for prefix, segment_boundary in _search_overlap_prefixes(
            pattern,
            canonicalize=canonicalize,
        ):
            if any(
                prefix == candidate
                or prefix.startswith(candidate + "/")
                or (
                    candidate.startswith(prefix + "/")
                    if segment_boundary
                    else candidate.startswith(prefix)
                )
                for candidate in candidates
            ):
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


def path_deny_policy_error(path: str) -> str:
    """Fail-closed error when ``permissions.deny.paths`` cannot be evaluated."""
    return (
        f"BLOCKED: permissions.deny.paths could not be evaluated for {path!r}. "
        "No backend content operation was attempted because deny-policy "
        "configuration and matching errors fail closed. Check config.yaml and "
        "retry after fixing the policy."
    )
