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
import re
from typing import Any, Iterable


@dataclass(frozen=True)
class DenyMatch:
    """A matched deny rule."""

    pattern: str
    source: str


class DenyPolicyError(ValueError):
    """Raised when a configured deny policy cannot be evaluated safely."""


def load_user_config() -> dict[str, Any]:
    """Load effective deny policy without mutations or fail-open fallback."""
    from hermes_cli.config import load_security_policy_config_readonly

    try:
        loaded = load_security_policy_config_readonly()
    except DenyPolicyError:
        raise
    except Exception as exc:
        raise DenyPolicyError(f"Hermes security policy could not be loaded: {exc}") from exc
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


def command_deny_patterns(config: dict[str, Any] | None = None) -> list[str]:
    """Return both command-deny aliases from one strict policy snapshot."""
    if config is None:
        config = load_user_config()
    if not isinstance(config, dict):
        raise DenyPolicyError("Hermes config must be a mapping")

    approvals = config.get("approvals")
    if approvals is None:
        approvals = {}
    if not isinstance(approvals, dict):
        raise DenyPolicyError("approvals must be a mapping")

    patterns = parse_deny_patterns(
        approvals.get("deny"),
        field="approvals.deny",
        require_list=True,
    )
    patterns.extend(permissions_deny_commands(config))
    return patterns


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


def _adapt_local_windows_drive(path: str) -> str:
    """Map MSYS/Cygwin/WSL drive spellings to the native Windows dialect.

    This adapter is only used for explicitly local matching. Remote backends
    keep ``/c/...`` lexical because it may be a legitimate remote POSIX path.
    """
    if os.name != "nt":
        return path
    slash_path = path.replace("\\", "/")
    match = re.match(
        r"^/(?:cygdrive/|mnt/)?([A-Za-z])(?:/|$)",
        slash_path,
    )
    if match is None:
        return path
    suffix = slash_path[match.end():]
    drive = match.group(1).upper()
    return f"{drive}:/{suffix}" if suffix else f"{drive}:/"


def _normalize_path_variants(path: str, *, canonicalize: bool = True) -> tuple[str, ...]:
    """Return lexical and canonical identities for stable deny matching.

    The policy intentionally matches case-insensitively so the same rule protects
    case-insensitive filesystems (Windows/macOS defaults). Lexical identity is
    retained so a wildcard-before-symlink rule cannot be erased by canonical
    resolution; canonical identity catches aliases to an otherwise denied path.
    """
    expanded = _expand_tilde(str(path))
    if canonicalize:
        expanded = _adapt_local_windows_drive(expanded)
    lexical = _normalize_slash_path(expanded)
    if not canonicalize:
        return (lexical,)
    try:
        canonical = _normalize_slash_path(os.path.realpath(expanded))
    except (OSError, ValueError, RuntimeError):
        canonical = lexical
    return tuple(dict.fromkeys((lexical, canonical)))


def _normalize_pattern_lexical(
    pattern: str,
    *,
    adapt_local_windows: bool = False,
) -> str:
    """Normalize a deny glob without losing case or glob characters."""
    expanded = _expand_tilde(pattern.strip()).replace("\\", "/")
    if adapt_local_windows:
        expanded = _adapt_local_windows_drive(expanded)
    normalized = os.path.normpath(expanded).replace("\\", "/")
    if pattern.rstrip().endswith("/") and not normalized.endswith("/"):
        normalized += "/"
    return normalized


def _normalize_pattern_for_match(pattern: str) -> str:
    """Normalize and case-fold a user-supplied deny glob for comparison."""
    return _normalize_pattern_lexical(
        pattern,
        adapt_local_windows=True,
    ).casefold()


def _patterns_with_local_base(
    patterns: list[str],
    base_path: str | os.PathLike[str] | None,
) -> list[tuple[str, str]]:
    """Pair match variants with their configured source rules.

    Implicit readers resolve candidates to absolute local paths before policy
    matching. Keeping both the raw rule and an anchored variant preserves the
    lexical identity while making a rule such as ``secret/**`` mean the same
    thing for implicit reads as it does for explicit file tools. The second
    tuple item always remains the user's configured rule for truthful errors.
    """
    variants: list[tuple[str, str]] = [(pattern, pattern) for pattern in patterns]
    if base_path is None:
        return variants

    base = _expand_tilde(str(base_path))
    if not (
        os.path.isabs(base)
        or bool(re.match(r"^[A-Za-z]:[\\/]", base))
        or base.startswith("\\\\")
    ):
        raise DenyPolicyError("path-policy base_path must be absolute")

    for pattern in patterns:
        expanded = _expand_tilde(pattern)
        if (
            os.path.isabs(expanded)
            or bool(re.match(r"^[A-Za-z]:[\\/]", expanded))
            or expanded.startswith("\\\\")
        ):
            continue
        anchored = os.path.normpath(os.path.join(base, expanded))
        pair = (anchored, pattern)
        if pair not in variants:
            variants.append(pair)
    return variants


def _normalize_pattern_variants(
    pattern: str,
    *,
    canonicalize: bool = True,
) -> tuple[str, ...]:
    """Return lexical and safely canonicalized identities for a deny glob."""
    lexical_original_case = _normalize_pattern_lexical(
        pattern,
        adapt_local_windows=canonicalize,
    )
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


def _fixed_width_search_prefix_compatible(
    candidate: str,
    pattern: str,
    *,
    canonicalize: bool,
) -> bool:
    """Reject impossible overlap before applying the conservative prefix test.

    ``?`` and character classes consume exactly one path-segment character.
    Truncating at those metacharacters makes ``foo?`` appear to overlap
    ``foobar``. Compare every fully constrained segment up to the first ``*``;
    after a variable-width star, overlap remains intentionally conservative.
    """

    candidate_parts = candidate.strip("/").split("/") if candidate.strip("/") else []
    for normalized in _normalize_pattern_variants(
        pattern,
        canonicalize=canonicalize,
    ):
        pattern_parts = normalized.strip("/").split("/") if normalized.strip("/") else []
        compatible = True
        for index, pattern_part in enumerate(pattern_parts):
            if "*" in pattern_part:
                break
            if index >= len(candidate_parts):
                break
            candidate_part = candidate_parts[index]
            if any(char in pattern_part for char in "*?["):
                if not fnmatch.fnmatchcase(candidate_part, pattern_part):
                    compatible = False
                    break
            elif candidate_part != pattern_part:
                compatible = False
                break
        else:
            if len(candidate_parts) > len(pattern_parts):
                compatible = False
        if compatible:
            return True
    return False


def _path_matches_normalized_pattern(candidate: str, pat: str) -> bool:
    """Return whether normalized *candidate* is denied by normalized *pat*."""
    if not pat:
        return False
    # A configured trailing slash declares directory intent. Normalized paths
    # deliberately omit that slash, so compare against the directory spelling
    # while retaining the original rule for diagnostics.
    match_pat = pat.rstrip("/") or "/"
    if fnmatch.fnmatchcase(candidate, match_pat):
        return True

    # A rule that matches a directory also denies everything below it, even
    # when the directory spelling contains glob metacharacters. Without this,
    # ``private?`` blocks ``private1`` itself but permits
    # ``private1/SOUL.md`` and explicit file references beneath it. Plain
    # directory rules already have descendant semantics below; globbed
    # directory rules must preserve the same invariant.
    ancestor = candidate.rstrip("/")
    while "/" in ancestor:
        ancestor = ancestor.rsplit("/", 1)[0]
        if not ancestor:
            ancestor = "/"
        if fnmatch.fnmatchcase(ancestor, match_pat):
            return True
        if ancestor == "/":
            break

    # Treat a plain directory pattern as "that directory and everything below".
    has_glob = any(ch in match_pat for ch in "*?[")
    if not has_glob:
        base = match_pat.rstrip("/")
        return candidate == base or candidate.startswith(base + "/")

    # Common spelling: /secret/** should also block /secret itself.
    if match_pat.endswith("/**"):
        base = match_pat[:-3].rstrip("/")
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
    base_path: str | os.PathLike[str] | None = None,
    canonicalize: bool = True,
    source: str = "permissions.deny.paths",
) -> DenyMatch | None:
    """Return the matching path deny rule for *path*, or ``None``.

    Matching evaluates slash-normalized lexical spellings first. When
    ``canonicalize`` is enabled, filesystem-canonical aliases are evaluated only
    after the lexical phase allows the candidate. Empty patterns are ignored.
    """
    if patterns is None:
        patterns = permissions_deny_paths()
    globs = parse_deny_patterns(patterns, field=source)
    pattern_variants = _patterns_with_local_base(globs, base_path)
    if not pattern_variants:
        return None
    # Lexical policy is the first security boundary. Only touch filesystem
    # topology (realpath) after every lexical spelling has been allowed.
    phases = (False, True) if canonicalize else (False,)
    for phase_canonicalize in phases:
        candidates = _normalize_path_variants(
            path,
            canonicalize=phase_canonicalize,
        )
        for match_pattern, source_pattern in pattern_variants:
            if any(
                _path_matches_pattern(
                    candidate,
                    match_pattern,
                    canonicalize=phase_canonicalize,
                )
                for candidate in candidates
            ):
                return DenyMatch(pattern=source_pattern, source=source)
    return None


def match_permissions_deny_search_root(
    path: str,
    *,
    patterns: list[str] | None = None,
    base_path: str | os.PathLike[str] | None = None,
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
    pattern_variants = _patterns_with_local_base(globs, base_path)
    if not pattern_variants:
        return None

    phases = (False, True) if canonicalize else (False,)
    for phase_canonicalize in phases:
        candidates = tuple(
            candidate.rstrip("/")
            for candidate in _normalize_path_variants(
                path,
                canonicalize=phase_canonicalize,
            )
        )
        for match_pattern, source_pattern in pattern_variants:
            if any(
                _path_matches_pattern(
                    candidate,
                    match_pattern,
                    canonicalize=phase_canonicalize,
                )
                for candidate in candidates
            ):
                return DenyMatch(pattern=source_pattern, source=source)
            if root_is_file:
                continue
            for prefix, segment_boundary in _search_overlap_prefixes(
                match_pattern,
                canonicalize=phase_canonicalize,
            ):
                if any(
                    _fixed_width_search_prefix_compatible(
                        candidate,
                        match_pattern,
                        canonicalize=phase_canonicalize,
                    )
                    and (
                        prefix == candidate
                        or prefix.startswith(candidate + "/")
                        or (
                            candidate.startswith(prefix + "/")
                            if segment_boundary
                            else candidate.startswith(prefix)
                        )
                    )
                    for candidate in candidates
                ):
                    return DenyMatch(pattern=source_pattern, source=source)
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
