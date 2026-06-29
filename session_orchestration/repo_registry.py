"""
session_orchestration/repo_registry.py — resolves a repo name to an absolute
path and optional default agent.

Resolution order (criterion #9):
  1. Exact alias match in manual overrides (config repos: section)
  2. Exact name match in auto-scan cache (~/dev/* and ~/.hermes git repos)
  3. Fuzzy match (case-insensitive suffix on hyphen-split or substring) in
     overrides first, then scan cache
  4. Literal path: if the name looks like an absolute/home-expanded path and
     points at a git repo directory
  5. UnresolvedRepo sentinel — caller should ask the user for clarification

The scan result is cached at the module level; call ``invalidate_scan_cache()``
to reset (tests should inject ``_injected_scan`` instead of touching the cache).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

DEFAULT_AGENT: str = "omp"

_DEFAULT_SCAN_ROOTS: Sequence[str] = ("~/dev", "~/.hermes")


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoEntry:
    """One entry in the registry: absolute path + default agent."""

    path: str
    default_agent: str = DEFAULT_AGENT


@dataclass(frozen=True)
class ResolvedRepo:
    """Successful resolution result."""

    path: str
    default_agent: str
    matched_name: str  # the alias / canonical name that matched
    match_kind: str    # "exact" | "fuzzy" | "literal_path"


@dataclass(frozen=True)
class UnresolvedRepo:
    """Sentinel: the name could not be resolved; caller should ask the user."""

    name: str


# ---------------------------------------------------------------------------
# Scanning helpers
# ---------------------------------------------------------------------------


def _is_git_repo(path: Path) -> bool:
    """Return True if *path* contains a ``.git`` entry."""
    return (path / ".git").exists()


def scan_for_repos(
    scan_roots: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """Scan *scan_roots* for git repositories.

    Returns ``{lowercase_basename: absolute_path}``.  When multiple repos
    share a basename the first one found wins (overrides take precedence over
    the scan result anyway, so collisions here are low-risk).

    Parameters
    ----------
    scan_roots:
        Paths to examine.  Each is expanded via ``Path.expanduser()``.  If the
        root itself is a git repo it is included; immediate children that are
        git repos are also included.  Defaults to ``~/dev`` and ``~/.hermes``.
    """
    if scan_roots is None:
        scan_roots = _DEFAULT_SCAN_ROOTS

    found: Dict[str, str] = {}

    for root_str in scan_roots:
        root = Path(root_str).expanduser()
        if not root.exists():
            continue

        # Root itself may be a git repo (e.g. ~/.hermes)
        if root.is_dir() and _is_git_repo(root):
            key = root.name.lower()
            found.setdefault(key, str(root))

        # Immediate children
        try:
            for child in sorted(root.iterdir()):
                if child.is_dir() and _is_git_repo(child):
                    key = child.name.lower()
                    found.setdefault(key, str(child))
        except PermissionError:
            logger.debug("repo_registry.scan: permission denied reading %s", root)

    return found


# ---------------------------------------------------------------------------
# Module-level scan cache
# ---------------------------------------------------------------------------

_scan_cache: Optional[Dict[str, str]] = None


def _get_cached_scan(scan_roots: Optional[Sequence[str]] = None) -> Dict[str, str]:
    global _scan_cache
    if _scan_cache is None:
        _scan_cache = scan_for_repos(scan_roots)
    return _scan_cache


def invalidate_scan_cache() -> None:
    """Clear the module-level scan cache.

    Useful after a new repo is cloned.  Tests should prefer ``_injected_scan``
    over calling this so they don't depend on filesystem state.
    """
    global _scan_cache
    _scan_cache = None


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------


def _fuzzy_name_matches(query: str, candidate: str) -> bool:
    """True if *query* is a suffix-segment or substring of *candidate*.

    Examples
    --------
    "agent"   matches "hermes-agent"  (suffix-segment on ``-``)
    "hermes"  matches "hermes-agent"  (prefix-segment; also substring)
    "agent"   matches "my-agent-repo" (suffix-segment)
    "oops"    does NOT match "hermes-agent"
    """
    if not query:
        return False
    # Suffix-segment match: "agent" matches "hermes-agent"
    parts = candidate.split("-")
    for i in range(len(parts)):
        if "-".join(parts[i:]) == query:
            return True
    # Substring match
    return query in candidate


# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------


class RepoRegistry:
    """Resolves a repo name or alias to a ``ResolvedRepo`` or ``UnresolvedRepo``.

    Parameters
    ----------
    overrides:
        Manual alias → ``RepoEntry`` mapping from the config ``repos:`` section.
        These always win over the auto-scan result.
    scan_roots:
        Override the default filesystem scan roots (``~/dev`` and ``~/.hermes``).
        Injected here so tests can point at a temp directory.
    _injected_scan:
        Pre-built ``{name: path}`` dict that fully replaces the filesystem scan
        (no disk access at all).  Use this in unit tests.
    """

    def __init__(
        self,
        overrides: Optional[Dict[str, RepoEntry]] = None,
        scan_roots: Optional[Sequence[str]] = None,
        _injected_scan: Optional[Dict[str, str]] = None,
    ) -> None:
        self._overrides: Dict[str, RepoEntry] = overrides or {}
        self._scan_roots = scan_roots
        self._injected_scan = _injected_scan

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan(self) -> Dict[str, str]:
        if self._injected_scan is not None:
            return self._injected_scan
        return _get_cached_scan(self._scan_roots)

    def _fuzzy(self, lower: str, scan: Dict[str, str]) -> Optional[ResolvedRepo]:
        """Return the first fuzzy match, checking overrides before scan."""
        # Overrides first (they have priority)
        for alias, entry in self._overrides.items():
            if alias.lower() == lower:
                continue  # already handled by exact step
            if _fuzzy_name_matches(lower, alias.lower()):
                return ResolvedRepo(
                    path=entry.path,
                    default_agent=entry.default_agent,
                    matched_name=alias,
                    match_kind="fuzzy",
                )
        # Then scan
        for repo_name, path in scan.items():
            if _fuzzy_name_matches(lower, repo_name):
                return ResolvedRepo(
                    path=path,
                    default_agent=DEFAULT_AGENT,
                    matched_name=repo_name,
                    match_kind="fuzzy",
                )
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, name: str) -> Union[ResolvedRepo, UnresolvedRepo]:
        """Resolve *name* to a repo entry.

        Returns ``UnresolvedRepo`` when the name cannot be resolved — the
        caller should prompt the user for clarification.
        """
        stripped = name.strip()
        lower = stripped.lower()

        # 1. Exact alias in overrides (case-sensitive alias key)
        if stripped in self._overrides:
            entry = self._overrides[stripped]
            return ResolvedRepo(
                path=entry.path,
                default_agent=entry.default_agent,
                matched_name=stripped,
                match_kind="exact",
            )

        # 2. Exact name in scan cache (case-insensitive)
        scan = self._scan()
        if lower in scan:
            return ResolvedRepo(
                path=scan[lower],
                default_agent=DEFAULT_AGENT,
                matched_name=lower,
                match_kind="exact",
            )

        # 3. Fuzzy match
        fuzzy = self._fuzzy(lower, scan)
        if fuzzy is not None:
            return fuzzy

        # 4. Literal path (absolute or home-relative that exists as a git repo)
        try:
            candidate = Path(stripped).expanduser()
            if candidate.is_absolute() and candidate.exists() and _is_git_repo(candidate):
                return ResolvedRepo(
                    path=str(candidate),
                    default_agent=DEFAULT_AGENT,
                    matched_name=stripped,
                    match_kind="literal_path",
                )
        except Exception:  # noqa: BLE001 — Path expansion on untrusted input
            pass

        # 5. Unresolved
        return UnresolvedRepo(name=stripped)


# ---------------------------------------------------------------------------
# Factory — build from raw config dict
# ---------------------------------------------------------------------------


def _parse_repo_entry(alias: str, val: Any) -> Optional[RepoEntry]:
    """Parse one repos: entry (string shorthand or dict form)."""
    if isinstance(val, str):
        path = val.strip()
        if not path:
            logger.warning("repo_registry: alias %r has empty path — skipped", alias)
            return None
        return RepoEntry(path=path, default_agent=DEFAULT_AGENT)

    if isinstance(val, dict):
        path = str(val.get("path") or "").strip()
        if not path:
            logger.warning("repo_registry: alias %r missing 'path' — skipped", alias)
            return None
        raw_agent = val.get("default_agent")
        default_agent = str(raw_agent).strip() if raw_agent else DEFAULT_AGENT
        return RepoEntry(path=path, default_agent=default_agent or DEFAULT_AGENT)

    logger.warning(
        "repo_registry: alias %r has unrecognized value type %s — skipped",
        alias, type(val).__name__,
    )
    return None


def build_repo_registry(
    repos_cfg: Optional[Dict[str, Any]] = None,
    scan_roots: Optional[Sequence[str]] = None,
    _injected_scan: Optional[Dict[str, str]] = None,
) -> RepoRegistry:
    """Build a ``RepoRegistry`` from the ``repos`` sub-dict of the config section.

    Parameters
    ----------
    repos_cfg:
        The ``repos:`` dict from the ``session_orchestration`` config section.
        Keys are aliases; values are either an absolute path string or a dict
        with ``path`` (required) and ``default_agent`` (optional, defaults to
        ``omp``).
    scan_roots:
        Override the default scan roots (injectable for tests).
    _injected_scan:
        Pre-built scan dict; bypasses all filesystem access (for unit tests).
    """
    overrides: Dict[str, RepoEntry] = {}
    if isinstance(repos_cfg, dict):
        for alias, val in repos_cfg.items():
            entry = _parse_repo_entry(alias, val)
            if entry is not None:
                overrides[alias] = entry

    return RepoRegistry(
        overrides=overrides,
        scan_roots=scan_roots,
        _injected_scan=_injected_scan,
    )
