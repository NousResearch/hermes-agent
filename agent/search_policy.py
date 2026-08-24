"""Shared directory pruning policy for broad recursive scans.

These names identify version-control internals, dependency trees, generated
artifacts, caches, and backup copies that are not useful results for broad
agent-facing discovery. Ordinary search callers may still target an explicit
path; broad diagnostic probes should apply this policy to recursive walks.
"""

from __future__ import annotations


# Keep this policy conservative and name-based so it works for local and remote
# shell backends alike. The same set is used by context discovery and search
# probes; adding a directory here protects every broad recursive consumer.
SEARCH_PRUNE_DIR_NAMES = frozenset({
    # Version-control internals.
    ".git", ".hg", ".svn",
    # Dependency and vendored trees.
    "node_modules", "venv", ".venv", "site-packages", "dist-packages",
    "vendor", "third_party",
    # Generated/build output.
    "build", "dist", "target", "out", "coverage",
    ".next", ".turbo", ".parcel-cache", ".nuxt", ".svelte-kit",
    # Python and package-manager caches.
    "__pycache__", ".cache", ".Trash", ".tox", ".nox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".npm", ".yarn", ".pnpm-store",
    ".gradle", ".m2", ".nuget",
    # Backup copies.
    "backups", "backup", ".backups",
})
