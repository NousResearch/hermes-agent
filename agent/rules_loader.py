"""Markdown rule loader for Hermes Agent.

Discovers and loads rule files from two sources:

1. The active profile's ``rules/`` directory
   (``~/.hermes/profiles/<profile>/rules/``).
2. Project-level ``.hermes/rules/`` directories, searched from the
   current working directory up to the git repository root (mirroring
   how ``.hermes.md`` / ``HERMES.md`` are discovered).

Rules use YAML frontmatter to determine activation scope::

    alwaysApply: true          -> injected into system prompt on every session
    globs: ["*.vue", "*.css"] -> injected as additional context only when
                                  the agent touches a matching file
    (neither set)              -> always-on

Frontmatter schema::

    ---
    description: Short label for the rule picker
    alwaysApply: true
    globs: ["**/*.vue", "**/*.css"]
    ---

The loader walks ``rules/`` recursively and picks up ``*.md`` and
``*.mdc``. Nested directories are allowed; the relative path becomes
the rule's display name.

Resolution order:
    - Project ``.hermes/rules/`` entries nearest to cwd win first.
    - Profile ``rules/`` entries are merged in next.
    - When two rules share the same ``rel_id``, the more specific
      (nearer) project-level rule wins.
"""

from __future__ import annotations

import fnmatch
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """A single loaded rule file."""

    path: Path
    description: str
    always_apply: bool = False
    globs: list[str] = field(default_factory=list)
    body: str = ""

    @property
    def rel_id(self) -> str:
        """Stable identifier; unique across the merged rule set.

        Uses the file stem so that ``foo/bar.md`` and ``baz/bar.md``
        both resolve to ``bar`` -- the nearest one wins via precedence.
        Callers that need full disambiguation should use ``path``.
        """
        return self.path.stem

    @property
    def scope(self) -> str:
        """Return ``'project'`` if the rule lives under a ``.hermes/rules/``
        tree, otherwise ``'profile'``."""
        try:
            for parent in self.path.parents:
                if parent.name == "rules" and parent.parent.name == ".hermes":
                    return "project"
        except (OSError, ValueError):
            pass
        return "profile"


# ---------------------------------------------------------------------------
# Project rule discovery
# ---------------------------------------------------------------------------


def _find_git_root(cwd: Path) -> Optional[Path]:
    """Return the git repository root containing *cwd*, or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip()).resolve()
    except Exception:
        return None


def discover_project_rules_dirs(cwd: Optional[Path] = None) -> list[Path]:
    """Return ``.hermes/rules/`` directories from *cwd* up to the git root.

    The nearest directory is returned first. Directories are only
    considered if they actually exist.
    """
    if cwd is None:
        cwd = Path.cwd()
    try:
        cwd = cwd.resolve()
    except (OSError, ValueError):
        cwd = Path.cwd()

    stop_at = _find_git_root(cwd)
    dirs: list[Path] = []
    current = cwd

    while True:
        candidate = current / ".hermes" / "rules"
        if candidate.is_dir():
            dirs.append(candidate)
        if stop_at and current == stop_at:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
        if stop_at:
            try:
                current.relative_to(stop_at)
            except ValueError:
                break

    return dirs


# ---------------------------------------------------------------------------
# Parsing and loading
# ---------------------------------------------------------------------------


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Strip leading YAML frontmatter; return (meta, body).

    Missing or malformed frontmatter yields ``({}, full_text)`` so
    callers can treat the whole file as body.
    """
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        meta = yaml.safe_load(parts[1]) or {}
        if not isinstance(meta, dict):
            meta = {}
    except yaml.YAMLError as exc:
        logger.debug("Frontmatter parse failed: %s", exc)
        meta = {}
    return meta, parts[2].strip()


def load_rules(rules_dir: Path) -> list[Rule]:
    """Discover and load every rule under ``rules_dir``.

    Walks recursively. Accepts ``*.md`` and ``*.mdc``. Malformed files
    are skipped with a debug log so a single bad rule never breaks the
    loader.
    """
    rules: list[Rule] = []
    if not rules_dir or not rules_dir.exists():
        return rules
    for path in sorted(list(rules_dir.rglob("*.md")) + list(rules_dir.rglob("*.mdc"))):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.debug("Could not read rule %s: %s", path, exc)
            continue
        meta, body = parse_frontmatter(text)
        rules.append(
            Rule(
                path=path,
                description=str(meta.get("description", path.stem)),
                always_apply=bool(meta.get("alwaysApply", False)),
                globs=list(meta.get("globs", []) or []),
                body=body,
            )
        )
    return rules


# ---------------------------------------------------------------------------
# Partition and match
# ---------------------------------------------------------------------------


def partition_rules(rules: Iterable[Rule]) -> tuple[list[Rule], list[Rule]]:
    """Split rules into ``(always_on, glob_scoped)``.

    A rule with no globs is treated as always-on. A rule with globs
    but ``alwaysApply: true`` lands in both buckets -- always injected
    AND eligible for glob match.
    """
    always_on: list[Rule] = []
    glob_scoped: list[Rule] = []
    for rule in rules:
        if rule.always_apply or not rule.globs:
            always_on.append(rule)
        if rule.globs:
            glob_scoped.append(rule)
    return always_on, glob_scoped


def match_glob_rules(
    rules: Iterable[Rule], touched_paths: Iterable[str]
) -> list[Rule]:
    """Return glob-scoped rules whose patterns match any touched path."""
    touched = [str(p) for p in touched_paths if p]
    if not touched:
        return []
    matched: list[Rule] = []
    for rule in rules:
        for pattern in rule.globs:
            if any(fnmatch.fnmatch(p, pattern) for p in touched):
                matched.append(rule)
                break
    return matched


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_rules_for_prompt(
    rules: Iterable[Rule],
    header: str = "## Project Rules",
) -> str:
    """Render rules into a single system-prompt section."""
    rules = list(rules)
    if not rules:
        return ""
    lines = [header, ""]
    for rule in rules:
        lines.append(f"### {rule.rel_id}")
        if rule.description:
            lines.append(f"_{rule.description}_")
        lines.append("")
        lines.append(rule.body)
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_rules_dir(profile_dir: Path) -> Path:
    """Return the canonical rules directory for a profile."""
    return profile_dir / "rules"


def load_active_rules(profile_dir: Path, cwd: Optional[Path] = None) -> list[Rule]:
    """Load all rules that apply to the current session.

    Rules are loaded from:

    1. Every ``.hermes/rules/`` directory from *cwd* up to the git root
       (nearest first).
    2. The profile's own ``rules/`` directory.

    When two rules share the same ``rel_id``, project rules closer to
    *cwd* take precedence over more distant ones, and any project rule
    takes precedence over a profile rule. The final list preserves the
    nearest-first order for project rules, followed by profile rules.
    """
    seen: set[str] = set()
    rules: list[Rule] = []

    for project_dir in discover_project_rules_dirs(cwd):
        for rule in load_rules(project_dir):
            if rule.rel_id in seen:
                continue
            seen.add(rule.rel_id)
            rules.append(rule)

    for profile_rule in load_rules(resolve_rules_dir(profile_dir)):
        if profile_rule.rel_id in seen:
            continue
        seen.add(profile_rule.rel_id)
        rules.append(profile_rule)

    return rules
