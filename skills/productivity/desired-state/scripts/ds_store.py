"""
ds_store.py — CRUD over the desired-state artifact store.

The store is plain files at <HERMES_HOME>/state/desired/<domain>/<slug>.md,
one artifact per goal. Files are the source of truth: git-friendly,
hand-editable, and readable by the memory/notes tooling. This module owns
create / read / list / update / set-current / archive, stamping created_at
and updated_at via an injectable clock.

Not run directly — imported by ds.py (the CLI) and the tests.
Stdlib-only. Python 3.11+.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from _common import (
    GoalDoc,
    _as_number,
    desired_root,
    goal_path,
    iso_now,
    slugify,
)


class GoalNotFoundError(FileNotFoundError):
    """Raised when a (domain, slug) does not resolve to an artifact."""


class GoalExistsError(FileExistsError):
    """Raised when creating a goal that already exists without overwrite."""


def _root(root: Path | None) -> Path:
    return root if root is not None else desired_root()


def list_goals(*, root: Path | None = None, domain: str | None = None) -> list[GoalDoc]:
    """Load every goal (optionally filtered to one domain), sorted stably.

    Malformed artifacts are skipped rather than crashing the listing; a
    hand-edited file with bad frontmatter should not blind the whole report.
    """
    base = _root(root)
    if not base.exists():
        return []
    docs: list[GoalDoc] = []
    domains = [base / slugify(domain)] if domain else sorted(p for p in base.iterdir() if p.is_dir())
    for dom_dir in domains:
        if not dom_dir.is_dir():
            continue
        for md in sorted(dom_dir.glob("*.md")):
            try:
                docs.append(GoalDoc.from_file(md))
            except (ValueError, OSError, TypeError):
                continue
    docs.sort(key=lambda d: (d.domain, d.slug))
    return docs


def get_goal(domain: str, slug: str, *, root: Path | None = None) -> GoalDoc:
    path = goal_path(domain, slug, root=_root(root))
    if not path.exists():
        raise GoalNotFoundError(f"no goal at {domain}/{slugify(slug)}")
    return GoalDoc.from_file(path)


def create_goal(
    doc: GoalDoc,
    *,
    root: Path | None = None,
    overwrite: bool = False,
    now: datetime | None = None,
) -> GoalDoc:
    """Write a new goal artifact. Raises GoalExistsError unless `overwrite`.

    Returns the persisted doc (with `path`, `created_at`, `updated_at` set).
    Validation problems raise ValueError before anything touches disk.
    """
    problems = doc.validate()
    if problems:
        raise ValueError("; ".join(problems))
    doc.lock_direction()
    path = goal_path(doc.domain, doc.goal if doc.path is None else doc.path.stem, root=_root(root))
    if path.exists() and not overwrite:
        raise GoalExistsError(f"goal already exists at {path}; pass overwrite=True to replace")
    stamp = iso_now(now)
    doc.created_at = doc.created_at or stamp
    doc.updated_at = stamp
    doc.path = path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc.to_text(), encoding="utf-8")
    return doc


def update_goal(
    domain: str,
    slug: str,
    changes: dict[str, object],
    *,
    root: Path | None = None,
    now: datetime | None = None,
) -> GoalDoc:
    """Apply `changes` to an existing goal's frontmatter fields and persist.

    Only known GoalDoc fields are accepted; unknown keys raise KeyError so a
    typo never silently no-ops. Re-stamps updated_at and re-validates.
    """
    doc = get_goal(domain, slug, root=root)
    for key, value in changes.items():
        if key not in GoalDoc._FIELD_ORDER and key != "body":
            raise KeyError(f"unknown goal field: {key!r}")
        setattr(doc, key, value)
    problems = doc.validate()
    if problems:
        raise ValueError("; ".join(problems))
    # Freeze direction once tracking supplies the first reference value, so a
    # target-only decrease goal stays stable across later crossings.
    doc.lock_direction()
    doc.updated_at = iso_now(now)
    assert doc.path is not None
    doc.path.write_text(doc.to_text(), encoding="utf-8")
    return doc


def set_current(
    domain: str,
    slug: str,
    value: float | int | str,
    *,
    root: Path | None = None,
    now: datetime | None = None,
) -> GoalDoc:
    """Record a new measured current_value (the common tracking action).

    If the value meets or passes the target in the goal's direction, the
    status is NOT auto-flipped to achieved — that stays a human/agent
    decision — but `gap` will report pace="met" so it surfaces for closing.
    """
    num = _as_number(value)
    return update_goal(
        domain,
        slug,
        {"current_value": num if num is not None else value},
        root=root,
        now=now,
    )


def archive_goal(
    domain: str,
    slug: str,
    *,
    status: str = "dropped",
    root: Path | None = None,
    now: datetime | None = None,
) -> GoalDoc:
    """Soft-close a goal by setting a terminal status. Never deletes the file."""
    if status not in ("achieved", "dropped", "paused"):
        raise ValueError("archive status must be achieved | dropped | paused")
    return update_goal(domain, slug, {"status": status}, root=root, now=now)
