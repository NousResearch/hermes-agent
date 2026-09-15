"""Skill-pin hygiene for kanban cards (``tasks.skills``).

A card's ``skills`` list is force-loaded into the dispatched worker as repeated
``--skills`` pairs and resolved in the ASSIGNEE's home. A pin that matches no
installed skill is a live landmine: before this module an all-dead pin set
aborted the worker at agent init — ``Error: Unknown skill(s): seo, marketing`` —
so the card crashed on every dispatch and never did any work.

This module owns both ends of that bug:

* write time — :func:`unresolved_skill_pins` matches each pin against the
  installed catalog (exact name, frontmatter name, ``category/name`` path, or
  leaf basename) so a dead pin is flagged when the card is written instead of
  when it is dispatched;
* dispatch time — :func:`is_kanban_worker_context` is the gate the preload path
  uses to keep failing loudly for a human typo while never aborting a
  dispatcher-owned worker, and :func:`record_skill_pin_unresolved` turns that
  skip into a durable ``skill_pin_unresolved`` task event on the card (visible
  in ``hermes kanban show``).

The write-time flag is a warning, never a write refusal: category labels
(``research``, ``devops``, ``creative``) are common on existing fleet cards, and
rejecting them would break card creation rather than teach anyone anything. The
authoritative gate stays at dispatch time, which resolves against the worker's
real home and degrades safely.

Boundary: a ``plugin:skill`` pin is not locally verifiable, so it is never
flagged here. Catalog roots are the active home's skills dir, the assignee
profile's skills dir, the active config's project/external skill dirs, and the
shared ``~/.hermes/skills`` root when the active home is profile-scoped. A pin
that lives only in some other profile's config-declared external dir is
therefore flagged (a false positive) — the warning is advisory, and the worker
side handles it without failing.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# task_events kind recorded when a card's pin cannot be loaded. Task-scoped
# (no run_id) so the flag survives the attempt that produced it.
SKILL_PIN_UNRESOLVED = "skill_pin_unresolved"

_CATALOG_FILENAME = "SKILL.md"
# Never worth descending into, and ``_archived``/``.archive`` hold retired
# skills that must NOT keep a dead pin looking resolvable.
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "__pycache__", "node_modules",
    "site-packages", ".worktrees", "_archived", ".archive",
})
_MAX_DEPTH = 5
_FRONTMATTER_HEAD_BYTES = 4096
_NAME_RE = re.compile(r"^name:\s*(.+?)\s*$", re.MULTILINE)

# Root-identity+mtimes -> catalog. Keyed on each root's mtime so installing or
# removing a skill invalidates it without a TTL; the catalog is a directory
# listing plus a few KB of frontmatter, so a rescan is cheap but not free.
_CATALOG_CACHE: dict[tuple, frozenset[str]] = {}


def clear_skill_catalog_cache() -> None:
    """Drop the cached catalog (tests install fixtures mid-run)."""
    _CATALOG_CACHE.clear()


def _slugify(name: str) -> str:
    """``ASCII Art`` -> ``ascii-art``, matching the ``/command`` slug form."""
    return re.sub(r"-{2,}", "-", re.sub(r"[^\w-]", "", name.lower().replace(" ", "-").replace("_", "-"))).strip("-")


def _existing_dir(path: Optional[Path]) -> Optional[Path]:
    try:
        return path if path is not None and Path(path).is_dir() else None
    except OSError:
        return None


def _active_home_is_profile() -> bool:
    """True when HERMES_HOME points at ``<root>/profiles/<name>``.

    Profile homes are the ones that delegate to the shared ``<root>/skills``
    tree (the standard ``skills.external_dirs`` wiring), while a root-level home
    IS that tree — resolving it twice would just scan the same dir.
    """
    try:
        from hermes_constants import get_hermes_home

        return "profiles" in get_hermes_home().parts
    except Exception:
        return False


def candidate_skill_roots(assignee: Optional[str] = None) -> list[Path]:
    """Skill dirs a pin could resolve from, in runtime precedence order.

    ``assignee`` adds that profile's own skills dir: the worker resolves its
    pins in the ASSIGNEE's home, not in the creating profile's.
    """
    from hermes_constants import get_default_hermes_root, get_skills_dir

    roots: list[Path] = []
    seen: set[str] = set()

    def add(path: Optional[Path]) -> None:
        resolved = _existing_dir(path)
        if resolved is None:
            return
        # realpath dedup: profile homes commonly symlink the shared tree.
        key = os.path.realpath(resolved)
        if key in seen:
            return
        seen.add(key)
        roots.append(resolved)

    add(get_skills_dir())
    if _active_home_is_profile():
        add(get_default_hermes_root() / "skills")

    if assignee:
        try:
            from hermes_cli.profiles import get_profile_dir

            add(get_profile_dir(str(assignee)) / "skills")
        except Exception as exc:
            logger.debug("skill catalog: assignee %r skills dir unresolved (%s)", assignee, exc)

    try:
        from agent.skill_utils import get_external_skills_dirs, get_project_skills_dirs

        for getter in (get_project_skills_dirs, get_external_skills_dirs):
            try:
                for path in getter():
                    add(Path(path))
            except Exception as exc:  # a bad config entry must not break the check
                logger.debug("skill catalog: %s failed (%s)", getattr(getter, "__name__", getter), exc)
    except Exception as exc:
        logger.debug("skill catalog: skill dir helpers unavailable (%s)", exc)

    return roots


def _frontmatter_name(skill_md: Path) -> Optional[str]:
    """``name:`` from a SKILL.md frontmatter block, or None (best-effort)."""
    try:
        with skill_md.open("r", encoding="utf-8", errors="replace") as handle:
            head = handle.read(_FRONTMATTER_HEAD_BYTES)
    except OSError:
        return None
    if not head.startswith("---"):
        return None
    end = head.find("\n---", 3)
    block = head[3:end] if end != -1 else head[3:]
    match = _NAME_RE.search(block)
    if not match:
        return None
    return match.group(1).strip().strip("\"'") or None


def _scan_root_names(root: Path) -> set[str]:
    """Every identifier a pin may use to reach a skill under ``root``."""
    names: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        current = Path(dirpath)
        if len(current.relative_to(root).parts) >= _MAX_DEPTH:
            dirnames[:] = []
        if _CATALOG_FILENAME not in filenames:
            continue
        names.add(current.name)  # leaf basename
        names.add(current.relative_to(root).as_posix())  # category/name path
        frontmatter_name = _frontmatter_name(current / _CATALOG_FILENAME)
        if frontmatter_name:
            names.add(frontmatter_name)
            slug = _slugify(frontmatter_name)
            if slug:
                names.add(slug)
    return names


def skill_name_catalog(*, assignee: Optional[str] = None) -> frozenset[str]:
    """Identifiers of every skill installed for ``assignee`` (cached per root set)."""
    roots = candidate_skill_roots(assignee)
    key = tuple((str(root), _mtime(root)) for root in roots)
    cached = _CATALOG_CACHE.get(key)
    if cached is not None:
        return cached
    names: set[str] = set()
    for root in roots:
        names |= _scan_root_names(root)
    frozen = frozenset(names)
    # One entry per live root set: the key self-invalidates, so a stale catalog
    # can never be served (bounded growth, no TTL bookkeeping).
    _CATALOG_CACHE.clear()
    _CATALOG_CACHE[key] = frozen
    return frozen


def _mtime(path: Path) -> int:
    # Nanoseconds: a skill installed in the same second as the lookup must still
    # invalidate the key (second-granularity mtime froze the catalog in tests).
    try:
        return int(path.stat().st_mtime_ns)
    except OSError:
        return 0


def unresolved_skill_pins(skills: Optional[Iterable[str]], *, assignee: Optional[str] = None) -> list[str]:
    """Pins in ``skills`` that match no installed skill, in the order given.

    Empty when everything resolves (or nothing was pinned). ``plugin:skill``
    identifiers are skipped — plugin skills are not in the on-disk catalog.
    """
    pins = [str(s).strip() for s in (skills or ()) if s and str(s).strip()]
    if not pins:
        return []
    catalog = skill_name_catalog(assignee=assignee)
    return [pin for pin in pins if ":" not in pin and pin not in catalog]


def pin_warning_text(unresolved: Iterable[str]) -> str:
    """Shared operator-facing wording for a flagged pin list."""
    names = ", ".join(unresolved)
    return (
        f"Skill pin(s) resolve to no installed skill: {names}. "
        "The worker runs without them (a 'skill_pin_unresolved' task event is recorded). "
        "Re-point or drop the pin; `hermes skills list` shows what is installed."
    )


def is_kanban_worker_context() -> bool:
    """True inside a dispatcher-owned worker run (the fail-soft gate).

    A pin that resolves to nothing is a card-authoring mistake, so an
    interactive ``hermes -s <typo>`` still fails loudly — but a dispatched
    worker must never die for it: the card would crash on every attempt and no
    amount of retrying can fix a bad ``skills`` column.
    """
    if not os.environ.get("HERMES_KANBAN_TASK"):
        return False
    try:
        from agent.delegation_context import is_dispatcher_owned_worker_context

        return bool(is_dispatcher_owned_worker_context())
    except Exception:
        return True


def handle_unresolvable_pins(missing: Iterable[str], loaded: Iterable[str] = ()) -> bool:
    """Worker-side handling for pins that resolved to nothing.

    Returns True when the caller must CONTINUE without them — a dispatcher-owned
    worker, where the pins are logged and recorded on the card as a
    ``skill_pin_unresolved`` event. Returns False on every other surface (a
    human typo in ``hermes -s``), leaving the caller's fail-loud contract intact.
    """
    if not is_kanban_worker_context():
        return False
    missing_names = [str(n) for n in missing if n]
    loaded_names = [str(n) for n in loaded if n]
    if loaded_names:
        logger.warning(
            "Skill pin(s) resolve to no installed skill: %s. Continuing with: %s. "
            "The card is flagged with a '%s' task event.",
            ", ".join(missing_names), ", ".join(loaded_names), SKILL_PIN_UNRESOLVED,
        )
    else:
        logger.warning(
            "Skill pin(s) resolve to no installed skill: %s. Running WITHOUT preloaded "
            "skills; the card is flagged with a '%s' task event.",
            ", ".join(missing_names), SKILL_PIN_UNRESOLVED,
        )
    record_skill_pin_unresolved(missing_names, loaded_names, phase="preload")
    return True


def record_skill_pin_unresolved(
    missing: Iterable[str],
    loaded: Iterable[str] = (),
    *,
    task_id: Optional[str] = None,
    board: Optional[str] = None,
    phase: str = "preload",
) -> bool:
    """Append a ``skill_pin_unresolved`` event to the card; never raises.

    Best-effort by design: this runs on the worker's startup path, so a missing
    board, a lock or a bad slug must not turn the fail-soft into the crash it
    replaces. Needs a known board location (``HERMES_KANBAN_DB`` /
    ``HERMES_KANBAN_BOARD`` env or explicit args) — guessing the default board
    would create one as a side effect of a warning.
    """
    names = [str(n) for n in missing if n]
    if not names:
        return False
    tid = task_id or os.environ.get("HERMES_KANBAN_TASK")
    if not tid:
        return False
    board = board or os.environ.get("HERMES_KANBAN_BOARD") or None
    db_env = os.environ.get("HERMES_KANBAN_DB")
    if not board and not db_env:
        logger.debug("skill pin flag skipped: no board location in env")
        return False
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        payload = {
            "skills": names,
            "loaded": [str(n) for n in loaded if n],
            "phase": phase,
            "hint": "pin matched no installed skill in the worker's home",
        }
        run_id = os.environ.get("HERMES_KANBAN_RUN_ID")
        # HERMES_KANBAN_DB pins the exact board the dispatcher spawned us on;
        # only fall back to the slug when it is absent.
        with kbc.connect_closing(board=None if db_env else board) as conn:
            with kb.write_txn(conn):
                kb._append_event(
                    conn, str(tid), SKILL_PIN_UNRESOLVED, payload,
                    run_id=int(run_id) if (run_id or "").isdigit() else None,
                )
        return True
    except Exception as exc:
        logger.debug("skill pin flag skipped (%s)", exc)
        return False
