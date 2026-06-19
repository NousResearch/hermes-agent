"""Skill bundles — aliases that load multiple skills under one slash command.

A skill bundle is a small YAML file that names a set of skills to load
together. Invoking ``/<bundle-name>`` from the CLI or gateway loads every
referenced skill's full content into a single user message, the same way
``/<skill-name>`` does — but for N skills at once.

Storage
-------
Bundles live in ``~/.hermes/skill-bundles/*.yaml`` (and the equivalent
profile-aware directory under ``HERMES_HOME``). Each file looks like::

    name: backend-dev
    description: Backend feature work — code review, testing, PR workflow.
    skills:
      - github-code-review
      - test-driven-development
      - github-pr-workflow
    instruction: |
      Optional extra guidance to inject above the skill bodies.

The file's stem is treated as a fallback name when ``name:`` is absent, so
dropping a YAML into the directory is enough to register a new bundle.

Conflict resolution
-------------------
If a bundle and a skill share the same slash name, the bundle wins. The
slash command dispatch checks bundles first, then falls back to skills.
This is the intended behavior — a user who names a bundle ``research``
explicitly wants ``/research`` to mean their bundle, not whatever skill
happens to share the slug.

Public API
----------
- :func:`get_skill_bundles` — return ``{"/slug": bundle_info}``
- :func:`resolve_bundle_command_key` — map a user-typed command to its slug
- :func:`build_bundle_invocation_message` — produce the full user message
- :func:`reload_bundles` — re-scan disk and return a diff
- :func:`list_bundles` — return rich info for display (``hermes bundles``)
- :func:`save_bundle` / :func:`delete_bundle` — file-level operations
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Slug normalization — matches agent/skill_commands.py so a bundle and a
# skill called "Foo Bar" both resolve to "/foo-bar".
_BUNDLE_INVALID_CHARS = re.compile(r"[^a-z0-9-]")
_BUNDLE_MULTI_HYPHEN = re.compile(r"-{2,}")

_bundles_cache: Dict[str, Dict[str, Any]] = {}
_bundles_cache_mtime: Optional[float] = None


def _bundles_dir() -> Path:
    """Return the canonical bundles directory under HERMES_HOME.

    Honors ``HERMES_BUNDLES_DIR`` for tests; falls back to
    ``<HERMES_HOME>/skill-bundles``.
    """
    override = os.environ.get("HERMES_BUNDLES_DIR")
    if override:
        return Path(override).expanduser()
    return get_hermes_home() / "skill-bundles"


def _slugify(name: str) -> str:
    cmd = name.lower().replace(" ", "-").replace("_", "-")
    cmd = _BUNDLE_INVALID_CHARS.sub("", cmd)
    cmd = _BUNDLE_MULTI_HYPHEN.sub("-", cmd).strip("-")
    return cmd


def _iter_bundle_files() -> List[Path]:
    base = _bundles_dir()
    if not base.exists():
        return []
    files: List[Path] = []
    for ext in ("*.yaml", "*.yml"):
        files.extend(sorted(base.glob(ext)))
    return files


def _max_mtime(files: List[Path]) -> float:
    """Highest mtime across the bundle files plus the dir itself.

    Watching the directory mtime catches deletions; watching individual
    files catches edits. Together they're a cheap freshness check.
    """
    base = _bundles_dir()
    mtimes = []
    if base.exists():
        try:
            mtimes.append(base.stat().st_mtime)
        except OSError:
            pass
    for f in files:
        try:
            mtimes.append(f.stat().st_mtime)
        except OSError:
            continue
    return max(mtimes) if mtimes else 0.0


def _load_bundle_file(path: Path) -> Optional[Dict[str, Any]]:
    """Parse a single bundle YAML file. Returns ``None`` on any error.

    Errors are logged at WARNING level. We don't raise — a broken bundle
    shouldn't take down slash command discovery.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read bundle %s: %s", path, exc)
        return None
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        logger.warning("Invalid YAML in bundle %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("Bundle %s is not a mapping; skipping", path)
        return None

    name = str(data.get("name") or path.stem).strip()
    if not name:
        logger.warning("Bundle %s has no name; skipping", path)
        return None

    skills = data.get("skills") or []
    if not isinstance(skills, list) or not skills:
        logger.warning("Bundle %s has no skills list; skipping", path)
        return None
    skills = [str(s).strip() for s in skills if str(s).strip()]
    if not skills:
        logger.warning("Bundle %s has empty skills list; skipping", path)
        return None

    description = str(data.get("description") or "").strip()
    instruction = str(data.get("instruction") or "").strip()

    slug = _slugify(name)
    if not slug:
        logger.warning("Bundle %s yielded empty slug; skipping", path)
        return None

    return {
        "name": name,
        "slug": slug,
        "description": description or f"Load {len(skills)} skills as a bundle",
        "skills": skills,
        "instruction": instruction,
        "path": str(path),
    }


def scan_bundles() -> Dict[str, Dict[str, Any]]:
    """Scan the bundles directory and rebuild the cache.

    Returns the same mapping as :func:`get_skill_bundles` — ``"/slug"`` →
    bundle info dict. Later bundles with a duplicate slug are skipped with
    a warning (first wins, alphabetical order).
    """
    global _bundles_cache, _bundles_cache_mtime
    files = _iter_bundle_files()
    out: Dict[str, Dict[str, Any]] = {}
    for f in files:
        info = _load_bundle_file(f)
        if not info:
            continue
        key = f"/{info['slug']}"
        if key in out:
            logger.warning(
                "Duplicate bundle slug %s from %s; keeping %s",
                key, f, out[key]["path"],
            )
            continue
        out[key] = info
    _bundles_cache = out
    _bundles_cache_mtime = _max_mtime(files)
    return out


def get_skill_bundles() -> Dict[str, Dict[str, Any]]:
    """Return the current bundle mapping, rescanning when disk changed.

    Cheap to call repeatedly: only rescans when the bundles directory or
    any bundle file's mtime is newer than the cached snapshot.
    """
    files = _iter_bundle_files()
    current_mtime = _max_mtime(files)
    if not _bundles_cache or _bundles_cache_mtime != current_mtime:
        scan_bundles()
    return _bundles_cache


def resolve_bundle_command_key(command: str) -> Optional[str]:
    """Resolve a user-typed command to its canonical bundle slash key.

    Hyphens and underscores are treated interchangeably to mirror the
    skill-command behavior (Telegram converts hyphens to underscores in
    bot command names).
    """
    if not command:
        return None
    cmd_key = f"/{command.replace('_', '-')}"
    return cmd_key if cmd_key in get_skill_bundles() else None


def reload_bundles() -> Dict[str, Any]:
    """Re-scan the bundles directory and return a diff.

    Mirrors :func:`agent.skill_commands.reload_skills` so callers can use
    the same display logic. Returns a dict with ``added``, ``removed``,
    ``unchanged``, and ``total`` keys.
    """
    def _snapshot(cmds: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        return {k.lstrip("/"): (v or {}).get("description", "") for k, v in cmds.items()}

    before = _snapshot(_bundles_cache)
    new = scan_bundles()
    after = _snapshot(new)

    added_names = sorted(set(after) - set(before))
    removed_names = sorted(set(before) - set(after))
    unchanged = sorted(set(after) & set(before))

    return {
        "added": [{"name": n, "description": after[n]} for n in added_names],
        "removed": [{"name": n, "description": before[n]} for n in removed_names],
        "unchanged": unchanged,
        "total": len(after),
    }


def list_bundles() -> List[Dict[str, Any]]:
    """Return a sorted list of bundle info dicts for display."""
    bundles = get_skill_bundles()
    return sorted(bundles.values(), key=lambda b: b["slug"])


def build_bundle_invocation_message(
    cmd_key: str,
    user_instruction: str = "",
    task_id: str | None = None,
) -> Optional[Tuple[str, List[str], List[str]]]:
    """Build the user message content for a bundle slash command invocation.

    Returns ``(message, loaded_skill_names, missing_skill_names)`` or
    ``None`` if the bundle wasn't found.

    A bundle that references skills the user doesn't have installed still
    loads — the agent gets a note about which ones were skipped. This is
    the same forgiving stance ``build_preloaded_skills_prompt`` uses for
    ``-s`` CLI preloading.
    """
    bundles = get_skill_bundles()
    info = bundles.get(cmd_key)
    if not info:
        return None

    # Late import to avoid pulling tools/* at module import time and to
    # keep skill_bundles cheap to import in test environments.
    from agent.skill_commands import _load_skill_payload, _build_skill_message

    loaded_names: List[str] = []
    missing: List[str] = []
    skill_blocks: List[str] = []
    seen: set[str] = set()

    bundle_name = info["name"]
    skills = info["skills"]
    extra_instruction = info.get("instruction") or ""

    for skill_id in skills:
        identifier = (skill_id or "").strip()
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)

        loaded = _load_skill_payload(identifier, task_id=task_id)
        if not loaded:
            missing.append(identifier)
            continue
        loaded_skill, skill_dir, skill_name = loaded

        try:
            from tools.skill_usage import bump_use
            bump_use(skill_name)
        except Exception:
            pass

        activation_note = (
            f'[Loaded as part of the "{bundle_name}" skill bundle.]'
        )
        skill_blocks.append(
            _build_skill_message(
                loaded_skill,
                skill_dir,
                activation_note,
                session_id=task_id,
            )
        )
        loaded_names.append(skill_name)

    if not skill_blocks:
        return None

    # Header — tells the agent this is a bundle, lists the skills, and
    # provides any author-supplied instruction.
    header_lines = [
        f'[IMPORTANT: The user has invoked the "{bundle_name}" skill bundle, '
        f"loading {len(loaded_names)} skills together. Treat every skill below "
        "as active guidance for this turn.]",
        "",
        f"Bundle: {bundle_name}",
        f"Skills loaded: {', '.join(loaded_names)}",
    ]
    if missing:
        header_lines.append(f"Skills missing (skipped): {', '.join(missing)}")
    if extra_instruction:
        header_lines.extend(["", f"Bundle instruction: {extra_instruction}"])
    if user_instruction:
        header_lines.extend(
            ["", f"User instruction: {user_instruction}"]
        )

    header = "\n".join(header_lines)
    return ("\n\n".join([header, *skill_blocks]), loaded_names, missing)


# ---------------------------------------------------------------------------
# File-level CRUD helpers — used by `hermes bundles` CLI subcommand.
# ---------------------------------------------------------------------------


def bundle_path_for(name: str) -> Path:
    """Return the canonical filesystem path for a bundle name."""
    slug = _slugify(name)
    if not slug:
        raise ValueError(f"Bundle name {name!r} normalizes to an empty slug")
    return _bundles_dir() / f"{slug}.yaml"


def save_bundle(
    name: str,
    skills: List[str],
    description: str = "",
    instruction: str = "",
    overwrite: bool = False,
) -> Path:
    """Write a bundle to disk and invalidate the cache.

    Raises ``FileExistsError`` if the target exists and ``overwrite`` is
    False. Raises ``ValueError`` if the inputs are unusable.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("Bundle name is required")
    cleaned_skills = [str(s).strip() for s in skills if str(s).strip()]
    if not cleaned_skills:
        raise ValueError("Bundle must reference at least one skill")

    path = bundle_path_for(name)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Bundle already exists at {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"name": name, "skills": cleaned_skills}
    if description:
        payload["description"] = description
    if instruction:
        payload["instruction"] = instruction

    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    scan_bundles()  # refresh cache
    return path


def delete_bundle(name: str) -> Path:
    """Delete a bundle by name. Returns the deleted path.

    Raises ``FileNotFoundError`` if the bundle doesn't exist.
    """
    path = bundle_path_for(name)
    if not path.exists():
        raise FileNotFoundError(f"No bundle at {path}")
    path.unlink()
    scan_bundles()
    return path


def get_bundle(name: str) -> Optional[Dict[str, Any]]:
    """Look up a bundle by name (slug-normalized)."""
    slug = _slugify(name)
    return get_skill_bundles().get(f"/{slug}")
