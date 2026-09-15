"""Skill bundles — aliases that load multiple skills under one slash command.

YAML files in ``<HERMES_HOME>/skill-bundles/`` (``name``, ``description``,
``skills: [...]``, optional ``instruction``; file stem = fallback name).
``/<bundle>`` loads every member skill into one user message. If a bundle and a
skill share a slug, the bundle wins — slash dispatch checks bundles first, on purpose.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from hermes_constants import get_hermes_home
from agent.skill_commands import command_snapshot, diff_command_snapshots, resolve_slash_key, slugify_skill_name as _slugify

logger = logging.getLogger(__name__)

_bundles_cache: Dict[str, Dict[str, Any]] = {}
_bundles_cache_mtime: Optional[float] = None


def _bundles_dir() -> Path:
    """Bundles directory: ``HERMES_BUNDLES_DIR`` override (tests) or ``<HERMES_HOME>/skill-bundles``."""
    override = os.environ.get("HERMES_BUNDLES_DIR")
    return Path(override).expanduser() if override else get_hermes_home() / "skill-bundles"


def _iter_bundle_files() -> List[Path]:
    base = _bundles_dir()
    return [f for ext in ("*.yaml", "*.yml") for f in sorted(base.glob(ext))] if base.exists() else []


def _max_mtime(files: List[Path]) -> float:
    """Highest mtime across the bundle files plus the dir itself (dir mtime catches deletions)."""
    mtimes = []
    for f in (_bundles_dir(), *files):
        try:
            mtimes.append(f.stat().st_mtime)
        except OSError:
            continue
    return max(mtimes, default=0.0)


def _load_bundle_file(path: Path) -> Optional[Dict[str, Any]]:
    """Parse one bundle YAML; ``None`` (logged) on any error so a broken bundle can't break discovery."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        logger.warning("Could not read bundle %s: %s", path, exc)
        return None
    except yaml.YAMLError as exc:
        logger.warning("Invalid YAML in bundle %s: %s", path, exc)
        return None
    def _skip(reason: str) -> None:
        logger.warning("Bundle %s %s; skipping", path, reason)
    if not isinstance(data, dict):
        return _skip("is not a mapping")
    name = str(data.get("name") or path.stem).strip()
    if not name:
        return _skip("has no name")
    raw_skills = data.get("skills") or []
    if not isinstance(raw_skills, list) or not raw_skills:
        return _skip("has no skills list")
    skills = [str(s).strip() for s in raw_skills if str(s).strip()]
    if not skills:
        return _skip("has empty skills list")
    slug = _slugify(name)
    if not slug:
        return _skip("yielded empty slug")
    return {
        "name": name, "slug": slug, "skills": skills, "path": str(path),
        "description": str(data.get("description") or "").strip() or f"Load {len(skills)} skills as a bundle",
        "instruction": str(data.get("instruction") or "").strip(),
    }


def scan_bundles() -> Dict[str, Dict[str, Any]]:
    """Rebuild the ``"/slug"`` -> bundle info cache; duplicate slugs keep the first (alphabetical)."""
    global _bundles_cache, _bundles_cache_mtime
    files = _iter_bundle_files()
    out: Dict[str, Dict[str, Any]] = {}
    for f in files:
        info = _load_bundle_file(f)
        if not info:
            continue
        key = f"/{info['slug']}"
        if key in out:
            logger.warning("Duplicate bundle slug %s from %s; keeping %s", key, f, out[key]["path"])
            continue
        out[key] = info
    _bundles_cache = out
    _bundles_cache_mtime = _max_mtime(files)
    return out


def get_skill_bundles() -> Dict[str, Dict[str, Any]]:
    """Current bundle mapping; rescans only when a bundle file or the dir mtime changed."""
    current_mtime = _max_mtime(_iter_bundle_files())
    if not _bundles_cache or _bundles_cache_mtime != current_mtime:
        scan_bundles()
    return _bundles_cache


def resolve_bundle_command_key(command: str) -> Optional[str]:
    """Resolve a user-typed command to its ``/slug`` key (``_`` ≡ ``-``, as Telegram rewrites hyphens)."""
    return resolve_slash_key(command, get_skill_bundles())


def reload_bundles() -> Dict[str, Any]:
    """Re-scan and return an ``added``/``removed``/``unchanged``/``total`` diff (same shape as reload_skills)."""
    before = command_snapshot(_bundles_cache)
    return diff_command_snapshots(before, command_snapshot(scan_bundles()))


def list_bundles() -> List[Dict[str, Any]]:
    """Return a sorted list of bundle info dicts for display."""
    return sorted(get_skill_bundles().values(), key=lambda b: b["slug"])


def build_bundle_invocation_message(
    cmd_key: str, user_instruction: str = "", task_id: str | None = None, platform: str | None = None,
) -> Optional[Tuple[str, List[str], List[str]]]:
    """Build the user message for a bundle invocation: ``(message,
    loaded_skill_names, missing_skill_names)`` or ``None`` if the bundle wasn't
    found. Uninstalled members are skipped with a note; disabled ones too, since
    ``_load_skill_payload`` bypasses the scan-time filter (``platform`` scopes
    that check — gateway passes it, None resolves from env).

    Disabled skills are also skipped: bundles load members via ``_load_skill_payload`` directly, bypassing
    the scan-time disabled filter in ``get_skill_commands()``, so the disabled list must be re-applied here.
    ``platform`` scopes the check to a specific platform's ``skills.platform_disabled`` config (gateway
    dispatch passes it explicitly because the gateway handles multiple platforms in one process); when
    *None*, the platform resolves from session env vars and the global disabled list still applies. Mirrors
    the stacked-skill gate in gateway dispatch (#58888).
    """
    info = get_skill_bundles().get(cmd_key)
    if not info:
        return None
    # Late import keeps skill_bundles cheap to import (no tools/* at import time).
    from agent.skill_commands import _disabled_skill_names, _load_skill_blocks, _load_skill_payload, _scaffold_header
    bundle_name = info["name"]
    loaded_names, missing, disabled, skill_blocks = _load_skill_blocks(
        [(skill_id or "").strip() for skill_id in info["skills"]],
        lambda identifier: _load_skill_payload(identifier, task_id=task_id),
        lambda _name: f'[Loaded as part of the "{bundle_name}" skill bundle.]',
        task_id,
        disabled_names=_disabled_skill_names(platform),
    )
    if not skill_blocks:
        return None
    header = _scaffold_header(
        f'"{bundle_name}" skill bundle', loaded_names, lead_lines=[f"Bundle: {bundle_name}"], missing=missing,
        disabled=disabled, extra_instruction=info.get("instruction") or "", user_instruction=user_instruction,
    )
    return ("\n\n".join([header, *skill_blocks]), loaded_names, missing)


# File-level CRUD — used by `hermes bundles`.


def bundle_path_for(name: str) -> Path:
    """Return the canonical filesystem path for a bundle name."""
    slug = _slugify(name)
    if not slug:
        raise ValueError(f"Bundle name {name!r} normalizes to an empty slug")
    return _bundles_dir() / f"{slug}.yaml"


def save_bundle(name: str, skills: List[str], description: str = "", instruction: str = "", overwrite: bool = False) -> Path:
    """Write a bundle to disk and refresh the cache. Raises ``FileExistsError``
    if the target exists and not ``overwrite``; ``ValueError`` for unusable inputs."""
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
    payload.update({k: v for k, v in (("description", description), ("instruction", instruction)) if v})
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    scan_bundles()
    return path


def delete_bundle(name: str) -> Path:
    """Delete a bundle by name and return its path; ``FileNotFoundError`` if absent."""
    path = bundle_path_for(name)
    if not path.exists():
        raise FileNotFoundError(f"No bundle at {path}")
    path.unlink()
    scan_bundles()
    return path


def get_bundle(name: str) -> Optional[Dict[str, Any]]:
    """Look up a bundle by name (slug-normalized)."""
    return get_skill_bundles().get(f"/{_slugify(name)}")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import re  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
