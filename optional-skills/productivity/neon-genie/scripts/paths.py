#!/usr/bin/env python3
"""Resolve skill paths for full install and Hermes hub thin layout.

Hermes Hub only installs files under allowlisted dirs
(references/, templates/, scripts/, assets/, examples/) that are
explicitly path-referenced in SKILL.md. Full installs also keep
schemas/, profiles/, evals/, VERSION, and manifest.json at skill root.

Resolution prefers the full-tree location when present, then hub mirrors.
"""

from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent


def first_existing(*candidates: Path) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def require_existing(*candidates: Path) -> Path:
    found = first_existing(*candidates)
    if found is None:
        tried = ", ".join(str(p.relative_to(SKILL_ROOT)) for p in candidates)
        raise FileNotFoundError(f"missing required path; tried: {tried}")
    return found


def schemas_dir() -> Path:
    return require_existing(
        SKILL_ROOT / "schemas",
        SKILL_ROOT / "references" / "schemas",
    )


def profiles_dir() -> Path:
    return require_existing(
        SKILL_ROOT / "profiles",
        SKILL_ROOT / "references" / "profiles",
    )


def evals_dir() -> Path:
    """Prefer a layout that actually contains cases/ (hub may only mirror examples/evals)."""
    candidates = (
        SKILL_ROOT / "evals",
        SKILL_ROOT / "examples" / "evals",
    )
    for path in candidates:
        if (path / "cases").is_dir() or (path / "transcripts").is_dir() or (
            path / "behavioral"
        ).is_dir():
            # Prefer the candidate that has cases when both exist
            if (path / "cases").is_dir():
                return path
    for path in candidates:
        if path.is_dir():
            # If only incomplete evals/ exists, fall through to examples/evals
            if path.name == "evals" and path.parent == SKILL_ROOT:
                continue
            return path
    # Final: any existing, else raise
    return require_existing(*candidates)


def manifest_path() -> Path:
    return require_existing(
        SKILL_ROOT / "manifest.json",
        SKILL_ROOT / "references" / "manifest.json",
    )


def version_path() -> Path:
    return require_existing(
        SKILL_ROOT / "VERSION",
        SKILL_ROOT / "references" / "VERSION",
    )


def is_hub_layout() -> bool:
    """True when only hub-mirror layout is present (no root schemas/)."""
    has_mirror = (SKILL_ROOT / "references" / "schemas").is_dir()
    has_full = (SKILL_ROOT / "schemas").is_dir()
    return has_mirror and not has_full


def schema_file(name: str) -> Path:
    return schemas_dir() / name


def profile_file(name: str) -> Path:
    stem = name if name.endswith(".md") else f"{name}.md"
    return profiles_dir() / stem
