"""LOCAL GSSAI profile-manager dashboard plugin.

Mounted at /api/plugins/profile-manager/ by the dashboard plugin system.

This first surface is intentionally read-only and secret-free: it reports file
presence, size, mode, and coarse lifecycle/boundary labels for known local GSSAI
Hermes profiles. It never returns .env contents, token values, full process
arguments, or log contents.
"""

from __future__ import annotations

import os
import pwd
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter()

ACTIVE_LOCAL_PROFILES: tuple[str, ...] = (
    "ivan_bb",
    "storm",
    "neo",
    "gssai_admin",
    "assettiger",
)

ARCHIVED_LOCAL_PROFILES: tuple[str, ...] = (
    "business",
    "hr",
    "mailguard",
    "planner",
    "research",
)

PROFILE_BOUNDARIES: dict[str, str] = {
    "ivan_bb": "LOCAL GSSAI PROJECT — CEO/orchestrator",
    "storm": "LOCAL GSSAI PROJECT — CCO / communications",
    "neo": "LOCAL GSSAI PROJECT — CTO / technical escalation",
    "gssai_admin": "LOCAL GSSAI PROJECT — office/admin coordination",
    "assettiger": "LOCAL GSSAI PROJECT — AssetTiger/C-Track monitor/report-only",
}

EXCLUDED_NAME_FRAGMENTS: tuple[str, ...] = (
    "aria",
    "lilspa",
    "lil_pa",
    "lils-pa",
    "vps",
    "openclaw",
)

SECRET_FILE_NAMES: frozenset[str] = frozenset({".env", "auth.json"})


def _real_user_home() -> Path:
    """Return the OS user's real home, ignoring profile-virtualized HOME."""

    try:
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:
        return Path.home()


def _hermes_root() -> Path:
    override = os.environ.get("PROFILE_MANAGER_HERMES_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (_real_user_home() / ".hermes").resolve()


def _safe_stat(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {
        "exists": True,
        "size": stat.st_size,
        "mode": oct(stat.st_mode & 0o777),
        "mtime": stat.st_mtime,
    }


def _profile_lifecycle(profile_path: Path, profiles_root: Path) -> str:
    try:
        rel_parts = profile_path.resolve().relative_to(profiles_root.resolve()).parts
    except ValueError:
        rel_parts = profile_path.parts
    if "_archived" in rel_parts or profile_path.name.startswith("_archived"):
        return "archived"
    return "active"


def _is_excluded_profile(name: str, path: Path) -> bool:
    candidates = (name.lower(), path.name.lower())
    return any(fragment in candidate for candidate in candidates for fragment in EXCLUDED_NAME_FRAGMENTS)


def _profile_record(name: str, profile_path: Path, profiles_root: Path) -> dict[str, Any]:
    env_stat = _safe_stat(profile_path / ".env")
    return {
        "name": name,
        "path": str(profile_path),
        "boundary_label": PROFILE_BOUNDARIES.get(name, "NEEDS HUMAN REVIEW — PROJECT OWNERSHIP UNCLEAR"),
        "lifecycle": _profile_lifecycle(profile_path, profiles_root),
        "env": env_stat,
        "config_yaml": _safe_stat(profile_path / "config.yaml"),
        "soul_md": _safe_stat(profile_path / "SOUL.md"),
        "state_db": _safe_stat(profile_path / "state.db"),
        "gateway_log": _safe_stat(profile_path / "logs" / "gateway.log"),
    }


def _discover_profile_paths(profiles_root: Path) -> list[tuple[str, Path]]:
    profile_paths: list[tuple[str, Path]] = []
    for name in ACTIVE_LOCAL_PROFILES:
        path = profiles_root / name
        if path.exists() and not _is_excluded_profile(name, path):
            profile_paths.append((name, path))

    archived_root = profiles_root / "_archived"
    if archived_root.exists():
        for path in sorted(archived_root.iterdir()):
            if not path.is_dir():
                continue
            name = path.name
            if name not in ARCHIVED_LOCAL_PROFILES:
                continue
            if _is_excluded_profile(name, path):
                continue
            profile_paths.append((name, path))

    return profile_paths


@router.get("/inventory")
async def inventory() -> dict[str, Any]:
    """Return a read-only, secret-free LOCAL GSSAI profile inventory."""

    hermes_root = _hermes_root()
    profiles_root = hermes_root / "profiles"
    records = [
        _profile_record(name, path, profiles_root)
        for name, path in _discover_profile_paths(profiles_root)
    ]
    return {
        "scope": "LOCAL GSSAI PROJECT ONLY",
        "policy": "read-only; secret-free; file metadata only",
        "hermes_root": str(hermes_root),
        "profiles_root": str(profiles_root),
        "profiles": records,
    }


@router.get("/guardrails")
async def guardrails() -> dict[str, Any]:
    """Document the safety boundary enforced by this plugin."""

    return {
        "scope": "LOCAL GSSAI PROJECT ONLY",
        "read_only": True,
        "secret_values_returned": False,
        "excluded_name_fragments": list(EXCLUDED_NAME_FRAGMENTS),
        "secret_files_metadata_only": sorted(SECRET_FILE_NAMES),
        "guarded_actions_deferred": [
            "archive profile",
            "blank archived .env",
            "restore profile from backup",
            "rename profile",
            "delete/quarantine retired profile",
        ],
    }
