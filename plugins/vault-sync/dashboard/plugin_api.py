"""LOCAL GSSAI vault-sync dashboard plugin.

Mounted at /api/plugins/vault-sync/ by the dashboard plugin system.

This v0.1 surface is read-only and allowlist-bound. It reports the health of
known Obsidian/vault artefacts created during Phase 2.1 without syncing,
copying, moving, deleting, or reading secrets.
"""

from __future__ import annotations

import hashlib
import os
import pwd
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter()

MAX_HASH_BYTES = 16 * 1024 * 1024
CANONICAL_REL = Path("06-sops/phase-2-1-execution-checklist-approved-cleanup-with-controls.md")
TMP_REL = Path("phase2_1_execution_checklist_approved_cleanup_with_controls.md")
EXPECTED_DIRS = (
    "06-sops",
    "09-logs",
    "13-runtime-state",
    "agent-ecosystem",
)


def _real_user_home() -> Path:
    try:
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:
        return Path.home()


def _vault_root() -> Path:
    override = os.environ.get("VAULT_SYNC_VAULT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (_real_user_home() / ".hermes" / "obsidian-vault").resolve()


def _tmp_root() -> Path:
    override = os.environ.get("VAULT_SYNC_TMP_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (_real_user_home() / ".hermes" / "profiles" / "ivan_bb" / "tmp").resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        if path.stat().st_size > MAX_HASH_BYTES:
            return None
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _metadata(path: Path, allowed_roots: tuple[Path, ...]) -> dict[str, Any]:
    if path.is_symlink():
        return {"path": str(path), "allowed": False, "exists": False, "error": "symlink not permitted"}
    resolved = path.resolve()
    allowed = any(_is_relative_to(resolved, root) for root in allowed_roots)
    base: dict[str, Any] = {"path": str(resolved), "allowed": allowed}
    if not allowed:
        return {**base, "exists": False, "error": "path outside allowlist"}
    if not resolved.exists():
        return {**base, "exists": False}
    stat = resolved.stat()
    return {
        **base,
        "exists": True,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "mode": oct(stat.st_mode & 0o777),
        "sha256": _sha256(resolved),
    }


def _canonical_path() -> Path:
    return _vault_root() / CANONICAL_REL


def _tmp_copy_path() -> Path:
    return _tmp_root() / TMP_REL


@router.get("/status")
async def status() -> dict[str, Any]:
    vault = _vault_root()
    tmp = _tmp_root()
    return {
        "scope": "LOCAL GSSAI PROJECT ONLY",
        "policy": "read-only; allowlisted vault artefact metadata only; no sync/move/delete",
        "vault_root": str(vault),
        "vault_root_exists": vault.exists(),
        "tmp_root": str(tmp),
        "tmp_root_exists": tmp.exists(),
    }


@router.get("/expected-dirs")
async def expected_dirs() -> dict[str, Any]:
    vault = _vault_root()
    return {
        "scope": "LOCAL GSSAI PROJECT ONLY",
        "directories": [
            {"path": str((vault / rel).resolve()), "name": rel, "exists": (vault / rel).is_dir()}
            for rel in EXPECTED_DIRS
        ],
    }


@router.get("/artefacts")
async def artefacts() -> dict[str, Any]:
    vault = _vault_root()
    tmp = _tmp_root()
    allowed_roots = (vault, tmp)
    canonical = _metadata(_canonical_path(), allowed_roots)
    tmp_copy = _metadata(_tmp_copy_path(), allowed_roots)
    duplicate = bool(
        canonical.get("exists")
        and tmp_copy.get("exists")
        and canonical.get("sha256")
        and canonical.get("sha256") == tmp_copy.get("sha256")
    )
    return {
        "scope": "LOCAL GSSAI PROJECT ONLY",
        "policy": "metadata and hashes only; no file contents returned",
        "canonical": canonical,
        "temporary_copy": tmp_copy,
        "temporary_copy_duplicates_canonical": duplicate,
    }


@router.get("/duplicates")
async def duplicates() -> dict[str, Any]:
    artefact_data = await artefacts()
    duplicate = artefact_data["temporary_copy_duplicates_canonical"]
    return {
        "scope": "LOCAL GSSAI PROJECT ONLY",
        "duplicates": [
            {
                "canonical": artefact_data["canonical"]["path"],
                "temporary_copy": artefact_data["temporary_copy"]["path"],
                "same_sha256": duplicate,
                "action_required": "review before any move/delete" if duplicate else "none or manual review",
            }
        ],
    }
