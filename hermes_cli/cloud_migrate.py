"""Cloud migration export: package HERMES_HOME for Hermes Cloud.

Sits on top of ``hermes_cli.backup``: same walker, same exclusions, same
SQLite snapshot discipline, plus three migration-specific rules:

1. Secret files (``backup._SECRET_FILE_NAMES``) are excluded by default.
   ``--include-secrets`` puts ``.env``/``auth.json`` back in the archive;
   ``--include-history`` puts ``state.db`` back. The two flags are
   independent.
2. A ``migration-manifest.json`` entry sits at the archive root so the
   importing side can refuse an incompatible bundle before mutating the
   home directory.
3. After the archive closes, every text entry in it is scanned for
   recognizable secret values (API key shapes). Matches are reported, never
   dropped: deleting a file because it contains a documentation placeholder
   like ``ghp_xxxxxxxx`` would silently erase a skill.

The import side (``hermes_cli/cloud_import``, shipping separately) discards
secret files even when ``--include-secrets`` was used, so a secrets-bearing
bundle can still be produced locally but cannot poison a cloud instance.
"""

from __future__ import annotations

import json
import re
import socket
import sys
import zipfile
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from hermes_cli.backup import (
    BackupInProgressError,
    _SECRET_FILE_NAMES,
    _backup_operation_lock,
    _count_cron_jobs,
    _run_backup_locked,
)
from hermes_constants import display_hermes_home, get_default_hermes_root

MIGRATION_SCHEMA_VERSION = 1

DEFAULT_BUNDLE_PREFIX = "hermes_cloud_migration"

# Secret-value shapes scanned for in included text files. Kept deliberately
# narrow (documented prefixes and distinctive structural markers only) to
# keep the false-positive rate near zero on doc-placeholder hits.
_SECRET_VALUE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("openai api key", re.compile(r"sk-(?:live|proj)-[A-Za-z0-9_-]{10,}")),
    ("anthropic api key", re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}")),
    ("github token", re.compile(r"\b(?:ghp|gho)_[A-Za-z0-9]{30,}")),
    ("github fine-grained token", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
    ("slack token", re.compile(r"xox[bap]-[A-Za-z0-9-]{10,}")),
    ("aws access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]

# Text entries larger than this are skipped by the scan (state snapshots,
# base64 blobs, vendor bundles). Config and skill files are far smaller.
_SCAN_MAX_FILE_BYTES = 1_000_000

_manifest = dict


def _iter_env_key_names(hermes_root: Path) -> Iterator[str]:
    """Yield secret key names from the ROOT ``.env`` (names only, no values).

    Profile-level ``.env`` files are intentionally not indexed in v1: the
    root file is the one the cloud instance will need re-provisioned.
    """
    env_path = hermes_root / ".env"
    if not env_path.is_file():
        return
    try:
        with open(env_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key = line.split("=", 1)[0].strip()
                if key:
                    yield key
    except OSError:
        return


def _is_text_entry(data: bytes) -> bool:
    if b"\x00" in data[:4096]:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _scan_archive_for_secrets(out_path: Path) -> list[tuple[str, str]]:
    """Scan text entries of the finished archive for secret-value shapes.

    Returns ``(entry_name, pattern_label)`` pairs. Read-only.
    """
    hits: list[tuple[str, str]] = []
    with zipfile.ZipFile(out_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir() or info.filename == "migration-manifest.json":
                continue
            if info.file_size <= 0 or info.file_size > _SCAN_MAX_FILE_BYTES:
                continue
            try:
                raw = zf.read(info)
            except (OSError, zipfile.BadZipFile, RuntimeError):
                continue
            if not _is_text_entry(raw):
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            for label, pattern in _SECRET_VALUE_PATTERNS:
                if pattern.search(text):
                    hits.append((info.filename, label))
                    break
    return hits


def _count_skills_in_archive(out_path: Path) -> int:
    with zipfile.ZipFile(out_path, "r") as zf:
        return sum(
            1
            for name in zf.namelist()
            if name.endswith("/SKILL.md") and name.startswith("skills/")
        )


def _entry_bytes(zf: zipfile.ZipFile, name: str) -> Optional[int]:
    try:
        info = zf.getinfo(name)
    except KeyError:
        return None
    return info.file_size


def _build_manifest(out_path: Path, hermes_root: Path, *, args) -> _manifest:
    with zipfile.ZipFile(out_path, "r") as zf:
        state_db_bytes = _entry_bytes(zf, "state.db")
        skills_count = _count_skills_in_archive(out_path)

    try:
        from hermes_cli.config import check_config_version

        config_version = check_config_version()[0]
    except Exception:
        config_version = None

    try:
        tool_version = metadata.version("hermes-agent")
    except metadata.PackageNotFoundError:
        tool_version = "unknown"

    cron_path = hermes_root / "cron" / "jobs.json"
    cron_jobs = _count_cron_jobs(cron_path)

    secrets_included = bool(getattr(args, "include_secrets", False))
    secrets_excluded = (
        [] if secrets_included else sorted(_iter_env_key_names(hermes_root))
    )

    return {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "tool_version": tool_version,
        "config_version": config_version,
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "assets": {
            "state_db": {
                "included": state_db_bytes is not None,
                "bytes": state_db_bytes or 0,
            },
            "skills_count": skills_count,
            "cron_jobs": cron_jobs,
        },
        "secrets_included": secrets_included,
        "secrets_excluded": secrets_excluded,
    }


def _default_output_path() -> Path:
    host = re.sub(r"[^A-Za-z0-9_-]", "-", socket.gethostname().split(".")[0])
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return Path.cwd() / f"{DEFAULT_BUNDLE_PREFIX}_{host}_{stamp}.zip"


def run_cloud_export(args) -> int:
    """Write a migration bundle for ``hermes cloud export``."""
    hermes_root = get_default_hermes_root()

    if not hermes_root.is_dir():
        print(f"Error: Hermes home directory not found at {hermes_root}")
        return 1

    # Resolve the output path. A file path is the contract here (the bundle
    # is uploaded to the portal, not stashed next to backups), so unlike
    # ``hermes backup`` a directory argument is not supported.
    try:
        if args.output:
            out_path = Path(args.output).expanduser().resolve()
        else:
            out_path = _default_output_path()
        if out_path.suffix.lower() != ".zip":
            out_path = out_path.with_suffix(out_path.suffix + ".zip")
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        target = getattr(args, "output", None) or "-"
        print(f"Error: cannot write bundle to {target}: {exc}")
        return 1

    if out_path.exists() and not getattr(args, "force", False):
        print(f"Error: {out_path} already exists (use --force to overwrite)")
        return 1

    # Secret files are basename-excluded on top of the built-in backup
    # exclusions. ``state.db`` is governed by --include-history, the
    # credential pair by --include-secrets.
    extra_excludes = set(_SECRET_FILE_NAMES)
    if getattr(args, "include_history", False):
        extra_excludes.discard("state.db")
    if getattr(args, "include_secrets", False):
        extra_excludes.discard(".env")
        extra_excludes.discard("auth.json")

    # The backup walker keys off args.output; hand it a minimal shim so the
    # full-backup flow writes to our chosen path and nothing else.
    shim = type("_CloudExportArgs", (), {"output": str(out_path)})()

    try:
        with _backup_operation_lock(hermes_root):
            _run_backup_locked(shim, hermes_root, extra_exclude_names=extra_excludes)
    except BackupInProgressError as exc:
        print(f"Error: {exc}")
        return 2

    # Append the manifest, then scan the finished archive for stray secrets.
    manifest = _build_manifest(out_path, hermes_root, args=args)
    try:
        with zipfile.ZipFile(
            out_path, "a", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zf:
            zf.writestr("migration-manifest.json", json.dumps(manifest, indent=2))
    except (OSError, zipfile.BadZipFile) as exc:
        print(f"Error: could not finalize bundle manifest: {exc}")
        return 1

    scan_hits = _scan_archive_for_secrets(out_path)

    print()
    print(f"Migration bundle ready:")
    print(f"  {out_path}")
    print(f"  schema_version: {manifest['schema_version']}")
    print(f"  tool_version:   {manifest['tool_version']}")
    print(f"  config_version: {manifest['config_version']}")
    print(f"  skills:         {manifest['assets']['skills_count']}")
    print(f"  cron jobs:      {manifest['assets']['cron_jobs']}")
    if manifest["secrets_included"]:
        print()
        print(
            "  WARNING: secret files are included in this bundle. The cloud "
            "importer discards them, and this archive now carries live "
            "credentials. Handle it like a plaintext secret."
        )
    elif manifest["secrets_excluded"]:
        print(
            f"  secrets excluded ({len(manifest['secrets_excluded'])} key "
            "names recorded in the manifest): "
            + ", ".join(manifest["secrets_excluded"][:8])
            + (
                " ..."
                if len(manifest["secrets_excluded"]) > 8
                else ""
            )
        )
    else:
        print("  secrets:        none detected")

    if scan_hits:
        print()
        print(
            f"  WARNING: {len(scan_hits)} included file(s) match a known "
            "secret-value shape:"
        )
        for entry_name, label in scan_hits[:10]:
            print(f"    {entry_name}  ({label})")
        if len(scan_hits) > 10:
            print(f"    ... and {len(scan_hits) - 10} more")
        print(
            "  Nothing was dropped; review the files above before uploading. "
            "The cloud importer will discard .env and auth.json regardless."
        )

    print()
    print("Leaf note: this bundle is also a valid `hermes import` archive.")
    return 0