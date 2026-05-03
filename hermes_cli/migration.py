"""Portable Hermes profile migration export/import helpers.

This module implements the local, no-network portability layer used by
``hermes migrate``.  It is intentionally not a cloud drive for HERMES_HOME:
it exports a structured, portable subset of profile state and rejects runtime
state and plaintext secrets.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from hermes_constants import get_hermes_home
from utils import atomic_yaml_write


LEGACY_SYNC_FORMAT = "hermes-profile-sync"
MIGRATION_FORMAT = "hermes-profile-migration"
SUPPORTED_FORMATS = frozenset({LEGACY_SYNC_FORMAT, MIGRATION_FORMAT})
MIGRATION_VERSION = 1
MEMORY_CREATED_AT = "1970-01-01T00:00:00Z"

_MISSING = object()

_PORTABLE_ROOT_KEYS = frozenset(
    {
        "model",
        "providers",
        "fallback_providers",
        "credential_pool_strategies",
        "toolsets",
        "agent",
        "compression",
        "prompt_caching",
        "auxiliary",
        "display",
        "dashboard",
        "privacy",
        "tts",
        "stt",
        "voice",
        "human_delay",
        "context",
        "memory",
        "delegation",
        "skills",
        "curator",
        "timezone",
        "personalities",
        "security",
        "cron",
        "code_execution",
        "model_catalog",
        "network",
        "sessions",
        "onboarding",
        "updates",
        "_config_version",
    }
)

_DEVICE_ROOT_KEYS = frozenset(
    {
        "terminal",
        "browser",
        "bedrock",
        "discord",
        "whatsapp",
        "telegram",
        "slack",
        "mattermost",
        "prefill_messages_file",
        "approvals",
        "command_allowlist",
        "quick_commands",
        "hooks",
        "logging",
        "checkpoints",
        "file_read_max_chars",
        "tool_output",
        "honcho",
    }
)

_DEVICE_CONFIG_PATHS = {
    ("skills", "external_dirs"),
    ("auxiliary", "vision", "base_url"),
    ("auxiliary", "web_extract", "base_url"),
    ("auxiliary", "compression", "base_url"),
    ("auxiliary", "session_search", "base_url"),
    ("auxiliary", "skills_hub", "base_url"),
    ("auxiliary", "approval", "base_url"),
    ("auxiliary", "mcp", "base_url"),
    ("auxiliary", "title_generation", "base_url"),
    ("delegation", "base_url"),
}

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|secret|password|token|credential|private[_-]?key|client[_-]?secret)",
    re.IGNORECASE,
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9][A-Za-z0-9_-]{12,}"),
    re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[abprs]-[A-Za-z0-9-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)

_FORBIDDEN_NAMES = frozenset(
    {
        ".env",
        "auth.json",
        "state.db",
        "hermes_state.db",
        "response_store.db",
        "gateway.pid",
        "cron.pid",
        "gateway_state.json",
        "processes.json",
        "auth.lock",
        ".hermes_history",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
    }
)
_FORBIDDEN_DIRS = frozenset(
    {
        "logs",
        "cache",
        "tmp",
        "temp",
        "image_cache",
        "audio_cache",
        "document_cache",
        "browser_screenshots",
        "checkpoints",
        "sandboxes",
        ".ssh",
        "__pycache__",
        ".git",
        "node_modules",
    }
)
_FORBIDDEN_SUFFIXES = (
    ".db-wal",
    ".db-shm",
    ".db-journal",
    ".pyc",
    ".pyo",
    ".pid",
    ".sock",
    ".lock",
    ".tmp",
)
_SKILL_EXPORT_SKIP_DIRS = frozenset({".hub", ".archive", "__pycache__"})
_PROFILE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_DIAGNOSTIC_KEYS = (
    "migrated",
    "skipped",
    "manual_reauth",
    "possibly_incompatible",
)


@dataclass
class MigrationResult:
    """Result object returned by migration operations."""

    repo: Path
    ok: bool = True
    files_written: list[str] = field(default_factory=list)
    plan: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    diagnostics: dict[str, list[str]] = field(
        default_factory=lambda: {key: [] for key in _DIAGNOSTIC_KEYS}
    )

    def as_plan_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "repo": str(self.repo),
            "operations": self.plan,
            "warnings": self.warnings,
            "errors": self.errors,
            "diagnostics": self.diagnostics,
        }


def run_migrate(args) -> None:
    """Argparse entry point for ``hermes migrate``.

    This is the user-facing product workflow from #6078: local portability
    across machines, operating systems, reinstalls, and path changes without
    implying cloud service, network transfer, or background multi-device behavior.
    """
    action = getattr(args, "migrate_action", None)
    _run_profile_transfer_command(
        action,
        args,
        usage="hermes migrate {export,import,verify,doctor} ...",
        bundle_label="migration bundle",
        manifest_format=MIGRATION_FORMAT,
    )


def _run_profile_transfer_command(
    action: str | None,
    args,
    *,
    usage: str,
    bundle_label: str,
    manifest_format: str,
) -> None:
    """Dispatch shared portable profile export/import/validation commands."""
    try:
        if action == "export":
            result = export_migration_bundle(
                Path(args.out),
                device_id=getattr(args, "device_id", None),
                manifest_format=manifest_format,
                force=bool(getattr(args, "force", False)),
            )
            _print_export_result(result, bundle_label=bundle_label)
            if not result.ok:
                sys.exit(1)
            return

        if action == "import":
            result = import_migration_bundle(
                Path(getattr(args, "source_dir")),
                dry_run=bool(getattr(args, "dry_run", False)),
                device_id=getattr(args, "device_id", None),
            )
            if result.ok and _bundle_contains_skills(Path(getattr(args, "source_dir"))):
                _print_import_safety_notice()
            _print_plan_result(
                result,
                dry_run=bool(getattr(args, "dry_run", False)),
                bundle_label=bundle_label,
            )
            if not result.ok:
                sys.exit(1)
            return

        if action == "verify":
            result = doctor_migration_bundle(Path(args.repo))
            _print_verify_result(result, bundle_label=bundle_label)
            if not result.ok:
                sys.exit(1)
            return

        if action == "doctor":
            result = doctor_migration_bundle(Path(args.repo))
            _print_doctor_result(result, bundle_label=bundle_label)
            if not result.ok:
                sys.exit(1)
            return
    except MigrationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"usage: {usage}",
        file=sys.stderr,
    )
    sys.exit(2)


class MigrationError(RuntimeError):
    """Raised for invalid migration bundles or unsafe migration operations."""


def export_migration_bundle(
    out_dir: Path,
    *,
    hermes_home: Path | None = None,
    device_id: str | None = None,
    manifest_format: str = MIGRATION_FORMAT,
    force: bool = False,
) -> MigrationResult:
    """Export portable Hermes profile state into *out_dir*."""
    hermes_home = hermes_home or get_hermes_home()
    out_dir = out_dir.expanduser().resolve()
    device_id = normalize_device_id(device_id or default_device_id(hermes_home))
    if manifest_format not in SUPPORTED_FORMATS:
        raise MigrationError(f"unsupported migration bundle format: {manifest_format}")
    result = MigrationResult(repo=out_dir)

    _prepare_output_dir(out_dir, force=force, hermes_home=hermes_home)

    _export_home_state(
        hermes_home=hermes_home,
        bundle_root=out_dir,
        device_id=device_id,
        result=result,
        bundle_prefix="",
    )
    profile_names = _export_named_profiles(hermes_home, out_dir, device_id, result)

    manifest = {
        "format": manifest_format,
        "version": MIGRATION_VERSION,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_device": device_id,
        "profiles": profile_names,
    }
    _write_yaml_file(out_dir / "manifest.yaml", manifest)
    result.files_written.insert(0, "manifest.yaml")

    doctor = doctor_migration_bundle(out_dir)
    result.errors.extend(doctor.errors)
    result.ok = not result.errors
    return result


def import_migration_bundle(
    source_dir: Path,
    *,
    hermes_home: Path | None = None,
    device_id: str | None = None,
    dry_run: bool = False,
) -> MigrationResult:
    """Import portable profile state from *source_dir* into HERMES_HOME."""
    hermes_home = hermes_home or get_hermes_home()
    source_dir = source_dir.expanduser().resolve()
    device_id = normalize_device_id(device_id or default_device_id(hermes_home))
    result = doctor_migration_bundle(source_dir)
    result.repo = source_dir
    if not result.ok:
        return result

    _queue_home_import(
        result,
        source_root=source_dir,
        target_home=hermes_home,
        hermes_home=hermes_home,
        device_id=device_id,
    )
    for profile_name in _bundle_profile_names(source_dir, result):
        _queue_home_import(
            result,
            source_root=source_dir / "profiles" / profile_name,
            target_home=hermes_home / "profiles" / profile_name,
            hermes_home=hermes_home,
            device_id=device_id,
        )

    if result.ok and not dry_run:
        _apply_plan(result)
    return result


def doctor_migration_bundle(repo_dir: Path) -> MigrationResult:
    """Validate a local migration bundle."""
    repo_dir = repo_dir.expanduser().resolve()
    result = MigrationResult(repo=repo_dir)
    if not repo_dir.exists():
        result.ok = False
        result.errors.append(f"migration bundle does not exist: {repo_dir}")
        return result
    if not repo_dir.is_dir():
        result.ok = False
        result.errors.append(f"migration bundle is not a directory: {repo_dir}")
        return result

    manifest_path = repo_dir / "manifest.yaml"
    try:
        manifest = _read_yaml_file(manifest_path, required=True)
    except MigrationError as exc:
        result.ok = False
        result.errors.append(str(exc))
        return result

    bundle_format = manifest.get("format")
    if bundle_format not in SUPPORTED_FORMATS:
        result.errors.append("manifest.yaml has invalid or missing format")
    if manifest.get("version") != MIGRATION_VERSION:
        result.errors.append("manifest.yaml has unsupported version")

    for path in _iter_files_for_validation(repo_dir):
        rel = path.relative_to(repo_dir)
        if path.is_symlink():
            result.errors.append(f"symlink present in migration bundle: {rel.as_posix()}")
        elif is_forbidden_migration_path(rel):
            result.errors.append(f"forbidden path present in migration bundle: {rel.as_posix()}")
        elif _file_contains_secret(path):
            result.errors.append(f"plaintext secret-like value present in migration bundle: {rel.as_posix()}")

    _inspect_bundle_scope(repo_dir, result, label="active profile")
    profile_names = _validate_bundle_profiles(repo_dir, manifest, result)
    if profile_names:
        _diag(result, "migrated", f"{len(profile_names)} named profile(s): {', '.join(profile_names)}")
        for profile_name in profile_names:
            _inspect_bundle_scope(
                repo_dir / "profiles" / profile_name,
                result,
                label=f"profile {profile_name}",
            )
    else:
        _diag(result, "skipped", "No named profiles are included in this bundle")

    _add_standard_diagnostics(result)
    result.ok = not result.errors
    return result


def split_config_for_export(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Split raw config into portable and device-local migration config maps."""
    portable: dict[str, Any] = {}
    device: dict[str, Any] = {}
    warnings: list[str] = []

    for key, value in config.items():
        path = (str(key),)
        if _is_secret_key(str(key)):
            warnings.append(f"Skipped config.{key}: secret-like key")
            continue
        if key in _DEVICE_ROOT_KEYS:
            scrubbed = _scrub_export_value(value, path, warnings)
            if scrubbed is not _MISSING:
                device[key] = scrubbed
            continue
        if key not in _PORTABLE_ROOT_KEYS:
            warnings.append(f"Kept config.{key} local: unknown migration ownership")
            continue

        p_value, d_value = _split_portable_value(value, path, warnings)
        if p_value is not _MISSING:
            portable[key] = p_value
        if d_value is not _MISSING:
            device[key] = d_value

    return portable, device, warnings


def _export_home_state(
    *,
    hermes_home: Path,
    bundle_root: Path,
    device_id: str,
    result: MigrationResult,
    bundle_prefix: str,
) -> None:
    """Export the safe portable subset of one Hermes home into *bundle_root*."""
    portable_config, device_config, config_warnings = split_config_for_export(
        _read_raw_config_from_home(hermes_home)
    )
    result.warnings.extend(config_warnings)

    _write_yaml_file(bundle_root / "config" / "global.yaml", portable_config)
    result.files_written.append(_bundle_rel(bundle_prefix, "config/global.yaml"))
    if device_config:
        device_config_path = bundle_root / "config" / "devices" / f"{device_id}.yaml"
        _write_yaml_file(device_config_path, device_config)
        result.files_written.append(
            _bundle_rel(bundle_prefix, f"config/devices/{device_id}.yaml")
        )

    soul_src = hermes_home / "SOUL.md"
    if soul_src.is_file():
        if _file_contains_secret(soul_src):
            result.warnings.append(
                f"Skipped {_bundle_rel(bundle_prefix, 'SOUL.md')}: secret-like content"
            )
        else:
            soul_dst = bundle_root / "soul" / "SOUL.md"
            _copy_file(soul_src, soul_dst)
            result.files_written.append(_bundle_rel(bundle_prefix, "soul/SOUL.md"))

    memory_entries = _extract_local_memory_entries(
        hermes_home,
        device_id,
        result.warnings,
    )
    _write_memory_entries(bundle_root / "memory" / "entries.jsonl", memory_entries)
    result.files_written.append(_bundle_rel(bundle_prefix, "memory/entries.jsonl"))

    skill_manifest = _export_tree(
        hermes_home / "skills",
        bundle_root / "skills" / "files",
        result.warnings,
        skip_dirs=_SKILL_EXPORT_SKIP_DIRS,
    )
    _write_yaml_file(
        bundle_root / "skills" / "manifest.yaml",
        {"version": 1, "files": skill_manifest},
    )
    result.files_written.append(_bundle_rel(bundle_prefix, "skills/manifest.yaml"))
    result.files_written.extend(
        _bundle_rel(bundle_prefix, f"skills/files/{item['path']}")
        for item in skill_manifest
    )

    skin_manifest = _export_tree(
        hermes_home / "skins",
        bundle_root / "skins",
        result.warnings,
        skip_dirs=frozenset({"__pycache__"}),
    )
    result.files_written.extend(
        _bundle_rel(bundle_prefix, f"skins/{item['path']}") for item in skin_manifest
    )


def _export_named_profiles(
    hermes_home: Path,
    out_dir: Path,
    device_id: str,
    result: MigrationResult,
) -> list[str]:
    profiles_root = hermes_home / "profiles"
    if not profiles_root.is_dir():
        return []

    profile_names: list[str] = []
    for profile_dir in sorted(profiles_root.iterdir(), key=lambda path: path.name):
        if not profile_dir.is_dir() or profile_dir.is_symlink():
            continue
        profile_name = profile_dir.name
        if not _is_valid_profile_name(profile_name):
            result.warnings.append(f"Skipped profile {profile_name}: invalid profile name")
            continue
        _export_home_state(
            hermes_home=profile_dir,
            bundle_root=out_dir / "profiles" / profile_name,
            device_id=device_id,
            result=result,
            bundle_prefix=f"profiles/{profile_name}",
        )
        profile_names.append(profile_name)
    return profile_names


def _queue_home_import(
    result: MigrationResult,
    *,
    source_root: Path,
    target_home: Path,
    hermes_home: Path,
    device_id: str,
) -> None:
    local_config = _read_raw_config_from_home(target_home)
    remote_config = _compose_remote_config(source_root, device_id)
    merged_config = _deep_merge_dicts(local_config, remote_config)
    if local_config or remote_config:
        _queue_yaml_write(
            result,
            target_home / "config.yaml",
            merged_config,
            hermes_home=hermes_home,
        )

    soul_src = source_root / "soul" / "SOUL.md"
    if soul_src.is_file():
        _queue_text_or_conflict(
            result,
            source=soul_src,
            target=target_home / "SOUL.md",
            hermes_home=hermes_home,
            conflict_suffix=".migration-conflict",
        )

    memory_entries = _load_memory_entries(source_root / "memory" / "entries.jsonl")
    memory_entries.update(
        _extract_local_memory_entries(target_home, device_id, result.warnings)
    )
    rendered_memory = _render_memory_entries(memory_entries)
    for rel_path, content in rendered_memory.items():
        _queue_text_write(
            result,
            target_home / rel_path,
            content,
            hermes_home=hermes_home,
        )

    _queue_tree_import(
        result,
        source_root / "skills" / "files",
        target_home / "skills",
        hermes_home=hermes_home,
    )
    _queue_tree_import(
        result,
        source_root / "skins",
        target_home / "skins",
        hermes_home=hermes_home,
        skip_names={"manifest.yaml"},
    )


def _bundle_profile_names(source_dir: Path, result: MigrationResult) -> list[str]:
    try:
        manifest = _read_yaml_file(source_dir / "manifest.yaml", required=True)
    except MigrationError as exc:
        result.errors.append(str(exc))
        result.ok = False
        return []
    raw_profiles = manifest.get("profiles") or []
    if not isinstance(raw_profiles, list):
        result.errors.append("manifest.yaml profiles must be a list")
        result.ok = False
        return []
    return [name for name in raw_profiles if isinstance(name, str) and _is_valid_profile_name(name)]


def _bundle_contains_skills(source_dir: Path) -> bool:
    source_dir = source_dir.expanduser().resolve()
    if _tree_has_files(source_dir / "skills" / "files"):
        return True

    profiles_root = source_dir / "profiles"
    if not profiles_root.is_dir():
        return False
    for profile_dir in profiles_root.iterdir():
        if profile_dir.is_dir() and _tree_has_files(profile_dir / "skills" / "files"):
            return True
    return False


def _tree_has_files(root: Path) -> bool:
    if not root.is_dir():
        return False
    return any(path.is_file() for path in _iter_files(root))


def _inspect_bundle_scope(repo_dir: Path, result: MigrationResult, *, label: str) -> None:
    config_path = repo_dir / "config" / "global.yaml"
    if config_path.exists():
        _read_yaml_for_diagnostics(config_path, result)
        _diag(result, "migrated", f"Portable config for {label}")
    else:
        _diag(result, "skipped", f"Portable config for {label} is not present")

    devices_root = repo_dir / "config" / "devices"
    if devices_root.is_dir():
        for device_config in sorted(devices_root.glob("*.yaml")):
            _read_yaml_for_diagnostics(device_config, result)
            rel = device_config.relative_to(repo_dir).as_posix()
            _diag(result, "migrated", f"Device-local config for {label}: {rel}")
            _diag(
                result,
                "possibly_incompatible",
                f"{rel} may contain host-specific paths and is only applied when --device-id matches",
            )

    soul_path = repo_dir / "soul" / "SOUL.md"
    if soul_path.exists():
        if soul_path.is_file():
            _diag(result, "migrated", f"SOUL.md for {label}")
        else:
            result.errors.append(f"soul/SOUL.md for {label} is not a file")
    else:
        _diag(result, "skipped", f"SOUL.md for {label} is not present")

    memory_path = repo_dir / "memory" / "entries.jsonl"
    if memory_path.exists():
        try:
            memory_entries = _load_memory_entries(memory_path)
        except MigrationError as exc:
            result.errors.append(str(exc))
        else:
            _diag(result, "migrated", f"{len(memory_entries)} memory entries for {label}")
    else:
        _diag(result, "skipped", f"Memory entries for {label} are not present")

    _inspect_skill_manifest(repo_dir, result, label=label)

    skins_root = repo_dir / "skins"
    if skins_root.is_dir():
        skin_count = sum(1 for path in _iter_files(skins_root) if path.is_file())
        _diag(result, "migrated", f"{skin_count} skin files for {label}")
    else:
        _diag(result, "skipped", f"Skins for {label} are not present")


def _inspect_skill_manifest(repo_dir: Path, result: MigrationResult, *, label: str) -> None:
    manifest_path = repo_dir / "skills" / "manifest.yaml"
    files_root = repo_dir / "skills" / "files"
    if not manifest_path.exists():
        if files_root.exists():
            result.errors.append(f"skills/files exists without skills/manifest.yaml for {label}")
        else:
            _diag(result, "skipped", f"Skills for {label} are not present")
        return

    manifest = _read_yaml_for_diagnostics(manifest_path, result)
    raw_files = manifest.get("files") if isinstance(manifest, dict) else None
    if raw_files is None:
        raw_files = []
    if not isinstance(raw_files, list):
        result.errors.append(f"skills/manifest.yaml files must be a list for {label}")
        return

    validated = 0
    for idx, item in enumerate(raw_files):
        if not isinstance(item, dict):
            result.errors.append(f"skills/manifest.yaml files[{idx}] must be a mapping for {label}")
            continue
        rel = _safe_bundle_rel_path(item.get("path"), f"skills/manifest.yaml files[{idx}]")
        expected_sha = item.get("sha256")
        if rel is None:
            result.errors.append(f"unsafe skill path in skills/manifest.yaml for {label}: {item.get('path')}")
            continue
        if is_forbidden_migration_path(rel):
            result.errors.append(f"forbidden skill path in skills/manifest.yaml for {label}: {rel.as_posix()}")
            continue
        file_path = files_root / rel
        if not file_path.is_file():
            result.errors.append(f"missing skill file for {label}: {(Path('skills/files') / rel).as_posix()}")
            continue
        if expected_sha and expected_sha != _sha256_file(file_path):
            result.errors.append(f"sha256 mismatch for skill file in {label}: {rel.as_posix()}")
            continue
        validated += 1
    _diag(result, "migrated", f"{validated} skill files for {label}")


def _validate_bundle_profiles(
    repo_dir: Path,
    manifest: dict[str, Any],
    result: MigrationResult,
) -> list[str]:
    raw_profiles = manifest.get("profiles") or []
    if not isinstance(raw_profiles, list):
        result.errors.append("manifest.yaml profiles must be a list")
        return []

    profile_names: list[str] = []
    for raw_name in raw_profiles:
        if not isinstance(raw_name, str) or not _is_valid_profile_name(raw_name):
            result.errors.append(f"invalid profile name in manifest.yaml: {raw_name!r}")
            continue
        profile_dir = repo_dir / "profiles" / raw_name
        if not profile_dir.is_dir():
            result.errors.append(f"missing profile bundle directory: profiles/{raw_name}")
            continue
        profile_names.append(raw_name)

    profiles_root = repo_dir / "profiles"
    if profiles_root.is_dir():
        declared = set(profile_names)
        for child in profiles_root.iterdir():
            if child.is_dir() and child.name not in declared:
                result.errors.append(f"profile directory not listed in manifest.yaml: profiles/{child.name}")
    return profile_names


def _add_standard_diagnostics(result: MigrationResult) -> None:
    _diag(
        result,
        "skipped",
        ".env, auth.json, state.db, logs, caches, sockets, pid files, and lock files are excluded by design",
    )
    _diag(
        result,
        "manual_reauth",
        "API keys, OAuth tokens, provider credentials, and messaging tokens are not migrated; run setup/auth commands again on the target machine",
    )


def _read_yaml_for_diagnostics(path: Path, result: MigrationResult) -> dict[str, Any]:
    try:
        return _read_yaml_file(path, required=True)
    except MigrationError as exc:
        result.errors.append(str(exc))
        return {}


def _safe_bundle_rel_path(value: Any, context: str) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        return None
    parts = [part for part in posix_path.parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        return None
    try:
        return Path(*parts)
    except TypeError as exc:
        raise MigrationError(f"invalid path in {context}: {value}") from exc


def _diag(result: MigrationResult, key: str, message: str) -> None:
    entries = result.diagnostics.setdefault(key, [])
    if message not in entries:
        entries.append(message)


def _bundle_rel(prefix: str, rel_path: str) -> str:
    return f"{prefix}/{rel_path}" if prefix else rel_path


def _is_valid_profile_name(name: str) -> bool:
    return bool(_PROFILE_NAME_RE.match(name)) and name != "default"


def _prepare_output_dir(out_dir: Path, *, force: bool, hermes_home: Path) -> None:
    if out_dir.exists() and not out_dir.is_dir():
        raise MigrationError(f"migration bundle output path is not a directory: {out_dir}")

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        return

    if not any(out_dir.iterdir()):
        return

    if not force:
        raise MigrationError(
            "migration bundle output directory is not empty. "
            "Please choose an empty directory or rerun with --force to overwrite it."
        )

    _clear_existing_output_dir(out_dir, hermes_home=hermes_home)


def _clear_existing_output_dir(out_dir: Path, *, hermes_home: Path) -> None:
    out_dir = out_dir.resolve()
    hermes_home = hermes_home.resolve()
    home = Path.home().resolve()
    root = Path(out_dir.anchor).resolve()
    if out_dir in {root, home, hermes_home}:
        raise MigrationError(f"refusing to clear unsafe migration output directory: {out_dir}")

    for child in out_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def normalize_device_id(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    if not slug:
        raise MigrationError("device id cannot be empty")
    if len(slug) > 80:
        slug = slug[:80].rstrip("-_")
    return slug


def default_device_id(hermes_home: Path) -> str:
    host = platform.node() or "device"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", host.lower()).strip("-") or "device"
    digest = hashlib.sha256(str(hermes_home.expanduser().resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


def is_forbidden_migration_path(rel_path: Path) -> bool:
    parts = rel_path.parts
    if any(part in _FORBIDDEN_DIRS for part in parts):
        return True
    name = rel_path.name
    if name in _FORBIDDEN_NAMES:
        return True
    if name.endswith(_FORBIDDEN_SUFFIXES):
        return True
    if name.endswith((".history", "_history")) and "shell" in name:
        return True
    return False


def _split_portable_value(
    value: Any,
    path: tuple[str, ...],
    warnings: list[str],
) -> tuple[Any, Any]:
    if _is_secret_key(path[-1]):
        warnings.append(f"Skipped config.{'.'.join(path)}: secret-like key")
        return _MISSING, _MISSING
    if _is_secret_scalar(value):
        warnings.append(f"Skipped config.{'.'.join(path)}: secret-like value")
        return _MISSING, _MISSING
    if _is_device_config_path(path, value):
        scrubbed = _scrub_export_value(value, path, warnings)
        return _MISSING, scrubbed
    if isinstance(value, dict):
        p_map: dict[str, Any] = {}
        d_map: dict[str, Any] = {}
        for child_key, child_value in value.items():
            p_child, d_child = _split_portable_value(
                child_value, path + (str(child_key),), warnings
            )
            if p_child is not _MISSING:
                p_map[child_key] = p_child
            if d_child is not _MISSING:
                d_map[child_key] = d_child
        return (p_map if p_map else _MISSING, d_map if d_map else _MISSING)
    if isinstance(value, list):
        clean = []
        for idx, item in enumerate(value):
            scrubbed = _scrub_export_value(item, path + (str(idx),), warnings)
            if scrubbed is not _MISSING:
                clean.append(scrubbed)
        return (clean, _MISSING)
    return value, _MISSING


def _scrub_export_value(value: Any, path: tuple[str, ...], warnings: list[str]) -> Any:
    if _is_secret_key(path[-1]):
        warnings.append(f"Skipped config.{'.'.join(path)}: secret-like key")
        return _MISSING
    if _is_secret_scalar(value):
        warnings.append(f"Skipped config.{'.'.join(path)}: secret-like value")
        return _MISSING
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            scrubbed = _scrub_export_value(child, path + (str(key),), warnings)
            if scrubbed is not _MISSING:
                out[key] = scrubbed
        return out
    if isinstance(value, list):
        out_list = []
        for idx, child in enumerate(value):
            scrubbed = _scrub_export_value(child, path + (str(idx),), warnings)
            if scrubbed is not _MISSING:
                out_list.append(scrubbed)
        return out_list
    return value


def _is_device_config_path(path: tuple[str, ...], value: Any) -> bool:
    if path in _DEVICE_CONFIG_PATHS:
        return True
    key = path[-1].lower()
    if key in {"cwd", "path", "paths", "dir", "directory", "file", "files"}:
        return True
    if key.endswith(("_path", "_dir", "_file")):
        return True
    if isinstance(value, str):
        expanded = os.path.expanduser(os.path.expandvars(value))
        if os.path.isabs(expanded):
            return True
        lower = value.lower()
        if lower.startswith(("http://localhost", "https://localhost", "http://127.", "https://127.")):
            return True
    return False


def _is_secret_key(key: str) -> bool:
    if key.lower() in {
        "key_env",
        "requires_env",
        "env_passthrough",
        "credential_pool_strategies",
    }:
        return False
    return bool(_SECRET_KEY_RE.search(key))


def _is_secret_scalar(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS)


def _file_contains_secret(path: Path) -> bool:
    try:
        if path.stat().st_size > 2_000_000:
            return False
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return _is_secret_scalar(content)


def _compose_remote_config(source_dir: Path, device_id: str) -> dict[str, Any]:
    global_cfg = _read_yaml_file(source_dir / "config" / "global.yaml", required=False)
    device_cfg = _read_yaml_file(
        source_dir / "config" / "devices" / f"{device_id}.yaml",
        required=False,
    )
    return _deep_merge_dicts(global_cfg, device_cfg)


def _deep_merge_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _extract_local_memory_entries(
    hermes_home: Path,
    device_id: str,
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for filename, target in (("MEMORY.md", "memory"), ("USER.md", "user")):
        path = hermes_home / "memories" / filename
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            warnings.append(f"Skipped memories/{filename}: {exc}")
            continue
        for raw_line in lines:
            content = raw_line.strip()
            if not content or content.startswith("#"):
                continue
            if _is_secret_scalar(content):
                warnings.append(f"Skipped memories/{filename} entry: secret-like value")
                continue
            entry_id = _memory_id(target, content)
            entries[entry_id] = {
                "id": entry_id,
                "target": target,
                "content": content,
                "source_device": device_id,
                "created_at": MEMORY_CREATED_AT,
                "updated_at": "",
                "deleted": False,
            }
    return entries


def _memory_id(target: str, content: str) -> str:
    digest = hashlib.sha256(f"{target}\0{content}".encode("utf-8")).hexdigest()[:24]
    return f"mem_{digest}"


def _load_memory_entries(path: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return entries
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MigrationError(f"invalid memory JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(entry, dict):
                raise MigrationError(f"invalid memory entry at {path}:{line_no}: expected object")
            entry_id = str(entry.get("id") or "")
            target = entry.get("target")
            content = entry.get("content")
            if not entry_id or target not in {"memory", "user"} or not isinstance(content, str):
                raise MigrationError(f"invalid memory entry at {path}:{line_no}: missing id/target/content")
            entries[entry_id] = entry
    return entries


def _write_memory_entries(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    lines = [
        json.dumps(entries[key], sort_keys=True, ensure_ascii=False)
        for key in sorted(entries)
        if not entries[key].get("deleted")
    ]
    _atomic_text_write(path, "\n".join(lines) + ("\n" if lines else ""))


def _render_memory_entries(entries: dict[str, dict[str, Any]]) -> dict[Path, str]:
    by_target = {"memory": [], "user": []}
    for key in sorted(entries):
        entry = entries[key]
        if entry.get("deleted"):
            continue
        target = entry.get("target")
        content = str(entry.get("content", "")).strip()
        if target in by_target and content:
            by_target[target].append(content)

    rendered: dict[Path, str] = {}
    if by_target["memory"]:
        rendered[Path("memories/MEMORY.md")] = "# Memory\n\n" + "\n".join(by_target["memory"]) + "\n"
    if by_target["user"]:
        rendered[Path("memories/USER.md")] = "# User Profile\n\n" + "\n".join(by_target["user"]) + "\n"
    return rendered


def _export_tree(
    src_root: Path,
    dst_root: Path,
    warnings: list[str],
    *,
    skip_dirs: frozenset[str],
) -> list[dict[str, str]]:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    manifest: list[dict[str, str]] = []
    if not src_root.is_dir():
        return manifest

    for dirpath, dirnames, filenames in os.walk(src_root, followlinks=False):
        dirnames[:] = [name for name in dirnames if name not in skip_dirs]
        base = Path(dirpath)
        for filename in filenames:
            src = base / filename
            rel = src.relative_to(src_root)
            if src.is_symlink():
                warnings.append(f"Skipped {src_root.name}/{rel.as_posix()}: symlink")
                continue
            if is_forbidden_migration_path(rel):
                warnings.append(f"Skipped {src_root.name}/{rel.as_posix()}: forbidden path")
                continue
            if _file_contains_secret(src):
                warnings.append(f"Skipped {src_root.name}/{rel.as_posix()}: secret-like content")
                continue
            dst = dst_root / rel
            _copy_file(src, dst)
            manifest.append({"path": rel.as_posix(), "sha256": _sha256_file(src)})
    return sorted(manifest, key=lambda item: item["path"])


def _queue_tree_import(
    result: MigrationResult,
    src_root: Path,
    dst_root: Path,
    *,
    hermes_home: Path,
    skip_names: set[str] | None = None,
) -> None:
    if not src_root.is_dir():
        return
    skip_names = skip_names or set()
    for src in _iter_files(src_root):
        rel = src.relative_to(src_root)
        if src.name in skip_names or is_forbidden_migration_path(rel):
            continue
        _queue_bytes_write(result, src, dst_root / rel, hermes_home=hermes_home)


def _queue_yaml_write(
    result: MigrationResult,
    target: Path,
    data: dict[str, Any],
    *,
    hermes_home: Path,
) -> None:
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True)
    _queue_text_write(result, target, text, hermes_home=hermes_home)


def _queue_text_or_conflict(
    result: MigrationResult,
    *,
    source: Path,
    target: Path,
    hermes_home: Path,
    conflict_suffix: str,
) -> None:
    remote = source.read_text(encoding="utf-8")
    if not target.exists():
        _queue_text_write(result, target, remote, hermes_home=hermes_home)
        return
    local = target.read_text(encoding="utf-8", errors="ignore")
    if local == remote:
        return
    digest = hashlib.sha256(remote.encode("utf-8")).hexdigest()[:8]
    conflict_target = target.with_name(f"{target.name}{conflict_suffix}-{digest}")
    _queue_text_write(
        result,
        conflict_target,
        remote,
        hermes_home=hermes_home,
        action="write_conflict",
        source=str(source),
    )
    result.warnings.append(
        f"Conflict for {_display_local_path(target, hermes_home)}; wrote remote copy to "
        f"{_display_local_path(conflict_target, hermes_home)}"
    )


def _queue_text_write(
    result: MigrationResult,
    target: Path,
    text: str,
    *,
    hermes_home: Path,
    action: str = "write",
    source: str | None = None,
) -> None:
    old = None
    if target.exists():
        old = target.read_text(encoding="utf-8", errors="ignore")
    if old == text:
        return
    result.plan.append(
        {
            "action": action,
            "target": _display_local_path(target, hermes_home),
            "source": source,
            "bytes": len(text.encode("utf-8")),
            "_kind": "text",
            "_target": str(target),
            "_content": text,
        }
    )


def _queue_bytes_write(
    result: MigrationResult,
    source: Path,
    target: Path,
    *,
    hermes_home: Path,
) -> None:
    new_bytes = source.read_bytes()
    if target.exists() and target.read_bytes() == new_bytes:
        return
    result.plan.append(
        {
            "action": "write",
            "target": _display_local_path(target, hermes_home),
            "source": str(source),
            "bytes": len(new_bytes),
            "_kind": "bytes",
            "_target": str(target),
            "_source": str(source),
        }
    )


def _apply_plan(result: MigrationResult) -> None:
    for op in result.plan:
        target = Path(op["_target"])
        if op["_kind"] == "text":
            _atomic_text_write(target, op["_content"])
        elif op["_kind"] == "bytes":
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(op["_source"], target)
        result.files_written.append(op["target"])


def _display_local_path(path: Path, hermes_home: Path) -> str:
    try:
        return path.relative_to(hermes_home).as_posix()
    except ValueError:
        return str(path)


def _read_yaml_file(path: Path, *, required: bool) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise MigrationError(f"missing required file: {path}")
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise MigrationError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise MigrationError(f"invalid YAML in {path}: expected mapping")
    return data


def _read_raw_config_from_home(hermes_home: Path) -> dict[str, Any]:
    path = hermes_home / "config.yaml"
    if not path.exists():
        return {}
    return _read_yaml_file(path, required=False)


def _write_yaml_file(path: Path, data: dict[str, Any]) -> None:
    atomic_yaml_write(path, data, sort_keys=False)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _iter_files(root: Path):
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [name for name in dirnames if name not in {".git", "__pycache__"}]
        for filename in filenames:
            path = Path(dirpath) / filename
            if not path.is_symlink():
                yield path


def _iter_files_for_validation(root: Path):
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [name for name in dirnames if name not in {".git", "__pycache__"}]
        for filename in filenames:
            yield Path(dirpath) / filename


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _print_export_result(result: MigrationResult, *, bundle_label: str) -> None:
    if result.ok:
        print(f"Exported Hermes {bundle_label} to {result.repo}")
    else:
        print(f"Export failed for {result.repo}")
    if result.files_written:
        print("Files written:")
        for path in result.files_written:
            print(f"  {path}")
    _print_warnings_errors(result)


def _print_plan_result(
    result: MigrationResult,
    *,
    dry_run: bool,
    bundle_label: str,
) -> None:
    label = "Import dry-run plan" if dry_run else "Import plan"
    print(f"{label} for {bundle_label}:")
    printable = _public_plan(result)
    print(json.dumps(printable, indent=2, sort_keys=True))
    if result.ok and not dry_run:
        print(f"Applied {len(result.files_written)} file changes.")


def _print_import_safety_notice() -> None:
    print("Security notice:")
    print("  This migration bundle may modify local skills and agent behavior.")
    print("  Only import bundles you created yourself or trust.")
    print("  Secrets are not migrated; you may need to re-authenticate after import.")


def _print_verify_result(result: MigrationResult, *, bundle_label: str) -> None:
    title = bundle_label[:1].upper() + bundle_label[1:]
    if result.ok:
        print(f"{title} verified: {result.repo}")
        return

    print(f"{title} verification failed: {result.repo}")
    _print_warnings_errors(result)


def _print_doctor_result(result: MigrationResult, *, bundle_label: str) -> None:
    title = bundle_label[:1].upper() + bundle_label[1:]
    if result.ok:
        print(f"{title} OK: {result.repo}")
    else:
        print(f"{title} invalid: {result.repo}")
    _print_diagnostics(result)
    _print_warnings_errors(result)


def _print_warnings_errors(result: MigrationResult) -> None:
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  {warning}")
    if result.errors:
        print("Errors:", file=sys.stderr)
        for error in result.errors:
            print(f"  {error}", file=sys.stderr)


def _print_diagnostics(result: MigrationResult) -> None:
    print("Diagnostics:")
    labels = {
        "migrated": "Migrated",
        "skipped": "Skipped by design or absent",
        "manual_reauth": "Needs manual re-authentication",
        "possibly_incompatible": "May need review on this machine",
    }
    for key in _DIAGNOSTIC_KEYS:
        entries = result.diagnostics.get(key) or []
        print(f"  {labels[key]}:")
        if not entries:
            print("    - none")
            continue
        for entry in entries:
            print(f"    - {entry}")


def _public_plan(result: MigrationResult) -> dict[str, Any]:
    operations = []
    for op in result.plan:
        operations.append(
            {
                key: value
                for key, value in op.items()
                if not key.startswith("_") and value is not None
            }
        )
    return {
        "ok": result.ok,
        "repo": str(result.repo),
        "operations": operations,
        "warnings": result.warnings,
        "errors": result.errors,
        "diagnostics": result.diagnostics,
    }
