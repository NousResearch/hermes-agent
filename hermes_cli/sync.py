"""Portable Hermes profile sync export/import helpers.

This module implements the local, no-network phase of profile sync.  It is
intentionally not a cloud drive for HERMES_HOME: it exports a structured,
portable subset of profile state and rejects runtime state and plaintext
secrets.
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
from pathlib import Path
from typing import Any

import yaml

from hermes_constants import get_hermes_home
from utils import atomic_yaml_write


SYNC_FORMAT = "hermes-profile-sync"
SYNC_VERSION = 1
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
        "checkpoints",
        "sandboxes",
        ".ssh",
        "__pycache__",
        ".git",
        "node_modules",
    }
)
_FORBIDDEN_SUFFIXES = (".db-wal", ".db-shm", ".db-journal", ".pyc", ".pyo")
_SKILL_EXPORT_SKIP_DIRS = frozenset({".hub", ".archive", "__pycache__"})


@dataclass
class SyncResult:
    """Result object returned by sync operations."""

    repo: Path
    ok: bool = True
    files_written: list[str] = field(default_factory=list)
    plan: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_plan_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "repo": str(self.repo),
            "operations": self.plan,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def run_sync(args) -> None:
    """Argparse entry point for ``hermes sync``."""
    action = getattr(args, "sync_action", None)
    _run_profile_transfer_command(
        action,
        args,
        usage="hermes sync {export,import,doctor} ...",
    )


def run_migrate(args) -> None:
    """Entry point for the profile-transfer ``hermes migrate`` subcommands.

    ``sync`` is the implementation substrate; ``migrate`` is the user-facing
    workflow for moving portable Hermes profile state between machines.
    Dispatched from :func:`hermes_cli.migrate.cmd_migrate`, which owns the
    top-level ``migrate`` command (including ``migrate xai``).
    """
    action = getattr(args, "migrate_type", None)
    if action == "verify":
        action = "doctor"
    _run_profile_transfer_command(
        action,
        args,
        usage="hermes migrate {export,import,verify,doctor} ...",
    )


def _run_profile_transfer_command(action: str | None, args, *, usage: str) -> None:
    """Dispatch shared portable profile export/import/validation commands."""
    try:
        if action == "export":
            result = export_profile_sync(
                Path(args.out),
                device_id=getattr(args, "device_id", None),
            )
            _print_export_result(result)
            if not result.ok:
                sys.exit(1)
            return

        if action == "import":
            result = import_profile_sync(
                Path(getattr(args, "source_dir")),
                dry_run=bool(getattr(args, "dry_run", False)),
                device_id=getattr(args, "device_id", None),
            )
            _print_plan_result(result, dry_run=bool(getattr(args, "dry_run", False)))
            if not result.ok:
                sys.exit(1)
            return

        if action == "doctor":
            result = doctor_sync_repo(Path(args.repo))
            _print_doctor_result(result)
            if not result.ok:
                sys.exit(1)
            return
    except SyncError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"usage: {usage}",
        file=sys.stderr,
    )
    sys.exit(2)


class SyncError(RuntimeError):
    """Raised for invalid sync repos or unsafe sync operations."""


def export_profile_sync(
    out_dir: Path,
    *,
    hermes_home: Path | None = None,
    device_id: str | None = None,
) -> SyncResult:
    """Export portable Hermes profile state into *out_dir*."""
    hermes_home = hermes_home or get_hermes_home()
    out_dir = out_dir.expanduser().resolve()
    device_id = normalize_device_id(device_id or default_device_id(hermes_home))
    result = SyncResult(repo=out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    portable_config, device_config, config_warnings = split_config_for_export(
        _read_raw_config_from_home(hermes_home)
    )
    result.warnings.extend(config_warnings)

    _write_yaml_file(out_dir / "config" / "global.yaml", portable_config)
    result.files_written.append("config/global.yaml")
    if device_config:
        device_config_path = out_dir / "config" / "devices" / f"{device_id}.yaml"
        _write_yaml_file(device_config_path, device_config)
        result.files_written.append(f"config/devices/{device_id}.yaml")

    soul_src = hermes_home / "SOUL.md"
    if soul_src.is_file():
        if _file_contains_secret(soul_src):
            result.warnings.append("Skipped SOUL.md because it appears to contain a secret")
        else:
            soul_dst = out_dir / "soul" / "SOUL.md"
            _copy_file(soul_src, soul_dst)
            result.files_written.append("soul/SOUL.md")

    memory_entries = _load_memory_entries(out_dir / "memory" / "entries.jsonl")
    memory_entries.update(
        _extract_local_memory_entries(hermes_home, device_id, result.warnings)
    )
    _write_memory_entries(out_dir / "memory" / "entries.jsonl", memory_entries)
    result.files_written.append("memory/entries.jsonl")

    skill_manifest = _export_tree(
        hermes_home / "skills",
        out_dir / "skills" / "files",
        result.warnings,
        skip_dirs=_SKILL_EXPORT_SKIP_DIRS,
    )
    _write_yaml_file(
        out_dir / "skills" / "manifest.yaml",
        {"version": 1, "files": skill_manifest},
    )
    result.files_written.append("skills/manifest.yaml")
    result.files_written.extend(f"skills/files/{item['path']}" for item in skill_manifest)

    skin_manifest = _export_tree(
        hermes_home / "skins",
        out_dir / "skins",
        result.warnings,
        skip_dirs=frozenset({"__pycache__"}),
    )
    result.files_written.extend(f"skins/{item['path']}" for item in skin_manifest)

    manifest = {
        "format": SYNC_FORMAT,
        "version": SYNC_VERSION,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_device": device_id,
    }
    _write_yaml_file(out_dir / "manifest.yaml", manifest)
    result.files_written.insert(0, "manifest.yaml")

    doctor = doctor_sync_repo(out_dir)
    result.errors.extend(doctor.errors)
    result.ok = not result.errors
    return result


def import_profile_sync(
    source_dir: Path,
    *,
    hermes_home: Path | None = None,
    device_id: str | None = None,
    dry_run: bool = False,
) -> SyncResult:
    """Import portable profile state from *source_dir* into HERMES_HOME."""
    hermes_home = hermes_home or get_hermes_home()
    source_dir = source_dir.expanduser().resolve()
    device_id = normalize_device_id(device_id or default_device_id(hermes_home))
    result = doctor_sync_repo(source_dir)
    result.repo = source_dir
    if not result.ok:
        return result

    local_config = _read_raw_config_from_home(hermes_home)
    remote_config = _compose_remote_config(source_dir, device_id)
    merged_config = _deep_merge_dicts(local_config, remote_config)
    if local_config or remote_config:
        _queue_yaml_write(
            result,
            hermes_home / "config.yaml",
            merged_config,
            hermes_home=hermes_home,
        )

    soul_src = source_dir / "soul" / "SOUL.md"
    if soul_src.is_file():
        _queue_text_or_conflict(
            result,
            source=soul_src,
            target=hermes_home / "SOUL.md",
            hermes_home=hermes_home,
            conflict_suffix=".sync-conflict",
        )

    memory_entries = _load_memory_entries(source_dir / "memory" / "entries.jsonl")
    memory_entries.update(_extract_local_memory_entries(hermes_home, device_id, result.warnings))
    rendered_memory = _render_memory_entries(memory_entries)
    for rel_path, content in rendered_memory.items():
        _queue_text_write(
            result,
            hermes_home / rel_path,
            content,
            hermes_home=hermes_home,
        )

    _queue_tree_import(
        result,
        source_dir / "skills" / "files",
        hermes_home / "skills",
        hermes_home=hermes_home,
    )
    _queue_tree_import(
        result,
        source_dir / "skins",
        hermes_home / "skins",
        hermes_home=hermes_home,
        skip_names={"manifest.yaml"},
    )

    if result.ok and not dry_run:
        _apply_plan(result)
    return result


def doctor_sync_repo(repo_dir: Path) -> SyncResult:
    """Validate a local sync repository."""
    repo_dir = repo_dir.expanduser().resolve()
    result = SyncResult(repo=repo_dir)
    if not repo_dir.exists():
        result.ok = False
        result.errors.append(f"sync repo does not exist: {repo_dir}")
        return result
    if not repo_dir.is_dir():
        result.ok = False
        result.errors.append(f"sync repo is not a directory: {repo_dir}")
        return result

    manifest_path = repo_dir / "manifest.yaml"
    try:
        manifest = _read_yaml_file(manifest_path, required=True)
    except SyncError as exc:
        result.ok = False
        result.errors.append(str(exc))
        return result

    if manifest.get("format") != SYNC_FORMAT:
        result.errors.append("manifest.yaml has invalid or missing format")
    if manifest.get("version") != SYNC_VERSION:
        result.errors.append("manifest.yaml has unsupported version")

    # Pre-read pass: reject symlinks before any bundle file is opened.  A
    # symlinked entry would otherwise be followed by the content checks below
    # and by the import machinery.
    for rel in _find_symlinks(repo_dir):
        result.errors.append(f"symlink present in sync repo: {rel.as_posix()}")

    for path in _iter_files(repo_dir):
        if path.is_symlink():
            continue  # already rejected above; never read through a symlink
        rel = path.relative_to(repo_dir)
        if is_forbidden_sync_path(rel):
            result.errors.append(f"forbidden path present in sync repo: {rel.as_posix()}")
        elif _file_contains_secret(path):
            result.errors.append(f"plaintext secret-like value present in sync repo: {rel.as_posix()}")

    result.ok = not result.errors
    return result


def split_config_for_export(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Split raw config into portable and device-local sync config maps."""
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
            warnings.append(f"Kept config.{key} local: unknown sync ownership")
            continue

        p_value, d_value = _split_portable_value(value, path, warnings)
        if p_value is not _MISSING:
            portable[key] = p_value
        if d_value is not _MISSING:
            device[key] = d_value

    return portable, device, warnings


def normalize_device_id(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    if not slug:
        raise SyncError("device id cannot be empty")
    if len(slug) > 80:
        slug = slug[:80].rstrip("-_")
    return slug


def default_device_id(hermes_home: Path) -> str:
    host = platform.node() or "device"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", host.lower()).strip("-") or "device"
    digest = hashlib.sha256(str(hermes_home.expanduser().resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


def is_forbidden_sync_path(rel_path: Path) -> bool:
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
                raise SyncError(f"invalid memory JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(entry, dict):
                raise SyncError(f"invalid memory entry at {path}:{line_no}: expected object")
            entry_id = str(entry.get("id") or "")
            target = entry.get("target")
            content = entry.get("content")
            if not entry_id or target not in {"memory", "user"} or not isinstance(content, str):
                raise SyncError(f"invalid memory entry at {path}:{line_no}: missing id/target/content")
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
            if is_forbidden_sync_path(rel):
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
    result: SyncResult,
    src_root: Path,
    dst_root: Path,
    *,
    hermes_home: Path,
    skip_names: set[str] | None = None,
) -> None:
    if not src_root.is_dir():
        return
    # Pre-read pass: reject symlinked bundle entries before anything is read.
    # ``_queue_bytes_write`` reads every enumerated source, so a symlink here
    # would be followed during import even though ``doctor_sync_repo`` also
    # rejects it — keep this guard so tree imports are safe in isolation.
    symlinks = _find_symlinks(src_root)
    if symlinks:
        for rel in symlinks:
            result.errors.append(
                f"refusing to import symlinked bundle entry: "
                f"{src_root.name}/{rel.as_posix()}"
            )
        result.ok = False
        return
    skip_names = skip_names or set()
    for src in _iter_files(src_root):
        if src.is_symlink():
            continue  # defense in depth; the pre-read pass rejects these
        rel = src.relative_to(src_root)
        if src.name in skip_names or is_forbidden_sync_path(rel):
            continue
        _queue_bytes_write(result, src, dst_root / rel, hermes_home=hermes_home)


def _queue_yaml_write(
    result: SyncResult,
    target: Path,
    data: dict[str, Any],
    *,
    hermes_home: Path,
) -> None:
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True)
    _queue_text_write(result, target, text, hermes_home=hermes_home)


def _queue_text_or_conflict(
    result: SyncResult,
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
    result: SyncResult,
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
    result: SyncResult,
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


def _apply_plan(result: SyncResult) -> None:
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
            raise SyncError(f"missing required file: {path}")
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SyncError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SyncError(f"invalid YAML in {path}: expected mapping")
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
            yield Path(dirpath) / filename


def _find_symlinks(root: Path) -> list[Path]:
    """Return the relative paths of every symlinked entry under *root*.

    Symlinked files and directories are both reported.  Bundle contents must
    never be symlinks: a symlinked entry would be followed on read and could
    smuggle arbitrary local files (e.g. ``~/.ssh/id_ed25519``) into a
    validation pass or an import.  Mirrors the safety pattern of
    ``profile_distribution._reject_distribution_symlinks``.
    """
    found: list[Path] = []
    if not root.is_dir() or root.is_symlink():
        return found
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        for name in dirnames + filenames:
            entry = base / name
            if entry.is_symlink():
                found.append(entry.relative_to(root))
    return sorted(found)


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


def _print_export_result(result: SyncResult) -> None:
    if result.ok:
        print(f"Exported Hermes sync profile to {result.repo}")
    else:
        print(f"Export failed for {result.repo}")
    if result.files_written:
        print("Files written:")
        for path in result.files_written:
            print(f"  {path}")
    _print_warnings_errors(result)


def _print_plan_result(result: SyncResult, *, dry_run: bool) -> None:
    label = "Import dry-run plan" if dry_run else "Import plan"
    print(label + ":")
    printable = _public_plan(result)
    print(json.dumps(printable, indent=2, sort_keys=True))
    if result.ok and not dry_run:
        print(f"Applied {len(result.files_written)} file changes.")


def _print_doctor_result(result: SyncResult) -> None:
    if result.ok:
        print(f"Sync repo OK: {result.repo}")
    else:
        print(f"Sync repo invalid: {result.repo}")
    _print_warnings_errors(result)


def _print_warnings_errors(result: SyncResult) -> None:
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  {warning}")
    if result.errors:
        print("Errors:", file=sys.stderr)
        for error in result.errors:
            print(f"  {error}", file=sys.stderr)


def _public_plan(result: SyncResult) -> dict[str, Any]:
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
    }
