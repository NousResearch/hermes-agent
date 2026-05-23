"""Read-only memory capacity and deletion-readiness audit.

The audit intentionally reports metadata only: store existence, size, mode,
capacity headroom, and safe operator checklists. It never prints memory file
contents, session payloads, log lines, database rows, or credential values.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_cli.config import read_raw_config
from hermes_constants import get_hermes_home


SCHEMA_VERSION = 1
ENTRY_DELIMITER = "\n\u00a7\n"
DEFAULT_MEMORY_LIMIT = 2200
DEFAULT_USER_LIMIT = 1375
WARN_RATIO = 0.80
CRITICAL_RATIO = 0.95
DIR_SCAN_LIMIT = 5000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _display_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    text = str(path)
    try:
        home = str(Path.home())
        if text == home:
            return "~"
        if text.startswith(home + os.sep):
            return "~" + text[len(home):]
    except Exception:
        pass
    return text


def _mode_string(mode: int | None) -> str | None:
    if mode is None:
        return None
    return format(stat.S_IMODE(mode), "04o")


def _permission_warning(mode: int | None, *, directory: bool) -> str | None:
    if mode is None:
        return None
    actual = stat.S_IMODE(mode)
    if actual & 0o077:
        return "Expected owner-only permissions (0700 directory, 0600 file)"
    if directory and actual & stat.S_IWOTH:
        return "Directory is world-writable"
    return None


def _coerce_limit(value: Any, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


def _capacity_status(char_count: int, char_limit: int) -> str:
    if char_limit <= 0:
        return "unknown"
    ratio = char_count / char_limit
    if ratio >= CRITICAL_RATIO:
        return "critical"
    if ratio >= WARN_RATIO:
        return "warn"
    return "ok"


def _status_with_permissions(base_status: str, warning: str | None) -> str:
    if base_status in {"critical", "missing"}:
        return base_status
    if warning:
        return "warn"
    return base_status


def _read_markdown_metadata(path: Path, char_limit: int) -> dict[str, Any]:
    exists = path.exists()
    try:
        st = path.stat() if exists else None
    except OSError:
        st = None
        exists = False

    char_count = 0
    entry_count = 0
    line_count = 0
    decode_error = False
    read_error = False
    if exists:
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw = path.read_text(encoding="utf-8", errors="replace")
            decode_error = True
        except OSError:
            raw = ""
            read_error = True
        entries = [entry.strip() for entry in raw.split(ENTRY_DELIMITER) if entry.strip()]
        char_count = len(ENTRY_DELIMITER.join(entries)) if entries else 0
        entry_count = len(entries)
        line_count = raw.count("\n") + (1 if raw else 0)

    capacity_status = "missing" if not exists else "warn" if read_error else _capacity_status(char_count, char_limit)
    mode = st.st_mode if st is not None else None
    permission_warning = _permission_warning(mode, directory=False)
    status = _status_with_permissions(capacity_status, permission_warning)
    notes = []
    if decode_error:
        notes.append("UTF-8 decode warning while counting metadata")
    if read_error:
        notes.append("Could not read file content for metadata counts")
    if permission_warning:
        notes.append(permission_warning)
    if status in {"warn", "critical"} and exists:
        notes.append("Review and compact intentionally before more memory writes")

    return {
        "exists": exists,
        "size_bytes": st.st_size if st is not None else 0,
        "mode": _mode_string(mode),
        "recommended_mode": "0600",
        "permission_warning": permission_warning,
        "char_count": char_count,
        "char_limit": char_limit,
        "headroom_chars": max(0, char_limit - char_count),
        "usage_ratio": round(char_count / char_limit, 4) if char_limit > 0 else None,
        "entry_count": entry_count,
        "line_count": line_count,
        "status": status,
        "notes": notes,
    }


def _file_store_metadata(path: Path, *, recommended_mode: str = "0600") -> dict[str, Any]:
    exists = path.exists()
    try:
        st = path.stat() if exists else None
    except OSError:
        st = None
        exists = False
    mode = st.st_mode if st is not None else None
    permission_warning = _permission_warning(mode, directory=False)
    status = _status_with_permissions("ok" if exists else "missing", permission_warning)
    notes = [permission_warning] if permission_warning else []
    return {
        "exists": exists,
        "size_bytes": st.st_size if st is not None else 0,
        "mode": _mode_string(mode),
        "recommended_mode": recommended_mode,
        "permission_warning": permission_warning,
        "status": status,
        "notes": notes,
    }


def _dir_store_metadata(path: Path, *, recommended_mode: str = "0700") -> dict[str, Any]:
    exists = path.exists()
    try:
        st = path.stat() if exists else None
    except OSError:
        st = None
        exists = False
    file_count = 0
    dir_count = 0
    total_bytes = 0
    scan_truncated = False
    if exists and path.is_dir():
        for root, dirs, files in os.walk(path, followlinks=False):
            dir_count += len(dirs)
            for name in files:
                file_count += 1
                if file_count > DIR_SCAN_LIMIT:
                    scan_truncated = True
                    break
                try:
                    total_bytes += (Path(root) / name).stat().st_size
                except OSError:
                    pass
            if scan_truncated:
                break
    mode = st.st_mode if st is not None else None
    permission_warning = _permission_warning(mode, directory=True)
    status = _status_with_permissions("ok" if exists else "missing", permission_warning)
    notes = [permission_warning] if permission_warning else []
    if scan_truncated:
        notes.append(f"Directory scan capped at {DIR_SCAN_LIMIT} files")
    return {
        "exists": exists,
        "size_bytes": total_bytes,
        "mode": _mode_string(mode),
        "recommended_mode": recommended_mode,
        "permission_warning": permission_warning,
        "file_count": min(file_count, DIR_SCAN_LIMIT),
        "dir_count": dir_count,
        "scan_truncated": scan_truncated,
        "status": status,
        "notes": notes,
    }


def _store(
    *,
    store_id: str,
    kind: str,
    label: str,
    path: Path,
    metadata: dict[str, Any],
    deletion_domain: str,
    contains_private_data: bool = True,
    mutating_command: str | None = None,
) -> dict[str, Any]:
    return {
        "id": store_id,
        "kind": kind,
        "label": label,
        "path": _display_path(path),
        "status": metadata.get("status", "unknown"),
        "contains_private_data": contains_private_data,
        "content_sample": None,
        "metadata_only": True,
        "deletion_domain": deletion_domain,
        "safe_status_command": "hermes memory audit --json --redact",
        "mutating_command": mutating_command,
        "requires_explicit_user_confirmation": bool(contains_private_data or mutating_command),
        **metadata,
    }


def _forget_checklist(hermes_home: Path) -> list[dict[str, Any]]:
    home = _display_path(hermes_home)
    return [
        {
            "id": "markdown_agent_memory",
            "store_ids": ["builtin.memory_md"],
            "scope": f"{home}/memories/MEMORY.md",
            "required_action": "Review, export if needed, then replace/remove matching entries or reset the file.",
            "safe_audit_command": "hermes memory audit --json --redact",
            "destructive_command": "hermes memory reset --target memory",
            "requires_confirmation": True,
            "notes": ["Do not run destructive reset without a user-specific forget request."],
        },
        {
            "id": "markdown_user_profile",
            "store_ids": ["builtin.user_md"],
            "scope": f"{home}/memories/USER.md",
            "required_action": "Review, export if needed, then replace/remove matching entries or reset the file.",
            "safe_audit_command": "hermes memory audit --json --redact",
            "destructive_command": "hermes memory reset --target user",
            "requires_confirmation": True,
            "notes": ["User preference deletion must also be reconciled with structured memory."],
        },
        {
            "id": "holographic_memory_store",
            "store_ids": ["structured.holographic_db"],
            "scope": f"{home}/memory_store.db",
            "required_action": "Delete or reconcile structured facts by provider-specific fact IDs, then vacuum/checkpoint if supported.",
            "safe_audit_command": "hermes memory audit --json --redact",
            "destructive_command": None,
            "requires_confirmation": True,
            "notes": ["Provider-specific deletion is intentionally not automated by this audit command."],
        },
        {
            "id": "session_index_db",
            "store_ids": ["session.state_db"],
            "scope": f"{home}/state.db",
            "required_action": "Prune or delete matching session rows and related message/tool records.",
            "safe_audit_command": "hermes sessions stats",
            "destructive_command": "hermes sessions delete <session-id>",
            "requires_confirmation": True,
            "notes": ["Use session IDs and redacted exports; do not grep raw private prompts into tickets."],
        },
        {
            "id": "session_transcripts",
            "store_ids": ["session.sessions_dir"],
            "scope": f"{home}/sessions/",
            "required_action": "Remove matching transcript JSONL files only after confirming the target session scope.",
            "safe_audit_command": "hermes sessions list --limit 20",
            "destructive_command": None,
            "requires_confirmation": True,
            "notes": ["Transcript deletion should stay session-scoped, not broad directory deletion."],
        },
        {
            "id": "response_store",
            "store_ids": ["response.response_store_db"],
            "scope": f"{home}/response_store.db",
            "required_action": "Purge response/cache rows tied to the deleted session or memory fact if the schema supports it.",
            "safe_audit_command": "hermes memory audit --json --redact",
            "destructive_command": None,
            "requires_confirmation": True,
            "notes": ["Cache purge requires schema-aware code, not ad hoc file deletion while Hermes is running."],
        },
        {
            "id": "logs",
            "store_ids": ["ops.logs_dir"],
            "scope": f"{home}/logs/",
            "required_action": "Redact or rotate logs containing the forgotten data; preserve operational receipts when possible.",
            "safe_audit_command": "hermes logs gateway --since 30m --level WARNING",
            "destructive_command": None,
            "requires_confirmation": True,
            "notes": ["Summarize logs instead of copying raw private payloads into docs."],
        },
        {
            "id": "general_cache",
            "store_ids": ["cache.root_dir"],
            "scope": f"{home}/cache/",
            "required_action": "Inspect cache metadata and remove only scoped files that contain the forgotten data.",
            "safe_audit_command": "hermes memory audit --json --redact",
            "destructive_command": None,
            "requires_confirmation": True,
            "notes": ["Cache deletion should be scoped by request and artifact type, not broad cache wipes."],
        },
        {
            "id": "media_image_cache",
            "store_ids": ["cache.images_dir", "cache.legacy_image_dir", "media.media_dir"],
            "scope": f"{home}/cache/images/, {home}/image_cache/, and {home}/media/",
            "required_action": "Remove generated or inbound image/media files tied to the forget request.",
            "safe_audit_command": "hermes memory audit --json --redact",
            "destructive_command": None,
            "requires_confirmation": True,
            "notes": ["Generated images and inbound media can contain private prompts or uploads."],
        },
        {
            "id": "screenshot_cache",
            "store_ids": ["cache.screenshots_dir", "cache.legacy_browser_screenshots_dir"],
            "scope": f"{home}/cache/screenshots/ and {home}/browser_screenshots/",
            "required_action": "Remove browser/computer-use screenshots tied to the forget request.",
            "safe_audit_command": "hermes memory audit --json --redact",
            "destructive_command": None,
            "requires_confirmation": True,
            "notes": ["Screenshots may contain private page content even when filenames are harmless."],
        },
        {
            "id": "audio_video_cache",
            "store_ids": [
                "cache.audio_dir",
                "cache.legacy_audio_dir",
                "cache.videos_dir",
                "cache.legacy_video_dir",
            ],
            "scope": f"{home}/cache/audio/, {home}/audio_cache/, {home}/cache/videos/, and {home}/video_cache/",
            "required_action": "Remove generated or inbound audio/video artifacts tied to the forget request.",
            "safe_audit_command": "hermes memory audit --json --redact",
            "destructive_command": None,
            "requires_confirmation": True,
            "notes": ["Audio/video artifacts can preserve private directives, transcripts, or uploads."],
        },
        {
            "id": "document_cache",
            "store_ids": ["cache.documents_dir", "cache.legacy_document_dir"],
            "scope": f"{home}/cache/documents/ and {home}/document_cache/",
            "required_action": "Remove uploaded or generated documents tied to the forget request.",
            "safe_audit_command": "hermes memory audit --json --redact",
            "destructive_command": None,
            "requires_confirmation": True,
            "notes": ["Document cache deletion should keep a receipt without copying document contents."],
        },
        {
            "id": "snapshots_backups",
            "store_ids": ["backups.backups_dir", "backups.checkpoints_dir"],
            "scope": f"{home}/backups/ and {home}/checkpoints/",
            "required_action": "Identify backups/snapshots that can reintroduce deleted memory and expire or quarantine them.",
            "safe_audit_command": "hermes memory audit --json --redact",
            "destructive_command": None,
            "requires_confirmation": True,
            "notes": ["Backups may be append-only; document residual retention when deletion is infeasible."],
        },
        {
            "id": "profile_memory_stores",
            "store_ids": ["profiles.profiles_dir"],
            "scope": f"{home}/profiles/*/memories/ and profile-specific stores",
            "required_action": "Repeat the audit for each profile HERMES_HOME before declaring deletion complete.",
            "safe_audit_command": "hermes --profile <profile> memory audit --json --redact",
            "destructive_command": None,
            "requires_confirmation": True,
            "notes": ["Profile stores are separate deletion domains."],
        },
    ]


def build_memory_audit(
    *,
    config: dict[str, Any] | None = None,
    hermes_home: Path | None = None,
    redacted: bool = True,
) -> dict[str, Any]:
    """Build a metadata-only audit of Hermes memory stores."""
    home = hermes_home or get_hermes_home()
    cfg = config if config is not None else read_raw_config()
    mem_cfg = cfg.get("memory", {}) if isinstance(cfg.get("memory"), dict) else {}
    memory_limit = _coerce_limit(mem_cfg.get("memory_char_limit"), DEFAULT_MEMORY_LIMIT)
    user_limit = _coerce_limit(mem_cfg.get("user_char_limit"), DEFAULT_USER_LIMIT)
    active_provider = str(mem_cfg.get("provider") or "").strip()

    memories = home / "memories"
    stores = [
        _store(
            store_id="builtin.memory_md",
            kind="markdown_memory",
            label="Built-in agent memory",
            path=memories / "MEMORY.md",
            metadata=_read_markdown_metadata(memories / "MEMORY.md", memory_limit),
            deletion_domain="markdown_agent_memory",
            mutating_command="hermes memory reset --target memory",
        ),
        _store(
            store_id="builtin.user_md",
            kind="markdown_memory",
            label="Built-in user profile",
            path=memories / "USER.md",
            metadata=_read_markdown_metadata(memories / "USER.md", user_limit),
            deletion_domain="markdown_user_profile",
            mutating_command="hermes memory reset --target user",
        ),
        _store(
            store_id="structured.holographic_db",
            kind="sqlite_memory",
            label="Structured holographic memory",
            path=home / "memory_store.db",
            metadata=_file_store_metadata(home / "memory_store.db"),
            deletion_domain="holographic_memory_store",
            mutating_command=None,
        ),
        _store(
            store_id="session.state_db",
            kind="sqlite_session",
            label="Session index database",
            path=home / "state.db",
            metadata=_file_store_metadata(home / "state.db"),
            deletion_domain="session_index_db",
            mutating_command="hermes sessions delete <session-id>",
        ),
        _store(
            store_id="session.sessions_dir",
            kind="session_transcripts",
            label="Session transcript directory",
            path=home / "sessions",
            metadata=_dir_store_metadata(home / "sessions"),
            deletion_domain="session_transcripts",
            mutating_command=None,
        ),
        _store(
            store_id="response.response_store_db",
            kind="sqlite_response_cache",
            label="Response store database",
            path=home / "response_store.db",
            metadata=_file_store_metadata(home / "response_store.db"),
            deletion_domain="response_store",
            mutating_command=None,
        ),
        _store(
            store_id="ops.logs_dir",
            kind="logs",
            label="Hermes logs",
            path=home / "logs",
            metadata=_dir_store_metadata(home / "logs"),
            deletion_domain="logs",
            mutating_command=None,
        ),
        _store(
            store_id="cache.root_dir",
            kind="cache",
            label="Hermes general cache",
            path=home / "cache",
            metadata=_dir_store_metadata(home / "cache"),
            deletion_domain="general_cache",
            mutating_command=None,
        ),
        _store(
            store_id="cache.images_dir",
            kind="image_media_cache",
            label="Image/media cache",
            path=home / "cache" / "images",
            metadata=_dir_store_metadata(home / "cache" / "images"),
            deletion_domain="media_image_cache",
            mutating_command=None,
        ),
        _store(
            store_id="cache.legacy_image_dir",
            kind="image_media_cache",
            label="Legacy image cache",
            path=home / "image_cache",
            metadata=_dir_store_metadata(home / "image_cache"),
            deletion_domain="media_image_cache",
            mutating_command=None,
        ),
        _store(
            store_id="media.media_dir",
            kind="media_cache",
            label="Generic media directory",
            path=home / "media",
            metadata=_dir_store_metadata(home / "media"),
            deletion_domain="media_image_cache",
            mutating_command=None,
        ),
        _store(
            store_id="cache.screenshots_dir",
            kind="screenshot_cache",
            label="Browser screenshot cache",
            path=home / "cache" / "screenshots",
            metadata=_dir_store_metadata(home / "cache" / "screenshots"),
            deletion_domain="screenshot_cache",
            mutating_command=None,
        ),
        _store(
            store_id="cache.legacy_browser_screenshots_dir",
            kind="screenshot_cache",
            label="Legacy browser screenshot cache",
            path=home / "browser_screenshots",
            metadata=_dir_store_metadata(home / "browser_screenshots"),
            deletion_domain="screenshot_cache",
            mutating_command=None,
        ),
        _store(
            store_id="cache.audio_dir",
            kind="audio_cache",
            label="Audio cache",
            path=home / "cache" / "audio",
            metadata=_dir_store_metadata(home / "cache" / "audio"),
            deletion_domain="audio_video_cache",
            mutating_command=None,
        ),
        _store(
            store_id="cache.legacy_audio_dir",
            kind="audio_cache",
            label="Legacy audio cache",
            path=home / "audio_cache",
            metadata=_dir_store_metadata(home / "audio_cache"),
            deletion_domain="audio_video_cache",
            mutating_command=None,
        ),
        _store(
            store_id="cache.videos_dir",
            kind="video_cache",
            label="Video cache",
            path=home / "cache" / "videos",
            metadata=_dir_store_metadata(home / "cache" / "videos"),
            deletion_domain="audio_video_cache",
            mutating_command=None,
        ),
        _store(
            store_id="cache.legacy_video_dir",
            kind="video_cache",
            label="Legacy video cache",
            path=home / "video_cache",
            metadata=_dir_store_metadata(home / "video_cache"),
            deletion_domain="audio_video_cache",
            mutating_command=None,
        ),
        _store(
            store_id="cache.documents_dir",
            kind="document_cache",
            label="Document cache",
            path=home / "cache" / "documents",
            metadata=_dir_store_metadata(home / "cache" / "documents"),
            deletion_domain="document_cache",
            mutating_command=None,
        ),
        _store(
            store_id="cache.legacy_document_dir",
            kind="document_cache",
            label="Legacy document cache",
            path=home / "document_cache",
            metadata=_dir_store_metadata(home / "document_cache"),
            deletion_domain="document_cache",
            mutating_command=None,
        ),
        _store(
            store_id="backups.backups_dir",
            kind="backups",
            label="Hermes backups",
            path=home / "backups",
            metadata=_dir_store_metadata(home / "backups"),
            deletion_domain="snapshots_backups",
            mutating_command=None,
        ),
        _store(
            store_id="backups.checkpoints_dir",
            kind="checkpoints",
            label="Hermes checkpoints",
            path=home / "checkpoints",
            metadata=_dir_store_metadata(home / "checkpoints"),
            deletion_domain="snapshots_backups",
            mutating_command=None,
        ),
        _store(
            store_id="profiles.profiles_dir",
            kind="profiles",
            label="Profile-scoped Hermes homes",
            path=home / "profiles",
            metadata=_dir_store_metadata(home / "profiles"),
            deletion_domain="profile_memory_stores",
            mutating_command=None,
        ),
    ]

    status_counts: dict[str, int] = {}
    for store in stores:
        status_counts[store["status"]] = status_counts.get(store["status"], 0) + 1

    capacity_warnings = [
        store["id"]
        for store in stores
        if store["kind"] == "markdown_memory" and store["status"] in {"warn", "critical"}
    ]
    permission_warnings = [
        store["id"]
        for store in stores
        if store.get("permission_warning")
    ]
    checklist = _forget_checklist(home)
    reconciliation_status = "required" if active_provider else "not_configured"
    reconciliation_notes = [
        "Memory write hook exists for provider mirroring, but deletion/removal must be provider-specific and auditable.",
        "The audit command is read-only and does not reconcile or delete structured facts.",
    ]
    if active_provider == "holographic":
        reconciliation_notes.append(
            "Active holographic provider should be reconciled after markdown memory compaction, replacement, or deletion."
        )

    recommendations = []
    if capacity_warnings:
        recommendations.append(
            "Compact MEMORY.md/USER.md through explicit review; this command intentionally does not rewrite private memory."
        )
    if permission_warnings:
        recommendations.append("Harden Hermes-owned memory/log/session paths to owner-only permissions in the security phase.")
    if active_provider:
        recommendations.append("Run a provider-specific reconciliation pass after any curated memory replacement or deletion.")

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "owner": "hermes-memory-plane",
        "redacted": redacted,
        "metadata_only": True,
        "scope": {
            "hermes_home": _display_path(home),
            "profile_scoped": home.parent.name == "profiles",
        },
        "summary": {
            "total_stores": len(stores),
            "status_counts": status_counts,
            "capacity_warnings": capacity_warnings,
            "permission_warnings": permission_warnings,
            "forget_domains": [item["id"] for item in checklist],
            "active_memory_provider": active_provider or "",
            "reconciliation_status": reconciliation_status,
        },
        "stores": stores,
        "forget_checklist": checklist,
        "reconciliation": {
            "active_provider": active_provider or "",
            "status": reconciliation_status,
            "requires_explicit_user_request": True,
            "notes": reconciliation_notes,
        },
        "retention_policy": {
            "principle": "Keep curated memory compact, useful, and explicitly deletable across every store.",
            "delete_request_rule": "A forget/delete request must check every listed deletion domain and document residual backups.",
            "safe_default": "read_only_audit",
        },
        "safe_next_actions": recommendations,
    }


def format_markdown(audit: dict[str, Any]) -> str:
    """Render a human-readable metadata-only memory audit."""
    summary = audit.get("summary", {})
    lines = [
        "# Hermes Memory Audit",
        "",
        f"- Generated: {audit.get('generated_at')}",
        f"- Schema version: {audit.get('schema_version')}",
        f"- Hermes home: {audit.get('scope', {}).get('hermes_home')}",
        f"- Metadata only: {audit.get('metadata_only')}",
        f"- Active provider: {summary.get('active_memory_provider') or '(built-in only)'}",
        f"- Reconciliation: {summary.get('reconciliation_status')}",
        "",
        "## Store Status",
        "",
    ]
    for store in audit.get("stores", []):
        details = []
        if "char_count" in store:
            details.append(f"{store.get('char_count')}/{store.get('char_limit')} chars")
        if "file_count" in store:
            details.append(f"{store.get('file_count')} files")
        if store.get("mode"):
            details.append(f"mode {store.get('mode')}")
        detail_text = f" ({'; '.join(details)})" if details else ""
        lines.append(f"- {store.get('id')}: {store.get('status')}{detail_text}")
    lines.extend(["", "## Forget Checklist", ""])
    for item in audit.get("forget_checklist", []):
        command = item.get("safe_audit_command") or "n/a"
        lines.append(f"- {item.get('id')}: check {', '.join(item.get('store_ids') or [])}; safe audit: `{command}`")
    lines.extend(["", "## Safe Next Actions", ""])
    next_actions = audit.get("safe_next_actions") or []
    if next_actions:
        lines.extend(f"- {action}" for action in next_actions)
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def memory_audit_command(args: argparse.Namespace) -> int:
    audit = build_memory_audit(redacted=True)
    output_format = getattr(args, "format", None) or "json"
    if output_format == "markdown":
        print(format_markdown(audit), end="")
    else:
        print(json.dumps(audit, indent=2, sort_keys=True))
    return 0
