"""Safe Obsidian bridge plugin for Hermes.

The bridge is deliberately conservative:

* reads are limited to Markdown-like vault files;
* sensitive paths are denied;
* owner-only zones (90-Owner-Private/) are denied unless HERMES_OWNER=1;
* writes are limited to 95-Inbox-Lab/review/ unless explicitly changed in code;
* every write and failed write is recorded in an audit log.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


_DEFAULT_MAX_CHARS = 20000
_SAFE_SUFFIXES = {".md", ".base", ".canvas", ".json", ".yaml", ".yml", ".txt"}
_SENSITIVE_PARTS = {
    ".git",
    ".obsidian",
    ".trash",
    ".env",
    "auth.json",
    "auth.lock",
    "config.yaml",
    "config.yaml.bak",
}

# โซนที่เขียนได้ (AI ส่งงานเข้าคิวรอตรวจเท่านั้น)
_REVIEW_WRITE_PREFIX = "95-Inbox-Lab/review/"

# โซนลับเฉพาะเจ้าของ อ่านได้เมื่อ HERMES_OWNER=1 เท่านั้น
_OWNER_ONLY_PREFIXES = ("90-Owner-Private/",)


def _is_owner_only(rel: Path) -> bool:
    posix = rel.as_posix()
    if any(posix.startswith(p) for p in _OWNER_ONLY_PREFIXES):
        return True
    # โฟลเดอร์ owner-private ภายในแต่ละแผนก เช่น 20-Departments/05-Accounting/owner-private/
    return "owner-private" in rel.parts


def _owner_mode() -> bool:
    return os.environ.get("HERMES_OWNER", "").strip() == "1"


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _vault_root() -> Path:
    configured = os.environ.get("HERMES_OBSIDIAN_VAULT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / "ObsidianVault" / "HermesAgent").resolve()


def _audit_log_path() -> Path:
    return get_hermes_home() / "obsidian-safe-bridge" / "audit.jsonl"


def _is_sensitive(relative_path: Path) -> bool:
    return any(part in _SENSITIVE_PARTS or part.startswith(".env") for part in relative_path.parts)


def _resolve_vault_relative_path(path: str) -> tuple[Path, Path]:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path is required")
    rel = Path(path.strip())
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("path must be a safe relative path inside the vault")
    if _is_sensitive(rel):
        raise ValueError("path is sensitive and cannot be accessed through this bridge")
    if _is_owner_only(rel) and not _owner_mode():
        raise ValueError("path is owner-only; set HERMES_OWNER=1 to access")
    if rel.suffix and rel.suffix.lower() not in _SAFE_SUFFIXES:
        raise ValueError(f"unsupported file type: {rel.suffix}")

    root = _vault_root()
    target = (root / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes the Obsidian vault") from exc
    return root, target


def _audit(action: str, path: str, ok: bool, reason: str = "") -> None:
    payload = {
        "ts": int(time.time()),
        "action": action,
        "path": path,
        "ok": ok,
        "reason": reason,
    }
    audit_path = _audit_log_path()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def obsidian_safe_read(args: dict[str, Any]) -> str:
    path = str(args.get("path", ""))
    max_chars = int(args.get("max_chars") or _DEFAULT_MAX_CHARS)
    try:
        root, target = _resolve_vault_relative_path(path)
        if not target.exists() or not target.is_file():
            return _json({"ok": False, "error": "file not found", "path": path})
        content = target.read_text(encoding="utf-8")
        truncated = len(content) > max_chars
        if truncated:
            content = content[:max_chars]
        rel = target.relative_to(root).as_posix()
        return _json({
            "ok": True,
            "path": rel,
            "content": content,
            "truncated": truncated,
            "size": target.stat().st_size,
        })
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "path": path})


def obsidian_safe_list(args: dict[str, Any]) -> str:
    prefix = str(args.get("prefix", "") or ".")
    max_results = int(args.get("max_results") or 100)
    try:
        root, target = _resolve_vault_relative_path(prefix)
        if not target.exists():
            return _json({"ok": True, "prefix": prefix, "files": []})
        base = target if target.is_dir() else target.parent
        files: list[str] = []
        for item in sorted(base.rglob("*.md")):
            try:
                rel = item.relative_to(root)
            except ValueError:
                continue
            if _is_sensitive(rel):
                continue
            if _is_owner_only(rel) and not _owner_mode():
                continue
            files.append(rel.as_posix())
            if len(files) >= max_results:
                break
        return _json({"ok": True, "prefix": prefix, "files": files})
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "prefix": prefix})


def obsidian_safe_write_review(args: dict[str, Any]) -> str:
    path = str(args.get("path", ""))
    content = str(args.get("content", ""))
    overwrite = bool(args.get("overwrite", False))
    try:
        root, target = _resolve_vault_relative_path(path)
        rel = target.relative_to(root).as_posix()
        if not rel.startswith(_REVIEW_WRITE_PREFIX):
            raise ValueError(
                f"writes are limited to {_REVIEW_WRITE_PREFIX} for owner review"
            )
        if target.exists() and not overwrite:
            raise ValueError("target exists; pass overwrite=true to replace it")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        _audit("write_review", rel, True)
        return _json({"ok": True, "path": rel, "bytes": target.stat().st_size})
    except Exception as exc:
        _audit("write_review", path, False, str(exc))
        return _json({"ok": False, "error": str(exc), "path": path})


def obsidian_safe_audit_log(args: dict[str, Any]) -> str:
    limit = int(args.get("limit") or 50)
    audit_path = _audit_log_path()
    if not audit_path.exists():
        return _json({"ok": True, "entries": []})
    lines = audit_path.read_text(encoding="utf-8").splitlines()[-limit:]
    entries = [json.loads(line) for line in lines if line.strip()]
    return _json({"ok": True, "entries": entries})


def register(ctx) -> None:
    tools = [
        (
            "obsidian_safe_read",
            "Read a safe file from the configured Obsidian vault without writing anything.",
            obsidian_safe_read,
            {"path": {"type": "string"}, "max_chars": {"type": "integer"}},
        ),
        (
            "obsidian_safe_list",
            "List Markdown files under a safe Obsidian vault prefix.",
            obsidian_safe_list,
            {"prefix": {"type": "string"}, "max_results": {"type": "integer"}},
        ),
        (
            "obsidian_safe_write_review",
            "Write only to Obsidian 95-Inbox-Lab/review/ so the owner can review before promotion.",
            obsidian_safe_write_review,
            {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "overwrite": {"type": "boolean"},
            },
        ),
        (
            "obsidian_safe_audit_log",
            "Read the local audit log for Obsidian bridge actions.",
            obsidian_safe_audit_log,
            {"limit": {"type": "integer"}},
        ),
    ]
    for name, description, handler, properties in tools:
        ctx.register_tool(
            name=name,
            toolset="obsidian_safe_bridge",
            schema={
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "additionalProperties": False,
                },
            },
            handler=lambda args, _handler=handler, **_: _handler(args),
        )
