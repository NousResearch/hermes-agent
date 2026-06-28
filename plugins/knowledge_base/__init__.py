"""File-backed knowledge-base plugin."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from hermes_constants import get_hermes_home
from tools.registry import tool_error

_TOOLSET = "knowledge_base"
_DEFAULT_DIR = "knowledge-base"
_MAX_CONTENT_BYTES = 1_000_000
_MAX_SEARCH_RESULTS = 50


def _tool_result(**payload: Any) -> str:
    return json.dumps({"success": True, **payload}, ensure_ascii=False, indent=2)


def _load_kb_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        data = load_config()
    except Exception:
        return {}
    cfg = data.get("knowledge_base") if isinstance(data, dict) else None
    return cfg if isinstance(cfg, dict) else {}


def _root() -> Path:
    cfg = _load_kb_config()
    raw = str(cfg.get("path") or "").strip()
    if not raw:
        return get_hermes_home() / _DEFAULT_DIR
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return expanded
    return get_hermes_home() / expanded


def _clean_parts(raw_path: str, *, allow_empty: bool) -> tuple[str, ...]:
    cleaned = str(raw_path or "").strip().replace("\\", "/")
    if not cleaned:
        if allow_empty:
            return ()
        raise ValueError("path is required")
    parsed = PurePosixPath(cleaned)
    if parsed.is_absolute():
        raise ValueError("path must be relative to the knowledge base")
    parts = parsed.parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("path must not contain empty, '.', or '..' segments")
    return parts


def _resolve_path(raw_path: str, *, note: bool, allow_empty: bool = False) -> tuple[Path, str]:
    root = _root().resolve(strict=False)
    parts = _clean_parts(raw_path, allow_empty=allow_empty)
    rel = Path(*parts) if parts else Path()
    if note:
        if not rel.suffix:
            rel = rel.with_suffix(".md")
        elif rel.suffix.lower() != ".md":
            raise ValueError("knowledge-base notes must be markdown files")
    candidate = (root / rel).resolve(strict=False)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes the knowledge base") from exc
    return candidate, relative.as_posix()


def _configured_root() -> tuple[Path, str]:
    root = _root().resolve(strict=False)
    return root, root.as_posix()


def _kb_read(args: dict[str, Any], **_: Any) -> str:
    try:
        note_path, rel = _resolve_path(args.get("path", ""), note=True)
        if not note_path.exists():
            return tool_error(f"knowledge note not found: {rel}", success=False)
        if not note_path.is_file():
            return tool_error(f"knowledge path is not a file: {rel}", success=False)
        content = note_path.read_text(encoding="utf-8")
        return _tool_result(path=rel, content=content)
    except Exception as exc:
        return tool_error(str(exc), success=False)


def _kb_write(args: dict[str, Any], **_: Any) -> str:
    try:
        content = args.get("content")
        if not isinstance(content, str):
            return tool_error("content is required", success=False)
        encoded = content.encode("utf-8")
        if len(encoded) > _MAX_CONTENT_BYTES:
            return tool_error("content is too large for a single note", success=False)
        mode = str(args.get("mode") or "overwrite").strip().lower()
        if mode not in {"overwrite", "append"}:
            return tool_error("mode must be 'overwrite' or 'append'", success=False)
        note_path, rel = _resolve_path(args.get("path", ""), note=True)
        note_path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append" and note_path.exists():
            existing = note_path.read_text(encoding="utf-8")
            separator = "" if existing.endswith("\n") or not existing else "\n"
            note_path.write_text(f"{existing}{separator}{content}", encoding="utf-8")
        else:
            note_path.write_text(content, encoding="utf-8")
        return _tool_result(path=rel, bytes=len(encoded), mode=mode)
    except Exception as exc:
        return tool_error(str(exc), success=False)


def _iter_notes(base: Path):
    if not base.exists():
        return
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".md":
            yield path


def _snippet(text: str, needle: str) -> str:
    lower = text.lower()
    index = lower.find(needle)
    if index < 0:
        return text[:240].strip()
    start = max(0, index - 90)
    end = min(len(text), index + len(needle) + 150)
    return text[start:end].strip()


def _title_for(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or path.stem
    return path.stem


def _kb_search(args: dict[str, Any], **_: Any) -> str:
    try:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required", success=False)
        raw_limit = args.get("limit", 10)
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            return tool_error("limit must be an integer", success=False)
        limit = max(1, min(limit, _MAX_SEARCH_RESULTS))
        base, _ = _resolve_path(args.get("path", ""), note=False, allow_empty=True)
        root, root_display = _configured_root()
        needle = query.lower()
        results: list[dict[str, Any]] = []
        for note_path in _iter_notes(base):
            text = note_path.read_text(encoding="utf-8", errors="replace")
            haystack = text.lower()
            score = haystack.count(needle)
            if score <= 0:
                continue
            rel = note_path.resolve(strict=False).relative_to(root).as_posix()
            results.append({
                "path": rel,
                "title": _title_for(note_path, text),
                "score": score,
                "snippet": _snippet(text, needle),
            })
        results.sort(key=lambda item: (-item["score"], item["path"]))
        return _tool_result(root=root_display, query=query, results=results[:limit])
    except Exception as exc:
        return tool_error(str(exc), success=False)


def _kb_list(args: dict[str, Any], **_: Any) -> str:
    try:
        folder, rel = _resolve_path(args.get("path", ""), note=False, allow_empty=True)
        root, root_display = _configured_root()
        if not folder.exists():
            return _tool_result(root=root_display, path=rel, directories=[], notes=[])
        if not folder.is_dir():
            return tool_error(f"knowledge path is not a directory: {rel}", success=False)
        directories = []
        notes = []
        for child in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
            child_rel = child.resolve(strict=False).relative_to(root).as_posix()
            if child.is_dir():
                directories.append(child_rel)
            elif child.is_file() and child.suffix.lower() == ".md":
                notes.append(child_rel)
        return _tool_result(root=root_display, path=rel, directories=directories, notes=notes)
    except Exception as exc:
        return tool_error(str(exc), success=False)


KB_READ_SCHEMA = {
    "name": "kb_read",
    "description": "Read a markdown note from the configured knowledge base.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative note path. .md is appended when omitted."},
        },
        "required": ["path"],
    },
}

KB_WRITE_SCHEMA = {
    "name": "kb_write",
    "description": "Create, replace, or append to a markdown note in the configured knowledge base.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative note path. .md is appended when omitted."},
            "content": {"type": "string", "description": "Markdown content to write."},
            "mode": {"type": "string", "enum": ["overwrite", "append"], "description": "Write mode. Defaults to overwrite."},
        },
        "required": ["path", "content"],
    },
}

KB_SEARCH_SCHEMA = {
    "name": "kb_search",
    "description": "Search markdown notes in the configured knowledge base.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Case-insensitive text to find in notes."},
            "path": {"type": "string", "description": "Optional relative folder to search under."},
            "limit": {"type": "integer", "description": "Maximum results. Defaults to 10, max 50."},
        },
        "required": ["query"],
    },
}

KB_LIST_SCHEMA = {
    "name": "kb_list",
    "description": "List folders and markdown notes in the configured knowledge base.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Optional relative folder to list."},
        },
        "required": [],
    },
}

_TOOLS = (
    ("kb_read", KB_READ_SCHEMA, _kb_read),
    ("kb_write", KB_WRITE_SCHEMA, _kb_write),
    ("kb_search", KB_SEARCH_SCHEMA, _kb_search),
    ("kb_list", KB_LIST_SCHEMA, _kb_list),
)


def register(ctx) -> None:
    for name, schema, handler in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset=_TOOLSET,
            schema=schema,
            handler=handler,
        )
