"""Markdown/QMD export helpers for IX Agency sessions.

This module is intentionally filesystem-only: it formats already-exported
SessionDB dictionaries and writes them to user-selected export directories. It
must not mutate state.db or call delete/prune/archive APIs.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPORTER_VERSION = "hermes sessions export (md/qmd) v1"
_SHA_LINE_RE = re.compile(r"- SHA256 of exported body: `([0-9a-f]{64})`")


def _iso_timestamp(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return str(value)
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _frontmatter_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(str(value), ensure_ascii=False)


def _frontmatter_line(key: str, value: Any) -> str:
    return f"{key}: {_frontmatter_value(value)}"


def _message_heading(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "message")
    label = role.capitalize()
    name = message.get("name") or message.get("tool_name")
    if role == "tool" and name:
        label = f"Tool — {name}"
    timestamp = _iso_timestamp(message.get("created_at") or message.get("timestamp"))
    return f"### {label}{' — ' + timestamp if timestamp else ''}"


def _render_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.rstrip()
    return "```json\n" + json.dumps(content, ensure_ascii=False, indent=2) + "\n```"


def _render_tool_calls(tool_calls: Any) -> str:
    if not tool_calls:
        return ""
    return "\n\n## Tool calls\n\n```json\n" + json.dumps(tool_calls, ensure_ascii=False, indent=2) + "\n```"


def _session_id(session: dict[str, Any]) -> str:
    return str(session.get("id") or session.get("session_id") or "unknown-session")


def _segments(session: dict[str, Any]) -> list[dict[str, Any]]:
    segments = session.get("segments")
    if isinstance(segments, list) and segments:
        return [s for s in segments if isinstance(s, dict)]
    return [session]


def _message_count(session: dict[str, Any]) -> int:
    return sum(len(seg.get("messages") or []) for seg in _segments(session))


def _render_messages(session: dict[str, Any]) -> str:
    parts: list[str] = ["## Messages\n"]
    segments = _segments(session)
    total_messages = _message_count(session)
    if total_messages == 0:
        parts.append("_No messages in this session._\n")
        return "\n".join(parts).rstrip() + "\n"

    multi_segment = len(segments) > 1
    for segment in segments:
        if multi_segment:
            parts.append(f"## Compression segment: {_session_id(segment)}\n")
        for message in list(segment.get("messages") or []):
            parts.append(_message_heading(message) + "\n")
            rendered_content = _render_content(message.get("content"))
            if rendered_content:
                parts.append(rendered_content + "\n")
            tool_calls = _render_tool_calls(message.get("tool_calls"))
            if tool_calls:
                parts.append(tool_calls + "\n")
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _export_body_without_hash(session: dict[str, Any], *, fmt: str, exported_at: float) -> str:
    session_id = _session_id(session)
    title = session.get("title") or session_id
    provider = session.get("billing_provider") or session.get("provider")
    started_at = _iso_timestamp(session.get("started_at") or session.get("created_at"))
    last_active = _iso_timestamp(session.get("last_active") or session.get("updated_at"))
    ended_at = _iso_timestamp(session.get("ended_at"))
    exported_iso = _iso_timestamp(exported_at)
    message_count = _message_count(session)

    frontmatter = [
        "---",
        _frontmatter_line("session_id", session_id),
        _frontmatter_line("title", session.get("title")),
        _frontmatter_line("source", session.get("source")),
        _frontmatter_line("created_at", started_at),
        _frontmatter_line("updated_at", last_active),
        _frontmatter_line("ended_at", ended_at),
        _frontmatter_line("model", session.get("model")),
        _frontmatter_line("provider", provider),
        _frontmatter_line("cwd", session.get("cwd")),
        _frontmatter_line("archived", bool(session.get("archived"))),
        _frontmatter_line("message_count", message_count),
        _frontmatter_line("tool_call_count", session.get("tool_call_count") or 0),
    ]
    if session.get("lineage_session_ids"):
        frontmatter.append(_frontmatter_line("lineage_session_ids", session.get("lineage_session_ids")))
    frontmatter.extend([
        _frontmatter_line("format", fmt),
        _frontmatter_line("exported_at", exported_iso),
        _frontmatter_line("exporter", EXPORTER_VERSION),
        "---",
        "",
    ])

    parts = ["\n".join(frontmatter), f"# {title}\n"]
    parts.append(f"Session ID: `{session_id}`\n")
    if session.get("source"):
        parts.append(f"Source: `{session.get('source')}`\n")
    if session.get("cwd"):
        parts.append(f"Working directory: `{session.get('cwd')}`\n")

    parts.append(_render_messages(session))
    parts.append("## Export verification\n")
    parts.append(f"- Session id: `{session_id}`")
    parts.append(f"- Exported messages: `{message_count}`")
    parts.append(f"- Source DB message count at export: `{session.get('message_count', message_count)}`")
    parts.append(f"- Exported at: `{exported_iso}`")
    parts.append("- SHA256 of exported body: `__SHA256_PLACEHOLDER__`")
    return "\n".join(parts).rstrip() + "\n"


def _body_for_digest(text: str) -> str:
    return _SHA_LINE_RE.sub("- SHA256 of exported body: `pending`", text)


def render_session_markdown(
    session: dict[str, Any], *, fmt: str = "md", include_verification: bool = True
) -> str:
    """Render a SessionDB export dictionary as Markdown/QMD text."""
    if fmt not in {"md", "qmd"}:
        raise ValueError("fmt must be 'md' or 'qmd'")
    exported_at = time.time()
    body = _export_body_without_hash(session, fmt=fmt, exported_at=exported_at)
    digest_body = body.replace("`__SHA256_PLACEHOLDER__`", "`pending`")
    digest = hashlib.sha256(digest_body.encode("utf-8")).hexdigest()
    if include_verification:
        return body.replace("__SHA256_PLACEHOLDER__", digest)
    before_verification = body.split("\n## Export verification\n", 1)[0].rstrip() + "\n"
    return before_verification


def safe_session_filename(session: dict[str, Any], *, fmt: str = "md") -> str:
    """Return a deterministic, path-safe filename for a session export."""
    if fmt not in {"md", "qmd"}:
        raise ValueError("fmt must be 'md' or 'qmd'")
    session_id = _session_id(session)
    title = str(session.get("title") or "session")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", title).strip(".-_").lower()
    if not slug:
        slug = "session"
    slug = slug[:60]
    return f"{session_id}-{slug}.{fmt}"


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_export_file(path: Path | str, session: dict[str, Any]) -> tuple[bool, str]:
    p = Path(path)
    if not p.exists():
        return False, "file missing"
    text = p.read_text(encoding="utf-8")
    match = _SHA_LINE_RE.search(text)
    if not match:
        return False, "sha256 marker missing"
    actual = hashlib.sha256(_body_for_digest(text).encode("utf-8")).hexdigest()
    if actual != match.group(1):
        return False, "sha256 mismatch"
    expected_count = _message_count(session)
    if f"- Exported messages: `{expected_count}`" not in text:
        return False, "message count mismatch"
    if f"- Session id: `{_session_id(session)}`" not in text:
        return False, "session id mismatch"
    return True, "ok"


def redact_session_data(session: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of a session export dict with secrets redacted.

    Runs every message's content and tool-call arguments through the
    force-mode redaction pass (``agent.redact.redact_sensitive_text``), so
    API keys, tokens, and credentials that appeared in tool output never
    land in plaintext export files. Force mode ignores the user's global
    ``security.redact_secrets`` preference — an explicit ``--redact`` export
    must never emit raw secrets.
    """
    from agent.redact import redact_sensitive_text

    def _clean(value: Any) -> Any:
        if isinstance(value, str):
            return redact_sensitive_text(value, force=True)
        if isinstance(value, list):
            return [_clean(v) for v in value]
        if isinstance(value, dict):
            return {k: _clean(v) for k, v in value.items()}
        return value

    redacted = dict(session)
    for key in ("messages", "segments"):
        if key in redacted and redacted[key] is not None:
            redacted[key] = _clean(redacted[key])
    return redacted


def write_session_markdown(
    session: dict[str, Any], output_dir: Path | str, *, fmt: str = "md", force: bool = False
) -> Path:
    """Write a Markdown/QMD export file and return its path.

    Raises FileExistsError when the destination exists and force=False.
    """
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / safe_session_filename(session, fmt=fmt)
    if path.exists() and not force:
        raise FileExistsError(str(path))
    path.write_text(render_session_markdown(session, fmt=fmt), encoding="utf-8")
    return path


def append_manifest_entry(output_dir: Path | str, session: dict[str, Any], path: Path | str, *, fmt: str) -> Path:
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    export_path = Path(path)
    entry = {
        "session_id": _session_id(session),
        "lineage_session_ids": session.get("lineage_session_ids") or [_session_id(session)],
        "path": str(export_path),
        "format": fmt,
        "message_count": _message_count(session),
        "sha256": file_sha256(export_path),
        "exported_at": time.time(),
    }
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return manifest
