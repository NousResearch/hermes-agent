"""Markdown/QMD export helpers for Hermes sessions.

Filesystem-only: formats already-exported SessionDB dicts and writes them to user-selected export
directories. Must not mutate state.db or call delete/prune/archive APIs.
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
_SHA_PLACEHOLDER = "__SHA256_PLACEHOLDER__"
_VERIFICATION_HEADING = "## Export verification"


def _iso_timestamp(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return str(value)
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _frontmatter_line(key: str, value: Any) -> str:
    if value is None or isinstance(value, bool):
        shown = {None: "null", True: "true", False: "false"}[value]
    else:
        shown = json.dumps(value if isinstance(value, (int, float, list)) else str(value), ensure_ascii=False)
    return f"{key}: {shown}"


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def _message_heading(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "message")
    name = message.get("name") or message.get("tool_name")
    label = f"Tool — {name}" if role == "tool" and name else role.capitalize()
    timestamp = _iso_timestamp(message.get("created_at") or message.get("timestamp"))
    return f"### {label}{' — ' + timestamp if timestamp else ''}"


def _session_id(session: dict[str, Any]) -> str:
    return str(session.get("id") or session.get("session_id") or "unknown-session")


def _segments(session: dict[str, Any]) -> list[dict[str, Any]]:
    segments = session.get("segments")
    return [s for s in segments if isinstance(s, dict)] if isinstance(segments, list) and segments else [session]


def _message_count(session: dict[str, Any]) -> int:
    return sum(len(seg.get("messages") or []) for seg in _segments(session))


def _render_messages(session: dict[str, Any]) -> str:
    segments = _segments(session)
    if _message_count(session) == 0:
        return "## Messages\n\n_No messages in this session._\n"
    parts: list[str] = ["## Messages\n"]
    for segment in segments:
        if len(segments) > 1:
            parts.append(f"## Compression segment: {_session_id(segment)}\n")
        for message in list(segment.get("messages") or []):
            parts.append(_message_heading(message) + "\n")
            content = message.get("content")
            rendered = "" if content is None else content.rstrip() if isinstance(content, str) else _json_block(content)
            if rendered:
                parts.append(rendered + "\n")
            if tool_calls := message.get("tool_calls"):
                parts.append("\n\n## Tool calls\n\n" + _json_block(tool_calls) + "\n")
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _export_body_without_hash(session: dict[str, Any], *, fmt: str, exported_at: float) -> str:
    session_id = _session_id(session)
    exported_iso = _iso_timestamp(exported_at)
    message_count = _message_count(session)
    fields = [
        ("session_id", session_id),
        ("title", session.get("title")),
        ("source", session.get("source")),
        ("created_at", _iso_timestamp(session.get("started_at") or session.get("created_at"))),
        ("updated_at", _iso_timestamp(session.get("last_active") or session.get("updated_at"))),
        ("ended_at", _iso_timestamp(session.get("ended_at"))),
        ("model", session.get("model")),
        ("provider", session.get("billing_provider") or session.get("provider")),
        ("cwd", session.get("cwd")),
        ("archived", bool(session.get("archived"))),
        ("message_count", message_count),
        ("tool_call_count", session.get("tool_call_count") or 0),
        *([("lineage_session_ids", session["lineage_session_ids"])] if session.get("lineage_session_ids") else []),
        ("format", fmt), ("exported_at", exported_iso), ("exporter", EXPORTER_VERSION),
    ]
    parts = [
        "\n".join(["---", *(_frontmatter_line(k, v) for k, v in fields), "---", ""]),
        f"# {session.get('title') or session_id}\n", f"Session ID: `{session_id}`\n",
        *([f"Source: `{session.get('source')}`\n"] if session.get("source") else []),
        *([f"Working directory: `{session.get('cwd')}`\n"] if session.get("cwd") else []),
        _render_messages(session),
        f"{_VERIFICATION_HEADING}\n",
        f"- Session id: `{session_id}`",
        f"- Exported messages: `{message_count}`",
        f"- Source DB message count at export: `{session.get('message_count', message_count)}`",
        f"- Exported at: `{exported_iso}`",
        f"- SHA256 of exported body: `{_SHA_PLACEHOLDER}`",
    ]
    return "\n".join(parts).rstrip() + "\n"


def _check_fmt(fmt: str) -> None:
    if fmt not in {"md", "qmd"}:
        raise ValueError("fmt must be 'md' or 'qmd'")


def render_session_markdown(
    session: dict[str, Any], *, fmt: str = "md", include_verification: bool = True
) -> str:
    """Render a SessionDB export dictionary as Markdown/QMD text."""
    _check_fmt(fmt)
    body = _export_body_without_hash(session, fmt=fmt, exported_at=time.time())
    if not include_verification:
        return body.split(f"\n{_VERIFICATION_HEADING}\n", 1)[0].rstrip() + "\n"
    # The digest covers the body with the SHA line set to `pending`, which is what verify recomputes.
    digest_body = body.replace(f"`{_SHA_PLACEHOLDER}`", "`pending`")
    return body.replace(_SHA_PLACEHOLDER, hashlib.sha256(digest_body.encode("utf-8")).hexdigest())


def safe_session_filename(session: dict[str, Any], *, fmt: str = "md") -> str:
    """Return a deterministic, path-safe filename for a session export."""
    _check_fmt(fmt)
    title = str(session.get("title") or "session")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", title).strip(".-_").lower() or "session"
    return f"{_session_id(session)}-{slug[:60]}.{fmt}"


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_export_file(path: Path | str, session: dict[str, Any]) -> tuple[bool, str]:
    if not Path(path).exists():
        return False, "file missing"
    text = Path(path).read_text(encoding="utf-8")
    match = _SHA_LINE_RE.search(text)
    if not match:
        return False, "sha256 marker missing"
    digest_body = _SHA_LINE_RE.sub("- SHA256 of exported body: `pending`", text)
    if hashlib.sha256(digest_body.encode("utf-8")).hexdigest() != match.group(1):
        return False, "sha256 mismatch"
    if f"- Exported messages: `{_message_count(session)}`" not in text:
        return False, "message count mismatch"
    if f"- Session id: `{_session_id(session)}`" not in text:
        return False, "session id mismatch"
    return True, "ok"


def redact_session_data(session: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a session export dict with secrets redacted.

    Every message's content and tool-call arguments go through the force-mode redaction pass
    (``agent.redact.redact_sensitive_text``) so credentials in tool output never land in exports.
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

    return {**session, **{k: _clean(session[k]) for k in ("messages", "segments") if session.get(k) is not None}}


def _export_dir(output_dir: Path | str) -> Path:
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_session_markdown(
    session: dict[str, Any], output_dir: Path | str, *, fmt: str = "md", force: bool = False
) -> Path:
    """Write a Markdown/QMD export file and return its path."""
    path = _export_dir(output_dir) / safe_session_filename(session, fmt=fmt)
    if path.exists() and not force:
        raise FileExistsError(str(path))
    path.write_text(render_session_markdown(session, fmt=fmt), encoding="utf-8")
    return path


def append_manifest_entry(output_dir: Path | str, session: dict[str, Any], path: Path | str, *, fmt: str) -> Path:
    entry = {
        "session_id": _session_id(session),
        "lineage_session_ids": session.get("lineage_session_ids") or [_session_id(session)],
        "path": str(Path(path)),
        "format": fmt,
        "message_count": _message_count(session),
        "sha256": file_sha256(path),
        "exported_at": time.time(),
    }
    manifest = _export_dir(output_dir) / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return manifest
