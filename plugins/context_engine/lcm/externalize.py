"""Helpers for optional large-payload externalization.

Externalization started as a pre-compaction serializer guard for oversized tool
outputs. The same durable payload format is also used by the ingest path so
obvious oversized content can be kept out of SQLite/FTS while still being
recoverable through the LCM inspection and expansion tools.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict

DEFAULT_LARGE_OUTPUT_DIRNAME = "lcm-large-outputs"
_EXTERNALIZED_REF_RE = re.compile(
    r"\[(?:Externalized|GC'd externalized) (?:tool output|payload):.*?;\s*ref=([^;\]\s]+)\]"
)


def _placeholder_metadata(value: Any) -> str:
    text = str(value or "?")
    safe = re.sub(r"[^A-Za-z0-9_.:/-]+", "-", text).strip("-")
    return (safe or "?")[:120]


logger = logging.getLogger(__name__)


def _tool_call_stub(tool_call_id: str) -> str:
    return (tool_call_id or "tool-result").replace("/", "-").replace(":", "-")[:48]


def _safe_stub(value: str, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", (value or "").strip())
    return (text or fallback)[:48]


def _content_digest_prefix(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()[:12]


def get_large_output_storage_dir(config, hermes_home: str = "", *, create: bool) -> Path:
    configured = getattr(config, "large_output_externalization_path", "") or ""
    if configured:
        path = Path(configured).expanduser().resolve()
        # Check containment for configured paths when LCM_HERMES_BASE_DIR is set
        env_base = os.environ.get("LCM_HERMES_BASE_DIR")
        if env_base:
            allowed_base = Path(env_base).expanduser().resolve()
            try:
                path.relative_to(allowed_base)
            except ValueError:
                raise ValueError(f"Path {path} is not within allowed base {allowed_base}")
    else:
        base = Path(hermes_home).expanduser().resolve() if hermes_home else Path("~/.hermes").expanduser().resolve()
        path = base / DEFAULT_LARGE_OUTPUT_DIRNAME
        # Check containment within allowed base for default/hermes_home-based paths
        # Only enforced when LCM_HERMES_BASE_DIR is explicitly set
        env_base = os.environ.get("LCM_HERMES_BASE_DIR")
        if env_base:
            allowed_base = Path(env_base).expanduser().resolve()
            try:
                path.relative_to(allowed_base)
            except ValueError:
                raise ValueError(f"Path {path} is not within allowed base {allowed_base}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
        try:
            path.chmod(0o700)
        except OSError as exc:
            logger.warning("Could not restrict LCM externalized payload directory permissions for %s: %s", path, exc)
    return path


def _write_externalized_payload(path: Path, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            fd = -1
    finally:
        if fd >= 0:
            os.close(fd)


def resolve_large_output_storage_dir(config, hermes_home: str = "") -> Path:
    return get_large_output_storage_dir(config, hermes_home=hermes_home, create=True)


def _externalized_summary(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ref": path.name,
        "kind": payload.get("kind", "tool_result"),
        "tool_call_id": payload.get("tool_call_id", ""),
        "role": payload.get("role", ""),
        "session_id": payload.get("session_id", ""),
        "field_path": payload.get("field_path", ""),
        "content_chars": payload.get("content_chars", len(payload.get("content", ""))),
        "content_bytes": payload.get("content_bytes", len((payload.get("content", "") or "").encode("utf-8"))),
        "created_at": payload.get("created_at"),
    }


def _build_externalized_placeholder(summary: Dict[str, Any]) -> str:
    kind = _placeholder_metadata(summary.get("kind", "tool_result") or "tool_result")
    if kind != "tool_result":
        role = _placeholder_metadata(summary.get("role") or "?")
        return (
            f"[Externalized payload: kind={kind}; role={role}; "
            f"chars={summary.get('content_chars', 0)}; bytes={summary.get('content_bytes', 0)}; "
            f"ref={summary.get('ref', '')}]"
        )
    return (
        f"[Externalized tool output: tool_call_id={_placeholder_metadata(summary.get('tool_call_id') or '?')}; "
        f"chars={summary.get('content_chars', 0)}; bytes={summary.get('content_bytes', 0)}; ref={summary.get('ref', '')}]"
    )


def build_transcript_gc_placeholder(summary: Dict[str, Any]) -> str:
    kind = _placeholder_metadata(summary.get("kind", "tool_result") or "tool_result")
    if kind != "tool_result":
        role = _placeholder_metadata(summary.get("role") or "?")
        return (
            f"[GC'd externalized payload: kind={kind}; role={role}; "
            f"chars={summary.get('content_chars', 0)}; ref={summary.get('ref', '')}]"
        )
    return (
        f"[GC'd externalized tool output: tool_call_id={_placeholder_metadata(summary.get('tool_call_id') or '?')}; "
        f"chars={summary.get('content_chars', 0)}; ref={summary.get('ref', '')}]"
    )


def extract_externalized_ref(text: str) -> str | None:
    refs = extract_externalized_refs(text)
    return refs[0] if refs else None


def extract_externalized_refs(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    refs: list[str] = []
    for match in _EXTERNALIZED_REF_RE.finditer(text):
        ref = match.group(1).strip()
        if not ref or "/" in ref or "\\" in ref or Path(ref).name != ref or ref in refs:
            continue
        refs.append(ref)
    return refs


def is_externalized_placeholder(text: str) -> bool:
    """Return true for a compact legacy externalized payload/tool marker."""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped or len(stripped) > 512:
        return False
    return bool(_EXTERNALIZED_REF_RE.fullmatch(stripped))


def load_externalized_payload(ref: str, *, config, hermes_home: str = "") -> Dict[str, Any] | None:
    if not ref or Path(ref).name != ref:
        return None
    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    if not storage_dir.exists() or not storage_dir.is_dir():
        return None
    path = storage_dir / ref
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    summary = _externalized_summary(path, payload)
    summary["content"] = payload.get("content", "")
    return summary


def reassign_externalized_payloads(
    old_session_id: str,
    new_session_id: str,
    *,
    config,
    hermes_home: str = "",
) -> int:
    """Move externalized payload session metadata across a logical session boundary."""
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return 0
    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    if not storage_dir.exists() or not storage_dir.is_dir():
        return 0

    moved = 0
    for path in storage_dir.glob("*.json"):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (payload.get("session_id") or "") != old_session_id:
            continue
        payload["session_id"] = new_session_id
        tmp_path = path.with_name(f"{path.name}.tmp")
        try:
            _write_externalized_payload(tmp_path, payload)
            tmp_path.replace(path)
        except OSError as exc:
            logger.warning("Externalized payload session reassignment skipped for %s: %s", path.name, exc)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        moved += 1
    return moved


def find_externalized_payload_for_message(
    content: str,
    *,
    tool_call_id: str = "",
    session_id: str = "",
    kind: str | None = "tool_result",
    role: str = "",
    config,
    hermes_home: str = "",
) -> Dict[str, Any] | None:
    if not content:
        return None
    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    if not storage_dir.exists() or not storage_dir.is_dir():
        return None

    digest_prefix = _content_digest_prefix(content)
    candidates = sorted(storage_dir.glob(f"*_{digest_prefix}_*.json"))
    fallback_match = None
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if kind is not None and payload.get("kind", "tool_result") != kind:
            continue
        if (payload.get("tool_call_id") or "") != (tool_call_id or ""):
            continue
        payload_role = payload.get("role") or ""
        if role and payload_role and payload_role != role:
            continue
        if payload.get("content") != content:
            continue
        summary = _externalized_summary(path, payload)
        payload_session_id = (payload.get("session_id") or "")
        if session_id:
            if payload_session_id == session_id:
                return summary
            continue
        if fallback_match is None:
            fallback_match = summary
    return fallback_match


def externalize_ingest_payload(
    content: str,
    *,
    role: str = "",
    session_id: str = "",
    field_path: str = "",
    config,
    hermes_home: str = "",
    kind: str = "ingest_payload",
) -> Dict[str, Any] | None:
    if not content:
        return None
    try:
        storage_dir = resolve_large_output_storage_dir(config, hermes_home=hermes_home)
    except OSError as exc:
        logger.warning("LCM ingest payload externalization skipped (non-blocking): %s", exc)
        return None

    digest_prefix = _content_digest_prefix(content)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    unique_suffix = f"{time.time_ns():x}"
    kind_stub = _safe_stub(kind, "ingest_payload")
    field_stub = re.sub(r"[^A-Za-z0-9_.-]+", "-", field_path or "payload")[:48]
    filename = f"{timestamp}_{kind_stub}_{field_stub}_{digest_prefix}_{unique_suffix}.json"
    path = storage_dir / filename
    payload = {
        "kind": kind,
        "role": role,
        "session_id": session_id,
        "field_path": field_path,
        "content": content,
        "content_chars": len(content),
        "content_bytes": len(content.encode("utf-8")),
        "created_at": time.time(),
    }
    try:
        _write_externalized_payload(path, payload)
    except OSError as exc:
        logger.warning("LCM ingest payload externalization skipped (non-blocking): %s", exc)
        return None

    summary = _externalized_summary(path, payload)
    placeholder = (
        f"[Externalized LCM ingest payload: kind={_placeholder_metadata(summary.get('kind') or kind)}; "
        f"field={_placeholder_metadata(summary.get('field_path') or '?')}; chars={summary.get('content_chars', 0)}; "
        f"bytes={summary.get('content_bytes', 0)}; ref={summary.get('ref', '')}]"
    )
    return {
        "placeholder": placeholder,
        "path": path,
        "payload": payload,
    }


def maybe_externalize_tool_output(
    content: str,
    *,
    tool_call_id: str = "",
    session_id: str = "",
    config,
    hermes_home: str = "",
) -> Dict[str, Any] | None:
    return maybe_externalize_payload(
        content,
        kind="tool_result",
        tool_call_id=tool_call_id,
        session_id=session_id,
        role="tool",
        config=config,
        hermes_home=hermes_home,
    )


def maybe_externalize_payload(
    content: str,
    *,
    kind: str = "raw_payload",
    tool_call_id: str = "",
    session_id: str = "",
    role: str = "",
    config,
    hermes_home: str = "",
) -> Dict[str, Any] | None:
    """Externalize one oversized normalized payload if configured.

    Returns a dict with a compact placeholder and the durable JSON payload path,
    or ``None`` when disabled, below threshold, or storage is unavailable. On
    storage failure callers should keep the original content so there is no
    silent data loss.
    """
    if not getattr(config, "large_output_externalization_enabled", False):
        return None

    threshold = max(1, int(getattr(config, "large_output_externalization_threshold_chars", 0) or 0))
    if not content or len(content) <= threshold:
        return None

    try:
        storage_dir = resolve_large_output_storage_dir(config, hermes_home=hermes_home)
    except OSError as exc:
        logger.warning("Large payload externalization skipped (non-blocking): %s", exc)
        return None

    existing = find_externalized_payload_for_message(
        content,
        tool_call_id=tool_call_id,
        session_id=session_id,
        kind=kind,
        role=role,
        config=config,
        hermes_home=hermes_home,
    )
    if existing is not None:
        return {
            "placeholder": _build_externalized_placeholder(existing),
            "path": storage_dir / existing["ref"],
            "payload": existing,
        }

    digest_prefix = _content_digest_prefix(content)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    unique_suffix = f"{time.time_ns():x}"
    if kind == "tool_result":
        # Keep the original filename shape for compatibility with existing
        # externalized tool-output stores and tests.
        filename = f"{timestamp}_{_tool_call_stub(tool_call_id)}_{digest_prefix}_{unique_suffix}.json"
    else:
        filename = (
            f"{timestamp}_{_safe_stub(kind, 'payload')}_"
            f"{_safe_stub(role, 'message')}_{digest_prefix}_{unique_suffix}.json"
        )
    path = storage_dir / filename

    payload = {
        "kind": kind,
        "tool_call_id": tool_call_id,
        "role": role,
        "session_id": session_id,
        "content": content,
        "content_chars": len(content),
        "content_bytes": len(content.encode("utf-8")),
        "created_at": time.time(),
    }
    try:
        _write_externalized_payload(path, payload)
    except OSError as exc:
        logger.warning("Large payload externalization skipped (non-blocking): %s", exc)
        return None

    placeholder = _build_externalized_placeholder(
        {
            "kind": kind,
            "tool_call_id": tool_call_id,
            "role": role,
            "content_chars": payload["content_chars"],
            "content_bytes": payload["content_bytes"],
            "ref": path.name,
        }
    )
    return {
        "placeholder": placeholder,
        "path": path,
        "payload": payload,
    }
