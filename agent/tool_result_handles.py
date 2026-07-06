"""Large tool-result handle storage.

When enabled via config, this module stores oversized tool results out-of-band
under HERMES_HOME and returns a compact JSON handle suitable for conversation
history replay. The full artifact remains recoverable by handle id.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

from hermes_constants import get_hermes_home

DEFAULT_THRESHOLD_BYTES = 24_000
DEFAULT_PREVIEW_BYTES = 2_000


def _tool_result_config() -> Mapping[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        section = cfg.get("tool_results") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def handles_enabled() -> bool:
    cfg = _tool_result_config()
    return bool(cfg.get("handles_enabled", False))


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except Exception:
        return default


def maybe_handle_tool_result(
    result: Any,
    *,
    tool_name: str,
    args: Optional[dict[str, Any]] = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
) -> Any:
    """Return a compact handle for large results when configured.

    Small results, non-string results, and disabled config are returned
    unchanged. The caller can use this as a final canonicalization step before
    appending a tool result into conversation history.
    """
    if not isinstance(result, str) or not handles_enabled():
        return result

    cfg = _tool_result_config()
    threshold = _positive_int(cfg.get("handle_threshold_bytes"), DEFAULT_THRESHOLD_BYTES)
    data = result.encode("utf-8", errors="replace")
    if len(data) <= threshold:
        return result

    preview_bytes = _positive_int(cfg.get("preview_bytes"), DEFAULT_PREVIEW_BYTES)
    head = data[:preview_bytes].decode("utf-8", errors="replace")
    tail = data[-preview_bytes:].decode("utf-8", errors="replace") if preview_bytes else ""
    sha = hashlib.sha256(data).hexdigest()
    handle_id = f"trh_{int(time.time())}_{uuid.uuid4().hex[:12]}"

    home = get_hermes_home()
    artifacts_dir = home / "tool-results" / time.strftime("%Y%m%d")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifacts_dir / f"{handle_id}.json"
    relative_path = artifact_path.relative_to(home).as_posix()

    artifact = {
        "id": handle_id,
        "type": "hermes_tool_result_artifact",
        "tool_name": tool_name,
        "args": args or {},
        "task_id": task_id,
        "session_id": session_id,
        "tool_call_id": tool_call_id,
        "bytes": len(data),
        "sha256": sha,
        "relative_path": relative_path,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "content": result,
    }
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    compact = {
        "type": "hermes_tool_result_handle",
        "id": handle_id,
        "tool_name": tool_name,
        "bytes": len(data),
        "sha256": sha,
        "relative_path": relative_path,
        "preview": {"head": head, "tail": tail},
        "message": "Large tool result stored out-of-band; use the id to retrieve exact full content if needed.",
    }
    return json.dumps(compact, ensure_ascii=False)


def load_artifact(handle_id: str) -> Optional[dict[str, Any]]:
    """Load a stored tool-result artifact by id from HERMES_HOME."""
    if not handle_id or "/" in handle_id or ".." in handle_id:
        return None
    root = get_hermes_home() / "tool-results"
    if not root.exists():
        return None
    for path in root.glob(f"*/{handle_id}.json"):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def recover_tool_result(handle_or_id: str | Mapping[str, Any]) -> Optional[str]:
    """Recover the exact full content for a stored tool-result handle.

    This is the intentionally small internal retrieval API used by tests and
    future user-facing surfaces. It accepts either a raw handle id, a parsed
    compact-handle dict, or the compact-handle JSON string returned by
    :func:`maybe_handle_tool_result`.
    """
    handle_id: str | None
    expected_sha: str | None = None

    if isinstance(handle_or_id, Mapping):
        handle_id = str(handle_or_id.get("id") or "")
        expected_sha = str(handle_or_id.get("sha256") or "") or None
    elif isinstance(handle_or_id, str):
        text = handle_or_id.strip()
        if text.startswith("{"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                handle_id = str(payload.get("id") or "")
                expected_sha = str(payload.get("sha256") or "") or None
            else:
                handle_id = ""
        else:
            handle_id = text
    else:
        return None

    artifact = load_artifact(handle_id or "")
    if not artifact:
        return None
    content = artifact.get("content")
    if not isinstance(content, str):
        return None
    if expected_sha:
        actual_sha = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
        if actual_sha != expected_sha:
            return None
    return content
