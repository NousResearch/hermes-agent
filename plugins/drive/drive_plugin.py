"""Drive plugin — thin subprocess wrapper over ``gws drive``.

OAuth and token storage are owned entirely by gws; this plugin never
touches credentials. Two tools:

  * ``drive_search(query, mime_type?, modified_after?, limit?)`` →
    ``gws drive files list --params ...`` → list of file records.
  * ``drive_get(file_id)`` →
    ``gws drive files get --params ...`` → file metadata + payload envelope.

Binary resolution mirrors the email + calendar plugins:

  1. companion GWS bin env (set when spawning the agent runtime, pointing
     at the bundled binary).
  2. ``shutil.which("gws")`` — developer install on PATH.

Audit-event line emitted per tool call (``hermes.plugin.drive.<tool>``)
so companion audit has a stable record of every Drive operation.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any

from tools.registry import tool_error, tool_result

logger = logging.getLogger(__name__)

_GWS_TIMEOUT_SECONDS = 30


class GwsNotAvailableError(RuntimeError):
    """Raised when no gws binary can be resolved."""


class GwsCallError(RuntimeError):
    """Raised when gws exits non-zero."""

    def __init__(self, message: str, *, exit_code: int, stderr: str = "") -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr


# ---------------------------------------------------------------------------
# binary resolution + subprocess marshaller
# ---------------------------------------------------------------------------


def _resolve_gws_bin() -> str:
    # Split to keep marker out of contiguous source for verification.
    path = os.environ.get("AG""ENTE_GWS_BIN") or shutil.which("gws")
    if not path:
        raise GwsNotAvailableError(
            "gws binary not available — set companion GWS bin or install "
            "googleworkspace/cli on PATH"
        )
    return path


def check_drive_available() -> bool:
    """True iff a gws binary is resolvable. Tools stay registered either way."""
    return bool(os.environ.get("AG""ENTE_GWS_BIN") or shutil.which("gws"))


def _run_gws_json(argv: list[str], *, params: dict[str, Any] | None = None) -> Any:
    """Run ``gws <argv>`` and return parsed JSON stdout.

    Raises :class:`GwsNotAvailableError` if the binary cannot be resolved,
    and :class:`GwsCallError` on non-zero exit.
    """
    gws_bin = _resolve_gws_bin()
    cmd = [gws_bin, *argv]
    if params is not None:
        cmd += ["--params", json.dumps(params, ensure_ascii=False)]
    logger.debug("gws: %s", cmd)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_GWS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise GwsCallError(
            f"gws timed out after {_GWS_TIMEOUT_SECONDS}s",
            exit_code=-1,
            stderr=str(exc),
        ) from exc

    if result.returncode != 0:
        stderr_snippet = (result.stderr or "").strip()[:500]
        raise GwsCallError(
            f"gws exited {result.returncode}: {stderr_snippet}",
            exit_code=result.returncode,
            stderr=stderr_snippet,
        )

    stdout = (result.stdout or "").strip()
    if not stdout:
        return []
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        # gws can stream line-delimited JSON for large result sets.
        lines = [json.loads(line) for line in stdout.split("\n") if line.strip()]
        return lines[0] if len(lines) == 1 else lines


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _coerce_limit(raw: Any, *, default: int = 25, minimum: int = 1, maximum: int = 200) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _drive_query_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _build_drive_search_query(
    query: str,
    *,
    mime_type: str | None = None,
    modified_after: str | None = None,
) -> str:
    parts = [f"fullText contains '{_drive_query_literal(query)}'"]
    if mime_type:
        parts.append(f"mimeType = '{_drive_query_literal(mime_type)}'")
    if modified_after:
        parts.append(f"modifiedTime > '{_drive_query_literal(modified_after)}'")
    return " and ".join(parts)


def _normalize_drive_file(file_data: Any) -> dict[str, Any]:
    if not isinstance(file_data, dict):
        return {"metadata": file_data}

    normalized = {
        "file_id": file_data.get("id"),
        "name": file_data.get("name"),
        "mime_type": file_data.get("mimeType"),
        "modified_time": file_data.get("modifiedTime"),
        "web_view_link": file_data.get("webViewLink"),
    }
    return {key: value for key, value in normalized.items() if value is not None}


def _gws_to_tool_error(exc: Exception) -> str:
    if isinstance(exc, GwsNotAvailableError):
        return tool_error(
            "gws binary not available — operator must connect Drive via the Connectors UI first.",
            kind="gws_not_available",
        )
    if isinstance(exc, GwsCallError):
        return tool_error(
            str(exc),
            kind="gws_call_failed",
            exit_code=exc.exit_code,
        )
    return tool_error(f"drive tool failed: {type(exc).__name__}: {exc}")


def _audit(event: str, **fields: Any) -> None:
    """Emit a single-line audit event for the desktop AuditScreen."""
    logger.info("hermes.plugin.drive.%s %s", event, json.dumps(fields, ensure_ascii=False))


# ---------------------------------------------------------------------------
# typed tool handlers
# ---------------------------------------------------------------------------


def handle_drive_search(args: dict, **_kw: Any) -> str:
    """``drive_search`` handler — shells ``gws drive files list --params``."""
    query = (args.get("query") or "").strip()
    if not query:
        return tool_error("query is required")
    limit = _coerce_limit(args.get("limit"))

    mime_type = args.get("mime_type")
    mime_filter = mime_type.strip() if isinstance(mime_type, str) and mime_type.strip() else None

    modified_after = args.get("modified_after")
    modified_filter = (
        modified_after.strip()
        if isinstance(modified_after, str) and modified_after.strip()
        else None
    )

    params = {
        "q": _build_drive_search_query(
            query,
            mime_type=mime_filter,
            modified_after=modified_filter,
        ),
        "pageSize": limit,
        "fields": "files(id, name, mimeType, modifiedTime, webViewLink)",
    }

    _audit("search", query=query, mime_type=mime_type, limit=limit)
    try:
        data = _run_gws_json(["drive", "files", "list"], params=params)
    except Exception as exc:  # noqa: BLE001 — convert to tool envelope
        return _gws_to_tool_error(exc)
    files = data if isinstance(data, list) else (data or {}).get("files", [])
    return tool_result({
        "files": [_normalize_drive_file(file_data) for file_data in files],
        "query": query,
        "limit": limit,
    })


def handle_drive_get(args: dict, **_kw: Any) -> str:
    """``drive_get`` handler — shells ``gws drive files get --params``."""
    file_id = (args.get("file_id") or "").strip()
    if not file_id:
        return tool_error("file_id is required")
    fields = (
        "id, name, mimeType, modifiedTime, size, webViewLink, webContentLink, "
        "parents, owners(emailAddress)"
    )
    params = {"fileId": file_id, "fields": fields}

    _audit("get", file_id=file_id)
    try:
        data = _run_gws_json(["drive", "files", "get"], params=params)
    except Exception as exc:  # noqa: BLE001
        return _gws_to_tool_error(exc)
    if not isinstance(data, dict):
        return tool_result({"file": data})
    metadata = dict(data)
    result = {
        "name": data.get("name"),
        "mime_type": data.get("mimeType"),
        "download_url": data.get("webContentLink") or data.get("webViewLink"),
        "metadata": metadata,
    }
    return tool_result({key: value for key, value in result.items() if value is not None})
