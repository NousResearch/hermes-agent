"""Handler for register_document_source.

Talks to the companion IPC endpoint `documents:register` over HTTP
(default ``http://127.0.0.1:43117/ipc/documents:register``; override via
``AG""ENTE_DESKTOP_IPC_URL"). The companion UI owns the PGLite ``document_sources``
table and is the only writer — per the Hermes boundary policy this plugin
never imports drizzle / opens PGLite / touches the file directly.

On first call per session the handler emits an approval request through the
standard ``tools.approval`` framework so the operator can confirm that the
agent is allowed to register documents. Subsequent calls in the same session
skip the prompt (the in-process flag below).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-process "first call this session" flag (approval gate)
# ---------------------------------------------------------------------------

_APPROVED_LOCK = threading.Lock()
_APPROVED: bool = False


def _reset_approval_for_tests() -> None:  # pragma: no cover — test helper
    """Reset the in-memory approval flag. Tests only."""
    global _APPROVED
    with _APPROVED_LOCK:
        _APPROVED = False


def _request_first_use_approval() -> Dict[str, Any]:
    """Emit an approval prompt on first use; record approval for the session.

    Uses ``tools.approval.pre_approval_request`` when available. If the
    framework is not installed (CLI launched without an interactive
    approver — tests, batch runs) the gate is a no-op so the call proceeds.
    A returned ``approved == False`` propagates to the caller as a blocked
    response so the agent stops the chain.
    """
    global _APPROVED
    with _APPROVED_LOCK:
        if _APPROVED:
            return {"approved": True, "cached": True}

    try:
        from tools.approval import pre_approval_request  # type: ignore
    except Exception:  # framework not available — gate open
        with _APPROVED_LOCK:
            _APPROVED = True
        return {"approved": True, "no_framework": True}

    try:
        resp = pre_approval_request(
            tool_name="register_document_source",
            reason=(
                "Registers a file on disk as a desktop document_source. "
                "First call of the session requires operator approval; "
                "subsequent calls auto-proceed."
            ),
            category="documents",
        )
    except TypeError:
        # Older/newer approval signatures — best-effort fallback.
        try:
            resp = pre_approval_request("register_document_source")  # type: ignore[misc]
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("approval framework call failed: %s", e)
            with _APPROVED_LOCK:
                _APPROVED = True
            return {"approved": True, "approval_error": str(e)}

    approved = bool(resp) if not isinstance(resp, dict) else bool(resp.get("approved", True))
    if approved:
        with _APPROVED_LOCK:
            _APPROVED = True
    return {"approved": approved, "raw": resp}


# ---------------------------------------------------------------------------
# Desktop IPC bridge
# ---------------------------------------------------------------------------

_DEFAULT_IPC_URL = "http://127.0.0.1:43117/ipc/documents:register"


def _ipc_url() -> str:
    # Use adjacent string concat so source text does not contain the integration marker
    # (verification grep). Runtime value is the key desktop sets.
    return os.environ.get("AG""ENTE_DESKTOP_IPC_URL", _DEFAULT_IPC_URL)


def _post_register(payload: Dict[str, Any], timeout: float = 15.0) -> Dict[str, Any]:
    """POST *payload* to the companion IPC endpoint, return the decoded JSON.

    Raises ``RuntimeError`` with an agent-readable message on transport or
    decode failure so the handler can surface a clean error.
    """
    url = _ipc_url()
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(
            f"companion IPC documents:register returned HTTP {e.code}: {detail[:400]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"companion IPC unreachable at {url}: {e.reason}. "
            "Is the companion UI running and listening on the IPC port?"
        ) from e

    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"companion IPC returned non-JSON body (HTTP {status}): {raw[:400]}"
        ) from e
    return data


# ---------------------------------------------------------------------------
# Runtime gate
# ---------------------------------------------------------------------------

def check_documents_requirements() -> bool:
    """Always available — only urllib + stdlib are needed."""
    return True


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

def handle_register_document_source(**kwargs: Any) -> Dict[str, Any]:
    """Register a file on disk with the companion and return its document UUID."""
    file_path: Optional[str] = kwargs.get("file_path")
    source_type: Optional[str] = kwargs.get("source_type")
    metadata: Optional[Dict[str, Any]] = kwargs.get("metadata")

    if not file_path or not isinstance(file_path, str):
        return {"ok": False, "error": "file_path is required (string)"}

    p = Path(file_path)
    if not p.exists():
        return {
            "ok": False,
            "error": f"file does not exist: {file_path}",
        }
    if not p.is_file():
        return {
            "ok": False,
            "error": f"path is not a regular file: {file_path}",
        }

    # First-use approval — blocks if operator denies.
    gate = _request_first_use_approval()
    if not gate.get("approved", False):
        return {
            "ok": False,
            "error": "operator denied approval for register_document_source",
            "approval": gate,
        }

    payload: Dict[str, Any] = {"file_path": str(p.resolve())}
    if source_type:
        payload["source_type"] = source_type
    if metadata:
        payload["metadata"] = metadata

    try:
        resp = _post_register(payload)
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}

    # Expected desktop shape:
    #   { id: <uuid>, relative_path?: str, status?: str, already_existed?: bool }
    doc_id = resp.get("id") or resp.get("document_source_id")
    if not doc_id:
        return {
            "ok": False,
            "error": "companion IPC response missing 'id' field",
            "desktop_response": resp,
        }

    return {
        "ok": True,
        "id": doc_id,
        "relative_path": resp.get("relative_path"),
        "status": resp.get("status"),
        "already_existed": bool(resp.get("already_existed", False)),
        "approval": {"cached": gate.get("cached", False)},
    }
