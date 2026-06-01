from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


REQUEST_TYPE = "simplex_native_outbound_call"


def control_root(root: Path | None = None) -> Path:
    return Path(root) if root is not None else get_hermes_home() / "run" / "simplex-native-outbound"


def requests_dir(root: Path | None = None) -> Path:
    return control_root(root) / "requests"


def processing_dir(root: Path | None = None) -> Path:
    return control_root(root) / "processing"


def responses_dir(root: Path | None = None) -> Path:
    return control_root(root) / "responses"


def _ensure_dirs(root: Path | None = None) -> None:
    requests_dir(root).mkdir(parents=True, exist_ok=True)
    processing_dir(root).mkdir(parents=True, exist_ok=True)
    responses_dir(root).mkdir(parents=True, exist_ok=True)


def _response_path(request_id: str, root: Path | None = None) -> Path:
    return responses_dir(root) / f"{request_id}.json"


def enqueue_simplex_outbound_call_request(
    *,
    contact_id: str,
    reason: str = "",
    root: Path | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    contact_id = str(contact_id or "").strip()
    if not contact_id:
        raise ValueError("contact_id is required")
    _ensure_dirs(root)
    request_id = str(request_id or uuid.uuid4().hex)
    payload = {
        "id": request_id,
        "type": REQUEST_TYPE,
        "contact_id": contact_id,
        "reason": str(reason or ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    final_path = requests_dir(root) / f"{request_id}.json"
    tmp_path = final_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(final_path)
    return {
        "ok": True,
        "queued": True,
        "request_id": request_id,
        "contact_id": contact_id,
        "request_path": str(final_path),
        "response_path": str(_response_path(request_id, root)),
    }


def pending_simplex_outbound_call_requests(root: Path | None = None) -> list[Path]:
    _ensure_dirs(root)
    return sorted(requests_dir(root).glob("*.json"), key=lambda path: path.stat().st_mtime)


def claim_simplex_outbound_call_request(
    path: Path,
    *,
    root: Path | None = None,
) -> tuple[Path, dict[str, Any]] | None:
    _ensure_dirs(root)
    claim_path = processing_dir(root) / path.name
    try:
        path.rename(claim_path)
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        payload = json.loads(claim_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return claim_path, payload


def write_simplex_outbound_call_response(
    request_id: str,
    payload: dict[str, Any],
    *,
    root: Path | None = None,
) -> Path:
    _ensure_dirs(root)
    response_path = _response_path(str(request_id), root)
    tmp_path = response_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(response_path)
    return response_path


def wait_for_simplex_outbound_call_response(
    request_id: str,
    *,
    timeout_seconds: float,
    interval_seconds: float = 0.25,
    root: Path | None = None,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    response_path = _response_path(str(request_id), root)
    while True:
        if response_path.exists():
            try:
                payload = json.loads(response_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {"ok": False, "error": "invalid gateway response"}
            return payload if isinstance(payload, dict) else {"ok": False}
        if time.monotonic() >= deadline:
            return None
        time.sleep(max(0.05, float(interval_seconds)))
