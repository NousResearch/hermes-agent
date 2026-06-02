"""Private Photon phone-management client.

The Spectrum SDK does not currently expose typed project-user management
helpers, so the Hermes Photon CLI keeps the Node adapter boundary and asks the
private sidecar to perform management-plane calls over stdio JSON lines.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


_SIDECAR_DIR = Path(__file__).parent / "sidecar"
_SIDECAR_ENTRYPOINT = _SIDECAR_DIR / "index.mjs"
_DEFAULT_TIMEOUT_SECONDS = 45.0


@dataclass
class PhotonPhoneManagementError(RuntimeError):
    """Structured failure returned by the private Photon sidecar."""

    code: str
    message: str
    detail: str = ""
    retryable: bool = False
    status: Optional[int] = None

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)


def list_phones(project_id: str, project_secret: str) -> dict[str, Any]:
    """Return normalized Photon/Spectrum project users."""
    return _run_management_command(
        {"type": "phones_list"},
        project_id=project_id,
        project_secret=project_secret,
    )


def add_phone(project_id: str, project_secret: str, phone: str) -> dict[str, Any]:
    """Create one shared Photon/Spectrum user for ``phone``."""
    return _run_management_command(
        {"type": "phones_add", "phone": phone},
        project_id=project_id,
        project_secret=project_secret,
    )


def remove_phone(project_id: str, project_secret: str, phone: str) -> dict[str, Any]:
    """Remove one Photon/Spectrum project user by submitted phone."""
    return _run_management_command(
        {"type": "phones_remove", "phone": phone},
        project_id=project_id,
        project_secret=project_secret,
    )


def _run_management_command(
    command: dict[str, Any],
    *,
    project_id: str,
    project_secret: str,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    node_bin = os.getenv("PHOTON_NODE_BIN") or "node"
    if not shutil.which(node_bin):
        raise PhotonPhoneManagementError(
            code="MISSING_NODE",
            message="Node.js is required for Photon phone management",
            retryable=False,
        )
    if not _SIDECAR_ENTRYPOINT.exists():
        raise PhotonPhoneManagementError(
            code="MISSING_SIDECAR",
            message="Photon sidecar entrypoint is missing",
            detail=str(_SIDECAR_ENTRYPOINT),
            retryable=False,
        )

    env = os.environ.copy()
    env["PHOTON_PROJECT_ID"] = project_id
    env["PHOTON_PROJECT_SECRET"] = project_secret
    body = dict(command)
    body.setdefault("requestId", "phone-management")

    try:
        proc = subprocess.run(  # noqa: S603
            [node_bin, str(_SIDECAR_ENTRYPOINT), "--management"],
            cwd=str(_SIDECAR_DIR),
            input=json.dumps(body) + "\n",
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise PhotonPhoneManagementError(
            code="MANAGEMENT_TIMEOUT",
            message="Photon phone management command timed out",
            detail=str(exc),
            retryable=True,
        ) from exc
    except OSError as exc:
        raise PhotonPhoneManagementError(
            code="MANAGEMENT_SIDECAR_FAILED",
            message="could not start Photon phone management sidecar",
            detail=str(exc),
            retryable=True,
        ) from exc

    response = _last_response(proc.stdout)
    if response is None:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise PhotonPhoneManagementError(
            code="NO_SIDECAR_RESPONSE",
            message="Photon phone management sidecar returned no response",
            detail=detail,
            retryable=proc.returncode != 0,
        )
    if response.get("ok"):
        data = response.get("data")
        return data if isinstance(data, dict) else {}

    error = response.get("error") if isinstance(response.get("error"), dict) else {}
    raise PhotonPhoneManagementError(
        code=str(error.get("code") or "MANAGEMENT_ERROR"),
        message=str(error.get("message") or "Photon phone management failed"),
        detail=str(error.get("detail") or proc.stderr or "").strip(),
        retryable=bool(error.get("retryable")),
        status=_coerce_optional_int(error.get("status")),
    )


def _last_response(stdout: str) -> Optional[dict[str, Any]]:
    response: Optional[dict[str, Any]] = None
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("type") == "response":
            response = payload
    return response


def _coerce_optional_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
