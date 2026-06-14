"""macOS Hermes permission broker client.

This module intentionally starts with the narrowest safe live route:
``permission.status``. Protected actions can be added only after the native
broker implements and tests them.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import socket
import time
import uuid
from pathlib import Path
from typing import Any

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover - fallback for isolated imports
    def get_hermes_home() -> Path:  # type: ignore[misc]
        return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()

BROKER_PROTOCOL_VERSION = 1
DEFAULT_TTL_MS = 30_000
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_SOCKET_NAME = "mac-permission-broker.sock"
DEFAULT_TOKEN_NAME = "mac-permission-broker.token"
ALLOWED_METHODS = {"permission.status"}


class MacPermissionBrokerError(RuntimeError):
    """Raised when the broker request cannot be completed."""


def broker_runtime_dir() -> Path:
    path = get_hermes_home() / "run"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def default_socket_path() -> Path:
    return broker_runtime_dir() / DEFAULT_SOCKET_NAME


def default_token_path() -> Path:
    return broker_runtime_dir() / DEFAULT_TOKEN_NAME


def load_or_create_broker_token(path: Path | None = None) -> str:
    token_path = path or default_token_path()
    if token_path.exists():
        return token_path.read_text(encoding="utf-8").strip()
    token_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    token_path.write_text(token, encoding="utf-8")
    token_path.chmod(0o600)
    return token


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sign(envelope: dict[str, Any], token: str) -> str:
    payload = dict(envelope)
    payload.pop("signature", None)
    return hmac.new(token.encode("utf-8"), _stable_json(payload).encode("utf-8"), hashlib.sha256).hexdigest()


def create_broker_request(
    method: str,
    params: dict[str, Any] | None = None,
    *,
    token: str,
    now_ms: int | None = None,
    ttl_ms: int = DEFAULT_TTL_MS,
    nonce: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    if method not in ALLOWED_METHODS:
        raise MacPermissionBrokerError(f"unsupported broker method: {method}")
    if len(token) < 32:
        raise MacPermissionBrokerError("broker token must be at least 32 characters")
    issued_at = int(now_ms if now_ms is not None else time.time() * 1000)
    envelope: dict[str, Any] = {
        "version": BROKER_PROTOCOL_VERSION,
        "id": request_id or str(uuid.uuid4()),
        "method": method,
        "params": params or {},
        "issuedAt": issued_at,
        "ttlMs": ttl_ms,
        "nonce": nonce or secrets.token_hex(16),
    }
    envelope["signature"] = _sign(envelope, token)
    return envelope


def send_broker_request(
    request: dict[str, Any],
    *,
    socket_path: Path | str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    path = str(socket_path or default_socket_path())
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect(path)
            client.sendall(_stable_json(request).encode("utf-8") + b"\n")
            chunks: list[bytes] = []
            while True:
                chunk = client.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
    except OSError as exc:
        raise MacPermissionBrokerError(f"macOS permission broker unavailable at {path}: {exc}") from exc
    raw = b"".join(chunks).split(b"\n", 1)[0]
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise MacPermissionBrokerError(f"invalid broker response: {raw[:200]!r}") from exc
    if not isinstance(payload, dict):
        raise MacPermissionBrokerError("broker response must be a JSON object")
    return payload


def broker_permission_status(
    *,
    socket_path: Path | str | None = None,
    token: str | None = None,
    token_path: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    broker_token = token or load_or_create_broker_token(token_path)
    request = create_broker_request("permission.status", token=broker_token)
    return send_broker_request(request, socket_path=socket_path, timeout=timeout)


def cmd_macos_permission_broker(args: Any) -> None:
    """CLI entrypoint for ``hermes mac-broker``."""
    command = getattr(args, "mac_broker_command", None)
    if command in (None, "status"):
        token_file = getattr(args, "token_file", None)
        payload: dict[str, Any]
        try:
            payload = broker_permission_status(
                socket_path=getattr(args, "socket", None),
                token_path=Path(token_file).expanduser() if token_file else None,
                timeout=float(getattr(args, "timeout", DEFAULT_TIMEOUT_SECONDS)),
            )
        except MacPermissionBrokerError as exc:
            payload = {"ok": False, "error": str(exc)}
        print(json.dumps(payload, sort_keys=True))
        return
    print(json.dumps({"ok": False, "error": f"unknown mac-broker command: {command}"}, sort_keys=True))


__all__ = [
    "MacPermissionBrokerError",
    "broker_permission_status",
    "cmd_macos_permission_broker",
    "create_broker_request",
    "default_socket_path",
    "default_token_path",
    "load_or_create_broker_token",
    "send_broker_request",
]
