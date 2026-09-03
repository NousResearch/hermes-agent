"""Authenticated local client for the governed builder adapter."""

from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import os
import secrets
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from uuid import uuid4

from .canonical import canonical_json_bytes, sha256_bytes, signed_material
from .errors import AdapterError
from .runtime import RuntimeSettings, _read_owner_json


MAX_RESPONSE_BYTES = 1_000_000


@dataclass(frozen=True)
class OperatorKey:
    key_id: str
    secret: bytes


def load_operator_key(settings: RuntimeSettings, key_id: str | None = None) -> OperatorKey:
    """Select an active key authorized for the current local process."""
    auth = _read_owner_json(settings.auth_file, exact_mode=0o600)
    uid = os.geteuid()
    gid = os.getegid()
    matches = []
    for item in auth.get("keys", []):
        if key_id is not None and item.get("key_id") != key_id:
            continue
        if not item.get("active", True) or int(item.get("allowed_uid", -1)) != uid:
            continue
        allowed_gid = item.get("allowed_gid")
        if allowed_gid is not None and int(allowed_gid) != gid:
            continue
        secret_env = str(item.get("secret_env", ""))
        if not secret_env.startswith("HERMES_BUILDER_ADAPTER_SECRET_"):
            continue
        secret = os.environ.get(secret_env, "").encode()
        if len(secret) < 32:
            raise AdapterError(
                "AUTHENTICATION_FAILED",
                f"approved secret source did not supply {secret_env}",
            )
        matches.append(OperatorKey(str(item["key_id"]), secret))
    if len(matches) != 1:
        detail = "no matching active key" if not matches else "multiple active keys; pass --key-id"
        raise AdapterError("AUTHENTICATION_FAILED", detail)
    return matches[0]


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: Path, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self.socket_path))


Transport = Callable[[str, str, bytes, dict[str, str]], tuple[int, bytes]]


class BuilderAdapterClient:
    def __init__(
        self,
        socket_path: Path,
        key: OperatorKey | None,
        *,
        timeout: float = 15.0,
        clock: Callable[[], float] = time.time,
        transport: Transport | None = None,
    ):
        self.socket_path = socket_path
        self.key = key
        self.timeout = timeout
        self.clock = clock
        self.transport = transport or self._send

    def _headers(self, method: str, path: str, body: bytes) -> dict[str, str]:
        if self.key is None:
            return {}
        timestamp = str(int(self.clock()))
        nonce = secrets.token_hex(16)
        digest = sha256_bytes(body)
        signature = hmac.new(
            self.key.secret,
            signed_material(method, path, timestamp, nonce, digest),
            hashlib.sha256,
        ).hexdigest()
        return {
            "X-Hermes-Timestamp": timestamp,
            "X-Hermes-Nonce": nonce,
            "X-Hermes-Key-Id": self.key.key_id,
            "X-Hermes-Signature": signature,
        }

    def _send(
        self, method: str, target: str, body: bytes, headers: dict[str, str]
    ) -> tuple[int, bytes]:
        connection = _UnixHTTPConnection(self.socket_path, self.timeout)
        try:
            connection.request(method, target, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise AdapterError("INTERNAL_ERROR", "adapter response exceeded size limit")
            return response.status, raw
        except (OSError, http.client.HTTPException) as exc:
            raise AdapterError("PROVIDER_UNAVAILABLE", "local adapter is unavailable", retryable=True) from exc
        finally:
            connection.close()

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body = canonical_json_bytes(payload) if payload is not None else b""
        headers = self._headers(method, path, body)
        headers.update({"Accept": "application/json", "Connection": "close"})
        if payload is not None:
            headers["Content-Type"] = "application/json"
        target = path
        if query:
            target = f"{path}?{urlencode(query)}"
        status, raw = self.transport(method, target, body, headers)
        try:
            result = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdapterError("INTERNAL_ERROR", "adapter returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise AdapterError("INTERNAL_ERROR", "adapter returned an invalid result")
        if status >= 400:
            error = (result.get("errors") or [{}])[0]
            raise AdapterError(
                str(error.get("code", "INTERNAL_ERROR")),
                str(error.get("message", "adapter request failed")),
                retryable=bool(error.get("retryable", False)),
            )
        return result

    def health(self) -> dict[str, Any]:
        return self.request("GET", "/v1/health")

    def start(self, cycle_id: str, cycle: dict[str, Any], *, dispatch_id: str | None = None) -> dict[str, Any]:
        dispatch_id = dispatch_id or str(uuid4())
        payload = {
            "schema_version": "1.0.0",
            "dispatch_id": dispatch_id,
            "idempotency_key": f"operator:{cycle_id}:{dispatch_id}",
            "cycle_id": cycle_id,
            "contract_id": cycle["contract_id"],
            "repository_id": cycle["repository_id"],
            "builder_role": "primary_builder",
            "expected_cycle_revision": cycle["revision"],
            "completion_schema_version": "1.0.0",
        }
        return self.request("POST", "/v1/dispatches", payload)

    def status(self, dispatch_id: str, cycle_id: str) -> dict[str, Any]:
        return self.request(
            "GET", f"/v1/dispatches/{dispatch_id}", query={"cycle_id": cycle_id}
        )

    def cancel(self, dispatch_id: str, cycle_id: str, reason_code: str) -> dict[str, Any]:
        return self.request(
            "POST",
            f"/v1/dispatches/{dispatch_id}/cancel",
            {"cycle_id": cycle_id, "reason_code": reason_code},
        )

