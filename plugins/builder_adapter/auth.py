"""HMAC authentication, durable replay rejection, and Darwin peer identity."""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import os
import platform
import time
from dataclasses import dataclass
from socket import socket

from .canonical import signed_material
from .errors import AdapterError
from .store import DispatchStore


@dataclass(frozen=True)
class PrincipalKey:
    principal: str
    key_id: str
    secret: bytes
    allowed_uid: int
    allowed_gid: int | None = None
    active: bool = True


def darwin_peer_credentials(sock: socket) -> tuple[int, int]:
    if platform.system() != "Darwin":
        raise AdapterError(
            "AUTHENTICATION_FAILED",
            "Darwin peer credentials are unavailable on this platform",
        )
    uid = ctypes.c_uint()
    gid = ctypes.c_uint()
    libc = ctypes.CDLL(None, use_errno=True)
    getpeereid = libc.getpeereid
    getpeereid.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
    ]
    getpeereid.restype = ctypes.c_int
    if getpeereid(sock.fileno(), ctypes.byref(uid), ctypes.byref(gid)) != 0:
        raise AdapterError(
            "AUTHENTICATION_FAILED",
            f"peer credential lookup failed with errno {ctypes.get_errno()}",
        )
    return int(uid.value), int(gid.value)


class HMACAuthenticator:
    def __init__(
        self,
        keys: list[PrincipalKey],
        store: DispatchStore,
        *,
        replay_window_seconds: int = 300,
        clock=time.time,
    ):
        self._keys = {key.key_id: key for key in keys}
        if len(self._keys) != len(keys) or any(
            len(key.secret) < 32 for key in keys
        ):
            raise AdapterError(
                "AUTHENTICATION_FAILED",
                "authentication keys must be unique and at least 32 bytes",
            )
        self._store = store
        self._window = replay_window_seconds
        self._clock = clock

    def verify(
        self,
        *,
        method: str,
        path: str,
        timestamp: str,
        nonce: str,
        request_sha256: str,
        key_id: str,
        signature: str,
        peer_uid: int,
        peer_gid: int,
    ) -> str:
        key = self._keys.get(key_id)
        if not key or not key.active:
            raise AdapterError("AUTHENTICATION_FAILED", "unknown or inactive key")
        if peer_uid != key.allowed_uid or (
            key.allowed_gid is not None and peer_gid != key.allowed_gid
        ):
            raise AdapterError(
                "AUTHORIZATION_FAILED", "peer credentials are not authorized"
            )
        try:
            timestamp_value = int(timestamp)
        except (TypeError, ValueError) as exc:
            raise AdapterError(
                "AUTHENTICATION_FAILED", "invalid request timestamp"
            ) from exc
        now = int(self._clock())
        if abs(now - timestamp_value) > self._window:
            raise AdapterError("REPLAY_REJECTED", "request timestamp is stale")
        if len(nonce) < 16 or len(nonce) > 200:
            raise AdapterError("REPLAY_REJECTED", "invalid nonce")
        expected = hmac.new(
            key.secret,
            signed_material(method, path, timestamp, nonce, request_sha256),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise AdapterError(
                "AUTHENTICATION_FAILED", "request signature is invalid"
            )
        self._store.consume_nonce(
            key_id, nonce, timestamp_value + self._window, now=now
        )
        return key.principal
