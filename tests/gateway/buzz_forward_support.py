"""Shared fixtures for Buzz forward-mode tests (not collected by pytest)."""
from __future__ import annotations

import json
import socket
import tempfile
import threading
from pathlib import Path

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter
from tests.gateway.test_buzz_adapter import (
    CHANNEL,
    DM_CHANNEL,
    OTHER_PUBKEY,
    SELF_PUBKEY,
    _ScriptedCli,
    _event,
    _make_adapter,
)

_buzz_mod = load_plugin_adapter("buzz")
BuzzAdapter = _buzz_mod.BuzzAdapter
_standalone_send = _buzz_mod._standalone_send
_forward_dedupe_key = _buzz_mod._forward_dedupe_key
_encode_len_prefixed = _buzz_mod._encode_len_prefixed
_DESTINATION_DENIED = _buzz_mod._DESTINATION_DENIED

_ENV_VARS = (
    "BUZZ_ALLOWED_DESTINATIONS",
    "BUZZ_FORWARD_ONLY",
    "BUZZ_FORWARD_SOCKET",
    "BUZZ_FORWARD_ACK_TIMEOUT",
    "BUZZ_CLI_PATH",
    "BUZZ_CLI_SHA256",
    "BUZZ_ALLOWED_USERS",
    "BUZZ_CHANNELS",
)


@pytest.fixture(autouse=True)
def _clean_new_env(monkeypatch, tmp_path):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(_buzz_mod, "_DEFAULT_CREDENTIALS_DIR", tmp_path / "no-creds")
    yield


def _ack_frame(event_id: str, status: str = "accepted") -> bytes:
    return _encode_len_prefixed({"ack": event_id, "status": status})


class ForwardServer:
    def __init__(self, path: str, *, replies=None, hang: bool = False, hang_ids=None):
        self.path = path
        self.replies = list(replies or [])
        self.hang = hang
        self.hang_ids = set(hang_ids or [])
        self.frames = []
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._thread = None
        Path(path).unlink(missing_ok=True)
        self._sock.bind(path)
        self._sock.listen(8)
        self._sock.settimeout(2)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        try:
            while True:
                try:
                    conn, _addr = self._sock.accept()
                except socket.timeout:
                    continue
                with conn:
                    header = conn.recv(4)
                    if len(header) < 4:
                        return
                    size = int.from_bytes(header, "big")
                    payload = b""
                    while len(payload) < size:
                        chunk = conn.recv(size - len(payload))
                        if not chunk:
                            break
                        payload += chunk
                    self.frames.append(json.loads(payload.decode("utf-8")))
                    event_id = self.frames[-1].get("event", {}).get("id")
                    if self.hang:
                        threading.Event().wait(timeout=2)
                        return
                    if event_id in self.hang_ids:
                        continue
                    reply = self.replies.pop(0) if self.replies else _ack_frame(event_id)
                    conn.sendall(reply)
        except OSError:
            return

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass
        Path(self.path).unlink(missing_ok=True)


def short_sock_path() -> str:
    return tempfile.mktemp(prefix="bz-", suffix=".sock", dir="/tmp")


@pytest.fixture
def forward_socket():
    server = ForwardServer(short_sock_path())
    server.start()
    yield server
    server.close()


def forward_adapter(socket_path, extra=None):
    adapter = _make_adapter(
        {
            "forward_only": True,
            "forward_socket": socket_path,
            "forward_ack_timeout": 0.4,
            "allowed_users": [OTHER_PUBKEY],
            "allowed_destinations": [CHANNEL, DM_CHANNEL],
            "channels": [CHANNEL],
            **(extra or {}),
        }
    )
    from unittest.mock import AsyncMock

    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    adapter._self_pubkey = SELF_PUBKEY
    adapter._channel_state[CHANNEL] = adapter._new_channel_state("group")
    adapter._input_scope = {CHANNEL}
    return adapter
