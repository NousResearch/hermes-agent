from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest

from hermes_cli.macos_permission_broker import (
    MacPermissionBrokerError,
    broker_permission_status,
    create_broker_request,
    load_or_create_broker_token,
    send_broker_request,
)


def test_create_broker_request_signs_stable_envelope() -> None:
    token = "0123456789abcdef0123456789abcdef"
    request = create_broker_request(
        "permission.status",
        token=token,
        now_ms=1_000,
        nonce="nonce-1",
        request_id="request-1",
    )

    assert request["version"] == 1
    assert request["method"] == "permission.status"
    assert request["issuedAt"] == 1_000
    assert request["signature"] == "864c883ef84899c9fa811cd7e315b0c8919bdfef11fde3ad62e8eea3ee659204"


def test_create_broker_request_rejects_unsupported_method() -> None:
    with pytest.raises(MacPermissionBrokerError, match="unsupported broker method"):
        create_broker_request("ui.click", token="0123456789abcdef0123456789abcdef")


def test_load_or_create_broker_token_creates_private_token_file(tmp_path: Path) -> None:
    token_file = tmp_path / "broker.token"

    first = load_or_create_broker_token(token_file)
    second = load_or_create_broker_token(token_file)

    assert first == second
    assert len(first) == 64
    assert token_file.stat().st_mode & 0o777 == 0o600


def _serve_one(socket_path: Path, response: dict[str, object], seen: list[dict[str, object]]) -> threading.Thread:
    def run() -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(socket_path))
            server.listen(1)
            conn, _ = server.accept()
            with conn:
                raw = b""
                while not raw.endswith(b"\n"):
                    raw += conn.recv(4096)
                seen.append(json.loads(raw.decode("utf-8")))
                conn.sendall(json.dumps(response, sort_keys=True).encode("utf-8") + b"\n")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def short_tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="hmb-", dir="/tmp"))


def wait_for_socket(socket_path: Path) -> None:
    deadline = time.time() + 2
    while time.time() < deadline:
        if socket_path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"socket was not created: {socket_path}")


def test_send_broker_request_round_trips_json_over_unix_socket() -> None:
    socket_path = short_tmp_dir() / "b.sock"
    seen: list[dict[str, object]] = []
    thread = _serve_one(socket_path, {"ok": True, "permissions": {"screenCapture": "authorized"}}, seen)
    wait_for_socket(socket_path)
    request = create_broker_request("permission.status", token="0123456789abcdef0123456789abcdef")

    response = send_broker_request(request, socket_path=socket_path)

    thread.join(timeout=2)
    assert response == {"ok": True, "permissions": {"screenCapture": "authorized"}}
    assert seen[0]["method"] == "permission.status"
    assert "signature" in seen[0]


def test_broker_permission_status_uses_token_file_and_socket() -> None:
    tmp_dir = short_tmp_dir()
    socket_path = tmp_dir / "b.sock"
    token_file = tmp_dir / "b.token"
    token_file.write_text("0123456789abcdef0123456789abcdef", encoding="utf-8")
    seen: list[dict[str, object]] = []
    thread = _serve_one(socket_path, {"ok": True, "method": "permission.status"}, seen)
    wait_for_socket(socket_path)

    response = broker_permission_status(socket_path=socket_path, token_path=token_file)

    thread.join(timeout=2)
    assert response == {"ok": True, "method": "permission.status"}
    assert seen[0]["method"] == "permission.status"


def test_send_broker_request_reports_missing_socket(tmp_path: Path) -> None:
    with pytest.raises(MacPermissionBrokerError, match="broker unavailable"):
        send_broker_request({"ok": True}, socket_path=tmp_path / "missing.sock", timeout=0.1)
