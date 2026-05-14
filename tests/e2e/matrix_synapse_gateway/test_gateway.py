"""Opt-in Matrix gateway integration tests against a real Synapse homeserver."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import pytest
import pytest_asyncio

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType

pytestmark = [pytest.mark.integration, pytest.mark.matrix_synapse]

HS = os.environ.get("HERMES_MATRIX_SYNAPSE_URL", "http://127.0.0.1:28448").rstrip("/")
SHARED_SECRET = os.environ.get(
    "HERMES_MATRIX_SYNAPSE_REGISTRATION_SECRET",
    "test-shared-secret",
)


@dataclass
class MatrixUser:
    user_id: str
    access_token: str
    device_id: str
    username: str
    password: str


def _json_request(
    method: str,
    path: str,
    *,
    token: str | None = None,
    body: dict[str, Any] | None = None,
    data: bytes | None = None,
    content_type: str = "application/json",
) -> dict[str, Any]:
    headers = {"Content-Type": content_type}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = data if data is not None else (
        json.dumps(body or {}).encode("utf-8") if body is not None else None
    )
    req = urllib.request.Request(
        f"{HS}{path}",
        data=payload,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8")) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise AssertionError(f"{method} {path} failed: {exc.code} {detail}") from exc


def _homeserver_reachable() -> bool:
    try:
        _json_request("GET", "/_matrix/client/versions")
        return True
    except Exception:
        return False


def _quote(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def _register_user(username: str, password: str, *, admin: bool = False) -> None:
    nonce = _json_request("GET", "/_synapse/admin/v1/register")["nonce"]
    admin_flag = "admin" if admin else "notadmin"
    mac = hmac.new(
        SHARED_SECRET.encode("utf-8"),
        b"\x00".join(
            part.encode("utf-8")
            for part in (nonce, username, password, admin_flag)
        ),
        hashlib.sha1,
    ).hexdigest()
    _json_request(
        "POST",
        "/_synapse/admin/v1/register",
        body={
            "nonce": nonce,
            "username": username,
            "password": password,
            "admin": admin,
            "mac": mac,
        },
    )


def _login(username: str, password: str) -> MatrixUser:
    resp = _json_request(
        "POST",
        "/_matrix/client/v3/login",
        body={
            "type": "m.login.password",
            "identifier": {"type": "m.id.user", "user": username},
            "password": password,
            "initial_device_display_name": "Hermes Synapse integration",
        },
    )
    return MatrixUser(
        user_id=resp["user_id"],
        access_token=resp["access_token"],
        device_id=resp.get("device_id", ""),
        username=username,
        password=password,
    )


def _register_and_login(prefix: str) -> MatrixUser:
    suffix = secrets.token_hex(4)
    username = f"{prefix}{suffix}"
    password = secrets.token_urlsafe(18)
    _register_user(username, password)
    return _login(username, password)


def _create_room(
    creator: MatrixUser,
    invitee: MatrixUser,
    *,
    name: str | None = None,
    is_direct: bool = False,
) -> str:
    body: dict[str, Any] = {
        "preset": "trusted_private_chat",
        "invite": [invitee.user_id],
        "is_direct": is_direct,
    }
    if name:
        body["name"] = name
    resp = _json_request(
        "POST",
        "/_matrix/client/v3/createRoom",
        token=creator.access_token,
        body=body,
    )
    room_id = resp["room_id"]
    _json_request(
        "POST",
        f"/_matrix/client/v3/join/{_quote(room_id)}",
        token=invitee.access_token,
        body={},
    )
    return room_id


def _send_text(sender: MatrixUser, room_id: str, body: str) -> str:
    txn_id = secrets.token_hex(8)
    resp = _json_request(
        "PUT",
        f"/_matrix/client/v3/rooms/{_quote(room_id)}/send/m.room.message/{txn_id}",
        token=sender.access_token,
        body={"msgtype": "m.text", "body": body},
    )
    return resp["event_id"]


def _upload_media(sender: MatrixUser, filename: str, payload: bytes, content_type: str) -> str:
    query = urllib.parse.urlencode({"filename": filename})
    resp = _json_request(
        "POST",
        f"/_matrix/media/v3/upload?{query}",
        token=sender.access_token,
        data=payload,
        content_type=content_type,
    )
    return resp["content_uri"]


def _send_file(sender: MatrixUser, room_id: str, mxc_url: str) -> str:
    txn_id = secrets.token_hex(8)
    resp = _json_request(
        "PUT",
        f"/_matrix/client/v3/rooms/{_quote(room_id)}/send/m.room.message/{txn_id}",
        token=sender.access_token,
        body={
            "msgtype": "m.file",
            "body": "hello.txt",
            "url": mxc_url,
            "info": {
                "mimetype": "text/plain",
                "size": 11,
            },
        },
    )
    return resp["event_id"]


async def _wait_for(predicate, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.1)
    raise AssertionError("timed out waiting for Matrix gateway event")


@pytest.fixture(scope="module")
def synapse_available():
    if os.getenv("HERMES_MATRIX_SYNAPSE_INTEGRATION") != "1":
        pytest.skip("set HERMES_MATRIX_SYNAPSE_INTEGRATION=1 to run Synapse tests")
    if not _homeserver_reachable():
        pytest.skip(f"Synapse is not reachable at {HS}")
    return True


@pytest.fixture
def matrix_users(synapse_available):
    bot = _register_and_login("hermesbot")
    alice = _register_and_login("alice")
    return bot, alice


@pytest_asyncio.fixture
async def matrix_adapter(matrix_users, monkeypatch):
    try:
        from gateway.platforms.matrix import MatrixAdapter
    except ImportError as exc:
        pytest.skip(f"Matrix dependencies are not installed: {exc}")

    bot, _alice = matrix_users
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.setenv("MATRIX_E2EE_MODE", "off")
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token=bot.access_token,
            extra={
                "homeserver": HS,
                "user_id": bot.user_id,
                "device_id": bot.device_id,
                "e2ee_mode": "off",
            },
        )
    )
    adapter._text_batch_delay_seconds = 0
    captured = []

    async def capture(event):
        captured.append(event)

    adapter.handle_message = capture
    connected = await adapter.connect()
    if not connected:
        pytest.skip("Matrix adapter could not connect to Synapse")
    try:
        yield adapter, captured
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_dm_private_room_send_receive(matrix_adapter, matrix_users):
    adapter, captured = matrix_adapter
    bot, alice = matrix_users
    room_id = _create_room(alice, bot, is_direct=True)

    _send_text(alice, room_id, "hello from a private room")

    event = await _wait_for(lambda: captured[-1] if captured else None)
    assert event.text == "hello from a private room"
    assert event.source.chat_id == room_id
    assert event.source.user_id == alice.user_id


@pytest.mark.asyncio
async def test_room_invite_join_receive_and_respond(matrix_adapter, matrix_users):
    adapter, captured = matrix_adapter
    bot, alice = matrix_users
    room_id = _create_room(
        alice,
        bot,
        name="Hermes Synapse Integration",
        is_direct=False,
    )

    _send_text(alice, room_id, "room message for Hermes")

    event = await _wait_for(lambda: captured[-1] if captured else None)
    assert event.text == "room message for Hermes"
    assert event.source.chat_id == room_id

    result = await adapter.send(room_id, "Synapse integration pong")
    assert result.success is True

    sync = _json_request(
        "GET",
        "/_matrix/client/v3/sync?timeout=1000",
        token=alice.access_token,
    )
    joined = sync.get("rooms", {}).get("join", {}).get(room_id, {})
    events = joined.get("timeline", {}).get("events", [])
    assert any(
        ev.get("sender") == bot.user_id
        and ev.get("content", {}).get("body") == "Synapse integration pong"
        for ev in events
    )


@pytest.mark.asyncio
async def test_media_upload_download_reaches_gateway(matrix_adapter, matrix_users):
    _adapter, captured = matrix_adapter
    bot, alice = matrix_users
    room_id = _create_room(alice, bot, is_direct=True)
    mxc_url = _upload_media(alice, "hello.txt", b"hello world", "text/plain")

    _send_file(alice, room_id, mxc_url)

    event = await _wait_for(lambda: captured[-1] if captured else None)
    assert event.message_type == MessageType.DOCUMENT
    assert event.media_urls
    assert event.text == "hello.txt"


@pytest.mark.asyncio
async def test_startup_old_event_filtering(matrix_users, monkeypatch):
    try:
        from gateway.platforms.matrix import MatrixAdapter
    except ImportError as exc:
        pytest.skip(f"Matrix dependencies are not installed: {exc}")

    bot, alice = matrix_users
    room_id = _create_room(alice, bot, is_direct=True)
    _send_text(alice, room_id, "old startup event")
    await asyncio.sleep(6)

    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token=bot.access_token,
            extra={
                "homeserver": HS,
                "user_id": bot.user_id,
                "device_id": bot.device_id,
                "e2ee_mode": "off",
            },
        )
    )
    adapter._text_batch_delay_seconds = 0
    captured = []

    async def capture(event):
        captured.append(event)

    adapter.handle_message = capture
    connected = await adapter.connect()
    if not connected:
        pytest.skip("Matrix adapter could not connect to Synapse")
    try:
        await asyncio.sleep(1)
        assert captured == []
    finally:
        await adapter.disconnect()


@pytest.mark.matrix_e2ee
@pytest.mark.asyncio
async def test_encrypted_room_smoke_is_opt_in(synapse_available):
    if os.getenv("HERMES_MATRIX_SYNAPSE_E2EE") != "1":
        pytest.skip("set HERMES_MATRIX_SYNAPSE_E2EE=1 for encrypted-room smoke")
    try:
        from gateway.platforms import matrix as matrix_mod
    except ImportError as exc:
        pytest.skip(f"Matrix dependencies are not installed: {exc}")
    if not matrix_mod._check_e2ee_deps():
        pytest.skip("mautrix E2EE dependencies are not installed")

    # Full encrypted message exchange is intentionally not part of the default
    # Synapse harness. This smoke keeps the marker available and verifies the
    # local runtime can at least initialize the required crypto dependency path.
    assert matrix_mod._normalize_e2ee_mode("required") == "required"
