from __future__ import annotations

from pathlib import Path

import pytest

from gateway.config import PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from plugins.teams_context.relay_adapter import check_requirements, validate_config
from plugins.teams_context.relay_adapter import TeamsRelayAdapter
from plugins.teams_context.store import TeamsContextStore


class _FakeRequest:
    def __init__(
        self,
        payload=None,
        *,
        headers=None,
        content_length: int | None = None,
        json_error: Exception | None = None,
    ):
        self._payload = payload
        self._json_error = json_error
        self.headers = headers or {}
        self.content_length = content_length

    async def json(self):
        if self._json_error:
            raise self._json_error
        return self._payload


def _adapter(tmp_path: Path, *, secret: str = "relay-secret") -> TeamsRelayAdapter:
    if not platform_registry.is_registered("teams_relay"):
        platform_registry.register(
            PlatformEntry(
                name="teams_relay",
                label="Teams Relay",
                adapter_factory=lambda cfg: TeamsRelayAdapter(cfg),
                check_fn=check_requirements,
                validate_config=validate_config,
            )
        )
    return TeamsRelayAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "secret": secret,
                "store_path": str(tmp_path / "teams.sqlite"),
                "host": "127.0.0.1",
                "port": 0,
            },
        )
    )


@pytest.mark.anyio
async def test_relay_rejects_missing_secret(tmp_path: Path):
    adapter = _adapter(tmp_path)
    response = await adapter._handle_message(_FakeRequest({"message_id": "m1"}))
    assert response.status == 401


@pytest.mark.anyio
async def test_relay_rejects_invalid_json(tmp_path: Path):
    adapter = _adapter(tmp_path)
    response = await adapter._handle_message(
        _FakeRequest(
            headers={"X-Hermes-Relay-Secret": "relay-secret"},
            json_error=ValueError("bad json"),
        )
    )
    assert response.status == 400


@pytest.mark.anyio
async def test_relay_accepts_power_automate_custom_header(tmp_path: Path):
    adapter = _adapter(tmp_path)
    response = await adapter._handle_message(
        _FakeRequest(
            {
                "chat_id": "group-chat-1",
                "message_id": "msg-1",
                "text": "stored with CustomHeader1",
            },
            headers={"CustomHeader1": "relay-secret"},
        )
    )
    assert response.status == 202


@pytest.mark.anyio
async def test_relay_rejects_oversized_payload(tmp_path: Path):
    adapter = _adapter(tmp_path)
    response = await adapter._handle_message(
        _FakeRequest(
            {"message_id": "m1"},
            headers={"X-Hermes-Relay-Secret": "relay-secret"},
            content_length=999_999,
        )
    )
    assert response.status == 413


@pytest.mark.anyio
async def test_relay_stores_power_automate_payload(tmp_path: Path):
    adapter = _adapter(tmp_path)
    payload = {
        "source": "power_automate",
        "chat_id": "group-chat-1",
        "chat_name": "Example relay chat",
        "message_id": "msg-1",
        "sender_name": "Alice",
        "created_at": "2026-06-25T02:00:00Z",
        "text": "The relay should store this Teams message.",
        "web_url": "https://teams.example/messages/msg-1",
    }
    response = await adapter._handle_message(
        _FakeRequest(payload, headers={"X-Hermes-Relay-Secret": "relay-secret"})
    )

    assert response.status == 202
    store = TeamsContextStore(tmp_path / "teams.sqlite")
    results = store.search("relay", chat_id="group-chat-1")
    assert len(results) == 1
    assert results[0]["message_id"] == "msg-1"
    assert results[0]["sender_name"] == "Alice"


@pytest.mark.anyio
async def test_relay_generates_stable_id_when_power_automate_omits_message_id(tmp_path: Path):
    adapter = _adapter(tmp_path)
    payload = {
        "source": "power_automate",
        "chat_id": "group-chat-1",
        "sender_name": "Alice",
        "created_at": "2026-06-25T02:00:00Z",
        "text": "Power Automate did not provide a message id.",
    }
    response = await adapter._handle_message(
        _FakeRequest(payload, headers={"X-Hermes-Relay-Secret": "relay-secret"})
    )

    assert response.status == 202
    store = TeamsContextStore(tmp_path / "teams.sqlite")
    results = store.search("Automate", chat_id="group-chat-1")
    assert len(results) == 1
    assert results[0]["message_id"].startswith("power_automate:")
    assert results[0]["sender_name"] == "Alice"


@pytest.mark.anyio
async def test_relay_upserts_duplicate_message(tmp_path: Path):
    adapter = _adapter(tmp_path)
    headers = {"X-Hermes-Relay-Secret": "relay-secret"}
    first = {
        "chat_id": "group-chat-1",
        "message_id": "msg-1",
        "created_at": "2026-06-25T02:00:00Z",
        "text": "first body",
    }
    second = {**first, "text": "second body"}
    assert (await adapter._handle_message(_FakeRequest(first, headers=headers))).status == 202
    assert (await adapter._handle_message(_FakeRequest(second, headers=headers))).status == 202

    store = TeamsContextStore(tmp_path / "teams.sqlite")
    assert store.search("first", chat_id="group-chat-1") == []
    results = store.search("second", chat_id="group-chat-1")
    assert len(results) == 1
    assert results[0]["message_id"] == "msg-1"
