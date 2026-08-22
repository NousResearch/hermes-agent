"""Authorization and request tests for native WhatsApp group creation."""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

from gateway.config import PlatformConfig
from plugins.platforms.whatsapp import adapter as adapter_module
from plugins.platforms.whatsapp import tools


def _env(monkeypatch, **values):
    monkeypatch.setattr(
        tools,
        "_wenv",
        lambda name, default="": values.get(name, default),
    )


_TEST_CREDENTIAL = "x" * 32


def _config(monkeypatch, *, admins=(), participants=()):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "gateway": {
                "platforms": {
                    "whatsapp": {
                        "extra": {
                            "group_admin_users": list(admins),
                            "group_allowed_participants": list(participants),
                        }
                    }
                }
            }
        },
    )


def test_group_request_requires_exact_confirmation_and_allowlisted_jids(monkeypatch):
    _config(
        monkeypatch,
        participants=("15550001111@s.whatsapp.net", "998877@lid"),
    )
    valid = {
        "subject": "Codex Blockers",
        "confirmed_subject": "Codex Blockers",
        "operation_id": "codex-blockers-001",
        "participants": ["15550001111:2@c.us", "998877@lid"],
        "confirmed_participants": ["15550001111:2@c.us", "998877@lid"],
    }
    assert tools._validated_request(valid) == {
        "subject": "Codex Blockers",
        "confirmedSubject": "Codex Blockers",
        "operationId": "codex-blockers-001",
        "participants": ["15550001111@s.whatsapp.net", "998877@lid"],
        "confirmedParticipants": ["15550001111@s.whatsapp.net", "998877@lid"],
    }
    assert tools._validated_request({**valid, "confirmed_subject": "different"}) is None
    assert (
        tools._validated_request({**valid, "confirmed_participants": ["998877@lid"]})
        is None
    )
    assert (
        tools._validated_request({
            **valid,
            "participants": ["17770000000@s.whatsapp.net"],
        })
        is None
    )
    assert tools._validated_request({**valid, "operation_id": "short"}) is None


def test_tool_availability_fails_closed_without_each_private_boundary(monkeypatch):
    complete = {
        "WHATSAPP_GROUP_CONTROL_TOKEN": _TEST_CREDENTIAL,
    }
    _config(
        monkeypatch,
        admins=("15550001111@s.whatsapp.net",),
        participants=("15550001111@s.whatsapp.net",),
    )
    _env(monkeypatch, **complete)
    assert tools._check_available() is True
    _env(monkeypatch, WHATSAPP_GROUP_CONTROL_TOKEN="")
    assert tools._check_available() is False
    _env(monkeypatch, **complete)
    _config(monkeypatch, participants=())
    assert tools._check_available() is False
    _config(monkeypatch, participants=("15550001111@s.whatsapp.net",))
    assert tools._check_available() is False


def test_group_tool_sends_token_only_in_header_and_redacts_participants(monkeypatch):
    values = {
        "WHATSAPP_GROUP_CONTROL_TOKEN": _TEST_CREDENTIAL,
    }
    _env(monkeypatch, **values)
    _config(
        monkeypatch,
        admins=("15550001111@s.whatsapp.net",),
        participants=("15550001111@s.whatsapp.net",),
    )
    monkeypatch.setattr(tools, "_bridge_port", lambda: 3123)
    captured = {}

    class Response:
        status = 201

        async def json(self):
            return {"success": True, "status": "created", "groupId": "12036300@g.us"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, url, **kwargs):
            captured.update(url=url, **kwargs)
            return Response()

    fake_aiohttp = SimpleNamespace(
        ClientSession=Session,
        ClientTimeout=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    result = json.loads(
        asyncio.run(
            tools.whatsapp_create_group({
                "subject": "Codex Blockers",
                "confirmed_subject": "Codex Blockers",
                "operation_id": "codex-blockers-001",
                "participants": ["15550001111@s.whatsapp.net"],
                "confirmed_participants": ["15550001111@s.whatsapp.net"],
            })
        )
    )
    assert result == {
        "success": True,
        "status": "created",
        "operation_id": "codex-blockers-001",
        "subject": "Codex Blockers",
        "group_id": "12036300@g.us",
    }
    assert captured["url"] == "http://127.0.0.1:3123/groups/create"
    assert captured["headers"] == {"Authorization": f"Bearer {_TEST_CREDENTIAL}"}
    assert _TEST_CREDENTIAL not in json.dumps(captured["json"])
    assert "participants" not in json.dumps(result)


def test_adapter_exposes_admin_toolset_only_to_exact_direct_principal(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"platform_toolsets": {"whatsapp": ["web"]}},
    )
    monkeypatch.setattr(
        "gateway.whatsapp_identity.canonical_whatsapp_identifier",
        lambda value: str(value or "").split(":", 1)[0].split("@", 1)[0],
    )
    adapter = adapter_module.WhatsAppAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "session_name": "test",
                "group_admin_users": ["15550001111@s.whatsapp.net"],
            },
        )
    )

    owner = SimpleNamespace(
        chat_type="dm",
        user_id="15550001111:2@s.whatsapp.net",
        chat_id="15550001111@s.whatsapp.net",
    )
    stranger = SimpleNamespace(
        chat_type="dm",
        user_id="17770000000@s.whatsapp.net",
        chat_id="17770000000@s.whatsapp.net",
    )
    group = SimpleNamespace(
        chat_type="group",
        user_id="15550001111@s.whatsapp.net",
        chat_id="12036300@g.us",
    )

    assert "whatsapp_group_admin" in adapter.toolsets_for_source(owner)
    assert "whatsapp_group_admin" not in adapter.toolsets_for_source(stranger)
    assert "whatsapp_group_admin" not in adapter.toolsets_for_source(group)
    assert "web" in adapter.toolsets_for_source(owner)
    assert "web" in adapter.toolsets_for_source(stranger)
