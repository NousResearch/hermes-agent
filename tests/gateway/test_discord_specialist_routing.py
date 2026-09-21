from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _event(text: str = "Patch the confirmed failure") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        message_id="message-1",
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="project-updates",
            chat_type="group",
            user_id="operator-1",
        ),
    )


def _adapter(monkeypatch):
    import plugins.platforms.discord.adapter as discord_platform
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setattr(
        discord_platform.discord,
        "DMChannel",
        type("DMChannel", (), {}),
        raising=False,
    )
    value = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="fake-token",
            extra={
                "specialist_routing": {
                    "enabled": True,
                    "board": "project-maintenance",
                    "capabilities": {
                        "burndown-patch-steward": {
                            "domain": "repository-evidence",
                            "actions": ["audit", "inspect", "read", "review", "validate"],
                            "evidence_class": "diagnostic-only",
                            "requested_permissions": ["repository-evidence:read"],
                        }
                    },
                }
            },
        )
    )
    value._client = SimpleNamespace(user=SimpleNamespace(id=999))
    value.send = AsyncMock()
    return value


def test_specialist_route_creates_one_handoff_and_acknowledges(monkeypatch, tmp_path):
    import json
    from pathlib import Path
    from gateway.configured_board import configured_board_db_path
    from gateway.specialist_routing import RouteKind, SpecialistRouteDecision
    from hermes_cli.kanban_db_connect import connect_closing

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    profile_dir = tmp_path / ".hermes" / "profiles" / "task-orchestrator"
    profile_dir.mkdir(parents=True)
    # Create identity markers so named_profile_has_identity returns True
    (profile_dir / "config.yaml").write_text("")
    (profile_dir / "identity.json").write_text("{}")
    adapter = _adapter(monkeypatch)
    settings = adapter._specialist_routing_settings()
    settings["capabilities"]["task-orchestrator"] = dict(
        settings["capabilities"]["burndown-patch-steward"])
    registry = adapter._specialist_capability_registry(settings)
    registry.register_configured_profile("task-orchestrator")
    import threading
    loop_thread = threading.get_ident()
    registry_threads = []
    original_registry = adapter._specialist_capability_registry

    def tracked_registry(settings):
        registry_threads.append(threading.get_ident())
        return original_registry(settings)

    monkeypatch.setattr(adapter, "_specialist_capability_registry", tracked_registry)
    adapter._classify_specialist_event = AsyncMock(
        return_value=SpecialistRouteDecision(
            kind=RouteKind.SPECIALIST, profile="burndown-patch-steward",
            confidence=0.95, reason="bounded patch", title="Patch confirmed failure",
        )
    )
    assert asyncio.run(adapter._maybe_route_specialist_event(_event())) is True
    assert asyncio.run(adapter._maybe_route_specialist_event(_event())) is True
    with connect_closing(configured_board_db_path(settings["board"]), board=settings["board"]) as conn:
        tasks = conn.execute("SELECT body, assignee FROM tasks").fetchall()
        candidates = conn.execute("SELECT request_id, requested_profile_id FROM candidate_profile_requests").fetchall()
    assert registry_threads and all(thread != loop_thread for thread in registry_threads)
    assert len(tasks) == len(candidates) == 1
    assert tasks[0]["assignee"] == "task-orchestrator"
    assert json.loads(tasks[0]["body"])["candidate_request_id"] == candidates[0]["request_id"]
    assert candidates[0]["requested_profile_id"] == "burndown-patch-steward"
    signature = registry.configured_signature("burndown-patch-steward")
    assert registry.resolve(signature, profile_id="burndown-patch-steward").status == "no_match"


def test_general_route_preserves_normal_chat_path(monkeypatch):
    from gateway.specialist_routing import RouteKind, SpecialistRouteDecision

    adapter = _adapter(monkeypatch)
    adapter._classify_specialist_event = AsyncMock(
        return_value=SpecialistRouteDecision(
            kind=RouteKind.GENERAL,
            reason="ordinary conversation",
            confidence=0.0,
            audit_reason="general",
        )
    )

    assert asyncio.run(adapter._maybe_route_specialist_event(_event("Hello"))) is False
    adapter.send.assert_not_awaited()
