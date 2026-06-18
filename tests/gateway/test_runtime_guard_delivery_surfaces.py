from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.delivery import DeliveryRouter, DeliveryTarget, check_delivery_router_runtime_guard
from gateway.kanban_watchers import check_kanban_notification_runtime_guard
from gateway.runtime_guard import check_delivery_surface_policy


def test_delivery_surface_policy_allows_cron_and_kanban_by_default():
    config = {
        "runtime_guard": {
            "enabled": True,
            "dry_run": False,
            "scope": {"platforms": ["telegram"], "chat_ids": ["chat-1"]},
        }
    }

    cron_decision = check_delivery_surface_policy(
        config,
        "cron_delivery",
        platform="telegram",
        chat_id="chat-1",
    )
    kanban_decision = check_kanban_notification_runtime_guard(
        config,
        {"platform": "telegram", "chat_id": "chat-1", "thread_id": ""},
        task_id="task-1",
        event_kind="completed",
    )

    assert cron_decision.allowed is True
    assert cron_decision.status == "surface_allowed"
    assert kanban_decision.allowed is True
    assert kanban_decision.status == "surface_allowed"


def test_delivery_router_surface_policy_disabled_is_noop():
    decision = check_delivery_router_runtime_guard(
        None,
        DeliveryTarget(platform=Platform.TELEGRAM, chat_id="chat-1"),
        metadata={"job_id": "job-1"},
    )

    assert decision.allowed is True
    assert decision.status == "disabled"
    assert decision.surface == "delivery_router"


def test_delivery_router_surface_policy_can_block_when_scoped():
    config = {
        "runtime_guard": {
            "enabled": True,
            "dry_run": False,
            "scope": {"platforms": ["telegram"], "chat_ids": ["chat-1"]},
            "delivery_surfaces": {"delivery_router": "block"},
        }
    }

    decision = check_delivery_router_runtime_guard(
        config,
        DeliveryTarget(platform=Platform.TELEGRAM, chat_id="chat-1"),
    )

    assert decision.allowed is False
    assert decision.reason == "surface_policy_block"
    assert decision.status == "surface_blocked"


@pytest.mark.asyncio
async def test_delivery_router_live_delivery_blocks_before_adapter_send():
    adapter = AsyncMock()
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="***",
                extra={
                    "runtime_guard": {
                        "enabled": True,
                        "dry_run": False,
                        "scope": {"platforms": ["telegram"], "chat_ids": ["chat-1"]},
                        "delivery_surfaces": {"delivery_router": "block"},
                    }
                },
            )
        }
    )
    router = DeliveryRouter(config, adapters={Platform.TELEGRAM: adapter})

    results = await router.deliver(
        "visible output",
        [DeliveryTarget(platform=Platform.TELEGRAM, chat_id="chat-1")],
    )

    assert results["telegram:chat-1"]["success"] is False
    assert "surface_policy_block" in results["telegram:chat-1"]["error"]
    adapter.send.assert_not_called()
