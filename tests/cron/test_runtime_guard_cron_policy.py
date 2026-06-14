from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from cron.scheduler import _deliver_result, check_cron_delivery_runtime_guard
from gateway.config import Platform, PlatformConfig


def test_delivery_surface_policy_can_block_scoped_cron_delivery():
    config = {
        "runtime_guard": {
            "enabled": True,
            "dry_run": False,
            "scope": {"platforms": ["telegram"], "chat_ids": ["chat-1"]},
            "delivery_surfaces": {"cron_delivery": "block"},
        }
    }

    decision = check_cron_delivery_runtime_guard(
        config,
        {"platform": "telegram", "chat_id": "chat-1", "thread_id": "topic-1"},
        job={"id": "job-1", "name": "Nightly"},
    )

    assert decision.allowed is False
    assert decision.surface == "cron_delivery"
    assert decision.status == "surface_blocked"
    assert decision.reason == "surface_policy_block"


def test_cron_delivery_policy_disabled_is_noop():
    decision = check_cron_delivery_runtime_guard(
        {},
        {"platform": "telegram", "chat_id": "chat-1"},
        job={"id": "job-1"},
    )

    assert decision.allowed is True
    assert decision.status == "disabled"


def test_cron_delivery_block_policy_skips_standalone_send():
    pconfig = PlatformConfig(
        enabled=True,
        token="***",
        extra={
            "runtime_guard": {
                "enabled": True,
                "dry_run": False,
                "scope": {"platforms": ["telegram"], "chat_ids": ["chat-1"]},
                "delivery_surfaces": {"cron_delivery": "block"},
            }
        },
    )
    mock_cfg = SimpleNamespace(platforms={Platform.TELEGRAM: pconfig})
    job = {"id": "job-1", "name": "Nightly", "deliver": "telegram:chat-1"}

    with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock:
        error = _deliver_result(job, "visible cron output")

    assert error is not None
    assert "cron delivery blocked by runtime_guard" in error
    send_mock.assert_not_called()
