"""Regression coverage for centralized no-agent Telegram delivery."""

from unittest.mock import MagicMock, patch

from cron import scheduler_delivery


def _job():
    return {
        "id": "operational-watchdog",
        "name": "operational watchdog",
        "no_agent": True,
        "deliver": "telegram:8148316720",
    }


def _delivery_patches():
    return (
        patch.object(
            scheduler_delivery,
            "_resolve_delivery_targets",
            return_value=[{"platform": "telegram", "chat_id": "8148316720"}],
        ),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("gateway.config.load_gateway_config", return_value=MagicMock()),
        patch("gateway.platforms.base.BasePlatformAdapter.extract_media", return_value=([], "watchdog output")),
        patch("gateway.platforms.base.BasePlatformAdapter.filter_media_delivery_paths", side_effect=lambda files: files),
        patch("gateway.media_policy.apply_media_policy_env"),
        patch.object(scheduler_delivery, "_cron_mirror_delivery_enabled", return_value=False),
        patch.object(
            scheduler_delivery,
            "_prepare_target_delivery",
            side_effect=AssertionError("profile adapter path must be bypassed"),
        ),
    )


def test_no_agent_canonical_telegram_bypasses_profile_adapter():
    patches = _delivery_patches()
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], \
         patch("tools.operational_sender.send_operational_message", return_value={"ok": True}) as sender:
        result = scheduler_delivery._deliver_result(_job(), "watchdog output")

    assert result is None
    sender.assert_called_once_with("watchdog output")


def test_no_agent_canonical_telegram_sender_failure_is_reported():
    patches = _delivery_patches()
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], \
         patch("tools.operational_sender.send_operational_message", side_effect=RuntimeError("identity unavailable")):
        result = scheduler_delivery._deliver_result(_job(), "watchdog output")

    assert result == "operational Telegram delivery failed: identity unavailable"