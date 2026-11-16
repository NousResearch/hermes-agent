"""Per-job ``wrap_response`` vs the global ``cron.wrap_response`` in cron delivery."""

from unittest.mock import AsyncMock, MagicMock, patch

from cron.scheduler_delivery import _deliver_result


def _deliver(job, content, global_wrap):
    """Deliver ``job`` to a Telegram origin with ``cron.wrap_response`` set to ``global_wrap``."""
    from gateway.config import Platform

    pconfig = MagicMock()
    pconfig.enabled = True
    mock_cfg = MagicMock()
    mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

    with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
         patch("tools.send_message_tool._send_to_platform",
               new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("cron.scheduler.load_config",
               return_value={"cron": {"wrap_response": global_wrap}}):
        _deliver_result({
            "id": "test-job",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
            **job,
        }, content)

    send_mock.assert_called_once()
    return send_mock.call_args.kwargs.get("content") or send_mock.call_args[0][-1]


def test_per_job_wrap_response_false_overrides_global_true():
    """A per-job wrap_response=False delivers raw output even when the global config wraps."""
    sent = _deliver({"name": "morning-briefing", "wrap_response": False}, "Good morning.", True)

    assert sent == "Good morning."
    assert "Cronjob Response" not in sent


def test_per_job_wrap_response_true_overrides_global_false():
    """A per-job wrap_response=True forces wrapping even when the global config disables it."""
    sent = _deliver({"name": "noisy-job", "wrap_response": True}, "Output.", False)

    assert "Cronjob Response: noisy-job" in sent
    assert "To stop or manage this job" in sent


def test_delivery_skips_wrapping_when_config_disabled():
    """With no per-job value and cron.wrap_response false, content is delivered raw."""
    sent = _deliver({"name": "daily-report"}, "Clean output only.", False)

    assert sent == "Clean output only."
    assert "Cronjob Response" not in sent
    assert "The agent cannot see" not in sent
