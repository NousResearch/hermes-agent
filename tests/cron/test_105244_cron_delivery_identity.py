"""Regression tests for #105244 mechanism B.

A cron job pinned to a model different from the profile default delivers its
output into the user's chat. The delivery must carry the *job's* identity
(including the pinned model) so neither the user nor the agent mistakes the
job's model for the agent's own runtime model.

Old behavior preserved: an unpinned job's header/mirror bytes are unchanged,
and a pinned job still RUNS with its pin (pin precedence untouched).
"""

from unittest.mock import AsyncMock, MagicMock, patch

from cron.scheduler import _deliver_result, _load_cron_job_config
from cron.scheduler_delivery import _cron_mirror_message


def _run_delivery(job, content="Here is today's summary."):
    from gateway.config import Platform

    pconfig = MagicMock()
    pconfig.enabled = True
    mock_cfg = MagicMock()
    mock_cfg.platforms = {Platform.TELEGRAM: pconfig}
    with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
         patch("tools.send_message_tool._send_to_platform",
               new=AsyncMock(return_value={"success": True})) as send_mock:
        _deliver_result(job, content)
    send_mock.assert_called_once()
    return send_mock.call_args.kwargs.get("content") or send_mock.call_args[0][-1]


class TestPinnedModelAttribution:
    def test_delivery_header_names_pinned_model(self):
        sent = _run_delivery({
            "id": "test-job",
            "name": "weekly-research",
            "model": "qwen3.5:397b",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        })
        assert "Cronjob Response: weekly-research" in sent
        assert "(job_id: test-job" in sent
        assert "qwen3.5:397b" in sent

    def test_mirror_prefix_names_pinned_model(self):
        text = _cron_mirror_message(
            {"id": "abc123", "name": "weekly-research", "model": "qwen3.5:397b"},
            "payload",
        )
        assert text.startswith("[Cron delivery: weekly-research")
        assert "qwen3.5:397b" in text.split("\n", 1)[0]
        assert text.endswith("payload")


class TestUnpinnedBytesUnchanged:
    def test_delivery_header_without_pin_has_no_model(self):
        sent = _run_delivery({
            "id": "test-job",
            "name": "daily-report",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        })
        assert "(job_id: test-job)" in sent
        assert "model:" not in sent

    def test_mirror_prefix_without_pin_unchanged(self):
        assert _cron_mirror_message(
            {"id": "abc123", "name": "daily-report"}, "payload"
        ) == "[Cron delivery: daily-report]\npayload"


class TestPinStillEffective:
    def test_job_pin_wins_over_env_and_missing_config(self, monkeypatch, tmp_path):
        """Old behavior: the pin itself still decides the run's model."""
        monkeypatch.setenv("HERMES_MODEL", "deepseek-v4-pro:0813")
        # Point the home at an empty dir so no config.yaml can override the pin.
        import cron.scheduler as sched

        monkeypatch.setattr(sched, "_get_hermes_home", lambda: tmp_path)
        resolved = _load_cron_job_config(
            {"id": "j", "model": "qwen3.5:397b"}, "j", "weekly-research")
        assert resolved.model == "qwen3.5:397b"
