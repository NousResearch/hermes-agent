"""A manual run's completion carries the job output only when no delivery target got it.

The completion re-enters the session that ran ``cronjob(action='run')``, which can be another
topic (#97557), so output a chat already received is not copied there. Output that reached nobody
(every target failed) stays in it, or the agent has nothing to relay. A partial failure (one
target sent, another refused) already posted the output, so it stays status-only.
"""

from unittest.mock import MagicMock, patch

import pytest

from cron.scheduler import _deliver_result
from gateway.config import Platform, PlatformConfig
from tools.cronjob_tools import _manual_run_completion

GOOD, BAD = "-100111", "-100222"


def _run_and_complete(deliver):
    job = {"id": "j1", "name": "nightly", "deliver": deliver,
           "origin": {"platform": "telegram", "chat_id": GOOD}}
    config = MagicMock()
    config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda p: None
    stored: dict = {}

    async def _send(platform, pconfig, chat_id, text, **kwargs):
        return {"success": True, "message_id": "7"} if chat_id == GOOD else {"error": "chat not found"}

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("tools.send_message_tool._send_to_platform", _send), \
         patch("cron.jobs.update_job", side_effect=lambda _id, values: stored.update(values)):
        error = _deliver_result(job, "THE REPORT")
    refreshed = {**stored, "last_delivery_error": error}
    with patch("tools.cronjob_tools.get_job", return_value=refreshed), \
         patch("tools.cronjob_tools._latest_job_output_excerpt", return_value="THE REPORT"):
        return error, _manual_run_completion({"success": True}, "j1", "nightly", deliver, 0.0)["summary"]


@pytest.mark.parametrize("deliver, echoed", [
    (f"telegram:{BAD}", True),                    # failed everywhere: reached nobody
    (f"telegram:{GOOD},telegram:{BAD}", False),   # partial: already posted in GOOD
    (f"telegram:{GOOD}", False),                  # delivered
])
def test_completion_echoes_output_only_when_no_target_received_it(deliver, echoed):
    error, summary = _run_and_complete(deliver)
    assert (error is not None) == (BAD in deliver)
    assert ("THE REPORT" in summary) is echoed
