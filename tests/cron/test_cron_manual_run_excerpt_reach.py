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


def test_a_partial_standalone_split_send_counts_as_reached(monkeypatch):
    """The REAL Telegram standalone sender lands chunks 1-2 of a long report, then chunk 3 fails.
    The report's head is already in the target chat, so the manual-run completion must not echo it
    into the invoking conversation as if the delivery had reached nobody."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from plugins.platforms.telegram.standalone import _send_telegram
    from tests.tools.test_telegram_send_message_chunk_mdv2 import _install_telegram_mock, _no_proxy

    _no_proxy(monkeypatch)
    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=[SimpleNamespace(message_id=501), SimpleNamespace(message_id=502),
                                              RuntimeError("transport refused final chunk")])
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))
    deliver = f"telegram:{GOOD}"
    job = {"id": "j1", "name": "nightly", "deliver": deliver, "origin": {"platform": "telegram", "chat_id": GOOD}}
    config = MagicMock()
    config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda p: None
    stored: dict = {}

    async def _send(platform, pconfig, chat_id, text, **kwargs):
        return await _send_telegram("tok", chat_id, text)

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("tools.send_message_tool._send_to_platform", _send), \
         patch("cron.jobs.update_job", side_effect=lambda _id, values: stored.update(values)):
        error = _deliver_result(job, "status line ok\n" * 700)
    assert error and bot.send_message.await_count == 3
    with patch("tools.cronjob_tools.get_job", return_value={**stored, "last_delivery_error": error}), \
         patch("tools.cronjob_tools._latest_job_output_excerpt", return_value="THE REPORT"):
        summary = _manual_run_completion({"success": True}, "j1", "nightly", deliver, 0.0)["summary"]
    assert "THE REPORT" not in summary



@pytest.mark.parametrize("lost", ["ack", "raise"])
def test_a_report_whose_delivery_answer_was_lost_is_not_echoed(monkeypatch, lost):
    """The report may be on screen at the target: Telegram accepted it but the answer never came back
    (the REAL standalone sender reports ``delivery: uncertain``), or the sender itself raised."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from plugins.platforms.telegram.standalone import _send_telegram
    from tests.tools.test_telegram_send_message_chunk_mdv2 import _install_telegram_mock, _no_proxy

    _no_proxy(monkeypatch)
    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=OSError("connection reset after the request was sent"))
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))
    deliver = f"telegram:{GOOD}"
    job = {"id": "j1", "name": "nightly", "deliver": deliver, "origin": {"platform": "telegram", "chat_id": GOOD}}
    config = MagicMock()
    config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda p: None
    stored: dict = {}
    results: list = []

    async def _send(platform, pconfig, chat_id, text, **kwargs):
        if lost == "raise":
            raise TimeoutError("sender gave up waiting for Telegram's answer")
        results.append(await _send_telegram("tok", chat_id, text))
        return results[-1]

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("tools.send_message_tool._send_to_platform", _send), \
         patch("cron.jobs.update_job", side_effect=lambda _id, values: stored.update(values)):
        error = _deliver_result(job, "THE REPORT")
    assert error
    if lost == "ack":
        assert results[0].get("delivery") == "uncertain"
    with patch("tools.cronjob_tools.get_job", return_value={**stored, "last_delivery_error": error}), \
         patch("tools.cronjob_tools._latest_job_output_excerpt", return_value="THE REPORT"):
        summary = _manual_run_completion({"success": True}, "j1", "nightly", deliver, 0.0)["summary"]
    assert "THE REPORT" not in summary
