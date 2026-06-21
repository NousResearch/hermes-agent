"""Tests for threaded cron delivery (summary/detail split + threaded send)."""

import asyncio
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

from cron.scheduler import _deliver_result, _split_summary, _threaded_delivery_enabled


class TestSplitSummary:
    def test_tldr_marker_split_and_prefix_stripped(self):
        content = "TL;DR: Markets flat, two alerts fired.\n\nFull report body\nwith details."
        summary, detail = _split_summary(content)
        assert summary == "Markets flat, two alerts fired."
        assert detail == "Full report body\nwith details."

    def test_marker_variants_case_insensitive(self):
        for marker in ("TLDR:", "tl;dr:", "Summary:", "SUMMARY:"):
            summary, detail = _split_summary(f"{marker} Short version.\n\nLong version.")
            assert summary == "Short version.", marker
            assert detail == "Long version.", marker

    def test_no_marker_falls_back_to_first_paragraph(self):
        content = "First paragraph acts as summary.\n\nRest of the report."
        summary, detail = _split_summary(content)
        assert summary == "First paragraph acts as summary."
        assert detail == "Rest of the report."

    def test_marker_not_on_first_line_uses_first_paragraph(self):
        content = "Preamble line.\nTL;DR: buried marker.\n\nBody."
        summary, detail = _split_summary(content)
        assert summary == "Preamble line.\nTL;DR: buried marker."
        assert detail == "Body."

    def test_short_report_returns_no_detail(self):
        content = "TL;DR: everything fine."
        summary, detail = _split_summary(content)
        assert summary == "TL;DR: everything fine."
        assert detail is None

    def test_multiline_summary_paragraph(self):
        content = "TL;DR: line one\ncontinues here.\n\nDetail."
        summary, detail = _split_summary(content)
        assert summary == "line one\ncontinues here."
        assert detail == "Detail."

    def test_empty_and_whitespace_content(self):
        assert _split_summary("") == ("", None)
        assert _split_summary("   \n  ") == ("   \n  ", None)

    def test_whitespace_only_detail_is_none(self):
        summary, detail = _split_summary("TL;DR: brief.\n\n   \n")
        assert summary == "TL;DR: brief."
        assert detail is None

    def test_crlf_line_endings(self):
        content = "TL;DR: crlf summary.\r\n\r\nDetail line.\r\nMore."
        summary, detail = _split_summary(content)
        assert summary == "crlf summary."
        assert detail == "Detail line.\r\nMore."

    def test_bare_marker_first_paragraph_falls_back_flat(self):
        summary, detail = _split_summary("TL;DR:\n\nActual body here.")
        assert summary == "TL;DR:\n\nActual body here."
        assert detail is None

    def test_marker_without_space_after_colon(self):
        summary, detail = _split_summary("Summary:3 signals fired.\n\nBody.")
        assert summary == "3 signals fired."
        assert detail == "Body."


class TestThreadedDeliveryEnabled:
    def test_default_off_when_config_absent(self):
        with patch("cron.scheduler.load_config", return_value={}):
            assert _threaded_delivery_enabled({"id": "j"}) is False

    def test_enabled_via_config(self):
        with patch("cron.scheduler.load_config",
                   return_value={"cron": {"threaded_delivery": True}}):
            assert _threaded_delivery_enabled({"id": "j"}) is True

    def test_per_job_opt_out_beats_config(self):
        with patch("cron.scheduler.load_config",
                   return_value={"cron": {"threaded_delivery": True}}):
            assert _threaded_delivery_enabled({"id": "j", "thread": False}) is False

    def test_config_load_failure_means_off(self):
        with patch("cron.scheduler.load_config", side_effect=RuntimeError("boom")):
            assert _threaded_delivery_enabled({"id": "j"}) is False


THREAD_CFG = {"cron": {"threaded_delivery": True}}


def _fake_run_coro(coro, _loop):
    future = Future()
    future.set_result(asyncio.run(coro))
    return future


def _mk_env(platform_name="slack"):
    """Gateway config + loop mocks for a single enabled platform."""
    from gateway.config import Platform
    pconfig = MagicMock()
    pconfig.enabled = True
    mock_cfg = MagicMock()
    mock_cfg.platforms = {Platform(platform_name): pconfig}
    loop = MagicMock()
    loop.is_running.return_value = True
    return Platform(platform_name), mock_cfg, loop


def _mk_job(**extra):
    job = {
        "id": "tj-1",
        "name": "daily-scan",
        "deliver": "origin",
        "origin": {"platform": "slack", "chat_id": "C123"},
    }
    job.update(extra)
    return job


REPORT = "TL;DR: All clear today.\n\nLong detail line 1.\nLong detail line 2."


class TestThreadedDelivery:
    def _run(self, adapter, job, content, cfg=THREAD_CFG, platform_name="slack"):
        platform, mock_cfg, loop = _mk_env(platform_name)
        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value=cfg), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=_fake_run_coro):
            return _deliver_result(job, content,
                                   adapters={platform: adapter}, loop=loop)

    def test_threaded_sequence_parent_then_thread_detail(self):
        adapter = AsyncMock()
        adapter.send.side_effect = [
            MagicMock(success=True, message_id="1718000.111", raw_response={}),
            MagicMock(success=True, message_id="1718000.222", raw_response={}),
        ]
        err = self._run(adapter, _mk_job(), REPORT)
        assert err is None
        assert adapter.send.call_count == 2
        parent_call, detail_call = adapter.send.call_args_list
        parent_text = parent_call[0][1]
        assert "All clear today." in parent_text
        assert "daily-scan" in parent_text
        assert "(job_id:" not in parent_text          # slim parent
        assert "To stop or manage" not in parent_text
        detail_text = detail_call[0][1]
        assert "Long detail line 1." in detail_text
        assert "(job_id: tj-1)" in detail_text         # footer moved to thread
        assert detail_call[1]["metadata"]["thread_id"] == "1718000.111"

    def test_parent_send_without_ts_falls_back_flat(self):
        adapter = AsyncMock()
        adapter.send.side_effect = [
            MagicMock(success=True, message_id=None, raw_response={}),  # threaded attempt
            MagicMock(success=True, message_id="1", raw_response={}),   # flat fallback
        ]
        err = self._run(adapter, _mk_job(), REPORT)
        assert err is None
        assert adapter.send.call_count == 2
        flat_text = adapter.send.call_args_list[1][0][1]
        assert "All clear today." in flat_text
        assert "Long detail line 1." in flat_text      # full report, one message

    def test_detail_send_failure_falls_back_flat(self):
        adapter = AsyncMock()
        adapter.send.side_effect = [
            MagicMock(success=True, message_id="1718000.111", raw_response={}),
            MagicMock(success=False, message_id=None, error="boom", raw_response={}),
            MagicMock(success=True, message_id="2", raw_response={}),   # flat fallback
        ]
        err = self._run(adapter, _mk_job(), REPORT)
        assert err is None
        assert adapter.send.call_count == 3
        flat_text = adapter.send.call_args_list[2][0][1]
        assert "Long detail line 1." in flat_text      # report never lost

    def test_job_opt_out_posts_flat(self):
        adapter = AsyncMock()
        adapter.send.return_value = MagicMock(success=True, message_id="1", raw_response={})
        self._run(adapter, _mk_job(thread=False), REPORT)
        adapter.send.assert_called_once()
        text = adapter.send.call_args[0][1]
        assert "Cronjob Response: daily-scan" in text   # classic wrapper intact

    def test_config_off_posts_flat(self):
        adapter = AsyncMock()
        adapter.send.return_value = MagicMock(success=True, message_id="1", raw_response={})
        self._run(adapter, _mk_job(), REPORT, cfg={"cron": {}})
        adapter.send.assert_called_once()

    def test_short_report_posts_flat(self):
        adapter = AsyncMock()
        adapter.send.return_value = MagicMock(success=True, message_id="1", raw_response={})
        self._run(adapter, _mk_job(), "TL;DR: nothing else to say.")
        adapter.send.assert_called_once()

    def test_telegram_target_unaffected(self):
        adapter = AsyncMock()
        adapter.send.return_value = MagicMock(success=True, message_id="1", raw_response={})
        job = _mk_job(origin={"platform": "telegram", "chat_id": "777"})
        self._run(adapter, job, REPORT, platform_name="telegram")
        adapter.send.assert_called_once()
        assert "Cronjob Response: daily-scan" in adapter.send.call_args[0][1]

    def test_existing_origin_thread_id_kept_for_both_sends(self):
        adapter = AsyncMock()
        adapter.send.side_effect = [
            MagicMock(success=True, message_id="1718000.111", raw_response={}),
            MagicMock(success=True, message_id="1718000.222", raw_response={}),
        ]
        job = _mk_job(origin={"platform": "slack", "chat_id": "C123",
                              "thread_id": "1690.555"})
        self._run(adapter, job, REPORT)
        assert adapter.send.call_count == 2
        parent_meta = adapter.send.call_args_list[0][1]["metadata"]
        detail_meta = adapter.send.call_args_list[1][1]["metadata"]
        assert parent_meta["thread_id"] == "1690.555"
        assert detail_meta["thread_id"] == "1690.555"   # not the parent ts

    def test_media_sent_with_thread_metadata(self, tmp_path, monkeypatch):
        media = tmp_path / "chart.png"
        media.write_bytes(b"\x89PNG fake")
        monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
                            (tmp_path,))
        adapter = AsyncMock()
        adapter.send.side_effect = [
            MagicMock(success=True, message_id="1718000.111", raw_response={}),
            MagicMock(success=True, message_id="1718000.222", raw_response={}),
        ]
        adapter.send_image_file.return_value = MagicMock(success=True)
        content = REPORT + f"\nMEDIA:{media.resolve()}"
        self._run(adapter, _mk_job(), content)
        adapter.send_image_file.assert_called_once()
        meta = adapter.send_image_file.call_args[1]["metadata"]
        assert meta["thread_id"] == "1718000.111"
        for call in adapter.send.call_args_list:       # MEDIA tag never leaks
            assert "MEDIA:" not in call[0][1]
