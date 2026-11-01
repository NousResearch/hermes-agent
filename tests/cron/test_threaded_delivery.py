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


class _TimeoutFuture:
    """A future whose .result() times out; .cancel() returns a fixed verdict.

    cancel() -> True  models "never dispatched" (queued, safe to retry).
    cancel() -> False models "already in flight" (must NOT retry, would dupe).
    """

    def __init__(self, cancel_returns):
        self._cancel_returns = cancel_returns

    def result(self, timeout=None):
        raise TimeoutError()

    def cancel(self):
        return self._cancel_returns


def _seq_run_coro(*modes):
    """Sequence per-call scheduling behaviour for run_coroutine_threadsafe.

    Each mode is either ``"ok"`` (run the coroutine and resolve) or a
    ``("timeout", cancel_returns)`` tuple (the coroutine is not run; the future
    times out and its cancel() returns the given bool). Calls past the sequence
    default to ``"ok"``.
    """
    state = {"n": 0}

    def _factory(coro, _loop):
        i = state["n"]
        state["n"] += 1
        mode = modes[i] if i < len(modes) else "ok"
        if mode == "ok":
            future = Future()
            future.set_result(asyncio.run(coro))
            return future
        coro.close()  # timed-out leg: never run it
        return _TimeoutFuture(mode[1])

    return _factory


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
    def _run(self, adapter, job, content, cfg=THREAD_CFG, platform_name="slack",
             scheduler=_fake_run_coro):
        platform, mock_cfg, loop = _mk_env(platform_name)
        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value=cfg), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=scheduler):
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

    def test_parent_without_ts_sends_body_only_not_whole_report(self):
        # Parent summary landed but gave no thread anchor. Re-sending the whole
        # report would DUPLICATE the summary, so only the body goes out flat,
        # and the degradation is surfaced as a caveat (not a hard failure).
        adapter = AsyncMock()
        adapter.send.side_effect = [
            MagicMock(success=True, message_id=None, raw_response={}),  # parent, no ts
            MagicMock(success=True, message_id="1", raw_response={}),   # body-only flat
        ]
        err = self._run(adapter, _mk_job(), REPORT)
        assert "un-threaded" in (err or "")             # caveat surfaced
        assert adapter.send.call_count == 2
        flat_text = adapter.send.call_args_list[1][0][1]
        assert "Long detail line 1." in flat_text        # body delivered
        assert "All clear today." not in flat_text       # summary NOT duplicated

    def test_detail_send_failure_sends_body_only_not_whole_report(self):
        # Parent landed and threaded, but the detail send failed. The flat
        # fallback must re-send only the body — not the whole report — or the
        # summary the user already sees would be duplicated.
        adapter = AsyncMock()
        adapter.send.side_effect = [
            MagicMock(success=True, message_id="1718000.111", raw_response={}),
            MagicMock(success=False, message_id=None, error="boom", raw_response={}),
            MagicMock(success=True, message_id="2", raw_response={}),   # body-only flat
        ]
        err = self._run(adapter, _mk_job(), REPORT)
        assert "un-threaded" in (err or "")
        assert adapter.send.call_count == 3
        flat_text = adapter.send.call_args_list[2][0][1]
        assert "Long detail line 1." in flat_text        # body never lost
        assert "All clear today." not in flat_text       # summary NOT duplicated

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

    # --- Timeout semantics on the DeliveryRouter path (Teknium review) --------
    #
    # A slow confirmation is not a failed send. future.cancel() disambiguates:
    #   cancel()==True  -> queued/never dispatched -> nothing sent -> safe retry
    #   cancel()==False -> already in flight       -> must NOT retry (duplicate)

    def test_parent_timeout_queued_sends_full_report_flat(self):
        # Parent scheduling timed out but was never dispatched (cancel True):
        # nothing reached the wire, so the full report is safe to send flat.
        adapter = AsyncMock()
        adapter.send.return_value = MagicMock(success=True, message_id="9", raw_response={})
        err = self._run(adapter, _mk_job(), REPORT,
                        scheduler=_seq_run_coro(("timeout", True), "ok"))
        assert err is None
        assert adapter.send.call_count == 1                   # only the flat send ran
        flat_text = adapter.send.call_args_list[0][0][1]
        assert "All clear today." in flat_text                # summary present...
        assert "Long detail line 1." in flat_text             # ...with the body: full report

    def test_parent_timeout_in_flight_sends_body_only(self):
        # Parent is in flight (cancel False) but gave no anchor: the summary is
        # assumed delivered, so only the body goes out flat — never a second
        # copy of the summary.
        adapter = AsyncMock()
        adapter.send.return_value = MagicMock(success=True, message_id="9", raw_response={})
        err = self._run(adapter, _mk_job(), REPORT,
                        scheduler=_seq_run_coro(("timeout", False), "ok"))
        assert "un-threaded" in (err or "")
        assert adapter.send.call_count == 1
        flat_text = adapter.send.call_args_list[0][0][1]
        assert "Long detail line 1." in flat_text
        assert "All clear today." not in flat_text            # summary NOT duplicated

    def test_detail_timeout_in_flight_is_not_retried(self):
        # Parent posted; detail is in flight (cancel False). Retrying the detail
        # on any path would duplicate it, so we assume-deliver and do NOT send
        # a flat fallback.
        adapter = AsyncMock()
        adapter.send.return_value = MagicMock(
            success=True, message_id="1718000.111", raw_response={})
        err = self._run(adapter, _mk_job(), REPORT,
                        scheduler=_seq_run_coro("ok", ("timeout", False)))
        assert err is None
        assert adapter.send.call_count == 1                   # only the parent ran

    def test_detail_timeout_queued_sends_body_only(self):
        # Parent posted; detail was queued but never dispatched (cancel True):
        # nothing of the body reached the wire, so re-send the body flat — but
        # not the summary, which already landed.
        adapter = AsyncMock()
        adapter.send.side_effect = [
            MagicMock(success=True, message_id="1718000.111", raw_response={}),  # parent
            MagicMock(success=True, message_id="2", raw_response={}),            # body flat
        ]
        err = self._run(adapter, _mk_job(), REPORT,
                        scheduler=_seq_run_coro("ok", ("timeout", True), "ok"))
        assert "un-threaded" in (err or "")
        assert adapter.send.call_count == 2
        flat_text = adapter.send.call_args_list[1][0][1]
        assert "Long detail line 1." in flat_text
        assert "All clear today." not in flat_text

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

    def test_media_skipped_and_recorded_when_detail_in_flight(self, tmp_path, monkeypatch):
        # Detail send is in flight after a timeout (cancel False): the report is
        # assumed delivered, attachments are skipped as on the flat lane, and the
        # skip is surfaced in the delivery error rather than silently lost.
        media = tmp_path / "chart.png"
        media.write_bytes(b"\x89PNG fake")
        monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
                            (tmp_path,))
        adapter = AsyncMock()
        adapter.send.return_value = MagicMock(
            success=True, message_id="1718000.111", raw_response={})
        content = REPORT + f"\nMEDIA:{media.resolve()}"
        err = self._run(adapter, _mk_job(), content,
                        scheduler=_seq_run_coro("ok", ("timeout", False)))
        adapter.send_image_file.assert_not_called()
        assert adapter.send.call_count == 1                   # only the parent ran
        assert "1 media attachment(s) not delivered" in (err or "")


class TestThreadedParityWithFlatLane:
    """The threaded lane reuses the flat lane's transport and metadata, and
    yields to an already-opened handoff thread or the in_channel surface."""

    def test_authorized_transport_and_flat_metadata_forwarded(self):
        # The flat lane passes the target's authorized transport to
        # _deliver_to_platform (#115656); re-resolving cannot re-derive a
        # satellite grant. Both threaded sends must forward the same object and
        # carry the flat lane's route metadata (notify).
        from gateway.config import Platform
        from gateway.delivery import DeliveryRouter

        platform, mock_cfg, loop = _mk_env("slack")
        adapter = AsyncMock()
        transport = MagicMock(is_relay=False, adapter=adapter)
        calls = []

        async def fake_deliver(self, target, content, metadata, transport=None):
            calls.append((target, content, dict(metadata or {}), transport))
            return MagicMock(success=True, message_id=f"1718000.{len(calls)}",
                             raw_response={})

        resolved = (transport, mock_cfg.platforms[Platform.SLACK], adapter,
                    {platform: adapter})
        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value=THREAD_CFG), \
             patch("cron.scheduler_delivery._resolve_target_transport",
                   return_value=(resolved, None)), \
             patch.object(DeliveryRouter, "_deliver_to_platform", fake_deliver), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=_fake_run_coro):
            err = _deliver_result(_mk_job(), REPORT,
                                  adapters={platform: adapter}, loop=loop)
        assert err is None
        assert len(calls) == 2                              # parent, then detail
        assert all(c[3] is transport for c in calls)
        assert all("notify" in c[2] for c in calls)
        assert "All clear today." in calls[0][1]
        assert calls[1][2]["thread_id"] == "1718000.1"      # detail under parent

    def _slack_cfg(self, extra):
        from gateway.config import Platform
        pconfig = MagicMock()
        pconfig.enabled = True
        pconfig.extra = extra
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.SLACK: pconfig}
        return mock_cfg

    def _slack_adapter(self):
        adapter = AsyncMock()
        adapter.send.return_value = MagicMock(
            success=True, message_id="1718000.900", raw_response=None)
        adapter.supports_inchannel_continuable = True
        adapter.supports_inchannel_continuable_for_platform = None
        adapter._session_store = MagicMock()
        return adapter

    def _run(self, extra, adapter, opened_thread=None):
        from gateway.config import Platform
        loop = MagicMock()
        loop.is_running.return_value = True
        job = _mk_job(
            origin={"platform": "slack", "chat_id": "C123", "user_id": "U_HUMAN"},
            attach_to_session=True,
        )
        cfg = {"cron": {"threaded_delivery": True, "wrap_response": False}}
        with patch("gateway.config.load_gateway_config",
                   return_value=self._slack_cfg(extra)), \
             patch("cron.scheduler.load_config", return_value=cfg), \
             patch("cron.scheduler_delivery._open_continuable_cron_thread",
                   return_value=opened_thread) as open_mock, \
             patch("cron.scheduler_delivery._seed_cron_thread_session") as thread_seed, \
             patch("cron.scheduler_delivery._seed_cron_channel_session",
                   return_value=True) as channel_seed, \
             patch("gateway.mirror.mirror_to_session", return_value=True), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=_fake_run_coro):
            _deliver_result(job, REPORT, adapters={Platform.SLACK: adapter}, loop=loop)
        return open_mock, thread_seed, channel_seed

    def test_in_channel_surface_delivers_flat_and_seeds_channel_session(self):
        adapter = self._slack_adapter()
        open_mock, thread_seed, channel_seed = self._run(
            {"cron_continuable_surface": "in_channel"}, adapter)
        open_mock.assert_not_called()
        assert adapter.send.call_count == 1                 # one flat send, no parent
        text = adapter.send.call_args[0][1]
        assert "All clear today." in text and "Long detail line 1." in text
        channel_seed.assert_called_once()

    def test_opened_handoff_thread_delivers_flat_into_it_and_seeds_it(self):
        adapter = self._slack_adapter()
        open_mock, thread_seed, channel_seed = self._run(
            {}, adapter, opened_thread="1719000.500")
        open_mock.assert_called_once()
        assert adapter.send.call_count == 1                 # full report, flat
        text = adapter.send.call_args[0][1]
        assert "All clear today." in text and "Long detail line 1." in text
        meta = adapter.send.call_args[1].get("metadata") or adapter.send.call_args[0][2]
        assert meta.get("thread_id") == "1719000.500"
        thread_seed.assert_called_once()
        assert thread_seed.call_args[0][4] == "1719000.500"
        channel_seed.assert_not_called()
