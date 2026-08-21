"""Per-job channel_summary: [SUMMARY] extraction + thread-seed routing.

A cron job may opt into a channel-summary delivery shape: the agent is taught
(via _build_job_prompt) to open its final response with ``[SUMMARY] <1-2
lines>``, _deliver_result lifts that block, and _open_continuable_cron_thread
forwards it as the seed message of the dedicated thread — so the channel-root
line is a real TL;DR instead of the fixed handoff label.

Contract under test:

- cron/scheduler.py::_extract_cron_summary — leading-marker-only parse,
  NON-destructive (the body keeps the summary as its opening line(s); only
  the marker token is removed), guardrails (>2 lines / >400 chars / empty
  remainder -> no summary, marker still stripped), byte-identical
  passthrough when no marker leads the text.
- cron/scheduler.py::_open_continuable_cron_thread — seed_text forwarded as
  a kwarg when truthy, legacy 2-arg call otherwise; a bind-time TypeError
  from an out-of-tree adapter without the kwarg retries the 2-arg call.
- cron/scheduler.py::_deliver_result — extraction gated on the per-job flag
  (every other job's content stays byte-identical), summary rides
  seed_text= into the thread opener.
- cron/jobs.py::create_job — explicit-bool-only conditional persist (absent
  key = pre-feature behavior), same rule as attach_to_session.
- tools/cronjob_tools.py — schema property, update action, list surfacing
  of explicit bools, and the registry-handler forwarding regression
  (attach_to_session was declared in CRONJOB_SCHEMA but silently dropped
  by the dispatch lambda before this feature fixed it).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cron.jobs import create_job, load_jobs
from cron.scheduler import (
    _build_job_prompt,
    _cron_silence_suppresses_delivery,
    _deliver_result,
    _extract_cron_summary,
    _open_continuable_cron_thread,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Isolate the cron store (same pattern as tests/cron/test_jobs.py)."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path / "cron"


def _create(**kw):
    kw.setdefault("prompt", "say hi")
    kw.setdefault("schedule", "every 1h")
    return create_job(**kw)


class TestExtractCronSummary:
    def test_no_marker_is_byte_identical_passthrough(self):
        text = "  Daily digest:\n\nAll feeds nominal."
        summary, body = _extract_cron_summary(text)
        assert summary is None
        assert body == text  # untouched, leading whitespace included

    def test_empty_text(self):
        assert _extract_cron_summary("") == (None, "")

    def test_one_line_summary(self):
        summary, body = _extract_cron_summary(
            "[SUMMARY] All 3 feeds healthy.\n\nFull report:\n- feed A ok"
        )
        assert summary == "All 3 feeds healthy."
        assert body == "All 3 feeds healthy.\n\nFull report:\n- feed A ok"

    def test_two_line_summary_joined_with_newline(self):
        summary, body = _extract_cron_summary(
            "[SUMMARY] Line one.\nLine two.\n\nReport body."
        )
        assert summary == "Line one.\nLine two."
        assert body == "Line one.\nLine two.\n\nReport body."

    def test_colon_variant(self):
        summary, _ = _extract_cron_summary("[SUMMARY]: Done.\n\nDetails.")
        assert summary == "Done."

    def test_lowercase_marker(self):
        summary, _ = _extract_cron_summary("[summary] fine.\n\nrest")
        assert summary == "fine."

    def test_leading_whitespace_before_marker(self):
        summary, body = _extract_cron_summary("\n  [SUMMARY] Ship it.\n\nBody")
        assert summary == "Ship it."
        assert body == "Ship it.\n\nBody"

    def test_marker_mid_text_not_honored(self):
        text = "Intro line\n[SUMMARY] not a lead marker\n\nrest"
        assert _extract_cron_summary(text) == (None, text)

    def test_three_line_block_rejected_marker_still_stripped(self):
        summary, body = _extract_cron_summary("[SUMMARY] a\nb\nc\n\nrest")
        assert summary is None
        assert body == "a\nb\nc\n\nrest"

    def test_over_400_chars_rejected_marker_still_stripped(self):
        long_line = "x" * 401
        summary, body = _extract_cron_summary(f"[SUMMARY] {long_line}\n\nrest")
        assert summary is None
        assert body == f"{long_line}\n\nrest"

    def test_exactly_400_chars_accepted(self):
        line = "x" * 400
        summary, _ = _extract_cron_summary(f"[SUMMARY] {line}\n\nrest")
        assert summary == line

    def test_empty_remainder_rejected_marker_still_stripped(self):
        summary, body = _extract_cron_summary("[SUMMARY]\n\nreport body")
        assert summary is None
        assert body == "\n\nreport body"

    def test_marker_only_response(self):
        assert _extract_cron_summary("[SUMMARY] All good.") == (
            "All good.",
            "All good.",
        )

    def test_non_destructive_body_opens_with_the_summary(self):
        summary, body = _extract_cron_summary(
            "[SUMMARY] Two alerts fired.\nBoth auto-resolved.\n\nTimeline: ..."
        )
        assert summary is not None
        assert body.startswith(summary)


class TestOpenContinuableThreadSeed:
    """seed_text forwarding contract of _open_continuable_cron_thread."""

    JOB = {"id": "j1", "name": "Brief"}

    @staticmethod
    def _run_now(coro, _loop):
        """Stand-in for safe_schedule_threadsafe: close the coro, hand back a
        ready future carrying the adapter's thread id (same double as
        tests/cron/test_scheduler.py::test_open_thread_returns_id_on_thread_platform)."""
        coro.close()
        fut = MagicMock()
        fut.result.return_value = "9001"
        return fut

    class _RecordingAdapter:
        """Modern adapter: accepts any signature, records (args, kwargs)."""

        def __init__(self):
            self.calls = []

        def create_handoff_thread(self, *args, **kwargs):
            self.calls.append((args, kwargs))

            async def _coro():
                return "9001"

            return _coro()

    class _LegacyAdapter:
        """Out-of-tree adapter predating the seed_text kwarg: 2-arg only."""

        def __init__(self):
            self.calls = []

        def create_handoff_thread(self, chat_id, name):
            self.calls.append((chat_id, name))

            async def _coro():
                return "9001"

            return _coro()

    def test_seed_text_forwarded_as_kwarg(self):
        adapter = self._RecordingAdapter()
        with patch("agent.async_utils.safe_schedule_threadsafe", side_effect=self._run_now):
            tid = _open_continuable_cron_thread(
                self.JOB, adapter, "123", loop=MagicMock(), seed_text="All green."
            )
        assert tid == "9001"
        assert adapter.calls == [(("123", "Hermes — Brief"), {"seed_text": "All green."})]

    def test_no_seed_text_keeps_legacy_two_arg_call(self):
        adapter = self._RecordingAdapter()
        with patch("agent.async_utils.safe_schedule_threadsafe", side_effect=self._run_now):
            tid = _open_continuable_cron_thread(
                self.JOB, adapter, "123", loop=MagicMock(), seed_text=None
            )
        assert tid == "9001"
        assert adapter.calls == [(("123", "Hermes — Brief"), {})]

    def test_legacy_adapter_type_error_retries_two_arg_call(self):
        """Bind-time TypeError (no coroutine created) must not lose the
        thread: retry without the kwarg, thread id still returned."""
        adapter = self._LegacyAdapter()
        with patch("agent.async_utils.safe_schedule_threadsafe", side_effect=self._run_now):
            tid = _open_continuable_cron_thread(
                self.JOB, adapter, "123", loop=MagicMock(), seed_text="TLDR line"
            )
        assert tid == "9001"
        # Exactly one successful bind — the failed 3-arg attempt never ran the body.
        assert adapter.calls == [("123", "Hermes — Brief")]

    # ── await-time TypeError (wrapper adapters that bind but blow up on run) ──

    @staticmethod
    def _run_coro(coro, _loop):
        """safe_schedule_threadsafe double that actually RUNS the coroutine,
        so a TypeError raised inside its body surfaces from future.result()."""
        import asyncio as _asyncio
        from concurrent.futures import Future

        fut = Future()
        try:
            fut.set_result(_asyncio.run(coro))
        except BaseException as e:  # noqa: BLE001
            fut.set_exception(e)
        return fut

    class _AwaitTimeLegacyAdapter:
        """Async *args/**kwargs wrapper around a legacy 2-arg implementation:
        binds seed_text fine, raises TypeError only when the coroutine runs."""

        def __init__(self):
            self.calls = []

        def create_handoff_thread(self, *args, **kwargs):
            self.calls.append((args, kwargs))

            async def _coro():
                if kwargs:
                    raise TypeError(
                        "create_handoff_thread() got an unexpected keyword "
                        "argument 'seed_text'"
                    )
                return "9001"

            return _coro()

    class _AlwaysTypeErrorAdapter:
        """A genuinely broken adapter: every awaited call raises TypeError."""

        def __init__(self):
            self.calls = []

        def create_handoff_thread(self, *args, **kwargs):
            self.calls.append((args, kwargs))

            async def _coro():
                raise TypeError("genuine bug inside the adapter")

            return _coro()

    def test_await_time_type_error_with_seed_retries_two_arg_call(self):
        """REGRESSION: a wrapper adapter defeats the bind-time catch — the
        TypeError only appears when the coroutine runs. Same degradation
        contract: retry the 2-arg call, thread still opens."""
        adapter = self._AwaitTimeLegacyAdapter()
        with patch("agent.async_utils.safe_schedule_threadsafe", side_effect=self._run_coro):
            tid = _open_continuable_cron_thread(
                self.JOB, adapter, "123", loop=MagicMock(), seed_text="TLDR line"
            )
        assert tid == "9001"
        assert adapter.calls == [
            (("123", "Hermes — Brief"), {"seed_text": "TLDR line"}),
            (("123", "Hermes — Brief"), {}),
        ]

    def test_await_time_type_error_without_seed_yields_none(self):
        """A genuine adapter TypeError on a NO-seed call is not a kwarg
        problem: no retry, propagates to the outer handler, returns None."""
        adapter = self._AlwaysTypeErrorAdapter()
        with patch("agent.async_utils.safe_schedule_threadsafe", side_effect=self._run_coro):
            tid = _open_continuable_cron_thread(
                self.JOB, adapter, "123", loop=MagicMock(), seed_text=None
            )
        assert tid is None
        assert len(adapter.calls) == 1  # exactly one attempt, no retry

    def test_await_time_type_error_on_both_attempts_yields_none(self):
        """Seed call AND its 2-arg retry both raise: exactly two attempts,
        then None — the retry never loops."""
        adapter = self._AlwaysTypeErrorAdapter()
        with patch("agent.async_utils.safe_schedule_threadsafe", side_effect=self._run_coro):
            tid = _open_continuable_cron_thread(
                self.JOB, adapter, "123", loop=MagicMock(), seed_text="TLDR line"
            )
        assert tid is None
        assert len(adapter.calls) == 2


class TestDeliverResultChannelSummaryGate:
    """_deliver_result: extraction gated on the per-job flag; summary rides
    seed_text= into the thread opener; other jobs' content is byte-identical."""

    def _run_delivery(self, job, content):
        """Drive _deliver_result down the live-adapter origin path (harness
        modeled on TestCronContinuableSurfaceInChannel in test_scheduler.py).
        Returns (_open_continuable_cron_thread mock, captured delivery)."""
        from concurrent.futures import Future
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        pconfig.extra = {}
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.SLACK: pconfig}

        loop = MagicMock()
        loop.is_running.return_value = True

        adapter = AsyncMock()
        adapter.send.return_value = MagicMock(
            success=True, message_id="msg_1", raw_response=None,
        )
        adapter.supports_inchannel_continuable_for_platform = None
        adapter._session_store = MagicMock()

        captured = {}

        class _SpyRouter:
            def __init__(self, *a, **k):
                pass

            async def _deliver_to_platform(self, target, text, metadata):
                captured["text"] = text
                return {"success": True, "message_id": "msg_1"}

        def fake_run_coro(coro, _loop):
            future = Future()
            try:
                import asyncio as _asyncio
                future.set_result(_asyncio.run(coro))
            except BaseException as _e:  # noqa: BLE001
                future.set_exception(_e)
            return future

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("gateway.delivery.DeliveryRouter", _SpyRouter), \
             patch("cron.scheduler._open_continuable_cron_thread", return_value=None) as open_mock, \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro), \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            _deliver_result(
                job, content,
                adapters={Platform.SLACK: adapter}, loop=loop,
            )
        return open_mock, captured

    def _job(self, **extra):
        return {
            "id": "brief-job",
            "name": "Daily Brief",
            "deliver": "origin",
            "origin": {"platform": "slack", "chat_id": "C123", "user_id": "U_HUMAN"},
            "attach_to_session": True,
            **extra,
        }

    def test_summary_becomes_the_thread_seed_and_body_keeps_it(self):
        open_mock, captured = self._run_delivery(
            self._job(channel_summary=True),
            "[SUMMARY] All 3 feeds healthy.\n\nFull report:\n- feed A ok",
        )
        open_mock.assert_called_once()
        assert open_mock.call_args.kwargs.get("seed_text") == "All 3 feeds healthy."
        # Non-destructive: the delivered body opens with the summary, marker gone.
        assert captured["text"].startswith("All 3 feeds healthy.")
        assert "[SUMMARY]" not in captured["text"]
        assert "Full report:" in captured["text"]

    def test_without_the_flag_content_is_byte_identical(self):
        content = "[SUMMARY] would-be summary\n\nFull report."
        open_mock, captured = self._run_delivery(self._job(), content)
        open_mock.assert_called_once()
        assert open_mock.call_args.kwargs.get("seed_text") is None
        # No extraction: the marker text delivers exactly as the agent wrote it.
        assert captured["text"] == content

    def test_media_tag_in_summary_is_dropped_from_the_seed(self, tmp_path):
        """REGRESSION: the seed rides create_handoff_thread's raw post, which
        bypasses the send pipeline where MEDIA tags are normally extracted —
        a MEDIA: line inside the summary block must not print a raw local
        filesystem path at the channel root. The body copy owns attachments."""
        report = tmp_path / "digest.pdf"
        report.write_bytes(b"%PDF-1.4 x")
        open_mock, _ = self._run_delivery(
            self._job(channel_summary=True),
            f"[SUMMARY] Digest ready.\nMEDIA:{report}\n\nFull report follows.",
        )
        open_mock.assert_called_once()
        seed = open_mock.call_args.kwargs.get("seed_text")
        assert seed == "Digest ready."
        assert str(report) not in seed

    def test_media_only_summary_collapses_to_no_seed(self, tmp_path):
        """A summary that was ONLY a MEDIA tag leaves no displayable text:
        seed_text collapses to None and the thread opens with the fixed label."""
        report = tmp_path / "digest.pdf"
        report.write_bytes(b"%PDF-1.4 x")
        open_mock, _ = self._run_delivery(
            self._job(channel_summary=True),
            f"[SUMMARY] MEDIA:{report}\n\nFull report follows.",
        )
        open_mock.assert_called_once()
        assert open_mock.call_args.kwargs.get("seed_text") is None


class TestJobStoreChannelSummary:
    """create_job: explicit-bool-only conditional persist (attach_to_session rule)."""

    def test_true_persisted_and_round_trips(self, tmp_cron_dir):
        job = _create(channel_summary=True)
        assert job["channel_summary"] is True
        assert load_jobs()[0]["channel_summary"] is True

    def test_false_persisted_explicitly(self, tmp_cron_dir):
        job = _create(channel_summary=False)
        assert job["channel_summary"] is False
        assert load_jobs()[0]["channel_summary"] is False

    def test_omitted_key_absent(self, tmp_cron_dir):
        job = _create()
        assert "channel_summary" not in job
        assert "channel_summary" not in load_jobs()[0]

    def test_non_bool_not_persisted(self, tmp_cron_dir):
        """Truthy non-bools are not explicit opt-ins: key stays absent."""
        job = _create(channel_summary="yes")
        assert "channel_summary" not in job


class TestCronjobToolChannelSummary:
    def test_schema_exposes_channel_summary(self):
        from tools.cronjob_tools import CRONJOB_SCHEMA

        prop = CRONJOB_SCHEMA["parameters"]["properties"]["channel_summary"]
        assert prop["type"] == "boolean"

    def test_update_sets_channel_summary(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob

        job = _create()
        out = json.loads(
            cronjob(action="update", job_id=job["id"], channel_summary=True)
        )
        assert out["success"] is True
        assert out["job"]["channel_summary"] is True
        assert load_jobs()[0]["channel_summary"] is True

    def test_update_can_turn_it_off(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob

        job = _create(channel_summary=True)
        out = json.loads(
            cronjob(action="update", job_id=job["id"], channel_summary=False)
        )
        assert out["success"] is True
        assert load_jobs()[0]["channel_summary"] is False

    def test_list_surfaces_explicit_bools_only(self, tmp_cron_dir):
        """_format_job: explicit channel_summary / attach_to_session render;
        absent keys mean "default" and must not render as false."""
        from tools.cronjob_tools import cronjob

        _create(channel_summary=True, attach_to_session=True)
        listed = json.loads(cronjob(action="list"))["jobs"][0]
        assert listed["channel_summary"] is True
        assert listed["attach_to_session"] is True

    def test_list_omits_unset_flags(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob

        _create()
        listed = json.loads(cronjob(action="list"))["jobs"][0]
        assert "channel_summary" not in listed
        assert "attach_to_session" not in listed

    def test_registry_handler_forwards_attach_and_summary(self, monkeypatch):
        """REGRESSION: attach_to_session was declared in CRONJOB_SCHEMA but
        the dispatch lambda never read it from the model's args, so agent-set
        values were silently dropped. Guard both flags through the handler."""
        import tools.cronjob_tools as mod

        recorded = {}

        def _spy(**kwargs):
            recorded.update(kwargs)
            return "{}"

        monkeypatch.setattr(mod, "cronjob", _spy)
        handler = mod.registry._tools["cronjob"].handler
        handler({"action": "list", "attach_to_session": True, "channel_summary": True})
        assert recorded["action"] == "list"
        assert recorded["attach_to_session"] is True
        assert recorded["channel_summary"] is True

    def test_registry_handler_passes_none_when_flags_omitted(self, monkeypatch):
        """Jobs NOT using the new flags are untouched by the handler fix:
        args.get() misses cleanly, cronjob() receives None for both — which
        create_job's explicit-bool conditional-persist turns into absent keys
        (pinned by TestJobStoreChannelSummary::test_omitted_key_absent)."""
        import tools.cronjob_tools as mod

        recorded = {}

        def _spy(**kwargs):
            recorded.update(kwargs)
            return "{}"

        monkeypatch.setattr(mod, "cronjob", _spy)
        handler = mod.registry._tools["cronjob"].handler
        handler({"action": "list"})
        assert recorded["attach_to_session"] is None
        assert recorded["channel_summary"] is None


class TestRunJobPersistedOutputMarkerStrip:
    """run_job's composite saved-output doc: channel_summary jobs persist the
    marker-stripped body in the "## Response" section (summary lines retained
    — extraction is non-destructive) so context_from / continuity chains
    reading the saved output never re-ingest the raw [SUMMARY] control token.
    Harness reused from test_scheduler.py's run_job model-resolution tests.

    NOTE: assertions are scoped to the "## Response" section — the doc's
    "## Prompt" section carries the assembled cron hint, which for flagged
    jobs legitimately contains the literal "[SUMMARY]" instruction text.
    """

    _RUNTIME = {
        "api_key": "***",
        "base_url": "https://example.invalid/v1",
        "provider": "openrouter",
        "api_mode": "chat_completions",
    }

    def _run(self, job, final_response, tmp_path):
        from cron.scheduler import run_job

        fake_db = MagicMock()
        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value=self._RUNTIME), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": final_response}
            mock_agent_cls.return_value = mock_agent
            success, output, response, error = run_job(job)
        assert success is True
        assert error is None
        return output, response

    RAW = "[SUMMARY] All good.\n\nDetail."

    def test_flagged_job_persists_marker_stripped_response(self, tmp_path):
        job = {"id": "sum-job", "name": "sum", "prompt": "check",
               "channel_summary": True}
        output, response = self._run(job, self.RAW, tmp_path)
        response_section = output.split("## Response", 1)[1]
        assert "All good." in response_section  # summary line retained
        assert "Detail." in response_section
        assert "[SUMMARY]" not in response_section
        # The RETURNED final_response stays raw — delivery does its own
        # extraction from it (seed lifting happens in _deliver_result).
        assert response == self.RAW

    def test_unflagged_job_persists_raw_response_byte_identical(self, tmp_path):
        job = {"id": "raw-job", "name": "raw", "prompt": "check"}
        output, _ = self._run(job, self.RAW, tmp_path)
        assert f"## Response\n\n{self.RAW}\n" in output


class TestSilenceGateChannelSummaryShortCircuit:
    """A valid [SUMMARY] lead on a channel_summary job affirmatively declares
    deliverable content: a stray trailing [SILENT] (the hint forbids the
    combination, but models pattern-match adjacent instructions) must not
    swallow the whole report. Non-summary responses keep the exact
    pre-existing suppression. Harness copied from
    tests/cron/test_scheduler.py::TestSilentDelivery (tick with run_job
    patched)."""

    STRAY = "[SUMMARY] 3 alerts fired.\n\nFull report: two resolved, one open.\n\n[SILENT]"

    def _job(self, **extra):
        return {
            "id": "monitor-job",
            "name": "monitor",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
            **extra,
        }

    def _tick(self, job, response):
        from cron.scheduler import tick

        with patch("cron.scheduler.get_due_jobs", return_value=[job]), \
             patch("cron.scheduler.claim_job_for_fire", return_value=True), \
             patch("cron.scheduler.run_job", return_value=(True, "# output", response, None)), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            tick(verbose=False)
        return deliver_mock

    def test_summary_led_response_with_stray_silent_still_delivers(self, caplog):
        import logging

        job = self._job(channel_summary=True)
        with patch("cron.scheduler.get_due_jobs", return_value=[job]), \
             patch("cron.scheduler.claim_job_for_fire", return_value=True), \
             patch("cron.scheduler.run_job", return_value=(True, "# output", self.STRAY, None)), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            with caplog.at_level(logging.INFO, logger="cron.scheduler"):
                tick(verbose=False)
        deliver_mock.assert_called_once()
        # The stray-marker log proves the silence gate actually fired and was
        # short-circuited — not that the response missed silence detection.
        assert any("stray" in r.message for r in caplog.records)

    def test_same_response_without_the_flag_is_still_suppressed(self):
        deliver_mock = self._tick(self._job(), self.STRAY)
        deliver_mock.assert_not_called()

    def test_silent_only_response_on_flagged_job_is_still_suppressed(self):
        """No valid [SUMMARY] lead -> the extractor returns None and the
        pre-existing suppression applies unchanged."""
        deliver_mock = self._tick(self._job(channel_summary=True), "[SILENT]")
        deliver_mock.assert_not_called()


class TestCronSilenceSuppressesDeliveryPredicate:
    """Helper-level contract of _cron_silence_suppresses_delivery: the
    channel-summary carve-out keys on MARKER PRESENCE plus substantive body,
    NOT on the summary passing the line/char guardrails — an over-limit
    summary block still fronts a real report, and a format miss must not
    turn into data loss. A marker-led response whose body is nothing but
    silence tokens stays suppressed."""

    FLAGGED = {"id": "j1", "channel_summary": True}
    PLAIN = {"id": "j1"}

    def test_over_limit_summary_with_stray_silent_delivers(self):
        """3-line block fails the 2-line guardrail (extraction returns
        summary=None) — but the marker-led response still fronts a real
        report, so the stray [SILENT] must not swallow it."""
        text = "[SUMMARY] line1\nline2\nline3\n\nreport body\n\n[SILENT]"
        assert _extract_cron_summary(text)[0] is None  # guardrail miss, per spec
        assert _cron_silence_suppresses_delivery(self.FLAGGED, text) is False

    def test_marker_led_but_content_free_stays_suppressed(self):
        assert _cron_silence_suppresses_delivery(
            self.FLAGGED, "[SUMMARY]\n\n[SILENT]"
        ) is True

    def test_silent_only_on_flagged_job_stays_suppressed(self):
        assert _cron_silence_suppresses_delivery(self.FLAGGED, "[SILENT]") is True

    def test_valid_summary_with_stray_silent_delivers(self):
        assert _cron_silence_suppresses_delivery(
            self.FLAGGED, "[SUMMARY] x\n\nreport\n\n[SILENT]"
        ) is False

    def test_unflagged_job_keeps_pre_existing_suppression(self):
        """The documented "why-quiet note\\n\\n[SILENT]" pattern, no flag."""
        assert _cron_silence_suppresses_delivery(
            self.PLAIN, "nothing changed today\n\n[SILENT]"
        ) is True

    def test_non_silent_response_never_suppressed(self):
        assert _cron_silence_suppresses_delivery(
            self.FLAGGED, "[SUMMARY] x\n\nreport"
        ) is False


class TestBuildJobPromptChannelSummaryHint:
    def test_flagged_job_gets_the_summary_instruction(self):
        result = _build_job_prompt({"prompt": "Check feeds", "channel_summary": True})
        assert "[CHANNEL SUMMARY" in result
        assert "[SUMMARY]" in result
        assert "Check feeds" in result

    def test_unflagged_job_hint_is_byte_identical(self):
        baseline = _build_job_prompt({"prompt": "Check feeds"})
        assert "[CHANNEL SUMMARY" not in baseline
        # channel_summary False is the same as absent — no hint drift.
        assert _build_job_prompt({"prompt": "Check feeds", "channel_summary": False}) == baseline
