"""llama-server prompt progress, sleep-aware stall timers, and the local stale-breaker reset.

A laptop running a local model reads long prompts for minutes and sometimes sleeps mid-request.
These tests pin the behaviour that keeps such requests alive: progress chunks count as signs of
life, local requests never fall back to a non-streamed (uncancellable) request, time asleep
doesn't count as a stall, and a recovered local server gets the next message.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.local_prompt_progress import prompt_progress, prompt_progress_percent, with_prompt_progress


def _delta(**fields):
    base = dict(role="assistant", content=None, tool_calls=None, reasoning_content=None, reasoning=None, refusal=None)
    base.update(fields)
    return SimpleNamespace(**base)


def _progress_chunk(processed, total=1000, cache=0):
    choice = SimpleNamespace(index=0, delta=_delta(), finish_reason=None)
    return SimpleNamespace(choices=[choice], model=None, usage=None, id="c1",
                           model_extra={"prompt_progress": {"total": total, "cache": cache,
                                                            "processed": processed, "time_ms": 1}})


def _text_chunk(content, finish_reason=None):
    choice = SimpleNamespace(index=0, delta=_delta(content=content), finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="qwen", usage=None, id="c1", model_extra={})


class TestPromptProgressHelpers:
    def test_progress_chunk_is_recognised(self):
        assert prompt_progress(_progress_chunk(250)) == {"total": 1000, "cache": 0, "processed": 250, "time_ms": 1}

    def test_wire_dict_chunk_is_recognised(self):
        chunk = {"choices": [{"delta": {"role": "assistant", "content": None}, "finish_reason": None}],
                 "prompt_progress": {"total": 10, "processed": 5}}
        assert prompt_progress(chunk) == {"total": 10, "processed": 5}

    def test_chunk_with_text_is_a_real_chunk(self):
        chunk = _text_chunk("hi")
        chunk.model_extra = {"prompt_progress": {"total": 10, "processed": 10}}
        assert prompt_progress(chunk) is None

    def test_chunk_with_finish_reason_is_a_real_chunk(self):
        chunk = _progress_chunk(10, total=10)
        chunk.choices[0].finish_reason = "stop"
        assert prompt_progress(chunk) is None

    def test_plain_chunk_has_no_progress(self):
        assert prompt_progress(_text_chunk("hi")) is None

    def test_with_prompt_progress_merges_extra_body(self):
        kwargs = {"model": "m", "extra_body": {"cache_prompt": True}}
        out = with_prompt_progress(kwargs)
        assert out["extra_body"] == {"cache_prompt": True, "return_progress": True}
        assert kwargs["extra_body"] == {"cache_prompt": True}  # caller's dict untouched

    def test_percent(self):
        assert prompt_progress_percent({"total": 200, "processed": 50}) == 25
        assert prompt_progress_percent({"total": 0, "processed": 50}) is None
        assert prompt_progress_percent({"total": "x"}) is None


def _local_agent(provider="llamacpp", base_url="http://127.0.0.1:18434/v1"):
    from run_agent import AIAgent

    agent = AIAgent(api_key="k", base_url=base_url, provider=provider, model="qwen",
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


class TestMainStreamProgress:
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_llamacpp_stream_asks_for_progress_and_skips_progress_chunks(self, _close, mock_create):
        chunks = [_progress_chunk(0), _progress_chunk(500), _progress_chunk(1000),
                  _text_chunk("Hello"), _text_chunk(" there", finish_reason="stop")]
        client = MagicMock()
        client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = client
        agent = _local_agent()
        deltas = []
        agent.stream_delta_callback = deltas.append

        response = agent._interruptible_streaming_api_call({"model": "qwen", "messages": [{"role": "user", "content": "hi"}]})

        sent = client.chat.completions.create.call_args.kwargs
        assert sent["extra_body"]["return_progress"] is True
        assert response.choices[0].message.content == "Hello there"
        assert "".join(deltas) == "Hello there"

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_cloud_stream_does_not_ask_for_progress(self, _close, mock_create):
        client = MagicMock()
        client.chat.completions.create.return_value = iter([_text_chunk("ok", finish_reason="stop")])
        mock_create.return_value = client
        agent = _local_agent(provider="openrouter", base_url="https://openrouter.ai/api/v1")

        agent._interruptible_streaming_api_call({"model": "qwen", "messages": [{"role": "user", "content": "hi"}]})

        sent = client.chat.completions.create.call_args.kwargs
        assert "return_progress" not in (sent.get("extra_body") or {})


class TestNoNonStreamingFallbackForLlamacpp:
    def _call(self, agent):
        from agent.chat_completion_helpers import _StreamingCall

        return SimpleNamespace(agent=agent, deltas_were_sent={"yes": False}, api_kwargs={"model": "qwen"},
                               _stream_stale_timeout=900.0, result={"error": None, "response": None})

    def test_server_error_probe_skipped(self):
        from agent.chat_completion_helpers import _StreamingCall

        agent = SimpleNamespace(_interrupt_requested=False, provider="llamacpp", base_url="http://127.0.0.1:18434/v1",
                                api_mode="chat_completions", _stream_5xx_probe_ts=None)
        err = RuntimeError("boom")
        err.status_code = 500
        with patch("agent.chat_completion_helpers.interruptible_api_call") as probe:
            assert _StreamingCall._unmask_server_error_with_nonstreaming(self._call(agent), err) is False
        probe.assert_not_called()

    def test_streaming_never_disabled(self):
        from agent.chat_completion_helpers import _StreamingCall

        agent = SimpleNamespace(provider="llamacpp", base_url="http://127.0.0.1:18434/v1",
                                api_mode="chat_completions", _disable_streaming=False)
        with patch("agent.chat_completion_helpers._is_provider_stream_empty_frame_error", return_value=True):
            _StreamingCall._maybe_disable_streaming(self._call(agent), RuntimeError("empty frame"))
        assert agent._disable_streaming is False


class TestAuxProgress:
    def test_managed_local_stream_asks_for_progress(self):
        from agent import auxiliary_client as ac

        client = SimpleNamespace(base_url="http://127.0.0.1:18434/v1")
        with patch.object(ac, "_is_managed_local_endpoint", return_value=True):
            stream_kwargs, _model, _ceiling = ac._stream_request_plan({"model": "qwen", "messages": []}, client)
        assert stream_kwargs["extra_body"]["return_progress"] is True
        with patch.object(ac, "_is_managed_local_endpoint", return_value=False):
            stream_kwargs, _model, _ceiling = ac._stream_request_plan({"model": "qwen", "messages": []}, client)
        assert "extra_body" not in stream_kwargs

    def test_progress_chunk_counts_as_forward_progress(self):
        from agent import auxiliary_client as ac

        acc = ac._ChatStreamAccumulator(model="qwen")
        with patch.object(ac, "_notify_aux_progress") as tick:
            # Keeps the compression fence alive without starting the watchdog's inter-token window.
            assert acc.feed(_progress_chunk(100)) is False
            tick.assert_called_once()
            acc.feed(_text_chunk("summary", finish_reason="stop"))
        assert acc.finish().choices[0].message.content == "summary"


class TestSleepClock:
    def test_sleep_watch_reports_only_real_sleeps(self):
        from agent import sleep_clock

        wall = [1000.0]
        with patch.object(sleep_clock.time, "time", lambda: wall[0]):
            watch = sleep_clock.SleepWatch(interval=0.3)
            wall[0] += 0.3
            assert watch.slept() == 0.0
            wall[0] += 1.3  # clock sync nudge
            assert watch.slept() == 0.0
            wall[0] += 1200.3  # 20 minutes asleep, or frozen by Modern Standby
            assert watch.slept() == pytest.approx(1200.0)
            wall[0] += 0.3
            assert watch.slept() == 0.0

    def test_awake_seconds_moves_forward(self):
        from agent.sleep_clock import awake_seconds

        first = awake_seconds()
        assert awake_seconds() >= first

    def test_monitor_moves_stall_clock_past_sleep(self):
        from agent.chat_completion_stream_monitor import StreamingWaitMonitor

        mon = StreamingWaitMonitor()
        mon.agent = MagicMock()
        mon.last_chunk_time = {"t": 100.0}
        mon._mon = SimpleNamespace(last_heartbeat=100.0)
        mon._skip_sleep(900.0)
        assert mon.last_chunk_time["t"] == 1000.0
        assert mon._mon.last_heartbeat == 1000.0
        mon._skip_sleep(0.0)
        assert mon.last_chunk_time["t"] == 1000.0
        # A chunk that landed during the reading is not pushed into the future.
        now = time.time()
        mon.last_chunk_time["t"] = now - 1.0
        mon._skip_sleep(900.0)
        assert mon.last_chunk_time["t"] <= time.time()


class TestPromptProgressNotice:
    def _monitor(self, first_chunk_at=None):
        from agent import chat_completion_wait_notice as wn
        from agent.chat_completion_stream_monitor import StreamingWaitMonitor

        mon = StreamingWaitMonitor()
        mon.agent = MagicMock()
        mon.clients = SimpleNamespace(diag={"first_chunk_at": first_chunk_at})
        mon.prompt_progress = {"progress": {"total": 1000, "processed": 400}, "t": 50.0}
        mon._mon = SimpleNamespace(last_load_poll=0.0, load_notice_shown=False, load_notice_misses=0,
                                   wait_notice_started_ts=None, wait_notice=wn.WaitNoticeState(), last_heartbeat=0.0)
        return mon

    def test_shows_percent_before_first_token(self):
        mon = self._monitor()
        assert mon._show_prompt_progress(51.0) is True
        mon.agent._emit_wait_notice.assert_called_once_with("⚙ processing prompt — 40%")
        assert mon._show_prompt_progress(51.5) is False  # at most once a second

    def test_clears_line_at_first_token(self):
        mon = self._monitor()
        mon._show_prompt_progress(51.0)
        mon.clients.diag["first_chunk_at"] = 52.0
        assert mon._show_prompt_progress(53.0) is False
        mon.agent._emit_wait_notice.assert_called_with("")


class TestLocalStaleBreakerReset:
    def test_healthy_local_server_gets_one_attempt(self):
        from agent import chat_completion_helpers as helpers
        from agent import chat_completion_local as h

        agent = SimpleNamespace(_consecutive_stale_streams=5)
        with patch.object(h, "_streams_llamacpp", return_value=True), \
                patch.object(h, "_local_server_answers", return_value=True):
            assert h.retry_recovered_local_server(agent) is True
        helpers._check_stale_giveup(agent)  # the next call goes through
        helpers._bump_stale_streak(agent)  # ...and one more stale attempt shuts the breaker
        with pytest.raises(RuntimeError, match="unresponsive"):
            helpers._check_stale_giveup(agent)

    def test_closed_breaker_skips_the_health_check(self):
        from agent import chat_completion_local as h

        agent = SimpleNamespace(_consecutive_stale_streams=2)
        with patch.object(h, "_streams_llamacpp", return_value=True), \
                patch.object(h, "_local_server_answers", return_value=True) as health:
            assert h.retry_recovered_local_server(agent) is False
            health.assert_not_called()
        assert agent._consecutive_stale_streams == 2

    def test_unhealthy_or_cloud_keeps_streak(self):
        from agent import chat_completion_local as h

        agent = SimpleNamespace(_consecutive_stale_streams=5)
        with patch.object(h, "_streams_llamacpp", return_value=True), \
                patch.object(h, "_local_server_answers", return_value=False):
            assert h.retry_recovered_local_server(agent) is False
        with patch.object(h, "_streams_llamacpp", return_value=False), \
                patch.object(h, "_local_server_answers", return_value=True) as health:
            assert h.retry_recovered_local_server(agent) is False
            health.assert_not_called()
        assert agent._consecutive_stale_streams == 5
