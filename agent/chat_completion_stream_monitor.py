"""Display and heartbeat phase of the request-local streaming monitor."""

import logging
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

from agent import chat_completion_wait_notice as wn
from agent.model_metadata import is_local_endpoint
from agent.proxy_bypass import is_loopback_host
from agent.sleep_clock import SleepWatch

logger = logging.getLogger(__name__)


class StreamingWaitMonitor:
    def _stale_kill_patience(self) -> float:
        """Wall-clock window a wedged attempt's re-kills may keep re-arming before it is
        abandoned: its own read timeout. That timeout is the last thing that legitimately
        ends a parked read by itself, so a re-kill only proves wedging once it has passed;
        and a short stale budget re-fires many times inside one read timeout, so the bound
        is wall clock, never a kill count (#141098)."""
        try:
            return float(self._stream_timeouts()[1])
        except (AttributeError, IndexError, TypeError, ValueError):
            return 120.0  # a monitor without the wire's timeouts (the bare mixin in tests)

    def _note_stale_kill(self, counted: bool, elapsed: float) -> None:
        """Book one stale kill of the current attempt; keep its display/liveness honest.

        The first kill of an attempt is the one that counts toward the cross-call breaker
        and claims liveness. A re-kill means the previous kill did NOT unblock the reader —
        there is no progress to report, so it must not refresh the activity heartbeat
        either, or the kanban wedge reclaim never sees the worker (#141098). Past the
        patience window the attempt is abandoned: no further kill can change its outcome.
        """
        with self.stream_attempt_lock:
            attempt = int(self.stream_attempt_state["current"])
        state = getattr(self, "_stale_rekill_state", None)
        if state is None:
            state = self._stale_rekill_state = {"attempt": attempt, "count": 0, "since": time.time()}
        if state["attempt"] != attempt:
            state["attempt"], state["count"], state["since"] = attempt, 0, time.time()
        state["count"] += 1
        # Never on the first kill: one kill is the normal recovery path, and the window
        # below runs from it. From the second kill on, only the attempt's own read patience
        # decides (a zero patience must not turn a single kill into an abandonment).
        if state["count"] > 1 and time.time() - state["since"] >= self._stale_kill_patience():
            self._abandon_stale_attempt(elapsed)
            return
        self.last_chunk_time["t"] = time.time()  # another full stale window before the next kill
        self.agent._emit_diagnostic_wait(f"⚠ no output from provider for {int(elapsed)}s — reconnecting...")
        if counted:
            self.agent._touch_activity(f"stale stream detected after {int(elapsed)}s, reconnecting")

    def _abandon_stale_attempt(self, elapsed: float) -> None:
        """A wedged attempt whose kills never unwound the reader: stop killing and stop
        claiming liveness. The worker path raises from ``run()``; the inline path leaves
        the activity heartbeat to go stale so the external reclaim owns the worker."""
        state = self._stale_rekill_state
        logger.error(
            "Stale stream: attempt %s was re-killed %d times over %.0fs — past its %.0fs read "
            "patience without the reader unwinding; abandoning it (the socket shutdown is not "
            "unblocking the worker). model=%s",
            state["attempt"], state["count"], time.time() - state["since"],
            self._stale_kill_patience(), self.api_kwargs.get("model", "unknown"),
        )
        self._stale_wedged = True

    def _attempt_is_wedged(self) -> bool:
        return bool(getattr(self, "_stale_wedged", False))

    def _raise_wedged_stream(self) -> None:
        """Fail the call so the retry/fallback path runs instead of looping forever."""
        state = self._stale_rekill_state
        raise RuntimeError(
            f"Provider stream is wedged: the stale watchdog killed attempt {state['attempt']} "
            f"{state['count']} times without the read unwinding — aborting this call so the "
            "retry/fallback path can run. Switch models or start a new session, then retry."
        )

    def _poll_local_load_notice(self, now: float) -> bool:
        """Managed local server: surface a cold model's weight-load progress
        instead of the 60s neutral "waiting on <model>" notice. Polled ~1s only while no
        REAL chunk arrived for 2s+ (never during healthy token flow); in-memory,
        no network. True while loading = heartbeat liveness, skip the rest of
        this iteration (the stale detector's local floor dwarfs any load)."""
        from agent.chat_completion_local import _managed_local_load_notice

        m = self._mon
        if self._show_prompt_progress(now):
            return True
        if now - self.last_chunk_time["t"] < 2.0 or now - m.last_load_poll < 1.0:
            return False
        m.last_load_poll = now
        _load_notice = _managed_local_load_notice(self.agent, self.api_kwargs)
        if _load_notice is not None:
            m.wait_notice_started_ts = None  # The local loader now owns the display.
            m.wait_notice.reset()
            self.agent._emit_wait_notice(_load_notice)
            self.agent._touch_activity("local model loading")
            m.load_notice_shown, m.load_notice_misses, m.last_heartbeat = True, 0, now  # loading IS liveness
            return True
        if m.load_notice_shown:
            # One missed sample is routine (probe timeout under load); clearing on it strobed the line.
            m.load_notice_misses += 1
            if m.load_notice_misses >= 3:
                m.load_notice_shown, m.load_notice_misses = False, 0
                self.agent._emit_wait_notice("")
        return False

    def _show_prompt_progress(self, now: float) -> bool:
        """Prompt progress straight from the stream's own progress chunks (llama-server with
        ``return_progress``), at most once a second until the first token. Clears the line
        when the first token arrives. True when it showed progress."""
        from agent.local_prompt_progress import prompt_progress_percent

        m = self._mon
        state = getattr(self, "prompt_progress", None)
        diag = getattr(getattr(self, "clients", None), "diag", None)
        first_token = isinstance(diag, dict) and bool(diag.get("first_chunk_at"))
        if not state or state["progress"] is None:
            return False
        if first_token:
            if m.load_notice_shown:
                m.load_notice_shown, m.load_notice_misses = False, 0
                self.agent._emit_wait_notice("")
            return False
        if now - state["t"] > 60.0 or now - m.last_load_poll < 1.0:
            return False
        m.last_load_poll = now
        percent = prompt_progress_percent(state["progress"])
        m.wait_notice_started_ts = None  # The prompt-progress line owns the display.
        m.wait_notice.reset()
        self.agent._emit_wait_notice(
            f"⚙ processing prompt — {percent}%" if percent is not None else "⚙ processing prompt")
        m.load_notice_shown, m.load_notice_misses, m.last_heartbeat = True, 0, now
        return True

    def _skip_sleep(self, slept: float) -> None:
        """The computer slept and froze this request with a server on the same machine: move the
        stall clock past the sleep, so the request gets its full patience again after wake. Never
        past now: a chunk can land between the sleep reading and this write."""
        if slept <= 0:
            return
        now = time.time()
        self.last_chunk_time["t"] = min(self.last_chunk_time["t"] + slept, now)
        self._mon.last_heartbeat = min(self._mon.last_heartbeat + slept, now)
        logger.info("Computer slept for %.0fs during a model request; the stall timer skips that time.", slept)
        self.agent._touch_activity(f"resumed after the computer slept for {int(slept)}s")

    def _heartbeat(self, waiting_secs: int) -> None:
        """Gateway inactivity heartbeat: the start-to-first-chunk gap (thinking,
        local prefill) can exceed the gateway timeout."""
        if waiting_secs >= 60.0:
            # No chunks for 60s+: say WHAT the wait is and WHEN recovery kicks in —
            # once per silence, not every heartbeat (#92550).
            stale = self._stream_stale_timeout
            watchdog = ("stream stale", stale - waiting_secs) if stale is not None and stale != float("inf") else None
            diag = getattr(getattr(self, "clients", None), "diag", None)
            phase = "post_chunk" if isinstance(diag, dict) and diag.get("first_chunk_at") else "first_chunk"
            if not self._mon.wait_notice.should_emit(phase, watchdog):
                self.agent._touch_activity(f"waiting for stream response ({waiting_secs}s, {phase})")
                return
            self._mon.wait_notice_started_ts = self._mon.last_heartbeat
            self.agent._emit_wait_notice(wn.wait_notice_text(
                self.api_kwargs.get('model', 'the provider'), waiting_secs, phase, watchdog))
        else:
            # Chunks are flowing — keep the tracker fresh, leave the display alone.
            self.agent._touch_activity(f"waiting for stream response ({waiting_secs}s, no chunks yet)")

    def _monitor_loop(self) -> None:
        _HEARTBEAT_INTERVAL = 30.0  # seconds between gateway activity touches
        self._mon = SimpleNamespace(
            last_heartbeat=time.time(), last_load_poll=0.0,
            load_notice_shown=False, load_notice_misses=0, wait_notice_started_ts=None,
            wait_notice=wn.WaitNoticeState(),
        )
        _is_local_base = bool(self.agent.base_url) and is_local_endpoint(self.agent.base_url)
        # Only a server on this machine sleeps with it; a remote one kept working (or died).
        sleep_watch = (SleepWatch(interval=0.3) if self.agent.base_url
                       and is_loopback_host(urlsplit(self.agent.base_url).hostname) else None)
        while not self._call_done.is_set():
            self._call_done.wait(timeout=0.3)
            if sleep_watch is not None:
                self._skip_sleep(sleep_watch.slept())
            _hb_now = time.time()
            if _is_local_base and self._poll_local_load_notice(_hb_now):
                continue
            # Reasoning callbacks do not clear the classic CLI spinner. The empty
            # protocol payload resets status without adding synthetic reasoning.
            if (self._mon.wait_notice_started_ts is not None
                    and self.last_chunk_time["t"] > self._mon.wait_notice_started_ts):
                self.agent._emit_wait_notice("")
                self._mon.wait_notice_started_ts = None
                self._mon.wait_notice.reset()
            if _hb_now - self._mon.last_heartbeat >= _HEARTBEAT_INTERVAL:
                self._mon.last_heartbeat = _hb_now
                self._heartbeat(int(_hb_now - self.last_chunk_time["t"]))
            _stale_elapsed = time.time() - self.last_chunk_time["t"]
            if _stale_elapsed > self._stream_stale_timeout:
                self._mon.wait_notice_started_ts = None  # Reconnect status has its own owner.
                self._mon.wait_notice.reset()
                self._kill_stale_stream(_stale_elapsed)
                if self._attempt_is_wedged():
                    # The reader survived every kill: no further kill can change that, and the
                    # heartbeat would keep claiming a live worker. Worker path: run() raises for
                    # the caller to retry/fall back; inline path: the external wedge reclaim now
                    # sees a stale worker instead of a fresh heartbeat (#141098).
                    return
            if self.agent._interrupt_requested:
                self._abort_for_interrupt(_stale_elapsed)
                return

