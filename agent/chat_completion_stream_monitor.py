"""Display and heartbeat phase of the request-local streaming monitor."""

import time
from types import SimpleNamespace

from agent import chat_completion_wait_notice as wn
from agent.model_metadata import is_local_endpoint


class StreamingWaitMonitor:
    def _poll_local_load_notice(self, now: float) -> bool:
        """Managed local server: surface a cold model's weight-load progress
        instead of the 60s neutral "waiting on <model>" notice. Polled ~1s only while no
        REAL chunk arrived for 2s+ (never during healthy token flow); in-memory,
        no network. True while loading = heartbeat liveness, skip the rest of
        this iteration (the stale detector's local floor dwarfs any load)."""
        from agent.chat_completion_helpers import _managed_local_load_notice

        m = self._mon
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
        while not self._call_done.is_set():
            self._call_done.wait(timeout=0.3)
            if self._call_done.is_set():
                # The worker finished (cleanly or via its own in-loop interrupt
                # handling, see the SSE loop's `_interrupt_requested` check)
                # while this wait was asleep. Racing our own interrupt-abort
                # below against a worker that already unwound double-aborts
                # the request client: the worker's own close/abort already
                # landed, and firing here too is at best redundant, at worst a
                # second abort chasing a client the worker already closed and
                # cached/rebuilt for the next request.
                return
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
            if self.agent._interrupt_requested:
                self._abort_for_interrupt(_stale_elapsed)
                return

