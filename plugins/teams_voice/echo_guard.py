"""Echo guard — stop the realtime model answering its own playback.

The realtime self-answer bug (openclaw #92081's headline fix): on a speakerphone
the bot's own audio loops back into the mic, the model's VAD hears it, and it
"replies" to itself. This guard decides, per inbound caller frame, whether to
forward it to the model.

Logic (ported from the openclaw realtime echo guard):

* Keep a **playout clock** — an estimate of when our outbound audio finishes. The
  model streams faster than realtime, so wall-clock send time != play time; we
  accumulate sent-frame durations instead.
* While we're "speaking" (now < playout_end + tail window), inbound audio is
  **dropped unless it is loud enough to be a real barge-in** (RMS >= threshold).
* **Until the caller's first real turn**, allow *no* barge-in at all — so the
  opening greeting echoing back can't make the bot interrupt and re-greet itself
  in a loop while the caller is silent.

Pure logic with an injectable clock so it is unit-testable.
"""

from __future__ import annotations

import time


def _now_ms() -> float:
    return time.monotonic() * 1000.0


class EchoGuard:
    def __init__(
        self,
        *,
        enabled: bool = True,
        tail_window_ms: int = 600,
        barge_in_rms: float = 0.04,
    ) -> None:
        self.enabled = enabled
        self.tail_window_ms = tail_window_ms
        self.barge_in_rms = barge_in_rms
        self._playout_end_ms = 0.0
        self._first_turn = False

    def note_output(self, duration_ms: float, now: float | None = None) -> None:
        """Record that ``duration_ms`` of audio was queued to the worker."""
        now = _now_ms() if now is None else now
        self._playout_end_ms = max(now, self._playout_end_ms) + duration_ms

    def collapse(self, now: float | None = None) -> None:
        """A barge-in was accepted — playback is cut and the caller has the floor.

        Pull the playout horizon fully behind the tail window so ``speaking()`` is
        immediately false and the caller's audio flows without further guarding.
        """
        now = _now_ms() if now is None else now
        self._playout_end_ms = now - self.tail_window_ms

    def mark_caller_turn(self) -> None:
        """The caller has spoken a real turn; future barge-in is allowed."""
        self._first_turn = True

    def speaking(self, now: float | None = None) -> bool:
        now = _now_ms() if now is None else now
        return now < self._playout_end_ms + self.tail_window_ms

    def allow_input(self, rms: float, now: float | None = None) -> bool:
        """Return True if this inbound caller frame should reach the model."""
        if not self.enabled:
            return True
        now = _now_ms() if now is None else now
        if not self.speaking(now):
            return True
        # We are (likely) hearing our own playback.
        if not self._first_turn:
            return False  # opening greeting: never let echo trigger a barge-in
        return rms >= self.barge_in_rms
