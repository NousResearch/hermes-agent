"""Clock-skew detection for the Matrix adapter's startup grace filter (#12614)."""

import logging
import time

logger = logging.getLogger("plugins.platforms.matrix.adapter")


class ClockSkewMixin:
    """State for the grace-filter clock-skew heuristic: a host clock set ahead of real time makes the
    startup grace check ``event_ts < startup_ts - 5`` silently discard every live message."""

    def _reset_clock_skew_detector(self) -> None:
        """State for _note_late_grace_drop: consecutive-drop count, their skew, and the once-only warning."""
        # Clock-skew detection: count grace-check drops that happen well after startup (i.e. not
        # initial-sync backfill). If the host's system clock is set ahead of real time, the startup grace
        # check `event_ts < startup_ts - 5` silently drops every live message. See #12614 — the symptom is
        # "bot joins rooms but never replies". Drops only count when their skew matches the first sampled
        # drop (within 60s), so varied-age backfill from freshly-invited rooms doesn't trip the heuristic.
        self._late_grace_drops: int = 0
        self._late_grace_skew: float = 0.0
        self._clock_skew_warned: bool = False

    def _note_late_grace_drop(self, event_ts: float) -> None:
        """Clock-skew heuristic for grace-check drops well after startup. A host clock set ahead of
        real time makes every live event look "older than startup" and the bot silently never
        replies. Warn once when drops keep happening >30s after startup with a *consistent* skew —
        unlike backfill from a freshly invited room, whose event ages vary widely and reset the counter."""
        if self._clock_skew_warned or time.time() - self._startup_ts <= 30:
            return
        skew = self._startup_ts - event_ts
        if not (5 < skew < 86400):  # ignore malformed/absurd timestamps
            return
        if self._late_grace_drops and abs(skew - self._late_grace_skew) < 60:
            self._late_grace_drops += 1
        else:
            self._late_grace_skew = skew
            self._late_grace_drops = 1
        if self._late_grace_drops >= 3:
            logger.warning(
                "Matrix: dropped %d consecutive live events as 'too old' more than 30s after startup "
                "(skew ≈ %.0fs). The host system clock is likely set ahead of real time, which causes "
                "the startup grace filter to silently discard every incoming message. Run "
                "`timedatectl set-ntp true` (or sync NTP) and restart the bot.", self._late_grace_drops, skew)
            self._clock_skew_warned = True
