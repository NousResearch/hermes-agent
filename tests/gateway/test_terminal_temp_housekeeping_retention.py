"""The shared gateway housekeeping loop must not override the terminal-temp
cache's own 72h retention with the other caches' 24h interval.

``cleanup_terminal_temp_cache`` was folded into ``MEDIA_CACHE_CLEANUPS`` for
convenience, but that loop calls every entry with a hardcoded
``max_age_hours=24`` -- silently pruning session temp artifacts (background
process logs/pid/exit files, code-execution sandboxes) up to 2 days earlier
than the documented/tested 72h policy (``TERMINAL_TEMP_MAX_AGE_HOURS``).
"""

import gateway.run as gateway_run


class _NTicksStopEvent:
    """Run exactly ``n`` housekeeping ticks, no sleep or background thread."""

    def __init__(self, n):
        self.n = n
        self.count = 0

    def is_set(self):
        return self.count >= self.n

    def wait(self, timeout=None):
        self.count += 1
        return True


def test_terminal_temp_cleanup_keeps_its_own_72h_retention(monkeypatch):
    import tools.environments.local as local_mod

    calls = []
    monkeypatch.setattr(
        local_mod,
        "cleanup_terminal_temp_cache",
        lambda max_age_hours=None: calls.append(max_age_hours) or 0,
    )

    # IMAGE_CACHE_EVERY (the media-cache cleanup gate) is 60 ticks.
    gateway_run._start_gateway_housekeeping(_NTicksStopEvent(60), interval=0)

    assert calls == [local_mod.TERMINAL_TEMP_MAX_AGE_HOURS]
    assert calls != [24], "terminal temp must not use the other caches' 24h interval"
