"""Dual-clock stamps for provider silence across host suspend."""

import time


class StreamLiveness:
    """Track the last provider event on both display and decision clocks."""

    def __init__(self) -> None:
        self._stamp: tuple[float, float] | None = None

    def touch(self) -> float:
        """Stamp both clocks and return wall time for existing diagnostics."""
        stamp = (time.time(), time.monotonic())
        self._stamp = stamp
        return stamp[0]

    def silence(self) -> tuple[float, float]:
        """Return awake silence and the portion attributable to host suspend."""
        stamp = self._stamp
        if stamp is None:
            return 0.0, 0.0
        wall_stamp, monotonic_stamp = stamp
        awake = time.monotonic() - monotonic_stamp
        wall = time.time() - wall_stamp
        return awake, max(0.0, wall - awake)


def describe_silence(awake_secs: float, suspend_secs: float) -> str:
    """Format provider silence without attributing host-sleep time to the provider."""
    awake_secs = max(0.0, awake_secs)
    suspend_secs = max(0.0, suspend_secs)
    if suspend_secs < 1.0:
        return f"{int(awake_secs)}s"
    return f"{int(round(awake_secs))}s awake ({int(round(suspend_secs))}s host suspend)"
