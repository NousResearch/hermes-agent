"""Waits on the wall clock the stores stamp leases and claims with."""

from __future__ import annotations

import time

from tests.waiting import wait_until


def wait_past(instant: float) -> None:
    """Return once the wall clock is strictly past ``instant``: a lease's expiry, or a stamp a later write must outrun."""
    wait_until(lambda: time.time() > instant, what=f"the clock to pass {instant}")
