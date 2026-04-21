"""Shared cross-process API rate limiter.

All Hermes profiles share a single rate-limit state file.  Before any
gateway makes an API request it calls ``acquire_api_slot()`` which:

1. Takes an exclusive flock on ``~/.hermes/api_rate.lock``
2. Reads the last-request timestamp from ``~/.hermes/api_rate``
3. If enough time has elapsed → writes the new timestamp, releases the
   lock, returns immediately.
4. If not enough time → releases the lock, sleeps the remaining
   interval, then retries.

The state file is a single line of text: the epoch timestamp (float)
of the last approved request.  No daemon process is needed — every
caller coordinates through the lock file.
"""

import fcntl
import os
import time
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

# Default minimum interval between API requests (seconds).
# Can be overridden with HERMES_API_RATE_INTERVAL env var.
DEFAULT_INTERVAL = 1.0

# How long to wait for the flock itself (seconds).
LOCK_TIMEOUT = 10.0

# Max retries before giving up.
MAX_RETRIES = 60


def _state_path() -> Path:
    return get_hermes_home() / "api_rate"


def _lock_path() -> Path:
    return get_hermes_home() / "api_rate.lock"


def acquire_api_slot(interval: Optional[float] = None) -> None:
    """Block until an API request slot is available.

    Call this right before every external API request (LLM inference,
    token refresh, agent key minting, etc.) to enforce a global
    minimum interval across all Hermes profiles.
    """
    interval = interval or float(os.getenv("HERMES_API_RATE_INTERVAL", DEFAULT_INTERVAL))
    state_path = _state_path()
    lock_path = _lock_path()

    # Ensure parent dir exists
    state_path.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(MAX_RETRIES):
        # Open / create the lock file
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            # Acquire exclusive lock (blocking with timeout)
            deadline = time.monotonic() + LOCK_TIMEOUT
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("rate_limit: timed out waiting for flock")
                    time.sleep(0.01)

            # Read last request time
            last_time = 0.0
            try:
                with open(state_path, "r") as f:
                    last_time = float(f.read().strip() or "0")
            except (FileNotFoundError, ValueError):
                pass

            now = time.time()
            elapsed = now - last_time

            if elapsed >= interval:
                # Slot available — record this request and return
                with open(state_path, "w") as f:
                    f.write(f"{now:.6f}")
                    f.flush()
                    os.fsync(f.fileno())
                return  # GO
            else:
                # Need to wait — release lock first, sleep, retry
                wait_time = interval - elapsed
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

        time.sleep(max(0.001, wait_time))

    # If we exhausted retries, just proceed (don't block forever)
