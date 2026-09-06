"""Fleet-scoped rate-limit circuit breaker for scheduled jobs.

Protect the cron fleet from provider-wide rate-limit storms while retaining
an auditable persisted state transition.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None


_state_file_lock = threading.RLock()


@dataclass(frozen=True)
class BreakerPermit:
    """Decision returned before a scheduled job attempts inference."""

    allowed: bool
    half_open_probe: bool = False
    token: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class BreakerUpdate:
    """Result of recording a terminal rate-limit failure."""

    tripped: bool
    circuit_open: bool
    cooldown_until: float | None = None


class FleetRateLimitCircuitBreaker:
    """Coordinate rate-limit failures across all jobs sharing a cron directory."""

    def __init__(
        self,
        state_file: Path,
        *,
        threshold: int = 5,
        window_seconds: float = 3600.0,
        cooldown_seconds: float = 3600.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if threshold < 1:
            raise ValueError("threshold must be positive")
        if window_seconds <= 0 or cooldown_seconds <= 0:
            raise ValueError("window and cooldown must be positive")
        self.state_file = Path(state_file)
        self.lock_file = self.state_file.with_suffix(self.state_file.suffix + ".lock")
        self.threshold = threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock

    def before_run(self, job_id: str) -> BreakerPermit:
        """Allow a closed-circuit run or claim the sole half-open probe."""
        now = self.clock()
        with self._locked_state() as state:
            state_name = state["state"]
            if state_name == "closed":
                return BreakerPermit(allowed=True)

            cooldown_until = float(state.get("cooldown_until") or 0.0)
            if state_name == "open" and now >= cooldown_until:
                token = uuid.uuid4().hex
                state.update(
                    state="half_open",
                    probe_token=token,
                    probe_job_id=str(job_id),
                    probe_started_at=now,
                )
                return BreakerPermit(
                    allowed=True,
                    half_open_probe=True,
                    token=token,
                )

            if state_name == "half_open":
                try:
                    probe_started_at = float(state.get("probe_started_at") or 0.0)
                except (TypeError, ValueError):
                    probe_started_at = 0.0
                # A crashed probe must not wedge the persisted fleet state forever.
                # Reuse the cooldown as a bounded probe lease, then atomically let
                # the next due job replace the abandoned token under the state lock.
                if now - probe_started_at >= self.cooldown_seconds:
                    token = uuid.uuid4().hex
                    state.update(
                        probe_token=token,
                        probe_job_id=str(job_id),
                        probe_started_at=now,
                    )
                    return BreakerPermit(
                        allowed=True,
                        half_open_probe=True,
                        token=token,
                    )
                return BreakerPermit(
                    allowed=False,
                    reason="fleet rate-limit circuit breaker is waiting for its half-open probe",
                )

            remaining = max(0, int(cooldown_until - now))
            return BreakerPermit(
                allowed=False,
                reason=(
                    "fleet rate-limit circuit breaker is open"
                    + (f" for another {remaining}s" if remaining else "")
                ),
            )

    def record_rate_limit(self, permit: BreakerPermit) -> BreakerUpdate:
        """Record one terminal 429 and open/re-open when required."""
        now = self.clock()
        cutoff = now - self.window_seconds
        with self._locked_state() as state:
            failures = [
                float(timestamp)
                for timestamp in state.get("rate_limit_failures", [])
                if float(timestamp) >= cutoff
            ]
            failures.append(now)
            was_probe = (
                permit.half_open_probe
                and permit.token is not None
                and permit.token == state.get("probe_token")
            )
            # Jobs admitted while the circuit was closed may finish after a
            # peer has already opened it. Keep their 429s in the rolling
            # evidence, but do not extend the cooldown or emit another trip.
            if state["state"] in {"open", "half_open"} and not was_probe:
                state["rate_limit_failures"] = failures
                return BreakerUpdate(
                    tripped=False,
                    circuit_open=True,
                    cooldown_until=state.get("cooldown_until"),
                )
            should_trip = was_probe or len(failures) >= self.threshold
            if should_trip:
                cooldown_until = now + self.cooldown_seconds
                state.update(
                    state="open",
                    rate_limit_failures=failures,
                    opened_at=now,
                    cooldown_until=cooldown_until,
                    probe_token=None,
                    probe_job_id=None,
                    probe_started_at=None,
                )
                return BreakerUpdate(
                    tripped=True,
                    circuit_open=True,
                    cooldown_until=cooldown_until,
                )

            state.update(
                state="closed",
                rate_limit_failures=failures,
                probe_token=None,
                probe_job_id=None,
                probe_started_at=None,
            )
            return BreakerUpdate(tripped=False, circuit_open=False)

    def record_non_rate_limit(self, permit: BreakerPermit) -> None:
        """Reset consecutive 429s; a successful probe closes the breaker."""
        with self._locked_state() as state:
            if state["state"] == "closed":
                state.clear()
                state.update(self._default_state())
                return
            if not permit.half_open_probe or permit.token != state.get("probe_token"):
                return
            state.clear()
            state.update(self._default_state())

    @staticmethod
    def _default_state() -> dict:
        return {
            "version": 1,
            "state": "closed",
            "rate_limit_failures": [],
            "opened_at": None,
            "cooldown_until": None,
            "probe_token": None,
            "probe_job_id": None,
            "probe_started_at": None,
        }

    def _read_state(self) -> dict:
        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return self._default_state()
        if not isinstance(raw, dict) or raw.get("state") not in {
            "closed",
            "open",
            "half_open",
        }:
            return self._default_state()
        merged = {**self._default_state(), **raw}
        try:
            for field in ("opened_at", "cooldown_until", "probe_started_at"):
                value = merged.get(field)
                merged[field] = None if value is None else float(value)
            failures = merged.get("rate_limit_failures", [])
            if not isinstance(failures, list):
                raise TypeError("rate_limit_failures must be a list")
            merged["rate_limit_failures"] = [float(timestamp) for timestamp in failures]
        except (TypeError, ValueError):
            return self._default_state()
        return merged

    def _write_state(self, state: dict) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            dir=self.state_file.parent,
            prefix=f".{self.state_file.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_name, 0o600)
            os.replace(temp_name, self.state_file)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    @contextlib.contextmanager
    def _locked_state(self) -> Iterator[dict]:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        # A new breaker object is built for each parallel cron run, so this
        # module-level lock (not an instance lock) protects same-process
        # read/modify/write cycles. The advisory lock extends that guarantee
        # across profile gateways and standalone CLI processes.
        with _state_file_lock:
            lock_handle = open(self.lock_file, "a+", encoding="utf-8")
            try:
                if fcntl is not None:
                    fcntl.flock(lock_handle, fcntl.LOCK_EX)
                elif msvcrt is not None:  # pragma: no cover - Windows
                    lock_handle.seek(0)
                    getattr(msvcrt, "locking")(
                        lock_handle.fileno(), getattr(msvcrt, "LK_LOCK"), 1
                    )
                state = self._read_state()
                yield state
                self._write_state(state)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_handle, fcntl.LOCK_UN)
                elif msvcrt is not None:  # pragma: no cover - Windows
                    lock_handle.seek(0)
                    getattr(msvcrt, "locking")(
                        lock_handle.fileno(), getattr(msvcrt, "LK_UNLCK"), 1
                    )
                lock_handle.close()
