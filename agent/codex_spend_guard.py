"""Proactive, global, cross-process Codex spend guard.

A sliding-window circuit breaker that caps how often (and how many tokens)
the Hermes watchers may spend against Codex.  The three watcher processes
(hermesgeneralist / 2 / 3) each call Codex independently with no proactive
spend cap, which is how Codex tokens got exhausted.  This module enforces a
GLOBAL cap across those separate OS processes by persisting the call/token
ledger to a SHARED file under ``~/.hermes/`` and serialising read-modify-write
with advisory file locking.

Design mirrors ``agent/nous_rate_guard.py`` (atomic temp-file write +
``os.replace`` via :func:`utils.atomic_replace`, JSON persistence under
``~/.hermes/``) and the cross-process ``fcntl.flock`` pattern in
``agent/shell_hooks.py``.

The guard is **always on** (config can only LOWER the hard ceilings, never
disable them) and **fails OPEN** on any internal error: a broken ledger or a
lock failure must never wedge the watchers — it logs a WARNING and allows the
call through.

This module is intentionally self-contained and is NOT wired into any
production call path here; a later integration task gates Codex calls on it.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from utils import atomic_replace

try:  # POSIX only; non-POSIX falls back to best-effort (no cross-process lock).
    import fcntl
except ImportError:  # pragma: no cover — non-POSIX fallback
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# --- Hard ceilings (un-disableable; config may only LOWER these) -----------
MAX_CALLS_PER_HOUR = 60
MAX_CALLS_PER_DAY = 400
MAX_TOKENS_PER_DAY = 5_000_000
HOUR_SECONDS = 3600
DAY_SECONDS = 86_400

_DEFAULT_LEDGER_NAME = "codex_spend.json"
_CONFIG_SECTION = "codex_spend_cap"


@dataclass
class Limits:
    """Effective limits (already clamped to the hard ceilings)."""

    max_calls_per_hour: int
    max_calls_per_day: int
    max_tokens_per_day: int


@dataclass
class Decision:
    """Result of a pure window evaluation."""

    allowed: bool
    reason: Optional[str] = None


@dataclass
class Reservation:
    """Result of a :meth:`CodexSpendGuard.reserve` attempt."""

    allowed: bool
    reason: Optional[str] = None
    failed_open: bool = False


class CodexSpendCapError(Exception):
    """Raised by the (future) gate when a Codex spend ceiling is reached."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Codex spend cap reached: {reason}")


# ---------------------------------------------------------------------------
# Pure window evaluation (no file I/O — unit-tested independently)
# ---------------------------------------------------------------------------


def evaluate(
    call_times: list[float],
    token_events: list[tuple[float, int]],
    now: float,
    limits: Limits,
) -> Decision:
    """Decide whether a new Codex call is allowed given the current ledger.

    Checks are evaluated in priority order and the first breach wins:
    per-hour calls, then per-day calls, then per-day tokens.  Entries
    outside their respective windows are ignored.
    """
    hour_cutoff = now - HOUR_SECONDS
    day_cutoff = now - DAY_SECONDS

    calls_last_hour = sum(1 for t in call_times if t >= hour_cutoff)
    if calls_last_hour >= limits.max_calls_per_hour:
        return Decision(False, "calls_per_hour")

    calls_last_day = sum(1 for t in call_times if t >= day_cutoff)
    if calls_last_day >= limits.max_calls_per_day:
        return Decision(False, "calls_per_day")

    tokens_last_day = sum(tok for (t, tok) in token_events if t >= day_cutoff)
    if tokens_last_day >= limits.max_tokens_per_day:
        return Decision(False, "tokens_per_day")

    return Decision(True, None)


# ---------------------------------------------------------------------------
# Limits resolution (clamp config to the hard ceilings)
# ---------------------------------------------------------------------------


def _clamp(configured: object, ceiling: int) -> int:
    """Return ``min(ceiling, configured)`` for a positive int, else ``ceiling``.

    Bad config (non-int, bool, <= 0) is ignored so it can never raise the cap
    or disable enforcement.  ``bool`` is explicitly rejected even though it is
    an ``int`` subclass.
    """
    if isinstance(configured, bool):
        return ceiling
    if isinstance(configured, int) and configured > 0:
        return min(ceiling, configured)
    return ceiling


def resolve_limits(config: Optional[dict]) -> Limits:
    """Resolve effective :class:`Limits` from optional config, clamped to ceilings.

    Reads the optional ``codex_spend_cap`` section.  Each value is clamped to
    its hard ceiling; absent/None/invalid values fall back to the ceiling.
    Enforcement is always on — there is no disable.
    """
    section = {}
    if isinstance(config, dict):
        raw = config.get(_CONFIG_SECTION)
        if isinstance(raw, dict):
            section = raw

    return Limits(
        max_calls_per_hour=_clamp(section.get("max_calls_per_hour"), MAX_CALLS_PER_HOUR),
        max_calls_per_day=_clamp(section.get("max_calls_per_day"), MAX_CALLS_PER_DAY),
        max_tokens_per_day=_clamp(section.get("max_tokens_per_day"), MAX_TOKENS_PER_DAY),
    )


# ---------------------------------------------------------------------------
# The file-backed, cross-process guard
# ---------------------------------------------------------------------------


_LEDGER_PATH_ENV = "HERMES_CODEX_SPEND_LEDGER"


def _default_ledger_path() -> Path:
    """Resolve the default ledger path.

    Honors the ``HERMES_CODEX_SPEND_LEDGER`` env var when set (used for
    per-test isolation so the shared ``~/.hermes/codex_spend.json`` is never
    polluted). This is a PATH override ONLY — it never disables enforcement.
    """
    override = os.environ.get(_LEDGER_PATH_ENV)
    if override:
        return Path(override)
    return Path(os.path.expanduser("~")) / ".hermes" / _DEFAULT_LEDGER_NAME


class CodexSpendGuard:
    """File-backed, cross-process sliding-window Codex spend breaker.

    All persistent operations serialise on an advisory ``fcntl`` lock held on
    a sibling ``.lock`` file so the three watcher processes cannot clobber one
    another's read-modify-write.  Every public method fails OPEN: on any
    internal error it logs a WARNING and degrades to "allow" / best-effort,
    never raising.
    """

    def __init__(
        self,
        ledger_path: Union[str, os.PathLike, None] = None,
        limits: Optional[Limits] = None,
    ):
        self.ledger_path = (
            Path(ledger_path) if ledger_path is not None else _default_ledger_path()
        )
        self.limits = limits if limits is not None else resolve_limits(None)

    # -- public API ---------------------------------------------------------

    def reserve(self, now: Optional[float] = None) -> Reservation:
        """Atomically check the window and, if allowed, record a call.

        On denial the pruned ledger is still persisted (so pruning sticks) but
        no call is appended.  On ANY internal error this fails OPEN, returning
        ``Reservation(allowed=True, failed_open=True)`` — the guard must never
        raise out of ``reserve``.
        """
        now = time.time() if now is None else now
        try:
            with self._locked():
                ledger = self._load_and_prune(now)
                decision = evaluate(
                    ledger["call_times"], ledger["token_events"], now, self.limits
                )
                if not decision.allowed:
                    # Persist the pruned ledger but do not append the call.
                    self._persist(ledger)
                    return Reservation(allowed=False, reason=decision.reason)

                ledger["call_times"].append(float(now))
                self._persist(ledger)
                return Reservation(allowed=True)
        except Exception as exc:  # noqa: BLE001 — fail-open is the contract.
            logger.warning("CodexSpendGuard.reserve failed open: %s", exc)
            return Reservation(allowed=True, failed_open=True)

    def record_tokens(self, total_tokens: int, now: Optional[float] = None) -> None:
        """Record a token-spend event (best-effort; never raises).

        Non-positive or non-int token counts are ignored.
        """
        if isinstance(total_tokens, bool) or not isinstance(total_tokens, int):
            return
        if total_tokens <= 0:
            return
        now = time.time() if now is None else now
        try:
            with self._locked():
                ledger = self._load_and_prune(now)
                ledger["token_events"].append([float(now), int(total_tokens)])
                self._persist(ledger)
        except Exception as exc:  # noqa: BLE001 — best-effort, fail-open.
            logger.warning("CodexSpendGuard.record_tokens failed open: %s", exc)

    def snapshot(self, now: Optional[float] = None) -> dict:
        """Return current window counts for ``/usage`` display and tests.

        Best-effort: on any error it fails open to zero counts.
        """
        now = time.time() if now is None else now
        limits_view = {
            "max_calls_per_hour": self.limits.max_calls_per_hour,
            "max_calls_per_day": self.limits.max_calls_per_day,
            "max_tokens_per_day": self.limits.max_tokens_per_day,
        }
        try:
            with self._locked():
                # Read-only: snapshot intentionally does NOT persist the pruned
                # ledger, so the /usage display path can never mutate the file.
                ledger = self._load_and_prune(now)
            hour_cutoff = now - HOUR_SECONDS
            day_cutoff = now - DAY_SECONDS
            calls_last_hour = sum(1 for t in ledger["call_times"] if t >= hour_cutoff)
            calls_last_day = sum(1 for t in ledger["call_times"] if t >= day_cutoff)
            tokens_last_day = sum(
                tok for (t, tok) in ledger["token_events"] if t >= day_cutoff
            )
            return {
                "calls_last_hour": calls_last_hour,
                "calls_last_day": calls_last_day,
                "tokens_last_day": tokens_last_day,
                "limits": limits_view,
            }
        except Exception as exc:  # noqa: BLE001 — fail-open to zeros.
            logger.warning("CodexSpendGuard.snapshot failed open: %s", exc)
            return {
                "calls_last_hour": 0,
                "calls_last_day": 0,
                "tokens_last_day": 0,
                "limits": limits_view,
            }

    # -- internals ----------------------------------------------------------

    def _locked(self):
        """Context manager holding an exclusive cross-process advisory lock.

        Mirrors ``agent/shell_hooks.py``: an ``fcntl.flock`` on a sibling
        ``.lock`` file for the duration of the read-modify-write.  On non-POSIX
        platforms (no ``fcntl``) it degrades to a no-op lock (best-effort).
        """
        guard = self

        class _Lock:
            def __enter__(self_inner):
                guard.ledger_path.parent.mkdir(parents=True, exist_ok=True)
                if fcntl is None:  # pragma: no cover — non-POSIX fallback
                    self_inner._fh = None
                    return self_inner
                lock_path = guard.ledger_path.with_suffix(
                    guard.ledger_path.suffix + ".lock"
                )
                self_inner._fh = open(lock_path, "a+", encoding="utf-8")
                fcntl.flock(self_inner._fh.fileno(), fcntl.LOCK_EX)
                return self_inner

            def __exit__(self_inner, *exc):
                fh = self_inner._fh
                if fh is not None:
                    try:
                        if fcntl is not None:
                            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
                    finally:
                        fh.close()
                return False

        return _Lock()

    def _load_and_prune(self, now: float) -> dict:
        """Load the ledger, drop entries outside the day window, return it.

        A missing file is treated as empty.  A corrupt/garbage file is treated
        as empty and logged (part of fail-open).
        """
        day_cutoff = now - DAY_SECONDS
        raw = self._load_raw()

        call_times = [
            float(t)
            for t in raw.get("call_times", [])
            if isinstance(t, (int, float)) and float(t) >= day_cutoff
        ]
        token_events: list[list] = []
        for ev in raw.get("token_events", []):
            if (
                isinstance(ev, (list, tuple))
                and len(ev) == 2
                and isinstance(ev[0], (int, float))
                and isinstance(ev[1], int)
                and float(ev[0]) >= day_cutoff
            ):
                token_events.append([float(ev[0]), int(ev[1])])

        return {"call_times": call_times, "token_events": token_events}

    def _load_raw(self) -> dict:
        """Read the raw ledger dict; missing → empty, corrupt → empty + log."""
        try:
            with open(self.ledger_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            logger.warning(
                "Codex spend ledger %s is not a JSON object; treating as empty.",
                self.ledger_path,
            )
            return {}
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
            logger.warning(
                "Codex spend ledger %s is corrupt (%s); treating as empty.",
                self.ledger_path,
                exc,
            )
            return {}

    def _persist(self, ledger: dict) -> None:
        """Atomically write the ledger via temp file + ``os.replace``."""
        target_dir = self.ledger_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(target_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(ledger, f)
            atomic_replace(tmp_path, str(self.ledger_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


# ---------------------------------------------------------------------------
# Process-wide singleton accessor
# ---------------------------------------------------------------------------

_singleton: Optional[CodexSpendGuard] = None


def get_codex_spend_guard() -> CodexSpendGuard:
    """Return the process-wide :class:`CodexSpendGuard`, building it lazily.

    Limits are derived from Hermes config (``codex_spend_cap`` section), which
    can only LOWER the hard ceilings. Config/import failures fall back to the
    hard ceilings (``resolve_limits(None)``) — construction must never raise.
    The ledger path honors ``HERMES_CODEX_SPEND_LEDGER`` via
    :func:`_default_ledger_path`.
    """
    global _singleton
    if _singleton is None:
        try:
            from hermes_cli.config import load_config

            # resolve_limits reads the ``codex_spend_cap`` section itself, so
            # pass the full config dict (not the pre-extracted section).
            limits = resolve_limits(load_config())
        except Exception as exc:  # noqa: BLE001 — never raise at construction.
            logger.warning(
                "CodexSpendGuard config load failed (%s); using hard ceilings.", exc
            )
            limits = resolve_limits(None)
        _singleton = CodexSpendGuard(limits=limits)
    return _singleton


def reset_codex_spend_guard_for_test() -> None:
    """Clear the process-wide singleton (tests only)."""
    global _singleton
    _singleton = None


__all__ = [
    "MAX_CALLS_PER_HOUR",
    "MAX_CALLS_PER_DAY",
    "MAX_TOKENS_PER_DAY",
    "HOUR_SECONDS",
    "DAY_SECONDS",
    "Limits",
    "Decision",
    "Reservation",
    "CodexSpendCapError",
    "CodexSpendGuard",
    "evaluate",
    "resolve_limits",
    "get_codex_spend_guard",
    "reset_codex_spend_guard_for_test",
]
