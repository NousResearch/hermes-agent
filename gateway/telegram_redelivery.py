"""Telegram boot-redelivery guard — durable high-water-mark (HWM) state.

Scope B (hard-kill duplicate-answer guard). See
``~/.hermes/plans/2026-07-01_telegram-redelivery-guard-SPEC.md``.

On a hard kill (SIGKILL/OOM/power) after the agent answered a Telegram update
but before PTB confirmed its offset to Telegram's server, Telegram re-delivers
the already-answered update on next boot. Without a guard the agent answers it
twice. This module holds the per-profile HWM — the highest ``update_id`` the
gateway has dispatched — used to SCOPE the boot-time answerability guard to
genuine candidate re-deliveries (INV-3):

  * ``update_id <= HWM``  -> candidate re-delivery, run the answerability check.
  * ``update_id >  HWM``  -> genuinely new, normal path, NO transcript query.

The HWM is tracked in memory on every dispatch (a cheap int compare) and
flushed to disk COALESCED — on graceful shutdown and via a throttled periodic
checkpoint (default <=30s) — never once-per-message (SPEC INV-4 / D-2, pass-1
B-3). A hard kill loses at most the last <=30s of HWM advance, which only
*widens* the candidate window (more fail-open answerability checks), never
narrows it — so it can never cause a missed dup or a message loss.

Read/parse fails OPEN (SPEC D-4/RC-3): a corrupt/partial/absent file is treated
as "no HWM" so the guard falls back to the 120s post-boot time cap, never a
crash and never a suppress.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("gateway.telegram_redelivery")

# Coalesced-flush throttle. The HWM advances in memory on every dispatch; it is
# only *written* to disk this often (plus on graceful shutdown). SPEC D-2.
DEFAULT_CHECKPOINT_INTERVAL_SECS = 30.0


def _hwm_path(hermes_home: Path, profile: str) -> Path:
    return hermes_home / "state" / f"telegram-last-dispatched-update-id.{profile}.json"


class TelegramHwmTracker:
    """Per-profile in-memory HWM with a coalesced, atomic disk flush.

    One instance per gateway process (the process serves one profile). Thread-
    unsafe by design: mutated only from the single asyncio event loop that
    dispatches Telegram updates.
    """

    def __init__(
        self,
        hermes_home: Path,
        profile: str,
        *,
        checkpoint_interval_secs: float = DEFAULT_CHECKPOINT_INTERVAL_SECS,
        clock=time.monotonic,
    ) -> None:
        self._path = _hwm_path(Path(hermes_home), profile)
        self._interval = max(0.0, float(checkpoint_interval_secs))
        self._clock = clock
        self._hwm: int = 0
        self._dirty = False
        self._last_flush_at: float = clock()

    # ── in-memory advance (hot path — no I/O) ─────────────────────────────

    def observe_dispatch(self, update_id: Optional[int]) -> None:
        """Record that ``update_id`` was dispatched. In-memory only; a later
        ``maybe_checkpoint`` / ``flush`` writes it. No-op for a None/lower id."""
        if update_id is None:
            return
        try:
            uid = int(update_id)
        except (TypeError, ValueError):
            return
        if uid > self._hwm:
            self._hwm = uid
            self._dirty = True

    @property
    def value(self) -> int:
        return self._hwm

    # ── coalesced flush (called from the dispatch loop + on shutdown) ─────

    def maybe_checkpoint(self, *, force: bool = False) -> bool:
        """Flush to disk iff dirty AND (forced OR the throttle window elapsed).
        Returns True if a write happened. Best-effort: a write failure logs and
        returns False, never raises (SPEC D-4)."""
        if not self._dirty:
            return False
        now = self._clock()
        if not force and (now - self._last_flush_at) < self._interval:
            return False
        return self._write(now)

    def flush(self) -> bool:
        """Unconditional flush (graceful shutdown). SPEC D-2."""
        if not self._dirty:
            return False
        return self._write(self._clock())

    def _write(self, now: float) -> bool:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + f".tmp-{os.getpid()}")
            tmp.write_text(f'{{"update_id": {self._hwm}}}', encoding="utf-8")
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, self._path)  # atomic
            self._dirty = False
            self._last_flush_at = now
            return True
        except Exception:
            logger.debug("telegram HWM flush failed (non-fatal)", exc_info=True)
            return False


def read_hwm(hermes_home: Path, profile: str) -> Optional[int]:
    """Boot-time HWM read. Returns the persisted ``update_id``, or None when the
    file is absent/corrupt/unparseable — fail-OPEN (SPEC D-4/RC-3): a None HWM
    means the caller falls back to the 120s time-cap, never crashes, never
    suppresses. Deliberately hand-parses one int so a partial write can't raise
    a surprising type."""
    path = _hwm_path(Path(hermes_home), profile)
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    try:
        import json

        data = json.loads(raw)
        uid = data.get("update_id")
        if isinstance(uid, bool):  # bool is an int subclass — reject explicitly
            return None
        if isinstance(uid, int):
            return uid
        return None
    except Exception:
        return None


# ── the boot-redelivery guard decision (SPEC §2 decision table) ───────────
#
# TWO id spaces, NEVER interchanged (SPEC §0):
#   * update_id  (event.platform_update_id) — envelope sequence, used ONLY to
#     SCOPE (is this a candidate re-delivery? update_id <= HWM).
#   * message_id (event.message_id)        — per-chat counter, stamped as
#     platform_message_id, used ONLY for the ANSWERABILITY lookup.

# Default post-boot time cap (seconds) used ONLY when there is no durable HWM
# (SPEC INV-3 fallback). SPEC D-2.
DEFAULT_NO_HWM_WINDOW_SECS = 120.0


def in_redelivery_scope(
    update_id: Optional[int],
    hwm: Optional[int],
    *,
    seconds_since_boot: float,
    no_hwm_window_secs: float = DEFAULT_NO_HWM_WINDOW_SECS,
) -> bool:
    """SPEC INV-3 (HWM-PRIMARY, time-cap FALLBACK — never an OR).

    * HWM present -> authoritative: candidate iff ``update_id <= hwm``,
      regardless of how long after boot.
    * HWM absent (None) -> the 120s post-boot time cap is the ONLY gate:
      candidate iff ``seconds_since_boot <= no_hwm_window_secs``.
    """
    if update_id is None:
        return False
    if hwm is not None:
        try:
            return int(update_id) <= int(hwm)
        except (TypeError, ValueError):
            return False
    # No HWM: brief post-boot window only.
    return seconds_since_boot <= no_hwm_window_secs


def decide_redelivery(
    *,
    in_scope: bool,
    session_id: Optional[str],
    message_id: Optional[str],
    is_edited: bool,
    answerable_fn,
) -> bool:
    """Return True == SUPPRESS this (re-delivered) message, False == PROCESS it.

    SPEC §2 decision table. The ONLY suppress case is a positive
    ``(answered=True, present=True)`` on a message that is (a) in the
    restart-recovery scope, (b) not an edited_message, (c) has a resolvable
    session. EVERY other outcome PROCESSES — the non-negotiable fail-open
    direction (INV-1): unanswerable authority, absent message, no session, out
    of scope, or an edit all fall through to PROCESS, because suppressing on
    uncertainty would convert a rare duplicate into a message LOSS.

    ``answerable_fn(session_id, message_id) -> (answered: bool, present: bool)``
    mirrors ``SessionStore.has_platform_message_id_answerable``.
    """
    if not in_scope:
        return False  # genuinely new (update_id > HWM / outside window) — no query
    if is_edited:
        return False  # an edit is a new event, same message_id — never a "dup" (SPEC D-6)
    if session_id is None or message_id is None:
        return False  # vacuously new / nothing to look up — fail open
    try:
        answered, present = answerable_fn(session_id, message_id)
    except Exception:
        logger.debug("redelivery answerability lookup raised — failing open", exc_info=True)
        return False  # authority raised — fail toward PROCESS (INV-1/INV-5)
    if answered and present:
        return True  # positively already answered — the one suppress case
    return False  # absent, or unanswerable (answered=False) — fail open


class RedeliverySuppressionCounter:
    """Per-boot-window suppression counter + observability (SPEC INV-5 / RC-4).

    A rising suppression rate is the named worst case (a false-positive
    suppression spike silently eating real messages), so it must be countable in
    prod. This holds the running count for the life of the process; the value is
    read by the anomaly-rate check and surfaced in the periodic summary.
    """

    def __init__(self) -> None:
        self._count = 0

    def record(self, *, update_id, message_id) -> int:
        self._count += 1
        # Content-free structured line (no message text) — SPEC INV-5.
        logger.warning(
            "PHASE=tg_redelivery_suppressed update_id=%s message_id=%s count=%s",
            update_id, message_id, self._count,
        )
        return self._count

    @property
    def count(self) -> int:
        return self._count


