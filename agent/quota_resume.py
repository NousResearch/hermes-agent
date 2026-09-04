"""When a subscription quota wall can be waited out — and until exactly when.

A subscription provider (Claude Pro/Max, ChatGPT Plus/Pro via Codex) answers an
exhausted window with a *deadline*, not a refusal: the quota reopens at a stated
time. Today that deadline is parsed for the credential pool's cooldown and then
dropped, so the turn dies with a Retry button the user has to babysit.

This module answers one question — "is this failure a quota wall with a
trustworthy reset time, and when is it?" — and answers it the same way for every
surface (Desktop/TUI gateway, messaging gateway). It decides nothing about *what*
to do at that time; scheduling and dispatch belong to the caller that owns the
session.

Deadlines are only ever *reported*, never guessed:

1. ``provider_error`` — the failing response carried a reset timestamp.
2. ``credential_pool`` — a pooled credential re-enters rotation at a known time.
3. ``usage_api``       — the provider publishes live window utilization
   (``agent.account_usage``), consulted only when the error itself was silent.

No deadline from those three means no plan, and the caller leaves the user in
control. That is deliberate: a wrong auto-resume time is worse than none, because
it either wastes hours of a working session or hammers a wall that is still up.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Only a periodic quota that the provider promised to reopen. `billing` is a
# spend wall that no amount of waiting clears, `overloaded` is capacity (retry in
# seconds, not hours), and auth/context failures repeat identically. Waiting on
# any of those burns the session for nothing.
RESUMABLE_FAILURE_REASONS = frozenset({"rate_limit"})

SOURCE_PROVIDER_ERROR = "provider_error"
SOURCE_CREDENTIAL_POOL = "credential_pool"
SOURCE_USAGE_API = "usage_api"

# Providers publishing a live usage/limits endpoint that names its own reset
# time. Anything absent here still auto-resumes when its *error* carries a
# deadline — this gates only the extra lookup, never eligibility.
USAGE_API_PROVIDERS = frozenset({"anthropic", "openai-codex"})

# Wait past this and resuming is a surprise, not a convenience: a weekly window
# can be days out, by which time the task's premises are stale and the user has
# long since moved on. Beyond the cap we still report the deadline so the UI can
# show "resets Tuesday" — we just refuse to sit on it.
DEFAULT_MAX_WAIT_SECONDS = 26 * 3600

# Provider clocks and ours disagree by a second or two, and a window that reopens
# "at" T often serves the first request a moment later. Resuming a hair early
# spends the retry on the same wall.
DEFAULT_GRACE_SECONDS = 45.0

# Treat a window as blocking only when it is effectively spent. Reset times for
# windows with room left describe a future rollover, not the wall we just hit —
# scheduling against those would park the session on the wrong clock entirely.
_USAGE_API_EXHAUSTED_PERCENT = 99.0


@dataclass(frozen=True)
class QuotaResumePlan:
    """Whether this failure can be waited out, and until when.

    ``eligible`` is the only field a caller must branch on. ``resume_at`` may be
    set while ``eligible`` is False (deadline known but too far out, or already
    past) so a UI can still say *when* the wall lifts without arming a timer.
    """

    eligible: bool = False
    resume_at: Optional[float] = None
    source: str = ""
    provider: str = ""
    reason: str = ""

    @property
    def seconds_until_resume(self) -> Optional[float]:
        return None if self.resume_at is None else max(0.0, self.resume_at - time.time())

    def to_dict(self) -> dict:
        """Wire form for the turn result / gateway payload (JSON-safe)."""
        out: dict = {"eligible": self.eligible}
        if self.resume_at is not None:
            out["resume_at"] = self.resume_at
        if self.source:
            out["source"] = self.source
        if self.provider:
            out["provider"] = self.provider
        if self.reason:
            out["reason"] = self.reason
        return out


def coerce_reset_timestamp(value: Any, *, now: Optional[float] = None) -> Optional[float]:
    """Best-effort epoch seconds from the shapes providers actually send.

    Accepts epoch seconds, epoch milliseconds (Codex sends both across
    endpoints), and ISO-8601 with ``Z`` or an offset. Returns None rather than
    raising: an unparseable deadline must degrade to "no plan", never to an
    exception on a path that is already handling a failure.
    """
    if value is None or isinstance(value, bool) or value == "":
        return None
    now = time.time() if now is None else now
    if isinstance(value, (int, float)):
        stamp = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            stamp = float(text)
        except ValueError:
            iso = text[:-1] + "+00:00" if text.endswith("Z") else text
            try:
                parsed = datetime.fromisoformat(iso)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
    else:
        return None
    # Milliseconds: anything ~100x the current epoch cannot be seconds. Compared
    # against `now` rather than a hardcoded year so this never expires.
    if stamp > now * 10:
        stamp /= 1000.0
    return stamp


def _reset_from_error_context(error_context: Any, *, now: float) -> Optional[float]:
    """Deadline the failing response itself reported."""
    if not isinstance(error_context, dict):
        return None
    stamp = coerce_reset_timestamp(error_context.get("reset_at"), now=now)
    if stamp is not None:
        return stamp
    # Relative form. `extract_api_error_context` normalizes absolute fields but
    # has no seconds-offset branch, and Codex sends `resets_in_seconds` beside
    # `resets_at` — a provider that sends only the offset would otherwise lose
    # its deadline here.
    raw = error_context.get("resets_in_seconds")
    if isinstance(raw, bool) or raw is None or raw == "":
        return None
    try:
        offset = float(raw)
    except (TypeError, ValueError):
        return None
    return now + offset if offset > 0 else None


def _reset_from_credential_pool(pool: Any, *, now: float) -> Optional[float]:
    """When the pool says some credential re-enters rotation.

    ``next_available_at`` returns None while any credential can serve right now,
    so this never parks a session that could have rotated its way out.
    """
    if pool is None:
        return None
    getter = getattr(pool, "next_available_at", None)
    if not callable(getter):
        return None
    try:
        return coerce_reset_timestamp(getter(), now=now)
    except Exception:
        logger.debug("quota-resume: credential pool reset lookup failed", exc_info=True)
        return None


def _reset_from_usage_api(provider: str, *, now: float) -> Optional[float]:
    """Earliest reset among the provider's *exhausted* windows.

    Authoritative where it exists: it reads the account's real utilization
    instead of trusting how the error happened to be labelled. Only windows at
    or above the exhaustion threshold count — a fresh window's reset time is a
    rollover, not a wall.
    """
    try:
        from agent.account_usage import fetch_account_usage

        snapshot = fetch_account_usage(provider)
    except Exception:
        logger.debug("quota-resume: usage API lookup failed for %s", provider, exc_info=True)
        return None
    if snapshot is None or not getattr(snapshot, "available", False):
        return None
    candidates: list[float] = []
    for window in getattr(snapshot, "windows", ()) or ():
        used = getattr(window, "used_percent", None)
        reset_at = getattr(window, "reset_at", None)
        if used is None or reset_at is None:
            continue
        try:
            if float(used) < _USAGE_API_EXHAUSTED_PERCENT:
                continue
        except (TypeError, ValueError):
            continue
        stamp = coerce_reset_timestamp(
            reset_at.timestamp() if isinstance(reset_at, datetime) else reset_at, now=now
        )
        if stamp is not None and stamp > now:
            candidates.append(stamp)
    return min(candidates) if candidates else None


def plan_quota_resume(
    *,
    failure_reason: Any,
    error_context: Any = None,
    provider: Any = "",
    credential_pool: Any = None,
    now: Optional[float] = None,
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
    allow_usage_api: bool = True,
) -> QuotaResumePlan:
    """Decide whether a failed turn can be resumed when its quota window reopens.

    Sources are tried cheapest-first and the first hit wins; the usage API is a
    fallback for a silent error, not a second opinion on one that already told us
    when it reopens.

    Never raises: every failure mode degrades to an ineligible plan, because this
    runs while a turn is already failing.
    """
    reason = str(getattr(failure_reason, "value", failure_reason) or "").strip().lower()
    provider_name = str(provider or "").strip().lower()
    if reason not in RESUMABLE_FAILURE_REASONS:
        return QuotaResumePlan(provider=provider_name, reason=reason)

    now = time.time() if now is None else now
    resume_at: Optional[float] = None
    source = ""
    for candidate_source, resolver in (
        (SOURCE_PROVIDER_ERROR, lambda: _reset_from_error_context(error_context, now=now)),
        (SOURCE_CREDENTIAL_POOL, lambda: _reset_from_credential_pool(credential_pool, now=now)),
    ):
        try:
            resume_at = resolver()
        except Exception:
            logger.debug("quota-resume: %s resolver failed", candidate_source, exc_info=True)
            resume_at = None
        if resume_at is not None:
            source = candidate_source
            break
    if resume_at is None and allow_usage_api and provider_name in USAGE_API_PROVIDERS:
        resume_at = _reset_from_usage_api(provider_name, now=now)
        if resume_at is not None:
            source = SOURCE_USAGE_API

    if resume_at is None:
        return QuotaResumePlan(provider=provider_name, reason=reason)

    # A deadline already past means the wall lifted between the failure and this
    # check (a stale pool entry, or a race with the reset). Report it, but don't
    # claim an "auto-resume" the caller would fire instantly — the user's own
    # Retry is the honest affordance there.
    if resume_at <= now:
        return QuotaResumePlan(resume_at=resume_at, source=source, provider=provider_name, reason=reason)

    scheduled_at = resume_at + max(0.0, grace_seconds)
    if scheduled_at - now > max(0.0, max_wait_seconds):
        # Too far out to sit on, but still worth telling the user about.
        return QuotaResumePlan(resume_at=resume_at, source=source, provider=provider_name, reason=reason)
    return QuotaResumePlan(
        eligible=True, resume_at=scheduled_at, source=source, provider=provider_name, reason=reason
    )


__all__ = [
    "DEFAULT_GRACE_SECONDS",
    "DEFAULT_MAX_WAIT_SECONDS",
    "QuotaResumePlan",
    "RESUMABLE_FAILURE_REASONS",
    "SOURCE_CREDENTIAL_POOL",
    "SOURCE_PROVIDER_ERROR",
    "SOURCE_USAGE_API",
    "USAGE_API_PROVIDERS",
    "coerce_reset_timestamp",
    "plan_quota_resume",
]
