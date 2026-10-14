"""Keep the compression no-progress guard from out-tightening the aux deadline.

Context compression runs an auxiliary summarisation call inside a
progress-aware wrapper (:func:`agent.conversation_compression.
run_compress_context_with_progress_timeout`).  Two independent deadlines
therefore bound the SAME call:

* the **outer** no-progress watchdog — ``compression.context_timeout_seconds``
  (default 120s).  It abandons the future when no progress event has landed
  for that long.
* the **inner** auxiliary deadline — ``auxiliary.compression.timeout``
  (default 300s, floored to 300s by
  ``agent.auxiliary_client._COMPRESSION_TIMEOUT_FLOOR_SECONDS`` for the
  compression task specifically, because summarising a large context
  legitimately takes minutes).  On the progress-hooked streamed path this
  behaves as an INTER-CHUNK idle timeout.

When the outer guard is TIGHTER than the inner one it is not a safety net —
it is a guaranteed kill:

* the outer watchdog fires first and abandons the worker;
* ``call_llm`` never raises, because nothing timed out at the HTTP layer;
* so the ``except`` branch in ``auxiliary_client`` that walks
  ``fallback_providers`` / the configured fallback chain is never reached.

The user-visible result is a compaction that always fails on a slow-but-alive
provider, with the declared fallback policy structurally unreachable.  This is
the same bug shape as the gateway hygiene 30s-vs-300s contradiction: an outer
wall-clock guard tighter than the inner deadline can never be rescued by
tuning the inner one, because the outer guard never consulted it.

This module owns the arithmetic as pure functions so the invariant can be
tested without a provider, a gateway, or a live compression run.
"""

from __future__ import annotations

from typing import Optional

# Headroom added on top of the inner deadline when deriving the outer guard.
# The outer watchdog exists to catch a genuinely DEAD connection, so it should
# trip only after the inner deadline has had its full budget plus a margin for
# the surrounding bookkeeping (client construction, retry backoff, the commit
# fence handshake).
DERIVED_IDLE_HEADROOM_SECONDS = 60.0

# Absolute bound for a DERIVED outer guard.  Without a cap, a very large
# configured aux timeout would produce an outer guard so wide that a truly
# wedged compression would hang a session for an unreasonable time.
DERIVED_IDLE_CAP_SECONDS = 900.0

# Headroom added when deriving the TOTAL ceiling.  The ceiling has to admit the
# idle window (the primary attempt's full no-progress budget) PLUS one complete
# fallback attempt, because ``run_compress_context_with_progress_timeout``
# starts the stall-fallback against the SAME remaining ceiling.  Without this,
# the fallback is admitted into a budget that cannot possibly contain it.
DERIVED_CEILING_HEADROOM_SECONDS = 60.0

# Absolute bound for a DERIVED total ceiling.  Same rationale as
# :data:`DERIVED_IDLE_CAP_SECONDS`: a pathological inner deadline must not be
# able to pin a session to a multi-hour compression.
DERIVED_CEILING_CAP_SECONDS = 1800.0


def _coerce_positive_float(value: object) -> Optional[float]:
    """Return ``value`` as a positive float, or ``None`` when unusable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN
        return None
    if parsed <= 0:
        return None
    return parsed


def reconcile_idle_timeout(
    idle_timeout_seconds: float,
    inner_deadline_seconds: Optional[float],
    *,
    explicit: bool = False,
) -> float:
    """Return an outer no-progress budget that never undercuts the inner one.

    ``idle_timeout_seconds`` is the resolved outer guard;
    ``inner_deadline_seconds`` is the auxiliary call's own deadline.

    Rules:

    * ``idle_timeout_seconds <= 0`` disables the wrapper entirely — an
      explicit opt-out that is always honoured verbatim.
    * ``explicit=True`` (the operator set ``context_timeout_seconds`` by hand)
      is honoured verbatim and unclamped.  Hermetic tests pin tiny values like
      ``0.01`` and must keep working, and an operator who names a number means
      it.
    * Otherwise the guard is raised to ``inner + headroom`` when it would
      otherwise fire before the inner deadline, capped at
      :data:`DERIVED_IDLE_CAP_SECONDS`.

    The function never LOWERS a configured guard; it only lifts a default that
    would pre-empt the deadline it is supposed to be backstopping.
    """
    if idle_timeout_seconds <= 0:
        return idle_timeout_seconds
    if explicit:
        return idle_timeout_seconds

    inner = _coerce_positive_float(inner_deadline_seconds)
    if inner is None:
        return idle_timeout_seconds
    if idle_timeout_seconds > inner:
        return idle_timeout_seconds

    derived = min(inner + DERIVED_IDLE_HEADROOM_SECONDS, DERIVED_IDLE_CAP_SECONDS)
    return max(idle_timeout_seconds, derived)


def reconcile_ceiling(
    total_ceiling_seconds: float,
    idle_timeout_seconds: float,
    inner_deadline_seconds: Optional[float],
    *,
    explicit: bool = False,
) -> float:
    """Return a total ceiling that can contain a primary AND one fallback.

    The progress-aware wrapper treats ``total_ceiling_seconds`` as a HARD stop
    on the whole compression pass.  When the primary attempt stalls, the
    stall-fallback path starts a second attempt against the *same* remaining
    ceiling.  A ceiling sized for only one attempt therefore kills the
    fallback mid-stream — and because the summary work happens in a detached
    executor thread, the worker keeps running and COMPLETES after the host
    stopped waiting.  The commit fence then (correctly) refuses the late
    commit, so a fully-successful summary is discarded and the user is told
    the summariser "stalled".

    The fix is the same shape as :func:`reconcile_idle_timeout`, one axis
    over: a DEFAULT ceiling is lifted so it admits ``idle + inner + headroom``
    — the primary's no-progress budget plus one complete fallback attempt.

    Rules mirror the idle reconciler:

    * ``explicit=True`` (the operator set ``context_total_ceiling_seconds``)
      is honoured verbatim — an operator who names a number means it, and
      hermetic tests pin tiny values.
    * ``total_ceiling_seconds <= 0`` is left alone (the caller's own
      ``max(ceiling, idle)`` invariant still applies downstream).
    * An unknown inner deadline is a no-op.
    * The function never LOWERS a configured ceiling.
    """
    if explicit:
        return total_ceiling_seconds
    if total_ceiling_seconds <= 0:
        return total_ceiling_seconds

    inner = _coerce_positive_float(inner_deadline_seconds)
    if inner is None:
        return total_ceiling_seconds

    idle = max(float(idle_timeout_seconds), 0.0)
    required = idle + inner + DERIVED_CEILING_HEADROOM_SECONDS
    derived = min(required, DERIVED_CEILING_CAP_SECONDS)
    return max(total_ceiling_seconds, derived)


def reconcile_timeouts(
    idle_timeout_seconds: float,
    total_ceiling_seconds: float,
    inner_deadline_seconds: Optional[float],
    *,
    explicit_idle: bool = False,
    explicit_ceiling: bool = False,
) -> tuple[float, float]:
    """Reconcile ``(idle, ceiling)`` against the inner auxiliary deadline.

    Applies :func:`reconcile_idle_timeout`, then :func:`reconcile_ceiling` so
    the total budget can contain a primary attempt AND one stall-fallback,
    then restores the existing invariant that the total ceiling is at least
    one idle window (a ceiling below the idle budget would make the idle
    budget unreachable).
    """
    idle = reconcile_idle_timeout(
        idle_timeout_seconds, inner_deadline_seconds, explicit=explicit_idle
    )
    ceiling = reconcile_ceiling(
        total_ceiling_seconds,
        idle,
        inner_deadline_seconds,
        explicit=explicit_ceiling,
    )
    if idle > 0:
        ceiling = max(ceiling, idle)
    return idle, ceiling
