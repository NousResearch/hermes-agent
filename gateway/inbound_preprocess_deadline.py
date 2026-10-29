"""Hard deadline for the gateway's PRE-TURN inbound enrichment pipeline.

Why this module exists (2026-10-23 gateway DM outage, kanban t_d2133284)
---------------------------------------------------------------------
The gateway's inbound pipeline has two stages:

1. **Pre-turn enrichment** (``gateway/run_inbound.py::_prepare_inbound_message_text``):
   image pre-analysis (``vision_analyze``), speech-to-text, ``@`` context-reference
   expansion. All of this runs BEFORE ``AIAgent.run_conversation()`` is ever called.
2. **The agent turn** (``run_conversation``): guarded by the turn-liveness watchdog
   (``agent/turn_liveness.py``), which is armed from ``agent/turn_facade_lease.py``.

Stage 1 is **outside every existing watchdog's scope**. On 2026-10-23 the gateway
ingested a Telegram DM at 22:18:16 and then went silent for 27 minutes: the last
log line was

    Image routing: text (mode=text). Pre-analyzing 1 image(s) via vision_analyze.

and the *next* log line was 27 minutes later, after systemd had escalated a
SIGTERM it could not deliver into a SIGKILL. The vision call never returned
(``tools/vision_tools.py::_prepare_image`` -> ``resolve_image_source``), the
turn never started, so no reply was ever produced for that chat.

The loop-liveness watchdog (``gateway/shutdown_watchdog.py``) correctly did NOT
fire: it probes with ``loop.call_soon_threadsafe``, which succeeds whenever the
loop is *running*, even while one task awaits a hung socket. The loop was fine;
one task was stuck. The turn-liveness watchdog did not fire either, because
``run_agent`` had not been reached yet.

So the defect is structural: **stage 1 is unbounded**. A single hung enrichment
await holds the whole inbound pipeline forever, with no supervisor able to
notice.

The fix
-------
Bound stage 1 with a hard deadline. On expiry the enrichment is abandoned and
the turn proceeds with the **raw message text** — degraded, but never lost. The
user still gets a reply; the enrichment is re-derivable from the same media on
a retry, whereas a dropped turn is not.

This is deliberately NOT a "kill the gateway" response. The 2026-10-23
recovery attempt (a worker-initiated ``systemd-run`` gateway cycle) had to wait
out a 7-minute SIGTERM->SIGKILL escalation and then a 37-minute restart, all
because the wedge was unrecoverable in-process. Making the wedge recoverable
in-process is strictly better than making the kill faster.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default wall-clock budget for the whole pre-turn enrichment pipeline.
#
# Sized against the slowest thing it legitimately contains: a vision pre-analysis
# (image download + decode + a vision-LLM round trip, itself capped at 120s by
# ``tools/vision_tools.py::_aux_call_kwargs``) plus STT and context expansion.
# 180s leaves generous headroom over the sum of those budgets while staying far
# below the 7-minute SIGTERM->SIGKILL escalation window that turned a wedge
# into a 37-minute outage on 2026-10-23.
DEFAULT_INBOUND_PREPROCESS_TIMEOUT_S = 180.0

# Floor: a budget below this is a configuration error, not an intent. 30s still
# allows a cached-image vision call to finish while catching a genuine hang.
MIN_INBOUND_PREPROCESS_TIMEOUT_S = 30.0

# Ceiling: keep the budget inside the window where an abandoned turn is still
# cheap to retry. Above ~10 minutes the gateway is effectively "down" for that
# chat, which is the outage class this module exists to prevent.
MAX_INBOUND_PREPROCESS_TIMEOUT_S = 600.0

ENV_TIMEOUT_S = "HERMES_INBOUND_PREPROCESS_TIMEOUT_S"

_FALSEY = frozenset({"0", "false", "no", "off"})


def _coerce_timeout(raw: Any, default: float = DEFAULT_INBOUND_PREPROCESS_TIMEOUT_S) -> float:
    """Coerce a duration knob, rejecting typos, NaN and Inf (never raises)."""
    import math

    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value):
        logger.warning("Invalid %s=%r — falling back to default %.1fs.",
                       ENV_TIMEOUT_S, raw, default)
        return default
    return value


def resolve_inbound_preprocess_timeout(
    config: Optional[Any] = None,
) -> float:
    """Resolve the pre-turn enrichment budget in seconds.

    Precedence: the ``inbound_preprocess_timeout_s`` attribute of ``config`` (the
    gateway config object), then the ``HERMES_INBOUND_PREPROCESS_TIMEOUT_S`` env
    var, then the shipped default. A value of ``0`` (or a negative value) opts
    out — the pipeline runs unbounded, exactly as it did before this module
    existed. That opt-out is a deliberate escape hatch, not an accident.

    Never raises.
    """
    value: Any = None
    if config is not None:
        value = getattr(config, "inbound_preprocess_timeout_s", None)
    if value is None:
        try:
            import os

            raw_env = (os.environ.get(ENV_TIMEOUT_S) or "").strip()
        except Exception:
            raw_env = ""
        if raw_env and raw_env.lower() not in _FALSEY:
            value = _coerce_timeout(raw_env)
    if value is None:
        return DEFAULT_INBOUND_PREPROCESS_TIMEOUT_S

    timeout = _coerce_timeout(value)
    if timeout <= 0:
        return 0.0  # explicit opt-out
    return min(max(timeout, MIN_INBOUND_PREPROCESS_TIMEOUT_S), MAX_INBOUND_PREPROCESS_TIMEOUT_S)


class InboundPreprocessTimeout(Exception):
    """Raised when the pre-turn enrichment pipeline exceeds its budget."""


class _Expired:
    """Sentinel type distinguishing "deadline expired" from "returned None"."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<inbound-preprocess-expired>"


#: Returned when the budget expired. Distinct from ``None``, which is a
#: legitimate result for several enrichment paths (and, for context-reference
#: expansion, means "refused — drop the message").
EXPIRED = _Expired()


async def run_with_inbound_preprocess_deadline(
    awaitable: Awaitable[T],
    *,
    timeout_s: Optional[float] = None,
    what: str = "inbound enrichment",
) -> Any:
    """Await ``awaitable`` under the pre-turn enrichment budget.

    Returns the result, or :data:`EXPIRED` when the budget expired. On expiry
    the awaitable is cancelled and the caller is expected to continue with
    degraded input (raw text) rather than fail the turn.

    A timeout is NOT raised as an exception: the enrichment is an
    *enhancement*, and losing the enhancement must not lose the turn. Callers
    that need to distinguish "expired" from "returned a falsy value" compare
    against :data:`EXPIRED` (never against ``None`` — several enrichment paths
    legitimately return ``None``, and for context-reference expansion ``None``
    means "refused — drop the message").

    Cancellation is honoured: a ``CancelledError`` raised by the caller
    (gateway shutdown) propagates rather than being swallowed as a timeout.
    """
    budget = resolve_inbound_preprocess_timeout() if timeout_s is None else float(timeout_s)
    if budget <= 0:
        # Opted out — run unbounded, exactly as before.
        return await awaitable

    try:
        return await asyncio.wait_for(awaitable, timeout=budget)
    except asyncio.TimeoutError:
        logger.warning(
            "Inbound %s exceeded its %.0fs budget; abandoning it and continuing with the "
            "raw message so the turn is not lost. (2026-10-23 DM outage, kanban t_d2133284)",
            what, budget,
        )
        return EXPIRED
    except asyncio.CancelledError:
        # A caller-initiated cancellation (shutdown) must propagate: do not
        # convert a shutdown into a "timeout".
        raise
