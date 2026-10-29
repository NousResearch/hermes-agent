"""Per-session, per-error-class deduplication for agent-error notices.

On 2026-10-28 a profile's ``config.yaml`` became unparseable (one duplicate YAML key).
Every turn that touched the terminal raised ``TerminalPolicyUnavailable``, and the gateway
sent the user **85 identical 282-character DMs** in a row — a notification hammer.

The failure was DETERMINISTIC: the same cause, the same error, every time. A retry
changes nothing, so telling the user about it once per event is pure noise. What the user
needs is one notice per distinct cause, plus a way to see that it is still happening.

This module owns that policy:

* an error class is the stable ``layer:code`` pair from ``agent.error_surface``
  (``gateway:TerminalPolicyUnavailable``), so a config parse failure and a rate limit are
  different classes and each gets its own notice;
* a **deterministic** class (one that will not fix itself) is notified at most once per
  cooldown window, per session;
* a **transient** class (rate limit, timeout, upstream error) is still notified every time
  — the storm this guards against is a crash LOOP, and suppressing a flaky provider's
  errors would hide real information.

The ledger is process-local on purpose: it lives on the gateway runner, so a crash loop
inside one gateway process is bounded. A gateway restart resets the ledger, which is the
right trade — a fresh process is a fresh attempt, and the user should hear about a new
failure.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

#: Default cooldown for a deterministic error class, per session. Five minutes is long
#: enough that a crash loop cannot produce a stream of DMs, and short enough that a user
#: who fixes the problem and hits it again shortly after still hears about it.
DEFAULT_COOLDOWN_SECONDS = 300.0

#: The failure classes that will not fix themselves. Derived from the error surface's
#: taxonomy rather than hard-coded per exception type: the surface already distinguishes
#: "the provider said no" (auth, billing, policy) from "the provider hiccuped" (rate
#: limit, timeout, outage).
_DETERMINISTIC_LAYERS = frozenset({"auth", "billing"})

#: Codes that are deterministic regardless of their layer. A local runtime failure with
#: one of these codes (a missing skill, an unreadable config) repeats identically until
#: somebody changes the thing that is broken.
_DETERMINISTIC_CODES = frozenset({
    "TerminalPolicyUnavailable",
    "ConfigParseError",
    "MissingSkillError",
    "ContextOverflow",
    "ContentPolicyBlocked",
    "ModelNotFound",
})


def error_class_key(exc: BaseException, *, provider: str = "", model: str = "") -> str:
    """A stable class key for *exc*: ``layer:code``, or ``exc:TypeName`` as a fallback.

    The key must be stable across repeats of the SAME failure and different for DIFFERENT
    failures — that is exactly what the error surface's layer/code pair gives.
    """
    try:
        from agent.error_surface import build_error_surface_from_exception

        surface = build_error_surface_from_exception(exc, provider=provider, model=model)
    except Exception:  # pragma: no cover — classification must never break the error path
        logger.debug("error notice classification failed", exc_info=True)
        surface = None
    if surface:
        layer = str(surface.get("layer") or "")
        code = str(surface.get("code") or "")
        if code:
            return f"{layer}:{code}" if layer else code
    return f"exc:{type(exc).__name__}"


def is_deterministic(exc: BaseException, *, provider: str = "", model: str = "") -> bool:
    """Whether *exc* is a deterministic failure — one a retry cannot fix.

    Conservative by design: an unclassifiable error is treated as TRANSIENT, so an
    unexpected new failure is still reported. Suppressing a message the user needed is
    worse than one duplicate message.
    """
    try:
        from agent.error_surface import build_error_surface_from_exception

        surface = build_error_surface_from_exception(exc, provider=provider, model=model)
    except Exception:  # pragma: no cover
        return False
    if not surface:
        return False
    layer = str(surface.get("layer") or "")
    code = str(surface.get("code") or "")
    if layer in _DETERMINISTIC_LAYERS:
        return True
    if code in _DETERMINISTIC_CODES:
        return True
    # The surface's own verdict: a non-retryable provider verdict is deterministic.
    if surface.get("retryable") is False and layer in {"provider", "endpoint"}:
        return True
    return False


class AgentErrorNoticeLedger:
    """Tracks which (session, error class) notices were recently sent.

    One ledger per gateway runner. The stamp is taken when a notice is SENT, so a
    suppressed turn does not extend the window — the user gets a fresh notice once the
    cooldown expires, regardless of how many turns failed in between.
    """

    def __init__(self, cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS) -> None:
        self._cooldown = float(cooldown_seconds)
        self._stamps: dict[tuple[str, str], float] = {}

    @property
    def cooldown_seconds(self) -> float:
        return self._cooldown

    def should_notify(
        self, session_key: str, error_class: str, *, deterministic: bool,
        now: Optional[float] = None,
    ) -> bool:
        """True when a notice for (session, error class) may be sent now.

        Transient errors always notify. A deterministic error notifies only when the
        cooldown window for its (session, class) has elapsed.
        """
        if not deterministic:
            return True
        stamp = self._stamps.get((session_key, error_class))
        if stamp is None:
            return True
        current = time.monotonic() if now is None else now
        return (current - stamp) >= self._cooldown

    def record(self, session_key: str, error_class: str, *, now: Optional[float] = None) -> None:
        """Stamp (session, error class) as notified."""
        self._stamps[(session_key, error_class)] = time.monotonic() if now is None else now

    def notify(
        self, session_key: str, exc: BaseException, *, provider: str = "", model: str = "",
        now: Optional[float] = None,
    ) -> tuple[bool, str]:
        """Decide whether to notify about *exc* and record the decision.

        Returns ``(send, error_class)``. When ``send`` is False the caller must NOT send
        a user-facing notice for this failure.
        """
        error_class = error_class_key(exc, provider=provider, model=model)
        deterministic = is_deterministic(exc, provider=provider, model=model)
        if not self.should_notify(session_key, error_class, deterministic=deterministic, now=now):
            logger.info(
                "Suppressing duplicate agent-error notice for session %s (class %s) — "
                "already notified within the %.0fs cooldown",
                session_key, error_class, self._cooldown,
            )
            return False, error_class
        self.record(session_key, error_class, now=now)
        return True, error_class

    def reset(self, session_key: Optional[str] = None) -> None:
        """Forget the ledger for one session (or all of them)."""
        if session_key is None:
            self._stamps.clear()
            return
        for key in [k for k in self._stamps if k[0] == session_key]:
            self._stamps.pop(key, None)
