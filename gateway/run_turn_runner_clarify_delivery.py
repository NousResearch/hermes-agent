"""Clarify prompt delivery on messaging platforms: send disposition, plain-text fallback, bounded wait.

A native clarify card (Telegram inline keyboard, Slack blocks, …) can fail to render in three ways
the user never sees (#112684): the platform rejects it at once, the send outruns the 15 s
disposition window and only then fails (pool / connect timeouts, a stale-thread retry), or there
is no chat surface at all. Each used to end as a full ``clarify_timeout`` of silence reported as
"the user did not respond". Here every definitive failure of the native card falls back to the
adapter's plain-text ``send_clarify`` (numbered list + text capture), and a card that fails AFTER
the window releases the wait with the delivery notice instead of the inactivity one.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

UNDELIVERED = "[clarify prompt could not be delivered]"
UNDELIVERED_DECLINED = "[clarify prompt could not be delivered: destination refused]"
# No status adapter (a run whose chat surface is gone): shares the ``[clarify prompt could not be
# delivered`` prefix every consumer already treats as a non-answer
# (agent/context_compressor.py::_CLARIFY_NON_RESPONSE_PREFIXES).
UNDELIVERED_NO_SURFACE = "[clarify prompt could not be delivered: no chat surface]"

# Seconds a scheduled card send may take before it is classified ``ambiguous`` (possibly posted).
SEND_ACK_WINDOW = 15


def text_fallback_coro(adapter, **send_kwargs):
    """The base numbered-text ``send_clarify`` as a fresh coroutine, or ``None`` when the adapter has
    no native override — its ``send_clarify`` already IS the text path, so retrying it would only
    repeat the failure."""
    from gateway.platforms.base import BasePlatformAdapter

    if not isinstance(adapter, BasePlatformAdapter) \
            or type(adapter).send_clarify is BasePlatformAdapter.send_clarify:
        return None
    return BasePlatformAdapter.send_clarify(adapter, **send_kwargs)


def _abort_for_outcome(outcome: str, *, session_key: str, clarify_mod) -> Optional[str]:
    """Map a send outcome to the abort sentinel (registration torn down) or ``None`` (proceed to wait).

    Only a DEFINITIVE failure tears down the registration; ``ambiguous`` (card may have posted) stays armed
    and proceeds to the bounded wait, whose response timeout covers a lost card."""
    if outcome == "declined":
        # P5(b): a connector DECLINE is MORE definitive than a failure — the
        # destination was authorized and refused, so the card cannot arrive and
        # no late reply can resolve it. Without this branch `declined` fell
        # through to the bounded wait and the agent blocked until
        # clarify_timeout (indefinitely when that is configured non-positive).
        logger.warning(
            "Clarify prompt DECLINED by the connector's egress guard; "
            "clearing registration"
        )
        clarify_mod.clear_session(session_key)
        return UNDELIVERED_DECLINED
    if outcome == "failed":
        # Undeliverable: clear the registration and return the sentinel so the agent falls back, not hangs.
        logger.warning("Clarify send failed definitively; clearing registration")
        clarify_mod.clear_session(session_key)
        return UNDELIVERED
    if outcome == "ambiguous":
        logger.warning(
            "Clarify prompt send timed out — treating as possibly-delivered "
            "(no teardown; the registration stays armed for a late reply)")
    return None


def _clarify_send_disposition(fut, *, session_key: str, clarify_mod) -> Optional[str]:
    """Decide whether a clarify prompt send aborts the wait; returns the abort sentinel or ``None``."""
    from gateway.run import _approval_send_outcome

    return _abort_for_outcome(
        _approval_send_outcome(fut, timeout=SEND_ACK_WINDOW), session_key=session_key, clarify_mod=clarify_mod)


def _clarify_send_then_wait(fut, *, clarify_id: str, session_key: str, clarify_mod,
                            fallback: Optional[Callable[[], Any]] = None) -> tuple[str, bool]:
    """Resolve a clarify prompt: send disposition, plain-text fallback, then the bounded wait.

    ``fallback()`` schedules the plain-text ``send_clarify`` and returns its future (or ``None``);
    it runs when the native card failed definitively — at once, or late, after the ack window —
    and never on a connector DECLINE (re-sending refused content as text is the exfiltration the
    egress guard exists to stop). A card that fails after the window releases the wait with the
    delivery notice, not the inactivity one.

    Returns ``(response, answered)``. ``answered`` is the only signal that a user reply arrived;
    callers must not infer it from the text (a real answer may start with '[' like a sentinel)."""
    from gateway.run import _approval_send_outcome

    outcome = _approval_send_outcome(fut, timeout=SEND_ACK_WINDOW)
    if outcome == "failed" and fallback is not None:
        # The text prompt is the last resort: a late failure of ITS send has nothing to retry.
        fut, fallback = fallback(), None
        # ``None`` = the fallback could not even be scheduled; the card failure already stands,
        # so re-classifying would only log a misleading "no scheduling future".
        if fut is not None:
            outcome = _approval_send_outcome(fut, timeout=SEND_ACK_WINDOW)
        if outcome == "sent":
            logger.info("Clarify card undeliverable; plain-text prompt sent instead (id=%s)", clarify_id)
    abort = _abort_for_outcome(outcome, session_key=session_key, clarify_mod=clarify_mod)
    if abort is not None:
        return abort, False
    late = _LateFailureWatch(fut, clarify_id=clarify_id, session_key=session_key,
                             clarify_mod=clarify_mod, fallback=fallback)
    timeout = clarify_mod.get_clarify_timeout()
    response = clarify_mod.wait_for_response(clarify_id, timeout=float(timeout))
    late.disarm()
    if late.undeliverable:
        return late.undeliverable, False
    if response is None or response == "":
        return f"[user did not respond within {int(timeout / 60)}m]", False
    return response, True


class _LateFailureWatch:
    """Watch a possibly-delivered card send: when it resolves as a definitive failure after the
    ack window, try the text fallback once, then release the waiter with the delivery notice.

    Callbacks run on the gateway loop thread (the send future completes there); ``clear_session``
    wakes the agent thread blocked in ``wait_for_response`` and ``undeliverable`` (the delivery
    sentinel, or ``None`` while nothing definitive happened) tells it why.
    Armed only while the future is still pending: a sent card needs no watch."""

    def __init__(self, fut, *, clarify_id: str, session_key: str, clarify_mod, fallback) -> None:
        self.undeliverable: Optional[str] = None
        self._armed = False
        self._clarify_id = clarify_id
        self._session_key = session_key
        self._clarify_mod = clarify_mod
        self._fallback = fallback
        # Test doubles hand the runner a bare ``.result()`` object; only a real pending
        # concurrent future can still resolve late.
        done = getattr(fut, "done", None)
        if callable(done) and not done():
            self._armed = True
            fut.add_done_callback(self._on_card_done)

    def disarm(self) -> None:
        self._armed = False

    @staticmethod
    def _outcome(fut) -> str:
        from gateway.run import _approval_send_outcome
        return _approval_send_outcome(fut, timeout=0)

    def _on_card_done(self, fut) -> None:
        if not self._armed:
            return
        outcome = self._outcome(fut)
        if outcome == "sent":
            return
        logger.warning("Clarify card send resolved %s after the ack window (id=%s)", outcome, self._clarify_id)
        if outcome == "ambiguous":
            # Lost ack (``raw_response.ambiguous``): the card may well have posted. Same invariant as
            # ``_abort_for_outcome`` — stay armed for the late button tap, never re-send, never
            # release; the bounded wait's own timeout covers a card that truly never arrived.
            return
        if outcome == "declined":
            # Refused destination: no text retry (see ``_clarify_send_then_wait``), and the notice
            # says so rather than the generic delivery failure.
            self._release(UNDELIVERED_DECLINED)
            return
        fallback_fut = self._fallback() if self._fallback is not None else None
        if fallback_fut is None:
            self._release()
            return
        fallback_fut.add_done_callback(self._on_fallback_done)

    def _on_fallback_done(self, fut) -> None:
        if not self._armed:
            return
        if self._outcome(fut) == "sent":
            logger.info("Clarify card undeliverable; plain-text prompt sent instead (id=%s)", self._clarify_id)
            return
        self._release()

    def _release(self, notice: str = UNDELIVERED) -> None:
        self.undeliverable = notice
        self._clarify_mod.clear_session(self._session_key)
