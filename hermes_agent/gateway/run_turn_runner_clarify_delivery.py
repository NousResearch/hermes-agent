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

import asyncio
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

UNDELIVERED = "[clarify prompt could not be delivered]"
UNDELIVERED_DECLINED = "[clarify prompt could not be delivered: destination refused]"
# No status adapter (a run whose chat surface is gone).
UNDELIVERED_NO_SURFACE = "[clarify prompt could not be delivered: no chat surface]"

# Seconds a scheduled card send may take before it is classified ``ambiguous`` (possibly posted).
# Also the ONE budget the clarify card's explanation gets, shared by the agent thread and the card:
# a send that never returns (a plugin adapter with no HTTP timeout) must not hold the question back.
SEND_ACK_WINDOW = 15
# ``raw_response`` marker of a card that was not posted because its question was withdrawn
# (cancelled, answered elsewhere, session reset) while it waited for its explanation.
WITHDRAWN = "clarify_withdrawn"


def text_fallback_coro(adapter, **send_kwargs):
    """The base numbered-text ``send_clarify`` as a fresh coroutine, or ``None`` when the adapter has
    no native override — its ``send_clarify`` already IS the text path, so retrying it would only
    repeat the failure."""
    from hermes_agent.gateway.platforms.base import BasePlatformAdapter

    if not isinstance(adapter, BasePlatformAdapter) \
            or type(adapter).send_clarify is BasePlatformAdapter.send_clarify:
        return None
    # A human-decision prompt: Telegram's "important" mode must push it, not deliver it silently (#132516).
    metadata = {**(send_kwargs.pop("metadata", None) or {}), "is_approval_prompt": True}
    return BasePlatformAdapter.send_clarify(adapter, metadata=metadata, **send_kwargs)


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
    from hermes_agent.gateway.run import _approval_send_outcome

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
    from hermes_agent.gateway.run import _approval_send_outcome

    outcome = _approval_send_outcome(fut, timeout=SEND_ACK_WINDOW)
    if outcome == "failed" and _withdrawn(fut):
        # Nothing was posted on purpose: no text fallback for a question nobody is waiting on.
        clarify_mod.wait_for_response(clarify_id, timeout=0.01)  # retire the withdrawn entry
        return clarify_mod.CANCELLED, False
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
    if response == clarify_mod.CANCELLED:
        return response, False
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
        from hermes_agent.gateway.run import _approval_send_outcome
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
        if _withdrawn(fut):
            return  # withdrawn before posting; whoever withdrew it already woke the waiter
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


def _withdrawn(fut) -> bool:
    """True when the card send resolved as deliberately not posted (see ``WITHDRAWN``)."""
    done = getattr(fut, "done", None)
    if fut is None or not callable(done) or not done() or fut.cancelled() or fut.exception() is not None:
        return False
    raw = getattr(fut.result(), "raw_response", None)
    return isinstance(raw, dict) and bool(raw.get(WITHDRAWN))


def withdrawn_result():
    from hermes_agent.gateway.platforms.base import SendResult
    return SendResult(success=False, error="clarify prompt withdrawn before posting",
                      raw_response={WITHDRAWN: True})


def clarify_card_still_wanted(runner, entry) -> bool:
    """After the card's awaited explanation: False once the run went stale or the question was
    withdrawn meanwhile (cancelled, or cleared by ``/new`` / ``/stop``). An answered question keeps
    posting as before: its waiter returns the answer either way."""
    still_current = getattr(runner._ctx, "_run_still_current", None)
    if callable(still_current) and not still_current():
        return False
    if entry is None:
        return True
    from tools import clarify_gateway
    return not (entry.event.is_set() and entry.response == clarify_gateway.CANCELLED)


def clarify_context_sink(runner) -> Optional[Callable[..., None]]:
    """The ``clarify_context_callback`` for a turn with interim messages off (#102413): prose the model
    wrote in the same message as a ``clarify`` call is what the user is asked to decide on, so it is
    sent before the card. The agent calls this on its own thread before the clarify tool runs, and it
    waits for the send, so the card can never land above its explanation."""
    from hermes_agent.gateway.bot_peer_notices import automatic_notices_allowed
    from hermes_agent.gateway.run_turn_progress import local_viewer_turn
    if local_viewer_turn(runner):  # a local viewer renders the transcript itself; its adapter sends nothing
        return None
    ctx = runner._ctx

    def send(text: str, *, already_streamed: bool = False) -> None:
        adapter = ctx._status_adapter
        if (already_streamed or adapter is None or not str(text or "").strip() or not ctx._run_still_current()
                or not automatic_notices_allowed(ctx.source)):
            return
        from hermes_agent.gateway.run import _interim_metadata
        fut = runner._schedule(
            adapter.send(ctx._status_chat_id, text, metadata=_interim_metadata(ctx._status_thread_metadata)),
            "clarify context send failed to schedule")
        if fut is None:
            return
        # The card waits for it too, but only for what is left of this one budget.
        runner._clarify_context_send = fut
        runner._clarify_context_deadline = deadline = time.monotonic() + SEND_ACK_WINDOW
        try:
            fut.result(timeout=max(0.0, deadline - time.monotonic()))
        except Exception:
            logger.warning("Clarify context could not be sent before the prompt", exc_info=True)
    return send


async def await_clarify_context(runner) -> None:
    """Hold the clarify card until its explanation's send has finished, within what is left of the
    explanation's ``SEND_ACK_WINDOW`` budget (the agent thread already spent part of it). The card path
    does not take the adapter's text-send lock, so a slow split send would otherwise let the buttons
    land above the rest of the explanation. A send that never returns still lets the card post once
    the budget is spent: the user must always get the question the turn is waiting on."""
    fut, runner._clarify_context_send = getattr(runner, "_clarify_context_send", None), None
    deadline, runner._clarify_context_deadline = getattr(runner, "_clarify_context_deadline", None), None
    if fut is None or fut.done():
        return
    remaining = SEND_ACK_WINDOW if deadline is None else deadline - time.monotonic()
    if remaining <= 0:
        logger.warning("Clarify context still sending after %ss; posting the prompt anyway", SEND_ACK_WINDOW)
        return
    try:
        await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(fut)), remaining)
    except TimeoutError:
        logger.warning("Clarify context still sending after %ss; posting the prompt anyway",
                       SEND_ACK_WINDOW)
    except Exception:  # a failed explanation send was already logged; the card still goes
        logger.debug("clarify context send failed before the prompt", exc_info=True)
