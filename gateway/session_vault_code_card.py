"""The browser vault's one-time-code prompt on a messaging turn whose adapter renders a code card.

``browser_vault_enter_code`` asks the surface for the code a site texted or emailed through
``unlock.set_code_prompt_callback``. ``vault_prompt_scope`` installs it only for a local viewer, so on a
chat platform the tool reported ``prompt_unavailable`` and an SMS- or email-only sign-in could not finish
from chat (#141844). An adapter that implements ``send_vault_code_prompt`` gets the callback here: it posts
a card only the turn's author can answer, and the code travels from the adapter's private input (Discord:
a modal) straight to the waiting agent thread through ``submit``. It is never sent as a message, journaled
or logged. Only the code prompt is offered here; a master password or a new login typed into a chat
platform is a separate decision.

The adapter method is optional and detected on the class::

    async def send_vault_code_prompt(self, chat_id, site, hint, *, user_id, submit, timeout_s,
                                     metadata=None) -> SendResult

``user_id`` is the turn's author, the only person the card may accept a code from. ``submit(code)`` hands
the code to the waiting thread and returns False when the request already ended (answered, timed out,
interrupted, turn over), in which case nothing was entered.
"""
from __future__ import annotations

import contextlib
import threading
import time


class _CodeRequest:
    """One pending code: settled once, by the author's submit or by the wait ending."""

    __slots__ = ("_event", "_lock", "_value")

    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._value = ""

    def submit(self, code: str) -> bool:
        code = (code or "").strip()
        with self._lock:
            if self._event.is_set() or not code:
                return False
            self._value = code
            self._event.set()
            return True

    def close(self) -> None:
        with self._lock:
            self._event.set()

    def cancel(self) -> None:
        """The turn ended: settle the request AND drop a code submitted but not yet consumed."""
        with self._lock:
            self._value = ""
            self._event.set()

    def wait(self, timeout_s: float, cancelled, heartbeat) -> str:
        """The submitted code, or "" on timeout or interrupt. Waits in 1 s slices so ``heartbeat``
        keeps the gateway's inactivity watchdog from killing a turn whose user is still typing."""
        deadline = time.monotonic() + timeout_s
        while not self._event.wait(min(1.0, max(0.0, deadline - time.monotonic()))):
            if cancelled() or time.monotonic() >= deadline:
                break
            heartbeat()
        self.close()
        # A /stop that raced the submit still wins: nothing is entered for a stopped turn.
        with self._lock:
            return "" if cancelled() else self._value


def _code_card_adapter(turn):
    adapter = getattr(turn._ctx, "_status_adapter", None)
    return adapter if getattr(type(adapter), "send_vault_code_prompt", None) is not None else None


@contextlib.contextmanager
def code_card_scope(turn, agent):
    """Install the code callback on this (agent) thread for a messaging turn whose adapter renders a
    code card; restore the previous callback and release any request still waiting when the turn ends."""
    ctx = turn._ctx
    adapter = _code_card_adapter(turn)
    user_id = str(getattr(getattr(ctx, "source", None), "user_id", "") or "")
    if adapter is None or not user_id:
        # No author to bind the card to: answering it would be open to anyone the adapter admits.
        yield
        return
    from agent.vault_backends import unlock
    from gateway.run import _approval_send_outcome
    from gateway.run_turn_runner_clarify_delivery import SEND_ACK_WINDOW
    from gateway.session_vault_prompt import CODE_TIMEOUT_S
    from tools.approval_human_wait import activity_heartbeat, human_wait_window
    open_requests: list[_CodeRequest] = []
    scope_lock = threading.Lock()
    scope_open = [True]

    def code_prompt(site, hint):
        # Re-read the adapter: a reconnect replaces the instance the turn started with.
        live_adapter = _code_card_adapter(turn)
        request = _CodeRequest()
        with scope_lock:
            if not scope_open[0] or live_adapter is None:
                # A tool worker that outlived its turn keeps this callback: it may not ask again.
                return ""
            open_requests.append(request)
        fut = turn._schedule(
            live_adapter.send_vault_code_prompt(
                ctx._status_chat_id, site, hint, user_id=user_id, submit=request.submit,
                timeout_s=CODE_TIMEOUT_S, metadata=ctx._status_thread_metadata),
            "Vault code card failed to schedule")
        # ``ambiguous`` = the card may have posted with a late ack: keep waiting, as clarify does.
        if _approval_send_outcome(fut, timeout=SEND_ACK_WINDOW) not in ("sent", "ambiguous"):
            request.close()
            return ""
        with human_wait_window(ctx.session_key or None):
            return request.wait(CODE_TIMEOUT_S, lambda: bool(getattr(agent, "_interrupt_requested", False)),
                                activity_heartbeat("waiting for a one-time code"))

    previous = unlock.get_code_prompt_callback()
    unlock.set_code_prompt_callback(code_prompt)
    try:
        yield
    finally:
        unlock.set_code_prompt_callback(previous)
        # A tool thread abandoned by its batch deadline can still be parked on a card: release it.
        with scope_lock:
            scope_open[0] = False
            pending = list(open_requests)
        for request in pending:
            request.cancel()
