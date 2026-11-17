"""A skill's missing secret, asked for on a messaging turn whose adapter renders a private entry card.

A skill whose ``required_environment_variables`` are missing asks the turn's secret-capture callback
(``tools.skills_tool.bind_turn_secret_capture``). ``secret_prompt_scope`` binds one only for a local viewer,
so a chat platform got the "secure secret entry is not supported over messaging" hint and the user had to
edit ``.env`` by hand. An adapter that implements ``send_secret_prompt`` AND has it switched on
(``secret_prompt_enabled()``; Slack: ``platforms.slack.extra.secret_entry_modal``) gets the callback here:
it posts a card only the turn's author can answer, the value travels from the adapter's private input
(Slack: a modal) to this thread through ``submit``, and is saved to the owning profile's ``.env`` with
``save_env_value_secure``. It is never sent as a message, journaled, logged or returned to the model.

The agent thread runs inside the turn's profile scope, so ``get_env_path()`` is the owning profile's
``.env`` even when one gateway serves several profiles.

The adapter methods are optional and detected on the class::

    def secret_prompt_enabled(self) -> bool
    async def send_secret_prompt(self, chat_id, env_var, prompt, *, user_id, submit, timeout_s,
                                 metadata=None) -> SendResult
"""
from __future__ import annotations

import contextlib
import logging
import threading

from gateway.session_vault_code_card import _CodeRequest

logger = logging.getLogger(__name__)

SECRET_CARD_TIMEOUT_S = 300  # typing a key in from a password manager on a phone takes longer than the CLI's 120 s


def _skipped(var_name, message="Secret setup was skipped."):
    return {"success": True, "reason": "cancelled", "stored_as": var_name, "validated": False,
            "skipped": True, "message": message}


def _secret_card_adapter(turn):
    adapter = getattr(turn._ctx, "_status_adapter", None)
    cls = type(adapter)
    capable = adapter is not None and all(getattr(cls, name, None) is not None
                                          for name in ("send_secret_prompt", "secret_prompt_enabled"))
    return adapter if capable and adapter.secret_prompt_enabled() else None


@contextlib.contextmanager
def secret_card_scope(turn, agent):
    """Bind the secret-capture callback for a messaging turn whose adapter renders a secret card;
    release any request still waiting when the turn ends."""
    ctx = turn._ctx
    user_id = str(getattr(getattr(ctx, "source", None), "user_id", "") or "")
    if _secret_card_adapter(turn) is None or not user_id:
        # No author to bind the card to: answering it would be open to anyone the adapter admits.
        yield
        return
    from gateway.run import _approval_send_outcome
    from gateway.run_turn_runner_clarify_delivery import SEND_ACK_WINDOW
    from tools.approval_human_wait import activity_heartbeat, human_wait_window
    from tools.skills_tool import bind_turn_secret_capture
    open_requests: list[_CodeRequest] = []
    scope_lock = threading.Lock()
    scope_open = [True]

    def ask(var_name, prompt, metadata):
        # Re-read the adapter: a reconnect replaces the instance the turn started with.
        live_adapter = _secret_card_adapter(turn)
        request = _CodeRequest()
        with scope_lock:
            if not scope_open[0] or live_adapter is None:
                # A tool worker that outlived its turn keeps this callback: it may not ask again.
                return ""
            open_requests.append(request)
        fut = turn._schedule(
            live_adapter.send_secret_prompt(
                ctx._status_chat_id, var_name, prompt, user_id=user_id, submit=request.submit,
                timeout_s=SECRET_CARD_TIMEOUT_S, metadata={**(ctx._status_thread_metadata or {}), **metadata}),
            "Secret entry card failed to schedule")
        if _approval_send_outcome(fut, timeout=SEND_ACK_WINDOW) not in ("sent", "ambiguous"):
            request.close()
            return ""
        with human_wait_window(ctx.session_key or None):
            return request.wait(SECRET_CARD_TIMEOUT_S, lambda: bool(getattr(agent, "_interrupt_requested", False)),
                                activity_heartbeat(f"waiting for {var_name}"))

    def callback(var_name, prompt, metadata=None):
        value = ask(var_name, prompt, dict(metadata or {}))
        if not value:
            return _skipped(var_name)
        from hermes_cli.config import save_env_value_secure
        try:
            saved = save_env_value_secure(var_name, value)
        except ValueError as exc:  # a locked key or a redacted mask: refused, message names the key only
            return {"success": False, "stored_as": var_name, "validated": False, "skipped": True,
                    "message": str(exc)}
        return {**saved, "skipped": False,
                "message": "Secret stored securely. The secret value was not exposed to the model."}

    try:
        with bind_turn_secret_capture(callback):
            yield
    finally:
        with scope_lock:
            scope_open[0] = False
            pending = list(open_requests)
        for request in pending:
            request.cancel()
