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
    async def send_secret_prompt(self, chat_id, env_var, prompt, *, user_id, submit, timeout_s, outcome,
                                 metadata=None) -> SendResult

``submit(value)`` hands the value to the waiting thread (False: the request already ended). ``outcome`` is a
``concurrent.futures.Future`` resolved to True once the value is saved, False when it was not, so the card
reports what actually happened.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import logging

from gateway.session_vault_code_card import CardAsker

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
    from tools.skills_tool import bind_turn_secret_capture
    asker = CardAsker(turn, agent)

    def callback(var_name, prompt, metadata=None):
        # Resolved after the save is attempted; the card's footer says saved or not saved from it.
        outcome: concurrent.futures.Future = concurrent.futures.Future()

        def send(request):
            # Re-read the adapter: a reconnect replaces the instance the turn started with.
            live_adapter = _secret_card_adapter(turn)
            return None if live_adapter is None else live_adapter.send_secret_prompt(
                ctx._status_chat_id, var_name, prompt, user_id=user_id, submit=request.submit,
                timeout_s=SECRET_CARD_TIMEOUT_S, outcome=outcome,
                metadata={**(ctx._status_thread_metadata or {}), **dict(metadata or {})})
        value = asker.ask(send, SECRET_CARD_TIMEOUT_S, f"waiting for {var_name}", "Secret entry")
        result = _save(var_name, value) if value else _skipped(var_name)
        outcome.set_result(not result.get("skipped"))
        return result

    try:
        with bind_turn_secret_capture(callback):
            yield
    finally:
        asker.close()


def _save(var_name, value):
    from hermes_cli.config import save_env_value_secure
    try:
        saved = save_env_value_secure(var_name, value)
    except (OSError, ValueError) as exc:  # locked key, redacted mask, unwritable .env: the message names the key only
        logger.warning("Saving %s from the secret card failed: %s", var_name, type(exc).__name__)
        return {"success": False, "stored_as": var_name, "validated": False, "skipped": True, "message": str(exc)}
    return {**saved, "skipped": False,
            "message": "Secret stored securely. The secret value was not exposed to the model."}
