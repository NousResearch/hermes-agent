"""Vault secret prompts for messaging surfaces.

The vault tools (unlock, save-login, one-time code) obtain secrets through
callbacks registered on :mod:`agent.vault_backends.unlock`. The CLI and the
TUI/Desktop gateway register them; the messaging gateway did not, so on Telegram
(and every other chat platform) ``can_prompt_here()`` was always False and every
vault tool could only answer ``prompt_unavailable`` — the user had no way to hand
the agent a password at all.

This module builds those three callbacks over the gateway's existing synchronous
clarify bridge. Reusing that bridge is deliberate: it already renders a prompt on
any platform (native card, plus a plain-text fallback), blocks the agent thread on
an event, and — the part that matters here — intercepts the user's reply in the
inbound path, so the answer is handed straight to the callback and never becomes a
conversation message. A secret typed in answer to one of these prompts therefore
never reaches session history, memory, or the model.

The prompts are marked secret, and a secret prompt is a promise that the answer
does not stay in the chat: the gateway posts one only where the bot can delete the
user's message, binds it to the person the turn belongs to, deletes the answer on
receipt (before the value is handed over) and says so in the chat when it cannot.
A surface that cannot keep that promise raises :class:`VaultPromptRefused`, which
the vault tool reports to the model instead of asking.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from agent.i18n import t
from agent.vault_backends.unlock import (
    set_code_prompt_callback,
    set_save_login_prompt_callback,
    set_unlock_prompt_callback,
)

# A surface's blocking ask: (question, secret, last) -> the user's reply, "" when unanswered.
# ``last`` tells the surface whether another question follows in this sequence, so it can
# hold the stream/typing re-arm for the final one (same contract as the clarify batch).
VaultAsk = Callable[..., str]


logger = logging.getLogger(__name__)


class VaultPromptRefused(RuntimeError):
    """This chat cannot take a secret safely (the bot cannot delete the answer, or nobody is bound
    to it). Raised instead of asking; the message says what would make it possible."""


def bound_owner(ctx, schedule) -> str:
    """The sender a vault card is bound to, once this turn's chat is shown able to take a secret.

    Checked before anything is posted: a known sender (a topic or a group can share one session)
    and an adapter that can delete that sender's message there (``can_delete_user_messages``,
    run on the gateway loop through ``schedule``), because the answer is deleted on receipt and a
    secret prompt must not be posted where that cannot happen. An adapter without the probe
    cannot. Raises ``VaultPromptRefused`` with what would make it possible."""
    source = ctx.source
    owner = str(source.user_id or "") if source is not None else ""
    if not owner:
        raise VaultPromptRefused(
            "This chat cannot take a secret: the sender of this turn is unknown, so the answer "
            "cannot be bound to them. Use `hermes vault add` on the server instead.")
    adapter = ctx._status_adapter
    if adapter is None:
        raise VaultPromptRefused("This turn has no chat to ask in. Use `hermes vault add` on the server instead.")
    probe = getattr(adapter, "can_delete_user_messages", None)
    allowed = False
    if callable(probe):
        fut = schedule(probe(ctx._status_chat_id, source.chat_type),
                       "Vault prompt deletion-rights probe failed to schedule")
        try:
            allowed = bool(fut.result(timeout=15.0)) if fut is not None else False
        except Exception:
            logger.warning("Vault prompt deletion-rights probe failed", exc_info=True)
    if not allowed:
        logger.warning("Vault prompt refused: the bot cannot delete user messages in chat %s", ctx._status_chat_id)
        raise VaultPromptRefused(
            "This chat cannot take a secret: the bot cannot delete the user's message here, and "
            "a password or code must not stay in the chat. On Telegram, make the bot an "
            "administrator of this group with the \"Delete messages\" right, or ask in a private "
            "chat with the bot; otherwise use `hermes vault add` on the server.")
    return owner


def _clean(value: Optional[str]) -> str:
    return (value or "").strip()


def build_vault_prompt_callbacks(ask: VaultAsk) -> dict[str, Callable]:
    """Build the three vault prompt callbacks on top of ``ask``.

    ``ask(question, secret=True)`` must block until the user answers and return
    their reply, or "" when none arrived. Each callback returns the shape its
    caller in :mod:`agent.vault_backends.unlock` documents, and answers "" / None
    on a decline so the calling tool reports a cancellation instead of storing an
    empty secret.
    """

    def unlock_prompt(backend: str, display_name: str) -> str:
        """Master password for an external password manager. Consumed by the
        manager's CLI to mint a session token; never written to disk."""
        name = display_name or backend
        return _clean(ask(t("gateway.vault_prompt.unlock", name=name, backend=backend), secret=True, last=True))

    def save_login_prompt(origin: str, site_label: str) -> Optional[dict[str, str]]:
        """Identifier + password for the page currently open. Stored in the local
        vault bound to ``origin`` and filled into the page at once.

        Two questions, so the text says so: a bare "Now the password" read as the
        first prompt repeating, and an unanswered first card left the caller
        looking like it had hung on a prompt the user never saw."""
        host = site_label or origin
        identifier = _clean(ask(t("gateway.vault_prompt.login_identifier", host=host), secret=False, last=False))
        if not identifier:
            return None
        password = ask(t("gateway.vault_prompt.login_password", host=host), secret=True, last=True)
        if not password:
            return None
        return {"identifier": identifier, "password": password}

    def code_prompt(site: str, hint: str) -> str:
        """One-time code the site just sent, for a login with no stored seed."""
        detail = f"\n\n{hint}" if hint else ""
        return _clean(ask(t("gateway.vault_prompt.code", site=site, detail=detail), secret=True, last=True))

    return {
        "unlock": unlock_prompt,
        "save_login": save_login_prompt,
        "code": code_prompt,
    }


def install_vault_prompt_callbacks(ask: VaultAsk) -> None:
    """Register all three callbacks for the CURRENT thread.

    ``agent.vault_backends.unlock`` stores its callbacks in a thread-local, so this
    must run on the thread that will execute the tools — the gateway wires it
    alongside ``agent.clarify_callback`` for exactly that reason.
    """
    callbacks = build_vault_prompt_callbacks(ask)
    set_unlock_prompt_callback(callbacks["unlock"])
    set_save_login_prompt_callback(callbacks["save_login"])
    set_code_prompt_callback(callbacks["code"])


def clear_vault_prompt_callbacks() -> None:
    """Drop the callbacks after a turn so a reused worker thread never keeps
    prompting for a session that has ended."""
    set_unlock_prompt_callback(None)
    set_save_login_prompt_callback(None)
    set_code_prompt_callback(None)
