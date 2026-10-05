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

The prompts are marked secret so the adapter deletes the user's own message once
the secret has been read, on platforms that support deletion.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from agent.vault_backends.unlock import (
    set_code_prompt_callback,
    set_save_login_prompt_callback,
    set_unlock_prompt_callback,
)

# A surface's blocking ask: (question, secret, last) -> the user's reply, "" when unanswered.
# ``last`` tells the surface whether another question follows in this sequence, so it can
# hold the stream/typing re-arm for the final one (same contract as the clarify batch).
VaultAsk = Callable[..., str]


def _clean(value: Optional[str]) -> str:
    return (value or "").strip()


def build_vault_prompt_callbacks(ask: VaultAsk) -> Dict[str, Callable]:
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
        return _clean(ask(
            f"🔐 Unlock {name}.\n\nSend your master password in your next message. "
            f"It is passed to `{backend}` once to open the vault and is not stored.",
            secret=True, last=True,
        ))

    def save_login_prompt(origin: str, site_label: str) -> Optional[Dict[str, str]]:
        """Identifier + password for the page currently open. Stored in the local
        vault bound to ``origin`` and filled into the page at once.

        Two questions, so the text says so: a bare "Now the password" read as the
        first prompt repeating, and an unanswered first card left the caller
        looking like it had hung on a prompt the user never saw."""
        host = site_label or origin
        identifier = _clean(ask(
            f"🔑 Login for {host} (1/2)\n\n"
            "What email or username should I save for this site?",
            secret=False, last=False,
        ))
        if not identifier:
            return None
        password = ask(
            f"🔑 Login for {host} (2/2)\n\n"
            "Now the password. It is stored in your local vault and typed straight "
            "into the page.",
            secret=True, last=True,
        )
        if not password:
            return None
        return {"identifier": identifier, "password": password}

    def code_prompt(site: str, hint: str) -> str:
        """One-time code the site just sent, for a login with no stored seed."""
        detail = f"\n\n{hint}" if hint else ""
        return _clean(ask(
            f"🔢 {site} sent you a one-time code.\n\nSend it here and I'll enter it "
            f"into the page.{detail}",
            secret=True, last=True,
        ))

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
