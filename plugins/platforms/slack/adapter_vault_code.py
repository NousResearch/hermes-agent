"""The Slack card for the browser vault's one-time-code prompt: a button that opens a private modal.

The gateway asks through ``send_vault_code_prompt`` (``gateway/session_vault_code_card.py``). Only the
turn's author, in the workspace the card was posted to, may open the modal, and the code goes from the
``view_submission`` payload to the gateway's ``submit`` callback: it is never posted, echoed or logged.
The button carries an opaque token, never the code or the session key. The card → modal machinery is
shared with the skill-secret card (``adapter_private_entry.py``).
"""
from __future__ import annotations

from typing import Callable, Optional

from gateway.platforms.base import SendResult

try:  # sibling modules; support both package and flat plugin-dir import
    from .adapter_private_entry import _HINT_LIMIT, VAULT_CODE, SlackPrivateEntryMixin, mrkdwn_escape, t
except ImportError:  # pragma: no cover - plugin loaded outside package context
    from adapter_private_entry import (  # type: ignore
        _HINT_LIMIT, VAULT_CODE, SlackPrivateEntryMixin, mrkdwn_escape, t)

OPEN_ACTION = VAULT_CODE.open_action
VIEW_CALLBACK = VAULT_CODE.view_callback
_BLOCK_ID = VAULT_CODE.block_id
_INPUT_ACTION = VAULT_CODE.input_action


class SlackVaultCodeMixin(SlackPrivateEntryMixin):
    """``send_vault_code_prompt``: the capability the gateway detects to offer the code prompt here."""

    def _vault_code_cards(self) -> dict:
        return self._entry_cards(VAULT_CODE)

    async def send_vault_code_prompt(self, chat_id: str, site: str, hint: str, *, user_id: str,
                                     submit: Callable[[str], bool], timeout_s: float,
                                     metadata: Optional[dict] = None) -> SendResult:
        body = t("platform.shared.vault_code.body", site=mrkdwn_escape(site))
        if hint:
            body += "\n" + mrkdwn_escape(hint[:_HINT_LIMIT])
        return await self._post_entry_card(
            VAULT_CODE, chat_id, metadata, title=t("platform.shared.vault_code.title", site=site), body=body,
            user_id=user_id, submit=submit, timeout_s=timeout_s,
            modal_label=t("platform.shared.vault_code.modal_title", site=site)[:2000])

    async def _handle_vault_code_open(self, ack, body, action) -> None:
        await self._handle_entry_open(ack, body, action)

    async def _handle_vault_code_submit(self, ack, body, view) -> None:
        await self._handle_entry_submit(ack, body, dict(view or {}, callback_id=VIEW_CALLBACK))
