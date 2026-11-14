"""The Slack card for the browser vault's one-time-code prompt: a button that opens a private modal.

The gateway asks through ``send_vault_code_prompt`` (``gateway/session_vault_code_card.py``). Only the
turn's author, in the workspace the card was posted to, may open the modal, and the code goes from the
``view_submission`` payload to the gateway's ``submit`` callback: it is never posted, echoed or logged.
The button carries an opaque token, never the code or the session key.
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from agent.i18n import t
from gateway.platforms.base import SendResult

logger = logging.getLogger("plugins.platforms.slack.adapter")

OPEN_ACTION = "hermes_vault_code_open"
VIEW_CALLBACK = "hermes_vault_code_submit"
_BLOCK_ID = "hermes_vault_code_block"
_INPUT_ACTION = "hermes_vault_code_value"
_HINT_LIMIT = 300
_MODAL_TITLE_LIMIT = 24  # Slack caps modal title / submit text at 24 characters
_REQUESTS_MAX = 200


@dataclass
class _CodeCard:
    submit: Callable[[str], bool]
    user_id: str
    team_id: str
    site: str
    channel_id: str = ""
    message_ts: str = ""
    text: str = ""
    expires_at: float = 0.0


def _mrkdwn_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SlackVaultCodeMixin:
    """``send_vault_code_prompt``: the capability the gateway detects to offer the code prompt here."""

    _app: Any

    def _vault_code_cards(self) -> dict:
        return self._lazy_attr("_vault_code_requests", dict)

    def _register_vault_code_handlers(self, app) -> None:
        app.action(OPEN_ACTION)(self._handle_vault_code_open)
        app.view(VIEW_CALLBACK)(self._handle_vault_code_submit)

    async def send_vault_code_prompt(self, chat_id: str, site: str, hint: str, *, user_id: str,
                                     submit: Callable[[str], bool], timeout_s: float,
                                     metadata: Optional[dict] = None) -> SendResult:
        token = secrets.token_urlsafe(18)
        card = _CodeCard(submit=submit, user_id=str(user_id), team_id=str(self._metadata_team_id(metadata) or ""),
                         site=site, expires_at=time.monotonic() + timeout_s)
        body = t("platform.shared.vault_code.body", site=_mrkdwn_escape(site))
        if hint:
            body += "\n" + _mrkdwn_escape(hint[:_HINT_LIMIT])
        card.text = f"*{_mrkdwn_escape(t('platform.shared.vault_code.title', site=site))}*\n{body}"[:3000]

        def _build() -> tuple[str, list]:
            blocks = [
                {"type": "section", "text": {"type": "mrkdwn", "text": card.text}},
                {"type": "actions", "elements": [self._button(
                    t("platform.shared.vault_code.button")[:75], OPEN_ACTION, token, style="primary")]}]
            return t("platform.shared.vault_code.title", site=site), blocks

        result = await self._send_interactive_prompt(chat_id, metadata, _build, "send_vault_code_prompt",
                                                     sanitize=False)
        if result.success and result.message_id:
            card.channel_id = str((result.raw_response or {}).get("channel") or chat_id)
            card.message_ts = result.message_id
            cards = self._vault_code_cards()
            cards[token] = card
            self._trim_oldest_dict_entries(cards, _REQUESTS_MAX)
            # The card retires itself when the request can no longer be answered (Discord's view timeout).
            asyncio.get_running_loop().call_later(
                timeout_s, lambda: asyncio.ensure_future(self._retire_vault_code_card(token, ended=True)))
        return result

    def _live_vault_code_card(self, token: str) -> Optional[_CodeCard]:
        card = self._vault_code_cards().get(token)
        if card is None or time.monotonic() >= card.expires_at:
            return None
        return card

    async def _retire_vault_code_card(self, token: str, *, ended: bool) -> None:
        card = self._vault_code_cards().pop(token, None)
        if card is None:
            return
        notice = t("platform.shared.vault_code.ended" if ended else "platform.shared.vault_code.submitted_footer")
        await self._finalize_interactive_message(card.channel_id, card.message_ts, card.text, notice,
                                                 "Verification code request", "vault code",
                                                 card.team_id or None, sanitize=False)

    def _vault_code_author_matches(self, card: _CodeCard, user_id: str, team_id: str) -> bool:
        return user_id == card.user_id and (not card.team_id or not team_id or team_id == card.team_id)

    async def _handle_vault_code_open(self, ack, body, action) -> None:
        """Open the private modal for the card's author; anyone else is told so, privately."""
        started = await self._begin_interaction(ack, body, action, "vault code")
        if started is None:
            return
        team_id, _action_id, token, message, _msg_ts, channel_id, _name, user_id = started
        card = self._live_vault_code_card(token)
        refusal = ("platform.shared.vault_code.ended" if card is None
                   else None if self._vault_code_author_matches(card, user_id, team_id)
                   else "platform.shared.vault_code.not_author")
        client = self._get_client(channel_id, team_id=team_id or None)
        if refusal is not None:
            await client.chat_postEphemeral(channel=channel_id, user=user_id, text=t(refusal),
                                            thread_ts=(message or {}).get("thread_ts"))
            return
        view = {
            "type": "modal", "callback_id": VIEW_CALLBACK, "private_metadata": token,
            "title": {"type": "plain_text", "text": t("platform.shared.vault_code.input_label")[:_MODAL_TITLE_LIMIT]},
            "submit": {"type": "plain_text", "text": t("platform.shared.vault_code.button")[:_MODAL_TITLE_LIMIT]},
            "blocks": [{
                "type": "input", "block_id": _BLOCK_ID,
                "label": {"type": "plain_text", "text": t("platform.shared.vault_code.modal_title",
                                                          site=card.site)[:2000]},
                "element": {"type": "plain_text_input", "action_id": _INPUT_ACTION, "max_length": 32},
            }],
        }
        try:
            await client.views_open(trigger_id=str(body.get("trigger_id") or ""), view=view)
        except Exception:  # health: allow BLE001 -- any Web API failure leaves the card live for a retry
            logger.warning("[Slack] Could not open the vault code modal", exc_info=True)

    async def _handle_vault_code_submit(self, ack, body, view) -> None:
        """Hand the modal value to the waiting agent thread; it never becomes a message."""
        token = str((view or {}).get("private_metadata") or "")
        values = ((view or {}).get("state") or {}).get("values") or {}
        code = str(((values.get(_BLOCK_ID) or {}).get(_INPUT_ACTION) or {}).get("value") or "").strip()
        user_id = str(((body or {}).get("user") or {}).get("id") or "")
        team_id = self._event_team_id({}, body or {}, install_set=getattr(self, "_team_clients", None))
        card = self._live_vault_code_card(token)
        if card is None or not self._vault_code_author_matches(card, user_id, team_id):
            key = "platform.shared.vault_code.ended" if card is None else "platform.shared.vault_code.not_author"
            await ack(response_action="errors", errors={_BLOCK_ID: t(key)})
            return
        accepted = bool(code) and card.submit(code)
        code = ""
        if not accepted:
            await ack(response_action="errors", errors={_BLOCK_ID: t("platform.shared.vault_code.ended")})
            await self._retire_vault_code_card(token, ended=True)
            return
        await ack()
        await self._retire_vault_code_card(token, ended=False)
