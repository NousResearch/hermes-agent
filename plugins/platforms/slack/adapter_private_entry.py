"""Slack cards that collect a private value through a modal: a browser vault one-time code, or a skill secret.

A card posts in the turn's thread with one button carrying an opaque token. Only the turn's author, in the
workspace the card was posted to, may open the modal (``views_open``); the ``view_submission`` value goes to
the gateway's ``submit`` callback and is never posted, echoed or logged. Each kind has its own action and
view callback ids so a submit can only ever answer the kind of card that opened it.

``send_secret_prompt`` is the gateway capability ``gateway/session_secret_card.py`` detects: a skill whose
``required_environment_variables`` are missing asks for the value here, and the gateway saves it to the
owning profile's ``.env``. The capability is opt-in (``platforms.slack.extra.secret_entry_modal``).
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from gateway.platforms.base import SendResult

logger = logging.getLogger("plugins.platforms.slack.adapter")

_HINT_LIMIT = 300
_MODAL_TITLE_LIMIT = 24  # Slack caps modal title / submit text at 24 characters
_REQUESTS_MAX = 200


@dataclass(frozen=True)
class EntryKind:
    """One kind of private-entry card: its Block Kit ids, i18n prefix and input cap."""

    open_action: str
    view_callback: str
    block_id: str
    input_action: str
    i18n: str
    store_attr: str
    max_length: int
    label: str


VAULT_CODE = EntryKind("hermes_vault_code_open", "hermes_vault_code_submit", "hermes_vault_code_block",
                       "hermes_vault_code_value", "platform.shared.vault_code", "_vault_code_requests", 32,
                       "vault code")
SECRET = EntryKind("hermes_secret_entry_open", "hermes_secret_entry_submit", "hermes_secret_entry_block",
                   "hermes_secret_entry_value", "platform.shared.secret_entry", "_secret_entry_requests", 3000,
                   "secret entry")
KINDS = (VAULT_CODE, SECRET)


class _DropEntryFrames(logging.Filter):
    """slack_sdk's Socket Mode client logs every received frame at DEBUG before Bolt dispatches it, so a
    modal submit would put the value in agent.log with ``logging.level: DEBUG``. Drop those frames."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(kind.view_callback in message for kind in KINDS)


# The Socket Mode client logs through the Bolt app's logger (``AsyncSocketModeHandler`` passes
# ``app.logger``); the module loggers cover a client built without one.
for _name in ("slack_bolt.AsyncApp", "slack_sdk.socket_mode.aiohttp", "slack_sdk.socket_mode.async_client"):
    _logger = logging.getLogger(_name)
    if not any(isinstance(f, _DropEntryFrames) for f in _logger.filters):
        _logger.addFilter(_DropEntryFrames())


@dataclass
class _EntryCard:
    kind: EntryKind
    submit: Callable[[str], bool]
    user_id: str
    team_id: str
    modal_label: str
    channel_id: str = ""
    message_ts: str = ""
    text: str = ""
    expires_at: float = 0.0


def t(key: str, **kwargs: Any) -> str:
    """The adapter facade's ``t``: plugins reach core i18n only through the facade's baselined import."""
    from plugins.platforms.slack.adapter import t as facade_t
    return facade_t(key, **kwargs)


def mrkdwn_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SlackPrivateEntryMixin:
    """The shared card → modal → submit machinery, plus the ``send_secret_prompt`` capability."""

    _app: Any
    config: Any

    def _entry_cards(self, kind: EntryKind) -> dict:
        return self._lazy_attr(kind.store_attr, dict)

    def _register_private_entry_handlers(self, app) -> None:
        for kind in KINDS:
            app.action(kind.open_action)(self._handle_entry_open)
            app.view(kind.view_callback)(self._handle_entry_submit)

    def secret_prompt_enabled(self) -> bool:
        """``platforms.slack.extra.secret_entry_modal`` (default off: the legacy refusal hint)."""
        return self._extra_flag("secret_entry_modal", default=False)

    async def send_secret_prompt(self, chat_id: str, env_var: str, prompt: str, *, user_id: str,
                                 submit: Callable[[str], bool], timeout_s: float,
                                 metadata: Optional[dict] = None) -> SendResult:
        """Ask the turn's author for ``env_var``; ``metadata`` may carry ``skill_name``/``help``."""
        details = metadata or {}
        name = mrkdwn_escape(env_var)
        body = t("platform.shared.secret_entry.body", name=name)
        for line in (prompt, details.get("help"), details.get("skill_name") and t(
                "platform.shared.secret_entry.skill", skill=details["skill_name"])):
            if line:
                body += "\n" + mrkdwn_escape(str(line)[:_HINT_LIMIT])
        title = t("platform.shared.secret_entry.title", name=env_var)
        return await self._post_entry_card(
            SECRET, chat_id, details, title=title, body=body, user_id=user_id, submit=submit,
            timeout_s=timeout_s, modal_label=str(prompt or env_var)[:2000])

    async def _post_entry_card(self, kind: EntryKind, chat_id: str, metadata: Optional[dict], *, title: str,
                               body: str, user_id: str, submit: Callable[[str], bool], timeout_s: float,
                               modal_label: str) -> SendResult:
        token = secrets.token_urlsafe(18)
        card = _EntryCard(kind=kind, submit=submit, user_id=str(user_id),
                          team_id=str(self._metadata_team_id(metadata) or ""), modal_label=modal_label,
                          expires_at=time.monotonic() + timeout_s)
        card.text = f"*{mrkdwn_escape(title)}*\n{body}"[:3000]

        def _build() -> tuple[str, list]:
            blocks = [
                {"type": "section", "text": {"type": "mrkdwn", "text": card.text}},
                {"type": "actions", "elements": [self._button(
                    t(f"{kind.i18n}.button")[:75], kind.open_action, token, style="primary")]}]
            return title, blocks

        result = await self._send_interactive_prompt(chat_id, metadata, _build, f"send {kind.label} prompt",
                                                     sanitize=False)
        if result.success and result.message_id:
            card.channel_id = str((result.raw_response or {}).get("channel") or chat_id)
            card.message_ts = result.message_id
            cards = self._entry_cards(kind)
            cards[token] = card
            self._trim_oldest_dict_entries(cards, _REQUESTS_MAX)
            # The card retires itself when the request can no longer be answered.
            asyncio.get_running_loop().call_later(
                timeout_s, lambda: asyncio.ensure_future(self._retire_entry_card(kind, token, ended=True)))
        return result

    def _live_entry_card(self, kind: EntryKind, token: str) -> Optional[_EntryCard]:
        card = self._entry_cards(kind).get(token)
        if card is None or time.monotonic() >= card.expires_at:
            return None
        return card

    async def _retire_entry_card(self, kind: EntryKind, token: str, *, ended: bool) -> None:
        card = self._entry_cards(kind).pop(token, None)
        if card is None:
            return
        notice = t(f"{kind.i18n}.ended" if ended else f"{kind.i18n}.submitted_footer")
        await self._finalize_interactive_message(card.channel_id, card.message_ts, card.text, notice,
                                                 kind.label, kind.label, card.team_id or None, sanitize=False)

    @staticmethod
    def _entry_author_matches(card: _EntryCard, user_id: str, team_id: str) -> bool:
        return user_id == card.user_id and (not card.team_id or not team_id or team_id == card.team_id)

    async def _handle_entry_open(self, ack, body, action) -> None:
        """Open the private modal for the card's author; anyone else is told so, privately."""
        kind = next((k for k in KINDS if k.open_action == (action or {}).get("action_id")), VAULT_CODE)
        started = await self._begin_interaction(ack, body, action, kind.label)
        if started is None:
            return
        team_id, _action_id, token, message, _msg_ts, channel_id, _name, user_id = started
        card = self._live_entry_card(kind, token)
        refusal = (f"{kind.i18n}.ended" if card is None
                   else None if self._entry_author_matches(card, user_id, team_id)
                   else f"{kind.i18n}.not_author")
        client = self._get_client(channel_id, team_id=team_id or None)
        if refusal is not None:
            await client.chat_postEphemeral(channel=channel_id, user=user_id, text=t(refusal),
                                            thread_ts=(message or {}).get("thread_ts"))
            return
        view = {
            "type": "modal", "callback_id": kind.view_callback, "private_metadata": token,
            "title": {"type": "plain_text", "text": t(f"{kind.i18n}.input_label")[:_MODAL_TITLE_LIMIT]},
            "submit": {"type": "plain_text", "text": t(f"{kind.i18n}.button")[:_MODAL_TITLE_LIMIT]},
            "blocks": [{
                "type": "input", "block_id": kind.block_id,
                "label": {"type": "plain_text", "text": card.modal_label},
                "element": {"type": "plain_text_input", "action_id": kind.input_action,
                            "max_length": kind.max_length},
            }],
        }
        try:
            await client.views_open(trigger_id=str(body.get("trigger_id") or ""), view=view)
        except Exception:  # health: allow BLE001 -- any Web API failure leaves the card live for a retry
            logger.warning("[Slack] Could not open the %s modal", kind.label, exc_info=True)

    async def _handle_entry_submit(self, ack, body, view) -> None:
        """Hand the modal value to the waiting agent thread; it never becomes a message."""
        kind = next((k for k in KINDS if k.view_callback == (view or {}).get("callback_id")), VAULT_CODE)
        token = str((view or {}).get("private_metadata") or "")
        values = ((view or {}).get("state") or {}).get("values") or {}
        value = str(((values.get(kind.block_id) or {}).get(kind.input_action) or {}).get("value") or "").strip()
        user_id = str(((body or {}).get("user") or {}).get("id") or "")
        team_id = self._event_team_id({}, body or {}, install_set=getattr(self, "_team_clients", None))
        card = self._live_entry_card(kind, token)
        if card is None or not self._entry_author_matches(card, user_id, team_id):
            key = f"{kind.i18n}.ended" if card is None else f"{kind.i18n}.not_author"
            await ack(response_action="errors", errors={kind.block_id: t(key)})
            return
        accepted = bool(value) and card.submit(value)
        value = ""
        if not accepted:
            await ack(response_action="errors", errors={kind.block_id: t(f"{kind.i18n}.ended")})
            await self._retire_entry_card(kind, token, ended=True)
            return
        await ack()
        await self._retire_entry_card(kind, token, ended=False)
