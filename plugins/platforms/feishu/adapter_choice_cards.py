"""Feishu clarify cards and the finite-choice picker card.

Mixed into ``FeishuAdapter``: the ``send_clarify`` / ``retire_clarify_card`` / ``send_choice_picker``
capabilities from ``BasePlatformAdapter`` plus their card-action handlers. The card builders,
``_send_interactive_card``, ``_validate_card_action`` and ``_card_response`` live on the facade, so
every click passes the same operator authorization and chat check as an exec-approval click before
it answers anything.
"""

from __future__ import annotations

import itertools
import json
import logging
from typing import Any, Optional

from agent.i18n import t
from gateway.platforms.base import SendResult

logger = logging.getLogger(__name__)

_BUTTON_LABEL_MAX = 40


class FeishuChoiceCardsMixin:
    """Clarify and choice-picker cards for ``FeishuAdapter`` (needs its card/state helpers)."""

    # Card-action value key -> handler; the first key a click's value carries routes it.
    _CARD_ACTION_ROUTES = (
        ("hermes_action", "_handle_approval_card_action"),
        ("hermes_update_prompt_action", "_handle_update_prompt_card_action"),
        ("hermes_clarify_action", "_handle_clarify_card_action"),
        ("hermes_choice_picker_action", "_handle_choice_picker_card_action"),
    )

    def _init_choice_card_state(self) -> None:
        # clarify_id -> {session_key, message_id, chat_id, question}
        self._clarify_state: dict[str, dict[str, str]] = {}
        # picker_id -> {session_key, message_id, chat_id, body, choices, on_choice_selected, metadata}
        self._choice_picker_state: dict[int, dict[str, Any]] = {}
        self._choice_picker_counter = itertools.count(1)

    # --- clarify ---
    async def send_clarify(
        self, chat_id: str, question: str, choices: Optional[list], clarify_id: str,
        session_key: str, metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Multiple choice: a card with one numbered button per option plus "Other" (flips the entry
        to text capture). Open-ended: the base text prompt, answered by the next message."""
        if not choices:
            return await super().send_clarify(
                chat_id=chat_id, question=question, choices=choices, clarify_id=clarify_id,
                session_key=session_key, metadata=metadata)
        if not self._client:
            return SendResult(success=False, error="Not connected")
        from plugins.platforms.feishu.adapter import _card_button
        from tools.clarify_gateway import get_pending, mark_awaiting_text

        lines = [t("gateway.clarify.option_line", index=i, choice=c) for i, c in enumerate(choices, start=1)]
        body = f"{question}\n\n" + "\n".join(lines)
        entry = get_pending(clarify_id)
        if entry is not None and entry.multi_select:
            # A tap answers one option; several (or a free-form answer, as the hint says) are typed,
            # so typed text must be captured as the answer like the base text prompt does.
            body += "\n\n" + t("gateway.clarify.hint_multi")
            mark_awaiting_text(clarify_id)
        actions = [
            _card_button(str(i), "primary" if i == 1 else "default",
                         {"hermes_clarify_action": str(i - 1), "clarify_id": clarify_id})
            for i in range(1, len(choices) + 1)]
        actions.append(_card_button(t("platform.feishu.clarify.other"), "default",
                                    {"hermes_clarify_action": "other", "clarify_id": clarify_id}))
        card = self._clarify_card(body, "blue", actions=actions)
        try:
            return await self._send_interactive_card(
                chat_id, card, metadata, "send_clarify failed", state_map=self._clarify_state,
                state_id=clarify_id, session_key=session_key, extra_state={"question": str(question)})
        except Exception as exc:
            logger.warning("[Feishu] send_clarify failed: %s", exc, exc_info=True)
            return SendResult(success=False, error=str(exc))

    @staticmethod
    def _clarify_card(body: str, template: str, *, actions: Optional[list] = None) -> dict[str, Any]:
        from plugins.platforms.feishu.adapter import _card
        card = _card(f"❓ {t('platform.feishu.clarify.title')}", template, body, actions=actions)
        # Shared card: PATCH (retire_clarify_card) only reaches every recipient when the sent AND
        # the updated card both declare it.
        card["config"]["update_multi"] = True
        return card

    @classmethod
    def _settled_clarify_card(cls, state: dict[str, str], line: str, template: str) -> dict[str, Any]:
        return cls._clarify_card(f"{state.get('question', '')}\n\n{line}", template)

    def _handle_clarify_card_action(self, *, event: Any, action_value: dict[str, Any], loop: Any) -> Any:
        """Answer a clarify from a tap and return the settled card inline (syncs every client).
        ``resolve_gateway_clarify`` is a locked Event set, safe from the SDK callback thread."""
        clarify_id = str(action_value.get("clarify_id") or "")
        state = self._clarify_state.get(clarify_id)
        if not state:
            logger.debug("[Feishu] Clarify %s already resolved or unknown", clarify_id or "<missing>")
            return self._card_response()
        checked = self._validate_card_action(event=event, state=state, label="clarify", ident=clarify_id)
        if checked is None:
            return self._card_response()
        user_name = checked[2]
        from tools import clarify_gateway

        entry = clarify_gateway.get_pending(clarify_id)
        token = str(action_value.get("hermes_clarify_action") or "")
        if entry is None or entry.session_key != state.get("session_key"):
            # The wait ended (timeout, /new) before the retire reached this card.
            self._clarify_state.pop(clarify_id, None)
            return self._card_response(self._settled_clarify_card(state, t("gateway.clarify.expired"), "grey"))
        if token == "other":
            # State stays so the typed answer's retire_clarify_card can settle this card.
            if not clarify_gateway.mark_awaiting_text(clarify_id):
                return self._card_response(self._settled_clarify_card(state, t("gateway.clarify.expired"), "grey"))
            awaiting = t("platform.feishu.clarify.awaiting_typed", user=user_name)
            return self._card_response(self._settled_clarify_card(state, f"*{awaiting}*", "blue"))
        choices = list(entry.choices or [])
        if not (token.isdigit() and int(token) < len(choices)):
            logger.debug("[Feishu] Clarify %s has invalid choice token=%r", clarify_id, token)
            return self._card_response()
        if self._clarify_state.pop(clarify_id, None) is None:
            return self._card_response()  # a concurrent tap won
        answer = str(choices[int(token)])
        if not clarify_gateway.resolve_gateway_clarify(clarify_id, answer):
            return self._card_response(self._settled_clarify_card(state, t("gateway.clarify.expired"), "grey"))
        logger.info("Feishu clarify button resolved (id=%s, choice=%r, user=%s)", clarify_id, answer, user_name)
        return self._card_response(self._settled_clarify_card(state, f"**{user_name}:** {answer}", "green"))

    async def retire_clarify_card(self, clarify_id: str, notice: str) -> None:
        """Gateway hook for a clarify that ended without a tap (timeout, reset, superseding prose,
        typed answer): rewrite the card button-less. The state pop runs before the first await, so a
        tap racing this call is refused."""
        state = self._clarify_state.pop(clarify_id, None)
        if not state or not state.get("message_id") or not self._client:
            return
        from plugins.platforms.feishu import adapter as facade

        card = self._settled_clarify_card(state, notice, "grey")
        body = facade._sdk_build(facade.PatchMessageRequestBody, content=json.dumps(card, ensure_ascii=False))
        request = facade._sdk_build(facade.PatchMessageRequest, message_id=state["message_id"], request_body=body)
        response = await self._run_blocking(self._client.im.v1.message.patch, request)
        if not self._response_succeeded(response):
            logger.debug("[Feishu] Retiring clarify card %s rejected: code=%s", clarify_id, getattr(response, "code", None))

    # --- finite-choice picker ---
    async def send_choice_picker(
        self, chat_id: str, title: str, choices: list, session_key: str, on_choice_selected,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """One-tap picker card for /reasoning, /fast, etc.; choice dicts are
        ``{"value", "label", "is_current"}``. One live picker per chat, like Telegram."""
        if not self._client:
            return SendResult(success=False, error="Not connected")
        if not choices:
            return SendResult(success=False, error="No choices")
        from plugins.platforms.feishu.adapter import _card, _card_button

        picker_id = next(self._choice_picker_counter)
        actions = []
        for i, choice in enumerate(choices):
            label = str(choice.get("label") or choice.get("value") or "")
            label = self._truncate_preview(f"✓ {label}" if choice.get("is_current") else label, _BUTTON_LABEL_MAX, "…")
            actions.append(_card_button(label, "primary" if choice.get("is_current") else "default",
                                        {"hermes_choice_picker_action": str(i), "picker_id": picker_id}))
        heading, _, rest = str(title or "").partition("\n")
        header = heading.replace("*", "").strip()
        body = rest.strip() or header
        try:
            result = await self._send_interactive_card(
                chat_id, _card(header, "blue", body, actions=actions), metadata, "send_choice_picker failed",
                state_map=self._choice_picker_state, state_id=picker_id, session_key=session_key,
                extra_state={"header": header, "body": body, "choices": choices,
                             "on_choice_selected": on_choice_selected, "metadata": metadata})
        except Exception as exc:
            logger.warning("[Feishu] send_choice_picker failed: %s", exc, exc_info=True)
            return SendResult(success=False, error=str(exc))
        if result.success:
            for stale in [k for k, v in self._choice_picker_state.items() if k != picker_id and v["chat_id"] == chat_id]:
                self._choice_picker_state.pop(stale, None)
        return result

    def _handle_choice_picker_card_action(self, *, event: Any, action_value: dict[str, Any], loop: Any) -> Any:
        """Settle the picker card inline and run the selection's applier on the adapter loop."""
        picker_id = action_value.get("picker_id")
        state = self._choice_picker_state.get(picker_id)
        if not state:
            logger.debug("[Feishu] Choice picker %s already resolved or unknown", picker_id)
            return self._card_response()
        checked = self._validate_card_action(event=event, state=state, label="choice picker", ident=picker_id)
        if checked is None:
            return self._card_response()
        token = str(action_value.get("hermes_choice_picker_action") or "")
        choices = state["choices"]
        if not (token.isdigit() and int(token) < len(choices)):
            return self._card_response()
        if self._choice_picker_state.pop(picker_id, None) is None:
            return self._card_response()  # a concurrent tap won
        choice = choices[int(token)]
        if not self._submit_on_loop(loop, self._apply_choice_picker_selection(state, str(choice.get("value") or ""))):
            self._choice_picker_state[picker_id] = state
            return self._card_response()
        from plugins.platforms.feishu.adapter import _card
        label = str(choice.get("label") or choice.get("value") or "")
        return self._card_response(_card(state["header"], "green", f"{state['body']}\n\n✓ **{label}** — {checked[2]}"))

    def _on_webhook_card_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        """A card tap over HTTP, handled as the SDK delivers it: the button value stays a dict (the
        routes key on it) and the settled card goes back as the response body."""
        data = self._namespace_from_mapping(payload)
        action = getattr(getattr(data, "event", None), "action", None)
        if action is not None:
            action.value = ((payload.get("event") or {}).get("action") or {}).get("value")
        card = getattr(self._on_card_action_trigger(data), "card", None)
        return {"card": {"type": card.type, "data": card.data}} if card is not None else {}

    async def _apply_choice_picker_selection(self, state: dict[str, Any], value: str) -> None:
        """Run the picker's applier and send its confirmation to the chat."""
        try:
            result_text = await state["on_choice_selected"](state["chat_id"], value)
        except Exception as exc:
            logger.exception("[Feishu] Choice picker selection failed: %s", exc)
            result_text = t("platform.feishu.picker.apply_error", error=str(exc))
        if result_text:
            await self.send(state["chat_id"], str(result_text), metadata=state.get("metadata"))
