"""Feishu's native model picker; resolution and persistence belong to the gateway."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from typing import Any

from gateway.platforms.base import SendResult

logger = logging.getLogger(__name__)
_PICKER_TTL = 15 * 60
_MAX_PICKERS = 128


def _card(title: str, text: str, actions: list | None = None) -> dict:
    elements = [{"tag": "markdown", "content": text}]
    for action in actions or []:
        elements.append({"tag": "action", "actions": [action]})
    return {"config": {"wide_screen_mode": True},
            "header": {"title": {"tag": "plain_text", "content": title}, "template": "blue"},
            "elements": elements}


def _button(label: str, value: dict, style: str = "default") -> dict:
    return {"tag": "button", "text": {"tag": "plain_text", "content": label},
            "type": style, "value": value}


def _menu(label: str, value: dict, options: list, current: str | None = None) -> dict:
    menu = {"tag": "select_static", "placeholder": {"tag": "plain_text", "content": label},
            "value": value, "options": [
                {"text": {"tag": "plain_text", "content": text[:100]}, "value": key}
                for key, text in options]}
    if current is not None:
        menu["initial_option"] = current
    return menu


def _disambiguate_duplicate_labels(catalog: list) -> None:
    """Two credential groups can share one display name (e.g. one relay, several keys).

    Tag the numbered route (slug ``...-2``) so the provider menu never shows twin entries."""
    counts: dict = {}
    for item in catalog:
        counts[item["name"]] = counts.get(item["name"], 0) + 1
    if not any(count > 1 for count in counts.values()):
        return
    for item in catalog:
        if counts[item["name"]] > 1:
            match = re.search(r"-(\d+)$", item["slug"])
            if match:
                item["name"] = f"{item['name']}（{match.group(1)}）"


def _picker_text(state: dict, key: str, **fmt) -> str:
    """Card copy from the i18n catalogs, in the language the card was sent in (state["lang"])."""
    from agent.i18n import t
    return t(f"gateway.model.picker.{key}", lang=(state or {}).get("lang"), **fmt)


class FeishuModelPickerMixin:
    supports_model_picker_context = True

    def _init_model_picker(self) -> None:
        self._model_picker_state: dict[str, dict[str, Any]] = {}
        # The SDK callback runs on its own thread; the commit runs on the adapter loop.
        self._model_picker_lock = threading.RLock()

    async def send_model_picker(
        self, chat_id: str, providers: list, current_model: str, current_provider: str,
        session_key: str, on_model_selected, metadata: dict | None = None,
        *, picker_context: dict | None = None,
    ) -> SendResult:
        if not self._client:
            return SendResult(success=False, error="Not connected")
        context = dict(picker_context or {})
        if not context.get("user_id") or not callable(on_model_selected):
            return SendResult(success=False, error="Model picker requires its requesting user")
        catalog = []
        for provider in providers or []:
            models = list(dict.fromkeys(m for m in provider.get("models", []) if isinstance(m, str) and m))[:20]
            if provider.get("slug") and models:
                catalog.append({"slug": provider["slug"], "name": provider.get("name") or "Provider",
                                "models": models, "total_models": provider.get("total_models", len(models))})
        if not catalog:
            return SendResult(success=False, error="No authenticated models are available")
        _disambiguate_duplicate_labels(catalog)
        try:  # the card keeps the language it was sent with for its whole lifetime
            from agent.i18n import get_language
            picker_language = get_language()
        except Exception:
            picker_language = None
        picker_id = uuid.uuid4().hex
        state = {"picker_id": picker_id, "session_key": session_key, "chat_id": chat_id,
                 "message_id": "", "owner_profile": self._owner_profile, "context": context,
                 "providers": catalog, "current_model": current_model,
                 "current_provider": current_provider, "provider_index": None, "model_index": None,
                 "effort": "", "on_model_selected": on_model_selected,
                 "metadata": dict(metadata or {}), "created": time.monotonic(), "pending": False,
                 "revision": 0, "update_lock": asyncio.Lock(), "guard_warning": "",
                 "lang": picker_language}
        try:
            response = await self._feishu_send_with_retry(
                chat_id=chat_id, msg_type="interactive",
                payload=json.dumps(self._model_picker_card(state), ensure_ascii=False),
                reply_to=None, metadata=metadata,
            )
            result = self._finalize_send_result(response, "Model picker send failed")
        except Exception:
            logger.warning("[Feishu] Model picker send failed", exc_info=True)
            return SendResult(success=False, error="Model picker send failed")
        if result.success and result.message_id:
            state["message_id"] = result.message_id
            with self._model_picker_lock:
                now = time.monotonic()
                self._model_picker_state = {
                    key: value for key, value in self._model_picker_state.items()
                    if now - value["created"] < _PICKER_TTL and value["session_key"] != session_key}
                while len(self._model_picker_state) >= _MAX_PICKERS:
                    self._model_picker_state.pop(next(iter(self._model_picker_state)))
                self._model_picker_state[picker_id] = state
        elif result.success:
            return SendResult(success=False, error="Model picker response has no message ID")
        return result

    @staticmethod
    def _model_picker_card(state: dict) -> dict:
        from agent.reasoning_effort import effort_display_label
        from hermes_constants import VALID_REASONING_EFFORTS

        def value(action):
            # Only opaque IDs/indices cross the wire, never session IDs or provider endpoints.
            return {"hermes_model_picker": action, "picker_id": state["picker_id"], "revision": state["revision"]}

        scope = _picker_text(state, "scope_session" if not state["context"].get("persist_global")
                             else "scope_profile")
        profile = state["context"].get("profile_label") or state["owner_profile"] or "default"
        text = _picker_text(state, "header", profile=profile, scope=scope,
                            model=state["current_model"] or "unknown") + "\n"
        index = state["provider_index"]
        if state.get("guard_warning") and index is not None:
            # Selection guard (cost / data-policy) fired on Apply: confirm in-card before the
            # switch runs — parity with the typed path's confirm and the other platform pickers'.
            provider = state["providers"][index]
            model_index = state["model_index"]
            model = provider["models"][model_index] if model_index is not None else ""
            text += _picker_text(state, "guard_confirm", warning=state["guard_warning"],
                                 model=model, provider=provider["name"])
            actions = [_button(_picker_text(state, "btn_continue"), value("apply_confirm"), "primary"),
                       _button(_picker_text(state, "btn_back"), value("back"))]
        elif index is None:
            options = [(str(i), p["name"]) for i, p in enumerate(state["providers"])]
            actions = [_menu(_picker_text(state, "menu_provider"), value("provider"), options[start:start + 100])
                       for start in range(0, len(options), 100)]
            text += _picker_text(state, "hint_pick_provider")
        else:
            provider = state["providers"][index]
            model_index = state["model_index"]
            model = provider["models"][model_index] if model_index is not None else ""
            options = [(str(i), m) for i, m in enumerate(provider["models"])]
            actions = [_menu(_picker_text(state, "menu_model"), value("model"), options,
                             str(model_index) if model_index is not None else None)]
            current = state["context"].get("reasoning_effort") or _picker_text(state, "effort_default")
            efforts = [("keep", _picker_text(state, "effort_keep", current=current)),
                       ("none", _picker_text(state, "effort_none"))]
            efforts += [(lv, effort_display_label(lv, provider["slug"], model)) for lv in VALID_REASONING_EFFORTS]
            actions.append(_menu(_picker_text(state, "menu_effort"), value("effort"), efforts, state["effort"] or "keep"))
            text += _picker_text(state, "hint_pick_model", provider=provider["name"])
            if provider["total_models"] > len(provider["models"]):
                text += "\n" + _picker_text(state, "hint_model_limit", count=len(provider["models"]))
            if model:
                actions.append(_button(_picker_text(state, "btn_apply"), value("apply"), "primary"))
            actions.append(_button(_picker_text(state, "btn_back_provider"), value("back")))
        actions.append(_button(_picker_text(state, "btn_cancel"), value("cancel")))
        return _card(_picker_text(state, "title"), text, actions)

    def _handle_model_picker_action(self, *, event, action_value, loop):
        picker_id = action_value.get("picker_id")
        if not isinstance(picker_id, str):
            return self._card_response()
        with self._model_picker_lock:
            state = self._model_picker_state.get(picker_id)
            if not state or state["pending"] or action_value.get("revision") != state["revision"]:
                return self._card_response()
            if time.monotonic() - state["created"] >= _PICKER_TTL:
                self._model_picker_state.pop(picker_id, None)
                return self._card_response()
            context = getattr(event, "context", None)
            operator = getattr(event, "operator", None)
            user_ids = {getattr(operator, name, None) for name in ("open_id", "user_id", "union_id")} - {None, ""}
            expected_ids = {state["context"].get("user_id"), state["context"].get("user_id_alt")} - {None, ""}
            if (not user_ids.intersection(expected_ids)
                    or getattr(context, "open_chat_id", None) != state["chat_id"]
                    or getattr(context, "open_message_id", None) != state["message_id"]
                    or state["owner_profile"] != self._owner_profile):
                return self._card_response()
            if self._is_sender_authorized(state["context"]["user_id"], state["context"].get("chat_type"), state["chat_id"],
                                          thread_id=state["context"].get("thread_id")) is not True:
                return self._card_response()
            action = action_value.get("hermes_model_picker")
            option = getattr(getattr(event, "action", None), "option", None)
            if action in {"provider", "model"}:
                if not isinstance(option, str) or not option.isdecimal():
                    return self._card_response()
                index = int(option)
                if action == "provider":
                    if index >= len(state["providers"]):
                        return self._card_response()
                    state["provider_index"], state["model_index"] = index, None
                else:
                    p_index = state["provider_index"]
                    if p_index is None or index >= len(state["providers"][p_index]["models"]):
                        return self._card_response()
                    state["model_index"] = index
                state["guard_warning"] = ""  # a different pick must be confirmed afresh
            elif action == "effort":
                from hermes_constants import VALID_REASONING_EFFORTS
                if option not in ("keep", "none", *VALID_REASONING_EFFORTS):
                    return self._card_response()
                state["effort"] = "" if option == "keep" else option
            elif action == "back":
                state["provider_index"], state["model_index"] = None, None
                state["guard_warning"] = ""
            elif action == "cancel":
                self._model_picker_state.pop(picker_id, None)
                self._submit_on_loop(loop, self._finish_model_picker(
                    state, _picker_text(state, "cancelled_title"), _picker_text(state, "cancelled_body")))
                return self._card_response()
            elif action in {"apply", "apply_confirm"}:
                if state["provider_index"] is None or state["model_index"] is None:
                    return self._card_response()
                state["pending"] = True
                # "apply" may bounce into the guard-confirm view; "apply_confirm" is that view's
                # own button — never re-guard a selection the user already confirmed.
                if not self._submit_on_loop(
                        loop, self._apply_model_picker(state, check_guard=action == "apply")):
                    state["pending"] = False
                    return self._card_response()
                # Leave the inline card alone: an asynchronous PATCH may finish before the
                # SDK's callback response, which would otherwise overwrite the final result.
                return self._card_response()
            else:
                return self._card_response()
            state["revision"] += 1
            self._submit_on_loop(loop, self._refresh_model_picker(state, state["revision"]))
            return self._card_response()

    async def _patch_model_picker(self, state: dict, card: dict) -> bool:
        # All updates (including completion) share one writer; inline callback
        # cards can arrive after a later PATCH and resurrect obsolete controls.
        try:
            from lark_oapi.api.im.v1 import PatchMessageRequest, PatchMessageRequestBody
            body = PatchMessageRequestBody.builder().content(json.dumps(card, ensure_ascii=False)).build()
            request = PatchMessageRequest.builder().message_id(state["message_id"]).request_body(body).build()
            response = await self._run_blocking(self._client.im.v1.message.patch, request)
            return self._response_succeeded(response)
        except Exception:
            logger.warning("[Feishu] Could not update model picker")
            return False

    async def _refresh_model_picker(self, state: dict, revision: int) -> None:
        async with state["update_lock"]:
            with self._model_picker_lock:
                if (self._model_picker_state.get(state["picker_id"]) is not state
                        or state["pending"] or state["revision"] != revision):
                    return
                card = self._model_picker_card(state)
            if not await self._patch_model_picker(state, card):
                await self.send(state["chat_id"], _picker_text(state, "update_failed"), metadata=state["metadata"])

    async def _finish_model_picker(self, state: dict, title: str, reply: str) -> None:
        async with state["update_lock"]:
            if await self._patch_model_picker(state, _card(title, reply)):
                return
        result = await self.send(state["chat_id"], reply, metadata=state["metadata"])
        if not result.success:
            logger.warning("[Feishu] Could not deliver model picker result")

    async def _model_picker_guard_warning(self, model: str, provider_slug: str) -> str:
        """Selection-guard text (cost / data-policy) for the in-card confirm step — parity with
        the typed path's guard and the Telegram/Discord pickers'. Empty string when clear."""
        try:
            from hermes_cli.model_selection_guards import combined_selection_warning
            # Pricing lookup can hit models.dev on a cache miss — keep it off the event loop.
            warning = await asyncio.to_thread(combined_selection_warning, model, provider=provider_slug)
        except Exception:
            warning = None
        if warning is None:
            return ""
        return f"{warning.title}\n{warning.message}"

    async def _apply_model_picker(self, state: dict, *, check_guard: bool = False) -> None:
        provider = state["providers"][state["provider_index"]]
        model = provider["models"][state["model_index"]]
        if check_guard:
            warning = await self._model_picker_guard_warning(model, provider["slug"])
            if warning:
                with self._model_picker_lock:
                    state["pending"] = False
                    state["guard_warning"] = warning
                    state["revision"] += 1
                    revision = state["revision"]
                await self._refresh_model_picker(state, revision)
                return
        try:
            reply = await state["on_model_selected"](
                state["chat_id"], model, provider["slug"], state["effort"]
            )
            if not isinstance(reply, str) or not reply.strip():
                reply = _picker_text(state, "no_reply")
        except Exception:
            logger.warning("[Feishu] Model picker callback failed", exc_info=True)
            reply = _picker_text(state, "switch_failed")
        finally:
            with self._model_picker_lock:
                self._model_picker_state.pop(state["picker_id"], None)
        await self._finish_model_picker(state, _picker_text(state, "result_title"), reply)
