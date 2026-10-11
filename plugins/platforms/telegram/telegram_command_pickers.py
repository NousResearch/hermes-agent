"""Bound native Telegram command picker admission and callback dispatch."""

import asyncio
import logging

from agent.i18n import t

logger = logging.getLogger("plugins.platforms.telegram.adapter")


class TelegramCommandPickersMixin:
    @staticmethod
    def _bound_picker_callback_state(query, chat_id: str, states: dict) -> tuple:
        """Resolve picker state only for the exact message, topic, and actor."""
        query_message = getattr(query, "message", None)
        query_message_id = getattr(query_message, "message_id", None)
        if query_message_id is None:
            return None, None, "expired"

        state_key = (str(chat_id), str(query_message_id))
        state = states.get(state_key)
        if not state:
            return state_key, None, "expired"

        actual_thread_id = getattr(query_message, "message_thread_id", None)
        actual_thread_id = (
            str(actual_thread_id) if actual_thread_id is not None else None
        )
        if (
            str(state.get("msg_id")) != str(query_message_id)
            or state.get("thread_id") != actual_thread_id
            or not str(state.get("session_key") or "").strip()
        ):
            return state_key, None, "expired"

        owner_user_id = str(state.get("owner_user_id") or "").strip()
        query_user = getattr(query, "from_user", None)
        actor_user_id = str(getattr(query_user, "id", "") or "").strip()
        if not owner_user_id or actor_user_id != owner_user_id:
            return state_key, None, "unauthorized"
        return state_key, state, None

    def _model_picker_callback_actor_source(self, query):
        """Build an actor-bearing source for model-picker authorization only."""
        user = getattr(query, "from_user", None)
        message = getattr(query, "message", None)
        chat = getattr(message, "chat", None)
        user_id = str(getattr(user, "id", "") or "").strip()
        chat_id = str(getattr(message, "chat_id", "") or "").strip()
        if not user_id or not chat_id or chat is None:
            return None

        chat_type = str(getattr(chat, "type", "dm") or "dm").strip().lower()
        thread_id_raw = getattr(message, "message_thread_id", None)
        if chat_type == "private":
            chat_type = "dm"
        elif chat_type == "supergroup":
            chat_type = "forum" if thread_id_raw is not None else "group"
        user_name = (
            str(
                getattr(user, "username", "")
                or getattr(user, "full_name", "")
                or getattr(user, "first_name", "")
                or ""
            ).strip()
            or None
        )
        return self.build_source(
            chat_id=chat_id, chat_type=chat_type, user_id=user_id, user_name=user_name,
            thread_id=str(thread_id_raw) if thread_id_raw is not None else None,
        )

    def _is_model_picker_callback_authorized(self, query) -> bool:
        """Apply Telegram intake auth and the runner's /model slash policy."""
        source = self._model_picker_callback_actor_source(query)
        if source is None:
            return False
        if not self._is_callback_user_authorized(
            source.user_id,
            chat_id=source.chat_id,
            chat_type=source.chat_type,
            thread_id=source.thread_id,
            user_name=source.user_name,
        ):
            return False

        slash_check = getattr(self, "_slash_access_check", None)
        if not callable(slash_check):
            return False
        try:
            return slash_check(source, "model") is None
        except Exception:
            logger.debug("[Telegram] Model picker slash-access check failed", exc_info=True)
            return False

    async def _handle_model_picker_callback(self, query, data: str, chat_id: str) -> None:
        """Handle model picker callbacks (mp:/mpg:/mpv:/mm:/mc:/mb/mx/mg:)."""
        from plugins.platforms.telegram.adapter import (
            InlineKeyboardButton, InlineKeyboardMarkup, ParseMode, _toast,
        )

        bound = await self._authorized_model_picker_state(query, chat_id)
        if bound is None:
            return
        state_key, state = bound
        get_label = self._provider_get_label()
        if data.startswith("mp:"):  # provider selected: show model buttons (page 0)
            provider_slug = data[3:]
            provider = next((p for p in state["providers"] if p["slug"] == provider_slug), None)
            if not provider:
                await query.answer(text=_toast("platform.telegram.picker.provider_not_found"))
                return
            state["selected_provider"] = provider_slug
            state["selected_provider_name"] = provider.get("name", provider_slug)
            state["model_list"] = provider.get("models", [])
            await self._picker_show_models(query, state, 0)
        elif data.startswith("mg:"):  # model page navigation
            page = await self._parse_page(query, data[3:])
            if page is not None:
                await self._picker_show_models(query, state, page)
        elif data.startswith("mpv:"):  # provider page navigation
            page = await self._parse_page(query, data[4:])
            if page is not None:
                state["provider_page"] = page
                await self._picker_show_providers(query, state, page, get_label)
        elif data.startswith("mc:"):  # expensive model confirmed: perform the switch
            sel = await self._picker_selection(query, state, data[3:])
            if sel is not None:
                _idx, model_id, provider_slug, callback = sel
                await self._picker_switch(query, chat_id, model_id, provider_slug, callback)
        elif data.startswith("mm:"):  # model selected: warn if expensive, else perform the switch
            sel = await self._picker_selection(query, state, data[3:])
            if sel is None:
                return
            idx, model_id, provider_slug, callback = sel
            try:
                from hermes_cli.model_selection_guards import combined_selection_warning
                # Pricing lookup may hit models.dev on a cache miss — keep it off the event loop.
                warning = await asyncio.to_thread(combined_selection_warning, model_id, provider=provider_slug)
            except Exception:
                logger.debug("Model selection warning lookup failed", exc_info=True)
                warning = None
            if warning is not None:
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton(t("platform.telegram.picker.switch_anyway"), callback_data=f"mc:{idx}")], self._picker_back_cancel_row()])
                await query.edit_message_text(
                    text=self.format_message(f"⚠ *{warning.title}*\n\n{warning.message}"),
                    parse_mode=ParseMode.MARKDOWN_V2, reply_markup=keyboard)
                await query.answer(text=_toast("platform.telegram.picker.confirm_selection"))
                return
            if self._model_picker_state.get(state_key) is not state:
                await query.answer(text="Picker expired — use /model again.")
                return
            await self._picker_switch(query, chat_id, model_id, provider_slug, callback)
        elif data.startswith("mpg:"):  # provider group selected: show member providers
            group_id = data[4:]
            try:
                from hermes_cli.models_catalog_static import PROVIDER_GROUPS
                _label, _desc, member_slugs = PROVIDER_GROUPS.get(group_id, ("", "", []))
            except Exception:
                logger.debug("Model provider group lookup failed", exc_info=True)
                _label, member_slugs = "", []
            by_slug = {p["slug"]: p for p in state["providers"]}
            members = [by_slug[m] for m in member_slugs if m in by_slug]
            if not members:
                await query.answer(text=_toast("platform.telegram.picker.group_not_found"))
                return
            rows = self._rows_of_two([self._provider_button(p) for p in members])
            rows.append(self._picker_back_cancel_row())
            await self._picker_edit(
                query,
                f"⚙ *{t('platform.telegram.picker.title')}*\n\n"
                f"{t('platform.telegram.picker.provider_family', family=f'*{_label or group_id}*')}\n\n"
                f"{t('platform.telegram.picker.select_provider')}",
                InlineKeyboardMarkup(rows))
        elif data == "mb":  # back to provider list (folds groups)
            await self._picker_show_providers(query, state, int(state.get("provider_page", 0) or 0), get_label)
        elif data == "mx":
            self._model_picker_state.pop((str(chat_id), str(query.message.message_id)), None)
            await query.edit_message_text(text=t("platform.telegram.picker.cancelled"), reply_markup=None)
            await query.answer()
        else:
            await query.answer()  # e.g. page-counter button "mx:noop"

    async def _authorized_model_picker_state(self, query, chat_id: str):
        """Bind and authorize a native model picker before dispatching its action."""
        from plugins.platforms.telegram.adapter import _toast

        state_key, state, binding_error = self._bound_picker_callback_state(query, chat_id, self._model_picker_state)
        if binding_error == "unauthorized":
            await query.answer(text="⛔ You are not authorized to change this setting.")
            return None
        if not state or state_key is None:
            await query.answer(text=_toast("platform.telegram.picker.expired_model"))
            return None
        if not self._is_model_picker_callback_authorized(query):
            await query.answer(text="⛔ You are not authorized to change this setting.")
            return None
        return state_key, state
