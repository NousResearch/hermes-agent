"""Telegram prompts methods; runtime dependencies remain on the adapter facade."""

from typing import Any, Dict, Optional
from gateway.platforms.base import SendResult
try:
    from telegram import Message, Update, InlineKeyboardButton
    from telegram.ext import ContextTypes
except ImportError:
    Message = Update = InlineKeyboardButton = Any
    class ContextTypes:
        DEFAULT_TYPE = Any


class TelegramPromptsMixin:
    async def _send_prompt(self, what: str, chat_id: str, metadata: Optional[Dict[str, Any]], build, *,
                           parse_mode: Any = None, thread_id: Any = None, reply_to_mode: Any = None) -> SendResult:
        """Shared control-prompt shell: not-connected guard, ``build()`` → ``(text, keyboard, on_sent)`` (or a
        SendResult to return as-is), routed send, state hook, redacted failure log."""
        from . import adapter as _adapter

        if not self._bot:
            return _adapter.SendResult(success=False, error="Not connected")
        try:
            built = build()
            if isinstance(built, _adapter.SendResult):
                return built
            text, keyboard, on_sent = built
            msg = await self._send_control_message(
                chat_id, text, parse_mode=parse_mode if parse_mode is not None else _adapter.ParseMode.MARKDOWN_V2,
                reply_markup=keyboard, thread_id=thread_id, metadata=metadata, reply_to_mode=reply_to_mode)
            if on_sent is not None:
                on_sent(msg)
            return _adapter.SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            _adapter.logger.warning("[%s] %s failed: %s", self.name, what, _adapter._redact_telegram_error_text(e))
            return _adapter.SendResult(success=False, error=_adapter._redact_telegram_error_text(e))

    @staticmethod
    def _rows_of_two(buttons: list) -> list:
        """2-per-row layout keeps labels readable on mobile (a 4-button row truncates)."""
        return [buttons[i:i + 2] for i in range(0, len(buttons), 2)]

    async def send_update_prompt(
        self, chat_id: str, prompt: str, default: str = "", session_key: str = "", metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send an inline-keyboard Yes/No prompt for the gateway ``/update`` watcher."""
        from . import adapter as _adapter

        def build():
            default_hint = f" (default: {default})" if default else ""
            text = self.format_message(f"⚕ *Update needs your input:*\n\n{prompt}{default_hint}")
            keyboard = _adapter.InlineKeyboardMarkup([[
                _adapter.InlineKeyboardButton("✓ Yes", callback_data="update_prompt:y"),
                _adapter.InlineKeyboardButton("✗ No", callback_data="update_prompt:n")]])
            return text, keyboard, None
        return await self._send_prompt(
            "send_update_prompt", chat_id, metadata, build, thread_id=self._metadata_thread_id(metadata), reply_to_mode=self._reply_to_mode)

    def _ea_escape(self, text: str) -> str:
        from . import adapter as _adapter

        return _adapter._html.escape(text)

    async def send_exec_approval(
        self, chat_id: str, command: str, session_key: str, description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None, allow_permanent: bool = True, allow_session: bool = True,
        smart_denied: bool = False) -> SendResult:
        """Send an inline-keyboard approval prompt; buttons call ``resolve_gateway_approval()`` like the
        text ``/approve`` flow."""
        from . import adapter as _adapter

        def build():
            text = self._format_exec_approval(command, description, smart_denied)
            # Short monotonic ids in callback_data map back to session_key.
            import itertools
            if not hasattr(self, "_approval_counter"):
                self._approval_counter = itertools.count(1)
            approval_id = next(self._approval_counter)
            buttons = [_adapter.InlineKeyboardButton("✅ Allow Once", callback_data=f"ea:once:{approval_id}")]
            if not smart_denied and allow_session:
                buttons.append(_adapter.InlineKeyboardButton("✅ Session", callback_data=f"ea:session:{approval_id}"))
                if allow_permanent:
                    buttons.append(_adapter.InlineKeyboardButton("✅ Always", callback_data=f"ea:always:{approval_id}"))
            buttons.append(_adapter.InlineKeyboardButton("❌ Deny", callback_data=f"ea:deny:{approval_id}"))
            return text, _adapter.InlineKeyboardMarkup(
                self._rows_of_two(buttons)), lambda msg: self._approval_state.__setitem__(approval_id, session_key)
        return await self._send_prompt(
            "send_exec_approval", chat_id, metadata, build, parse_mode=_adapter.ParseMode.HTML,
            thread_id=self._metadata_thread_id(metadata), reply_to_mode=self._reply_to_mode)

    async def send_slash_confirm(
        self, chat_id: str, title: str, message: str, session_key: str, confirm_id: str,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Render a three-button slash-command confirmation prompt."""
        from . import adapter as _adapter

        def build():
            keyboard = _adapter.InlineKeyboardMarkup([
                [
                    _adapter.InlineKeyboardButton("✅ Approve Once", callback_data=f"sc:once:{confirm_id}"),
                    _adapter.InlineKeyboardButton("🔒 Always Approve", callback_data=f"sc:always:{confirm_id}")],
                [_adapter.InlineKeyboardButton("❌ Cancel", callback_data=f"sc:cancel:{confirm_id}")],
           ])
            preview = self.format_message(self._truncate_preview(message, 3800))
            return preview, keyboard, lambda msg: self._slash_confirm_state.__setitem__(confirm_id, session_key)
        return await self._send_prompt(
            "send_slash_confirm", chat_id, metadata, build, thread_id=self._metadata_thread_id(metadata), reply_to_mode=self._reply_to_mode)

    async def send_clarify(
        self, chat_id: str, question: str, choices: Optional[list], clarify_id: str, session_key: str,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Render a clarify prompt: numbered buttons per choice plus "✏️ Other (type answer)" (flips to
        text-capture mode); without choices, plain question and the gateway text-intercept captures."""
        from . import adapter as _adapter

        def build():
            text = f"❓ {_adapter._html.escape(question)}"
            keyboard = None
            if choices:
                # Full option text in the body (mobile truncates button labels); buttons keep numeric labels.
                text += "\n\n" + "\n".join(f"{i + 1}. {_adapter._html.escape(str(c))}" for i, c in enumerate(choices))
                # Telegram caps callback_data at 64 bytes; keep "cl:<id>:<idx>" short.
                rows = [[_adapter.InlineKeyboardButton(str(idx + 1), callback_data=f"cl:{clarify_id}:{idx}")] for idx in range(len(choices))]
                rows.append([_adapter.InlineKeyboardButton("✏️ Other (type answer)", callback_data=f"cl:{clarify_id}:other")])
                keyboard = _adapter.InlineKeyboardMarkup(rows)
            return text, keyboard, lambda msg: self._clarify_state.__setitem__(clarify_id, session_key)
        return await self._send_prompt(
            "send_clarify", chat_id, metadata, build, parse_mode=_adapter.ParseMode.HTML, thread_id=self._metadata_thread_id(metadata))

    @staticmethod
    def _provider_get_label():
        from . import adapter as _adapter

        try:
            from hermes_cli.providers import get_label
        except ImportError:
            def get_label(slug):
                return slug
        return get_label

    async def send_model_picker(
        self, chat_id: str, providers: list, current_model: str, current_provider: str, session_key: str,
        on_model_selected, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send an inline-keyboard model picker: provider → model drill-down, edited in place."""
        from . import adapter as _adapter

        def build():
            keyboard, provider_page_info = self._build_provider_keyboard(providers, 0)
            text = self.format_message(
                self._provider_list_text(current_model, self._provider_get_label()(current_provider), provider_page_info)
            )

            def _remember(msg):
                self._model_picker_state[str(chat_id)] = {
                    "msg_id": msg.message_id, "providers": providers, "session_key": session_key, "on_model_selected": on_model_selected,
                    "current_model": current_model, "current_provider": current_provider, "provider_page": 0}
            return text, keyboard, _remember
        return await self._send_prompt(
            "send_model_picker", chat_id, metadata, build, thread_id=metadata.get("thread_id") if metadata else None,
            reply_to_mode=self._reply_to_mode)

    supports_choice_pages = True
    choice_pages_edit_in_place = True

    async def send_choice_picker(
        self,
        chat_id: str,
        title: str,
        choices: list,
        session_key: str,
        on_choice_selected,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a flat inline-keyboard choice picker (one tap → one value).

        Generic single-level companion to ``send_model_picker`` used by
        `/reasoning`, `/fast`, and any future finite-choice command. Each
        choice dict: ``{"value": str, "label": str, "is_current": bool}``.
        """
        from . import adapter as _adapter
        from . import choice_picker as _choice_picker

        return await _choice_picker.send_choice_picker(
            self,
            chat_id,
            title,
            choices,
            session_key,
            on_choice_selected,
            metadata,
            inline_keyboard_button=_adapter.InlineKeyboardButton,
            inline_keyboard_markup=_adapter.InlineKeyboardMarkup,
            parse_mode=_adapter.ParseMode,
            normalize_chat_id=_adapter.normalize_telegram_chat_id,
            redact_error=_adapter._redact_telegram_error_text,
            logger=_adapter.logger,
        )

    async def _edit_result_text(self, query, result_text: str) -> None:
        """Replace a picker message with ``result_text`` (MarkdownV2, then plain, then give up), keyboard removed."""
        from . import adapter as _adapter

        try:
            await query.edit_message_text(text=self.format_message(result_text), parse_mode=_adapter.ParseMode.MARKDOWN_V2, reply_markup=None)
        except Exception:
            with _adapter.contextlib.suppress(Exception):
                await query.edit_message_text(text=result_text, parse_mode=None, reply_markup=None)

    async def _handle_choice_picker_callback(
        self, query, data: str, chat_id: str
    ) -> None:
        """Handle choice picker button taps (cp:<index>)."""
        from . import adapter as _adapter
        from . import choice_picker as _choice_picker

        await _choice_picker.handle_choice_picker_callback(
            self,
            query,
            data,
            chat_id,
            parse_mode=_adapter.ParseMode,
            logger=_adapter.logger,
            inline_keyboard_button=_adapter.InlineKeyboardButton,
            inline_keyboard_markup=_adapter.InlineKeyboardMarkup,
        )

    @staticmethod
    def _provider_button(p: dict) -> "InlineKeyboardButton":
        from . import adapter as _adapter

        count = p.get("total_models", len(p.get("models", [])))
        label = f"{p['name']} ({count})"
        if p.get("is_current"):
            label = f"✓ {label}"
        return _adapter.InlineKeyboardButton(label, callback_data=f"mp:{p['slug']}")

    @staticmethod
    def _picker_nav_row(page: int, total_pages: int, prefix: str) -> list:
        """``◀ Prev | n/N | Next ▶`` row (``prefix`` = ``mpv``/``mg`` page callback)."""
        from . import adapter as _adapter

        nav: list = []
        if page > 0:
            nav.append(_adapter.InlineKeyboardButton("◀ Prev", callback_data=f"{prefix}:{page - 1}"))
        nav.append(_adapter.InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="mx:noop"))
        if page < total_pages - 1:
            nav.append(_adapter.InlineKeyboardButton("Next ▶", callback_data=f"{prefix}:{page + 1}"))
        return nav

    @staticmethod
    def _picker_back_cancel_row() -> list:
        from . import adapter as _adapter

        return [_adapter.InlineKeyboardButton("◀ Back", callback_data="mb"), _adapter.InlineKeyboardButton("✗ Cancel", callback_data="mx")]

    def _paged_keyboard(self, buttons: list, page_meta: dict, nav_prefix: str, tail_row: list) -> tuple:
        from . import adapter as _adapter

        rows = self._rows_of_two(buttons)
        if page_meta["total_pages"] > 1:
            rows.append(self._picker_nav_row(page_meta["page"], page_meta["total_pages"], nav_prefix))
        rows.append(tail_row)
        return _adapter.InlineKeyboardMarkup(rows), page_meta["page_info"]

    def _build_provider_keyboard(self, providers: list, page: int = 0) -> tuple:
        """Paginated top-level provider keyboard folding provider families (Kimi/Moonshot, MiniMax, xAI…)
        into one ``mpg:<gid>`` button via the shared ``group_providers`` fold; singles are ``mp:<slug>``."""
        from . import adapter as _adapter

        try:
            from hermes_cli.models_catalog_static import group_providers
        except Exception:
            group_providers = None
        by_slug = {p.get("slug"): p for p in providers}
        buttons: list = []
        if group_providers is not None:
            for row in group_providers([p.get("slug") for p in providers]):
                if row["kind"] == "group":
                    members = [by_slug[m] for m in row["members"] if m in by_slug]
                    count = sum(m.get("total_models", len(m.get("models", []))) for m in members)
                    label = f"{row['label']} ▸ ({count})"
                    if any(m.get("is_current") for m in members):
                        label = f"✓ {label}"
                    buttons.append(_adapter.InlineKeyboardButton(label, callback_data=f"mpg:{row['group_id']}"))
                else:
                    p = by_slug.get(row["slug"])
                    if p is not None:
                        buttons.append(self._provider_button(p))
        else:
            buttons = [self._provider_button(p) for p in providers]
        page_buttons, page_meta = self._format_choice_page(buttons, page, self._PROVIDER_PAGE_SIZE)
        return self._paged_keyboard(page_buttons, page_meta, "mpv", [_adapter.InlineKeyboardButton("✗ Cancel", callback_data="mx")])

    def _build_model_keyboard(self, models: list, page: int) -> tuple:
        """Build paginated model buttons. Returns (keyboard, page_info_text)."""
        from . import adapter as _adapter

        page_models, page_meta = self._format_choice_page(models, page, self._MODEL_PAGE_SIZE)
        start = page_meta["start"]
        buttons: list = []
        for i, model_id in enumerate(page_models):
            short = model_id.split("/")[-1] if "/" in model_id else model_id
            if len(short) > 38:
                short = short[:35] + "..."
            buttons.append(_adapter.InlineKeyboardButton(short, callback_data=f"mm:{start + i}"))
        return self._paged_keyboard(buttons, page_meta, "mg", self._picker_back_cancel_row())

    async def _picker_edit(self, query, text_md: str, keyboard) -> None:
        """Re-render the picker message in place (MarkdownV2) and ack the tap."""
        from . import adapter as _adapter

        await query.edit_message_text(text=self.format_message(text_md), parse_mode=_adapter.ParseMode.MARKDOWN_V2, reply_markup=keyboard)
        await query.answer()

    async def _picker_show_models(self, query, state: dict, page: int) -> None:
        """Render the model page for the provider currently selected in ``state``."""
        models = state.get("model_list", [])
        state["model_page"] = page
        keyboard, page_info = self._build_model_keyboard(models, page)
        pname = state.get("selected_provider_name", "")
        provider_slug = state.get("selected_provider", "")
        provider = next((p for p in state["providers"] if p["slug"] == provider_slug), None)
        total = provider.get("total_models", len(models)) if provider else len(models)
        shown = len(models)
        extra = f"\n_{total - shown} more available — type `/model <name>` directly_" if total > shown else ""
        await self._picker_edit(query, f"⚙ *Model Configuration*\n\nProvider: *{pname}*{page_info}\nSelect a model:{extra}", keyboard)

    @staticmethod
    def _provider_list_text(current_model: str, provider_label: str, page_info: str) -> str:
        return (f"⚙ *Model Configuration*\n\nCurrent model: `{current_model or 'unknown'}`\n"
                f"Provider: {provider_label}\n\nSelect a provider:{page_info}")

    async def _picker_show_providers(self, query, state: dict, page: int, get_label) -> None:
        """Render the (folded, paginated) provider list."""
        keyboard, provider_page_info = self._build_provider_keyboard(state["providers"], page)
        try:
            provider_label = get_label(state["current_provider"])
        except Exception:
            provider_label = state["current_provider"]
        await self._picker_edit(query, self._provider_list_text(state["current_model"], provider_label, provider_page_info), keyboard)

    async def _picker_selection(self, query, state: dict, raw_idx: str) -> Optional[tuple]:
        """Resolve ``mm:``/``mc:`` index → ``(idx, model_id, provider_slug, callback)``; answers + None on error."""
        from . import adapter as _adapter

        try:
            idx = int(raw_idx)
        except ValueError:
            await query.answer(text="Invalid selection.")
            return None
        model_list = state.get("model_list", [])
        if idx < 0 or idx >= len(model_list):
            await query.answer(text="Invalid model index.")
            return None
        callback = state.get("on_model_selected")
        if not callback:
            await query.answer(text="Picker expired.")
            return None
        return idx, model_list[idx], state.get("selected_provider", ""), callback

    async def _picker_switch(self, query, chat_id: str, model_id: str, provider_slug: str, callback) -> None:
        """Perform the model switch, render the result, and drop the picker state."""
        from . import adapter as _adapter

        switch_failed = False
        try:
            result_text = await callback(chat_id, model_id, provider_slug)
        except Exception as exc:
            _adapter.logger.error("Model picker switch failed: %s", exc)
            result_text = f"Error switching model: {exc}"
            switch_failed = True
        await self._edit_result_text(query, result_text)
        await query.answer(text="Switch failed." if switch_failed else "Model switched!")
        self._model_picker_state.pop(chat_id, None)

    @staticmethod
    async def _parse_page(query, raw: str) -> Optional[int]:
        from . import adapter as _adapter

        try:
            return int(raw)
        except ValueError:
            await query.answer(text="Invalid page.")
            return None

    async def _handle_model_picker_callback(self, query, data: str, chat_id: str) -> None:
        """Handle model picker callbacks (mp:/mpg:/mpv:/mm:/mc:/mb/mx/mg:)."""
        from . import adapter as _adapter

        state = self._model_picker_state.get(chat_id)
        if not state:
            await query.answer(text="Picker expired — use /model again.")
            return
        get_label = self._provider_get_label()
        if data.startswith("mp:"):  # provider selected: show model buttons (page 0)
            provider_slug = data[3:]
            provider = next((p for p in state["providers"] if p["slug"] == provider_slug), None)
            if not provider:
                await query.answer(text="Provider not found.")
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
                warning = await _adapter.asyncio.to_thread(combined_selection_warning, model_id, provider=provider_slug)
            except Exception:
                warning = None
            if warning is not None:
                keyboard = _adapter.InlineKeyboardMarkup([
                    [_adapter.InlineKeyboardButton("Switch anyway", callback_data=f"mc:{idx}")], self._picker_back_cancel_row()])
                await query.edit_message_text(
                    text=self.format_message(f"⚠ *{warning.title}*\n\n{warning.message}"),
                    parse_mode=_adapter.ParseMode.MARKDOWN_V2, reply_markup=keyboard)
                await query.answer(text="Confirm model selection")
                return
            await self._picker_switch(query, chat_id, model_id, provider_slug, callback)
        elif data.startswith("mpg:"):  # provider group selected: show member providers
            group_id = data[4:]
            try:
                from hermes_cli.models_catalog_static import PROVIDER_GROUPS
                _label, _desc, member_slugs = PROVIDER_GROUPS.get(group_id, ("", "", []))
            except Exception:
                _label, member_slugs = "", []
            by_slug = {p["slug"]: p for p in state["providers"]}
            members = [by_slug[m] for m in member_slugs if m in by_slug]
            if not members:
                await query.answer(text="Group not found.")
                return
            rows = self._rows_of_two([self._provider_button(p) for p in members])
            rows.append(self._picker_back_cancel_row())
            await self._picker_edit(
                query, f"⚙ *Model Configuration*\n\nProvider family: *{_label or group_id}*\n\nSelect a provider:",
                _adapter.InlineKeyboardMarkup(rows))
        elif data == "mb":  # back to provider list (folds groups)
            await self._picker_show_providers(query, state, int(state.get("provider_page", 0) or 0), get_label)
        elif data == "mx":
            self._model_picker_state.pop(chat_id, None)
            await query.edit_message_text(text="Model selection cancelled.", reply_markup=None)
            await query.answer()
        else:
            await query.answer()  # e.g. page-counter button "mx:noop"

    async def _notify_clarify_expired(self, query, user_display: str) -> None:
        """Tell the user a clarify tap arrived too late (entry evicted or gateway restarted) — otherwise
        the tap leaves a misleading ✓ the agent never sees."""
        from . import adapter as _adapter

        with _adapter.contextlib.suppress(Exception):
            await query.answer(text="⚠️ This prompt expired — please /retry.")
        await self._edit_html_quiet(
            query, f"❓ {_adapter._html.escape(query.message.text or '')}\n\n<i>⚠️ This question expired or the session reset — please /retry.</i>")

    @staticmethod
    async def _edit_html_quiet(query, text: str) -> None:
        """HTML edit with the keyboard removed; failures ignored (non-fatal)."""
        from . import adapter as _adapter

        with _adapter.contextlib.suppress(Exception):
            await query.edit_message_text(text=text, parse_mode=_adapter.ParseMode.HTML, reply_markup=None)

    async def _edit_md_quiet(self, query, text_md: str) -> None:
        """MarkdownV2 edit with the keyboard removed; failures ignored (non-fatal)."""
        from . import adapter as _adapter

        with _adapter.contextlib.suppress(Exception):
            await query.edit_message_text(text=self.format_message(text_md), parse_mode=_adapter.ParseMode.MARKDOWN_V2, reply_markup=None)

    async def _handle_inline_query(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        """Answer ``@botname <query>`` with a searchable command/skill picker (the ``/`` menu is capped at
        60 slots). Results are computed per keystroke, 50 per page; tapping sends ``/cmd`` text as the
        user, so dispatch flows through the normal command path. Inline queries arrive from ANY chat, so
        unauthorized users get an empty list (the skill catalog is not leaked)."""
        from . import adapter as _adapter

        inline_query = getattr(update, "inline_query", None)
        if inline_query is None:
            return
        from_user = getattr(inline_query, "from_user", None)
        user_id = str(getattr(from_user, "id", "") or "").strip()
        try:
            # No chat context on inline queries — authorize on user identity alone, DM-shaped.
            authorized = bool(user_id) and self._is_callback_user_authorized(
                user_id, chat_id=user_id, chat_type="private", user_name=getattr(from_user, "username", None))
        except Exception:
            _adapter.logger.debug("[%s] inline picker auth check failed", self.name, exc_info=True)
            authorized = False
        if not authorized:
            try:
                from plugins.platforms.telegram.inline_picker import CACHE_TIME_SECONDS as _deny_cache
                await inline_query.answer([], cache_time=_deny_cache, is_personal=True)
            except Exception:
                _adapter.logger.debug("[%s] inline picker empty answer failed", self.name, exc_info=True)
            return
        try:
            from telegram import InlineQueryResultArticle, InputTextMessageContent
            from plugins.platforms.telegram.inline_picker import CACHE_TIME_SECONDS as _CACHE, build_inline_results
            results, next_offset = build_inline_results(
                getattr(inline_query, "query", "") or "", offset=getattr(inline_query, "offset", "") or "")
            articles = [
                InlineQueryResultArticle(
                    id=r["id"], title=r["title"], description=r["description"],
                    input_message_content=InputTextMessageContent(r["message_text"]))
                for r in results
           ]
            # is_personal: catalogs differ per user (auth, disabled skills) — never share cached pages.
            await inline_query.answer(articles, cache_time=_CACHE, is_personal=True, next_offset=next_offset)
        except Exception:
            _adapter.logger.debug("[%s] inline picker answer failed", self.name, exc_info=True)

    @staticmethod
    def _callback_ctx(query) -> Dict[str, Any]:
        """Chat/thread/user context of a button tap, for the callback auth gate."""
        query_message = getattr(query, "message", None)
        query_chat = getattr(query_message, "chat", None)
        return {
            "chat_id": getattr(query_message, "chat_id", None), "chat_type": getattr(query_chat, "type", None),
            "thread_id": getattr(query_message, "message_thread_id", None), "user_name": getattr(query.from_user, "first_name", None)}

    async def _callback_authorized(self, query, cb: Dict[str, Any], denial_text: str) -> bool:
        """Gate a button tap on the callback allowlist; answers ``denial_text`` when refused."""
        from . import adapter as _adapter

        if self._is_callback_user_authorized(
            str(getattr(query.from_user, "id", "")), chat_id=cb["chat_id"],
            chat_type=str(cb["chat_type"]) if cb["chat_type"] is not None else None,
            thread_id=str(cb["thread_id"]) if cb["thread_id"] is not None else None, user_name=cb["user_name"]):
            return True
        await query.answer(text=denial_text)
        return False

    async def _handle_callback_query(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        """Dispatch inline keyboard button clicks on the callback_data prefix."""
        from . import adapter as _adapter

        query = update.callback_query
        if not query or not query.data:
            return
        data = query.data
        cb = self._callback_ctx(query)
        # Model picker / generic choice picker (/reasoning, /fast) need a chat id.
        for prefixes, handler in (
            (("mp:", "mpg:", "mpv:", "mm:", "mc:", "mb", "mx", "mg:"), self._handle_model_picker_callback),
            (("cp:",), self._handle_choice_picker_callback)):
            if data.startswith(prefixes):
                chat_id = str(query.message.chat_id) if query.message else None
                if chat_id:
                    await handler(query, data, chat_id)
                return
        for prefix, handler in (
            ("gt:", self._handle_gmail_triage_callback), ("ea:", self._handle_exec_approval_callback),
            ("sc:", self._handle_slash_confirm_callback), ("cl:", self._handle_clarify_callback),
            ("update_prompt:", self._handle_update_prompt_callback)):
            if data.startswith(prefix):
                await handler(query, data, cb)
                return

    async def _claim_callback_state(self, query, cb: Dict[str, Any], state: dict, key, denial: str, resolved: str, *, pop: bool = True):
        """Auth-gate a button tap, then claim its pending entry; None (after answering) when refused or expired."""
        if not await self._callback_authorized(query, cb, denial):
            return None
        session_key = state.pop(key, None) if pop else state.get(key)
        if not session_key:
            await query.answer(text=resolved)
        return session_key

    async def _handle_exec_approval_callback(self, query, data: str, cb: Dict[str, Any]) -> None:
        """``ea:<choice>:<approval_id>`` — resolve a pending exec approval."""
        from . import adapter as _adapter

        parts = data.split(":", 2)
        if len(parts) != 3:
            return
        choice = parts[1]  # once, session, always, deny
        try:
            approval_id = int(parts[2])
        except (ValueError, IndexError):
            await query.answer(text="Invalid approval data.")
            return
        session_key = await self._claim_callback_state(
            query, cb, self._approval_state, approval_id, "⛔ You are not authorized to approve commands.",
            "This approval has already been resolved.")
        if not session_key:
            return
        user_display = getattr(query.from_user, "first_name", "User")
        # Resolve FIRST (unblocks the agent thread), render after: a tap landing after the wait timed out
        # (count == 0) must NOT claim "Approved" — the command was already denied.
        try:
            # Rendering happens after so the message reflects what actually occurred: a tap that lands after
            # the approval wait timed out (count == 0) must NOT claim "Approved" — the command was already
            # denied and will not run (#63501 regression follow-up: 60s waits made stale taps common).
            from tools.approval import resolve_gateway_approval
            count = resolve_gateway_approval(session_key, choice)
            _adapter.logger.info(
                "Telegram button resolved %d approval(s) for session %s (choice=%s, user=%s)", count, session_key, choice, user_display)
        except Exception as exc:
            _adapter.logger.error("Failed to resolve gateway approval from Telegram button: %s", exc)
            count = 0
        if count:
            label_map = {
                "once": "✅ Approved once", "session": "✅ Approved for session", "always": "✅ Approved permanently", "deny": "❌ Denied",
            }
            label = label_map.get(choice, "Resolved")
            edit_text = f"{label} by {user_display}"
        else:
            label = "⌛ Approval expired"
            edit_text = f"{label} — no command was waiting. It already timed out (and was denied) or was resolved elsewhere."
        await query.answer(text=label)
        await self._edit_md_quiet(query, edit_text)
        # Typing was paused when the approval was sent; the text /approve and /deny paths resume it too.
        if count and cb["chat_id"] is not None:
            self.resume_typing_for_chat(str(cb["chat_id"]))

    async def _handle_slash_confirm_callback(self, query, data: str, cb: Dict[str, Any]) -> None:
        """``sc:<choice>:<confirm_id>`` — resolve a slash-command confirmation."""
        from . import adapter as _adapter

        parts = data.split(":", 2)
        if len(parts) != 3:
            return
        choice = parts[1]  # once, always, cancel
        confirm_id = parts[2]
        session_key = await self._claim_callback_state(
            query, cb, self._slash_confirm_state, confirm_id, "⛔ You are not authorized to answer this prompt.",
            "This prompt has already been resolved.")
        if not session_key:
            return
        label_map = {"once": "✅ Approved once", "always": "🔒 Always approve", "cancel": "❌ Cancelled"}
        user_display = getattr(query.from_user, "first_name", "User")
        label = label_map.get(choice, "Resolved")
        await query.answer(text=label)
        await self._edit_md_quiet(query, f"{label} by {user_display}")
        # The runner stored a handler keyed by session_key; run it and send any returned text as a follow-up.
        try:
            from tools import slash_confirm as _slash_confirm_mod
            result_text = await _slash_confirm_mod.resolve(session_key, confirm_id, choice)
            if result_text and query.message:
                # Inherit the prompt's topic: forums use message_thread_id; private DM-topic lanes need
                # both the topic id and the prompt reply anchor.
                thread_id = getattr(query.message, "message_thread_id", None)
                chat_type = getattr(getattr(query.message, "chat", None), "type", None)
                prompt_message_id = getattr(query.message, "message_id", None)
                send_kwargs: _adapter.Dict[str, _adapter.Any] = {
                    "chat_id": int(query.message.chat_id), "text": self.format_message(result_text),
                    "parse_mode": _adapter.ParseMode.MARKDOWN_V2, **self._link_preview_kwargs()}
                is_private_chat = str(getattr(chat_type, "value", chat_type)).lower() in {
                    "private", str(_adapter.ChatType.PRIVATE).lower(), str(getattr(_adapter.ChatType.PRIVATE, "value", _adapter.ChatType.PRIVATE)).lower()}
                if thread_id is not None:
                    meta: _adapter.Dict[str, _adapter.Any] = {"thread_id": str(thread_id)}
                    reply_to_id = None
                    if is_private_chat and prompt_message_id is not None:
                        reply_to_id = send_kwargs["reply_to_message_id"] = int(prompt_message_id)
                        meta["telegram_dm_topic_reply_fallback"] = True
                    send_kwargs.update(self._thread_kwargs_for_send(
                        str(
                            query.message.chat_id
                        ), str(thread_id), meta, reply_to_message_id=reply_to_id, reply_to_mode=self._reply_to_mode))
                await self._send_message_with_thread_fallback(**send_kwargs)
        except Exception as exc:
            _adapter.logger.error("[%s] slash-confirm callback failed: %s", self.name, exc, exc_info=True)

    async def _handle_clarify_callback(self, query, data: str, cb: Dict[str, Any]) -> None:
        """``cl:<clarify_id>:<idx|other>`` — resolve a clarify prompt or flip to text capture."""
        from . import adapter as _adapter

        parts = data.split(":", 2)
        if len(parts) != 3:
            return
        clarify_id = parts[1]
        choice_token = parts[2]
        session_key = await self._claim_callback_state(
            query, cb, self._clarify_state, clarify_id, "⛔ You are not authorized to answer this prompt.",
            "This prompt has already been resolved.", pop=False)
        if not session_key:
            return
        user_display = getattr(query.from_user, "first_name", "User")
        if choice_token == "other":
            # Flip to text-capture: the gateway's text-intercept resolves the clarify with the next message.
            # Do NOT pop _clarify_state yet — still needed if the entry gets cleared by something else.
            flipped = False
            try:
                from tools.clarify_gateway import mark_awaiting_text
                flipped = mark_awaiting_text(clarify_id)
            except Exception as exc:
                _adapter.logger.warning("[%s] mark_awaiting_text failed: %s", self.name, exc)
            if not flipped:
                # Entry evicted / gateway restarted — a typed answer would go nowhere.
                self._clarify_state.pop(clarify_id, None)
                await self._notify_clarify_expired(query, user_display)
                return
            await query.answer(text="✏️ Type your answer in the chat.")
            await self._edit_html_quiet(
                query, f"❓ {query.message.text or ''}\n\n<i>Awaiting typed response from {_adapter._html.escape(user_display)}…</i>")
            return
        # Numeric choice → resolve immediately with the chosen text
        try:
            idx = int(choice_token)
        except (ValueError, TypeError):
            await query.answer(text="Invalid choice.")
            return
        resolved_text: _adapter.Optional[str] = None
        try:
            from tools.clarify_gateway import _entries as _clarify_entries  # type: ignore
            entry = _clarify_entries.get(clarify_id)
            if entry and entry.choices and 0 <= idx < len(entry.choices):
                resolved_text = entry.choices[idx]
        except Exception:
            resolved_text = None
        if resolved_text is None:
            # Race (timeout / session reset): echo the index so the agent sees an intentional response.
            resolved_text = f"choice {idx + 1}"
        self._clarify_state.pop(clarify_id, None)
        try:
            from tools.clarify_gateway import resolve_gateway_clarify
            resolved = resolve_gateway_clarify(clarify_id, resolved_text)
        except Exception as exc:
            _adapter.logger.error("[%s] resolve_gateway_clarify failed: %s", self.name, exc)
            resolved = False
        if resolved:
            await query.answer(text=f"✓ {resolved_text[:60]}")
            await self._edit_html_quiet(
                query, f"❓ {_adapter._html.escape(query.message.text or '')}\n\n<b>{_adapter._html.escape(user_display)}:</b> {_adapter._html.escape(resolved_text)}")
            _adapter.logger.info("Telegram clarify button resolved (id=%s, choice=%r, user=%s)", clarify_id, resolved_text, user_display)
        else:
            # Entry evicted / gateway restarted between ask and tap.
            await self._notify_clarify_expired(query, user_display)
            _adapter.logger.warning("Telegram clarify button: resolve_gateway_clarify returned False (id=%s)", clarify_id)

    async def _handle_update_prompt_callback(self, query, data: str, cb: Dict[str, Any]) -> None:
        """``update_prompt:<y|n>`` — forward the answer to the update process."""
        from . import adapter as _adapter

        answer = data.split(":", 1)[1]  # "y" or "n"
        if not await self._callback_authorized(query, cb, "⛔ You are not authorized to answer update prompts."):
            return
        await query.answer(text=f"Sent '{answer}' to the update process.")
        await self._edit_md_quiet(query, f"⚕ Update prompt answered: *{'Yes' if answer == 'y' else 'No'}*")
        try:
            from hermes_constants import get_hermes_home
            response_path = get_hermes_home() / ".update_response"
            tmp = response_path.with_suffix(".tmp")
            tmp.write_text(answer, encoding="utf-8")
            tmp.replace(response_path)
            _adapter.logger.info("Telegram update prompt answered '%s' by user %s", answer, getattr(query.from_user, "id", "unknown"))
        except Exception as exc:
            _adapter.logger.error("Failed to write update response from callback: %s", exc)

    async def _handle_gmail_triage_callback(self, query, data: str, cb: Dict[str, Any]) -> None:
        """Dispatch a gmail-triage inline-button callback (gt:verb:arg)."""
        from . import adapter as _adapter

        parts = data.split(":", 2)
        if len(parts) != 3:
            await query.answer(text="Invalid gmail-triage data.")
            return
        verb, arg = parts[1], parts[2]
        if not await self._callback_authorized(query, cb, "⛔ You are not authorized to act on this email."):
            return
        entry = self._GT_VERB_DISPATCH.get(verb)
        if not entry:
            await query.answer(text=f"Unknown verb: {verb}")
            return
        script_name, extra_args, success_label, is_state_verb = entry
        script_path = _adapter._Path.home() / ".hermes" / "scripts" / "gmail-triage" / script_name
        if not script_path.exists():
            await query.answer(text=f"❌ {script_name} missing")
            _adapter.logger.error("[%s] gmail-triage script missing: %s", self.name, script_path)
            return
        success = False
        try:
            proc = await _adapter.asyncio.create_subprocess_exec(
                str(script_path), arg, *extra_args, stdout=_adapter.asyncio.subprocess.PIPE, stderr=_adapter.asyncio.subprocess.PIPE)
            _stdout_bytes, stderr_bytes = await _adapter.asyncio.wait_for(proc.communicate(), timeout=60)
            if proc.returncode == 0:
                label = success_label
                success = True
                _adapter.logger.info("[%s] gmail-triage callback ok: verb=%s arg=%s", self.name, verb, arg)
            else:
                stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
                last_line = stderr_text.splitlines()[-1] if stderr_text else f"exit {proc.returncode}"
                label = f"❌ {verb} failed: {last_line[:80]}"
                _adapter.logger.error(
                    "[%s] gmail-triage callback failed: verb=%s arg=%s rc=%s stderr=%s", self.name, verb, arg, proc.returncode, stderr_text)
        except _adapter.asyncio.TimeoutError:
            label = f"❌ {verb} timed out"
            _adapter.logger.error("[%s] gmail-triage callback timed out: verb=%s arg=%s", self.name, verb, arg)
        except Exception as exc:
            label = f"❌ {verb} error: {exc}"
            _adapter.logger.error("[%s] gmail-triage callback exception: verb=%s arg=%s err=%s", self.name, verb, arg, exc, exc_info=True)
        await query.answer(text=label)
        if not success:
            return
        original_text = (query.message.text or "") if query.message else ""
        appended = f"{original_text}\n— {label} by {getattr(query.from_user, 'first_name', 'User')}"
        # Sticky state verbs keep the keyboard so further actions can stack; one-shots strip it (can't fire twice).
        with _adapter.contextlib.suppress(Exception):
            await query.edit_message_text(text=appended, **({} if is_state_verb else {"reply_markup": None}))
