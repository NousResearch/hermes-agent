"""Wisdom controls and publication-card lifecycle for Telegram."""

from __future__ import annotations

import asyncio
import copy
import html as _html
import logging
from typing import Any, Dict, List, Optional, cast

from gateway.wisdom_command_consent import surface_context
from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id

logger = logging.getLogger("plugins.platforms.telegram.adapter")


class TelegramWisdomMixin:
    """Use the parent adapter's transport and authorization for Wisdom actions."""

    async def _handle_wisdom_agent_callback(self, query, data: str) -> None:
        """Replace legacy controls with current authenticated review/settings."""
        caller_id = str(getattr(query.from_user, "id", ""))
        message = getattr(query, "message", None)
        chat_id = getattr(message, "chat_id", None)
        chat_type = str(getattr(getattr(message, "chat", None), "type", "") or "")
        if not self._is_callback_user_authorized(
            caller_id, chat_id=chat_id, chat_type=chat_type,
            thread_id=str(getattr(message, "message_thread_id", None) or "") or None,
            user_name=getattr(query.from_user, "first_name", None), command="wisdom",
        ):
            await query.answer(text="You are not authorized to manage skills.")
            return
        def run():
            from hermes_wisdom.agent_led.actions import current_action_view
            from hermes_wisdom.service import WisdomService

            service = WisdomService()
            context = surface_context(self,
                user_id=caller_id, chat_id=str(chat_id or caller_id),
                profile=getattr(self, "_owner_profile", None),
                organization_id=service.store.active_org_id(),
                is_group=chat_type.lower() != "private",
                thread_id=str(getattr(message, "message_thread_id", None) or ""),
            )
            return current_action_view(data, service, context), context

        try:
            await query.answer(text="Opening current controls; nothing was changed.")
            view, context = await self._run_wisdom_profile_operation(run)
            await self._prepare_wisdom_command_view(view, context)
            await self._edit_wisdom_command_view(query, view)
        except Exception:
            await query.answer(text="Current controls are unavailable. Open /wisdom inbox or /wisdom mute to try again.", show_alert=True)

    async def send_wisdom_agent_recommendation(self, chat_id: str, event, *, metadata=None) -> None:
        """Legacy sender cannot bypass mediation ownership and receipt recovery."""
        raise RuntimeError("Legacy Wisdom delivery is retired; use the profile-owned mediation scheduler")

    async def _handle_wisdom_callback(
        self,
        query,
        data: str,
        *,
        query_chat_id,
        query_chat_type,
        query_thread_id,
        query_user_name,
    ) -> None:
        """Run a user-selected managed Wisdom operation in this bot's profile."""
        from plugins.platforms.telegram.adapter import _redact_telegram_error_text

        caller_id = str(getattr(query.from_user, "id", ""))
        if not self._is_callback_user_authorized(
            caller_id,
            chat_id=query_chat_id,
            chat_type=str(query_chat_type) if query_chat_type is not None else None,
            thread_id=str(query_thread_id) if query_thread_id is not None else None,
            user_name=query_user_name,
            command="wisdom" if data.startswith(("wi:cmd:", "wi:plan:", "wi:confirm:")) else None,
        ):
            await query.answer(text="⛔ You are not authorized to manage skills.")
            return

        if data.startswith("wi:agent:"):
            await query.answer(text="Checking current state")
            try:
                def resolve():
                    from hermes_wisdom.mediation_view import resolve_surface_action
                    from hermes_wisdom.service import WisdomService

                    return resolve_surface_action(
                        WisdomService(), data, platform="telegram", actor_id=caller_id,
                        chat_id=str(query_chat_id or ""), thread_id=str(query_thread_id or ""),
                    )
                view = await self._run_wisdom_profile_operation(resolve)
                await self._edit_wisdom_command_view(query, view, full_details=True)
            except Exception:
                await query.answer(text="This control is unavailable. Open /wisdom inbox to review current state.", show_alert=True)
            return

        if data.startswith("wi:cmd:"):
            token = data.removeprefix("wi:cmd:")
            try:
                await query.answer(text="Checking current state…")

                def command_action():
                    from gateway.wisdom_command import (
                        WisdomCommandController,
                    )
                    from hermes_wisdom.service import WisdomService

                    service = WisdomService()
                    context = surface_context(self,
                        user_id=caller_id,
                        chat_id=str(query_chat_id or caller_id),
                        profile=getattr(self, "_owner_profile", None),
                        organization_id=service.store.active_org_id(),
                        is_group=str(query_chat_type or "").lower()
                        in {"group", "supergroup", "channel", "forum"},
                        thread_id=str(query_thread_id or ""),
                    )
                    view = WisdomCommandController().execute_token(
                        token, service, context
                    )
                    return view, context

                view, command_context = await self._run_wisdom_profile_operation(
                    command_action
                )
                await self._prepare_wisdom_command_view(view, command_context)
                await self._edit_wisdom_command_view(query, view)
            except (PermissionError, ValueError) as exc:
                await query.answer(text=str(exc), show_alert=True)
            except Exception as exc:
                logger.warning(
                    "[%s] Collective Wisdom command action failed: %s",
                    self.name,
                    _redact_telegram_error_text(exc),
                )
                # Preserve the original card and controls so transient failures
                # can be retried without losing context.
                try:
                    await query.answer(
                        text="Collective Wisdom is temporarily unavailable. Try again.",
                        show_alert=True,
                    )
                except Exception:
                    pass
            return

        parts = data.split(":", 3)
        if data == "wi:cancel":
            await query.answer(text="Cancelled")
            try:
                await query.edit_message_text(
                    text="Collective Wisdom action cancelled.", reply_markup=None
                )
            except Exception:
                pass
            return
        if len(parts) == 3 and parts[1] in {
            "defer",
            "draft",
            "publish",
            "decline",
        }:
            action, event_id = parts[1], parts[2]
            await query.answer(
                text={
                    "draft": "Preparing more details...",
                    "publish": "Sharing...",
                    "defer": "Will ask later...",
                    "decline": "Declining…",
                }[action]
            )

            def candidate_action():
                from hermes_wisdom.service import WisdomService

                service = WisdomService()
                if action == "draft":
                    return service.draft_candidate(event_id)
                if action == "publish":
                    return service.approve_candidate(event_id)
                if action == "defer":
                    return service.defer_candidate_prompt(
                        event_id, surface="telegram"
                    )
                return service.decline_candidate(event_id)

            try:
                result = await self._run_wisdom_profile_operation(candidate_action)
            except Exception as exc:
                logger.warning(
                    "[%s] Collective Wisdom candidate action failed: %s",
                    self.name,
                    _redact_telegram_error_text(exc),
                )
                # A timeout or transient Gateway failure must not consume the
                # user's only action surface. Leave the original rich card and
                # its buttons intact so the exact action can be retried.
                return

            skill_name = str(result.get("skill_name") or "Local skill")
            qualification_reason = self._wisdom_candidate_qualification_reason(
                str(result.get("qualification") or "")
            )
            portal_url = result.get("portal_url")
            state = str(result.get("publication_state") or result.get("state") or "")
            already_advanced = bool(result.get("already_advanced"))
            view_action = (
                [{"label": "View", "url": str(portal_url)}]
                if isinstance(portal_url, str)
                else []
            )
            if action == "draft":
                if state == "ready":
                    actions = [
                        {
                            "label": "Not Now",
                            "callback_data": f"wi:defer:{event_id}",
                        },
                        *view_action,
                        {
                            "label": "Yes",
                            "callback_data": f"wi:publish:{event_id}",
                            "primary": True,
                        },
                    ]
                    status = (
                        "Private draft created. Nothing is shared until you approve it."
                        if result.get("created")
                        else "Private draft is ready. Nothing is shared until you approve it."
                    )
                else:
                    status, actions = self._wisdom_candidate_resolved_state(
                        state, already_advanced=already_advanced, view_action=view_action
                    )
            elif action == "publish":
                status, actions = self._wisdom_candidate_resolved_state(
                    state, already_advanced=already_advanced, view_action=view_action
                )
            elif action == "defer":
                status = (
                    "Not sharing right now. You can revisit this skill in "
                    "Collective Wisdom."
                )
                actions = []
            else:
                if state == "published":
                    status = "This skill is already published to your collective."
                    actions = view_action
                elif result.get("withdrawn"):
                    status = (
                        "Withdrawn from collective review and declined on this device. "
                        "These exact bytes will not be suggested again."
                    )
                    actions = view_action
                else:
                    status = (
                        "Declined on this device. These exact bytes will not be "
                        "suggested again."
                    )
                    actions = view_action
            await self._edit_wisdom_candidate_card(
                query,
                skill_name=skill_name,
                qualification_reason=qualification_reason,
                status=status,
                actions=actions,
            )
            return
        await query.answer(text="Checking current state...")
        try:
            def review():
                from hermes_wisdom.agent_led.actions import current_install_view
                from hermes_wisdom.service import WisdomService

                service = WisdomService()
                context = surface_context(self,
                    user_id=caller_id, chat_id=str(query_chat_id or caller_id),
                    profile=getattr(self, "_owner_profile", None),
                    organization_id=service.store.active_org_id(),
                    is_group=str(query_chat_type or "").lower() != "private",
                    thread_id=str(query_thread_id or ""),
                )
                return current_install_view(data, service, context), context

            view, context = await self._run_wisdom_profile_operation(review)
            await self._prepare_wisdom_command_view(view, context)
            await self._edit_wisdom_command_view(query, view, full_details=True)
        except Exception as exc:
            logger.warning("[%s] Collective Wisdom review failed: %s",
                           self.name, _redact_telegram_error_text(exc))
            await query.answer(
                text="Collective Wisdom is temporarily unavailable. Try again.",
                show_alert=True,
            )

    async def _run_wisdom_profile_operation(self, operation):
        """Run local Wisdom state access inside this Telegram bot's profile."""
        owner_profile = getattr(self, "_owner_profile", None)
        if not isinstance(owner_profile, str) or not owner_profile.strip():
            return await asyncio.to_thread(operation)

        # Secondary multiplexed bots must operate on the profile that owns
        # their Telegram credential, not whichever profile happens to be
        # active in the shared gateway process.
        from gateway.run import _profile_runtime_scope
        from hermes_cli.profiles import get_profile_dir

        def scoped():
            with _profile_runtime_scope(get_profile_dir(owner_profile)):
                return operation()

        return await asyncio.to_thread(scoped)

    @staticmethod
    def _wisdom_command_html(view, *, full_details: bool = False) -> str:
        """Render a presentation-neutral Wisdom view as one compact rich card."""
        def compact(value: Any, limit: int) -> str:
            text = str(value or "")
            return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"

        def button(action) -> str | None:
            raw_label = compact(action.label, 48)
            if action.url:
                raw_label = raw_label.removesuffix(" ↗")
            label = _html.escape(raw_label)
            if action.url:
                return (
                    '<tg-button type="url" url="'
                    f'{_html.escape(str(action.url), quote=True)}">{label}</tg-button>'
                )
            if action.callback_data:
                style = ' style="danger"' if action.destructive else (
                    ' style="primary"' if action.primary else ""
                )
                return (
                    f'<tg-button type="callback_data"{style} data="'
                    f'{_html.escape(str(action.callback_data), quote=True)}">'
                    f"{label}</tg-button>"
                )
            return None

        def button_row(actions) -> str:
            controls = [value for action in actions if (value := button(action))]
            if not controls:
                return ""
            return (
                '<tg-button-row align="left">'
                f"{' '.join(controls)}"
                "</tg-button-row>"
            )

        item_html: list[str] = []
        for item in view.items[:5]:
            candidate = (
                f"<p><b>{_html.escape(compact(item.title, 140))}</b>"
                f"<br/>{_html.escape(item.detail if full_details else compact(item.detail, 300)).replace(chr(10), '<br/>')}"
                "</p>"
                f"{button_row(item.actions)}"
            )
            if not full_details and sum(map(len, item_html)) + len(candidate) > 2600:
                break
            item_html.append(candidate)
        summary = (
            f"<p>{_html.escape(compact(view.summary, 600)).replace(chr(10), '<br/>')}</p>"
            if view.summary
            else ""
        )
        notice = (
            f"<p><i>{_html.escape(compact(view.notice, 400))}</i></p>"
            if view.notice
            else ""
        )
        navigation_html = button_row(getattr(view, "navigation_actions", []))
        action_html = button_row(view.actions)
        return (
            f"<h3>{_html.escape(compact(view.title, 120))}</h3>"
            f"{navigation_html}{summary}{''.join(item_html)}{notice}{action_html}"
        )

    @staticmethod
    def _wisdom_command_text(view) -> str:
        """Bound fallback text below Telegram's message-size ceiling."""
        value = view.to_text()
        return value if len(value) <= 3500 else value[:3499].rstrip() + "…"

    @staticmethod
    def _wisdom_command_error_text(exc: Exception) -> str:
        """Return a stable user-safe command error without upstream details."""
        from gateway.wisdom_command import command_error_text

        return command_error_text(exc)

    @staticmethod
    def _wisdom_command_keyboard(view) -> Optional["InlineKeyboardMarkup"]:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        rows = []
        action_groups = [
            *(
                [getattr(view, "navigation_actions", [])]
                if getattr(view, "navigation_actions", [])
                else []
            ),
            *(item.actions for item in view.items),
            view.actions,
        ]
        for actions in action_groups:
            row = []
            for action in actions:
                label = str(action.label)
                if action.url:
                    label = label.removesuffix(" ↗")
                if action.url:
                    row.append(InlineKeyboardButton(label, url=action.url))
                elif action.callback_data:
                    row.append(
                        InlineKeyboardButton(
                            label, callback_data=action.callback_data
                        )
                    )
            if row:
                rows.extend([row[index : index + 2] for index in range(0, len(row), 2)])
        return InlineKeyboardMarkup(rows) if rows else None

    async def _prepare_wisdom_command_view(self, view, context) -> None:
        """Turn private group actions into DM links, then bind callbacks.

        This runs for both initial cards and callback-driven navigation.  In
        particular, pagination must not turn a group-safe ``Continue in DM``
        action back into an in-group mutation callback.
        """
        from gateway.wisdom_command import bind_view_callbacks, issue_continuation

        continuation_actions = [
            action
            for action in [
                *view.actions,
                *(action for item in view.items for action in item.actions),
            ]
            if action.operation == "continue_dm"
        ]
        if continuation_actions:
            username = str(
                getattr(getattr(self, "_bot", None), "username", "") or ""
            )
            if not username:
                try:
                    bot_user = await self._bot.get_me()
                    username = str(getattr(bot_user, "username", "") or "")
                except Exception:
                    logger.debug(
                        "[%s] Could not resolve Telegram username for Wisdom DM link",
                        self.name,
                        exc_info=True,
                    )
            for action in continuation_actions:
                if username:
                    token = issue_continuation(
                        str(action.arguments.get("raw_args") or ""), context
                    )
                    action.url = f"https://t.me/{username}?start=wisdom_{token}"
                else:
                    action.label = "DM this bot, then run /wisdom"
                action.operation = None

        bind_view_callbacks(view, context)

    async def edit_wisdom_publication(self, receipt, view) -> None:
        """Edit only the durable receipt's message; never fall back to a new send."""
        from telegram.constants import ParseMode

        from telegram.error import BadRequest

        try:
            await self._bot.do_api_request(
                "editMessageText",
                api_kwargs={
                    "chat_id": normalize_telegram_chat_id(receipt["destination"]),
                    "message_id": int(receipt["message_id"]),
                    "rich_message": {"html": self._wisdom_command_html(view, full_details=True)},
                    "link_preview_options": {"is_disabled": True},
                },
            )
        except BadRequest as exc:
            if "message is not modified" in str(exc).lower():
                return
            await self._bot.edit_message_text(
                chat_id=normalize_telegram_chat_id(receipt["destination"]),
                message_id=int(receipt["message_id"]),
                text=_html.escape(self._wisdom_command_text(view)),
                parse_mode=ParseMode.HTML,
                reply_markup=self._wisdom_command_keyboard(view),
            )

    async def _edit_wisdom_command_view(self, query, view, *, full_details: bool = False) -> None:
        from telegram.constants import ParseMode
        from telegram.error import BadRequest
        from plugins.platforms.telegram.adapter import _redact_telegram_error_text

        message = getattr(query, "message", None)
        raw_request = getattr(getattr(self, "_bot", None), "do_api_request", None)
        if message is not None and callable(raw_request):
            try:
                await raw_request(
                    "editMessageText",
                    api_kwargs={
                        "chat_id": normalize_telegram_chat_id(message.chat_id),
                        "message_id": int(message.message_id),
                        "rich_message": {"html": self._wisdom_command_html(view, full_details=full_details)},
                        "link_preview_options": {"is_disabled": True},
                    },
                )
                return
            except BadRequest as exc:
                if "message is not modified" in str(exc).lower():
                    return
                logger.debug(
                    "[%s] Wisdom command rich edit failed: %s",
                    self.name,
                    _redact_telegram_error_text(exc),
                )
        try:
            await query.edit_message_text(
                text=_html.escape(self._wisdom_command_text(view)),
                parse_mode=ParseMode.HTML,
                reply_markup=self._wisdom_command_keyboard(view),
            )
        except BadRequest as exc:
            if "message is not modified" not in str(exc).lower():
                raise

    async def send_wisdom_command(self, raw_args: str, *, source) -> None:
        """Execute and render `/wisdom` inside this Telegram adapter's profile."""
        from telegram.constants import ParseMode
        from plugins.platforms.telegram.adapter import _redact_telegram_error_text

        from gateway.wisdom_command import (
            WisdomCommandController,
        )
        from hermes_wisdom.service import WisdomService

        user_id = str(getattr(source, "user_id", None) or "")
        chat_id = str(source.chat_id)
        is_group = str(getattr(source, "chat_type", "") or "").lower() in {
            "group",
            "supergroup",
            "channel",
            "forum",
        }

        def command_action():
            service = WisdomService()
            context = surface_context(self,
                user_id=user_id,
                chat_id=chat_id,
                profile=getattr(self, "_owner_profile", None),
                organization_id=service.store.active_org_id(),
                is_group=is_group,
                thread_id=str(getattr(source, "thread_id", None) or ""),
            )
            view = WisdomCommandController().execute(raw_args, service, context)
            return view, context

        try:
            view, command_context = await self._run_wisdom_profile_operation(
                command_action
            )
        except Exception as exc:
            logger.warning(
                "[%s] Collective Wisdom command failed: %s",
                self.name,
                _redact_telegram_error_text(exc),
            )
            text = (
                "<b>Collective Wisdom could not continue</b>\n"
                f"{_html.escape(self._wisdom_command_error_text(exc))}"
            )
            await self._send_message_with_thread_fallback(
                chat_id=normalize_telegram_chat_id(chat_id),
                text=text,
                parse_mode=ParseMode.HTML,
                **self._link_preview_kwargs(),
            )
            return

        await self._prepare_wisdom_command_view(view, command_context)
        delivered = False
        raw_request = getattr(getattr(self, "_bot", None), "do_api_request", None)
        if callable(raw_request):
            try:
                await raw_request(
                    "sendRichMessage",
                    api_kwargs={
                        "chat_id": normalize_telegram_chat_id(chat_id),
                        "rich_message": {"html": self._wisdom_command_html(view)},
                        "link_preview_options": {"is_disabled": True},
                    },
                )
                delivered = True
            except Exception as exc:
                logger.debug(
                    "[%s] Wisdom command rich send failed: %s",
                    self.name,
                    _redact_telegram_error_text(exc),
                )
        if not delivered:
            await self._send_message_with_thread_fallback(
                chat_id=normalize_telegram_chat_id(chat_id),
                text=_html.escape(self._wisdom_command_text(view)),
                parse_mode=ParseMode.HTML,
                reply_markup=self._wisdom_command_keyboard(view),
                **self._link_preview_kwargs(),
            )

    async def send_wisdom_continuation(self, token: str, *, source) -> None:
        """Resume a user-bound group `/wisdom` request inside its DM."""
        from gateway.wisdom_command import (
            resolve_continuation,
        )
        from hermes_wisdom.service import WisdomService

        def continuation_action() -> str:
            service = WisdomService()
            context = surface_context(self,
                user_id=str(getattr(source, "user_id", None) or ""),
                chat_id=str(source.chat_id),
                profile=getattr(self, "_owner_profile", None),
                organization_id=service.store.active_org_id(),
                is_group=False,
                thread_id=str(getattr(source, "thread_id", None) or ""),
            )
            return resolve_continuation(token, context)

        raw_args = await self._run_wisdom_profile_operation(continuation_action)
        await self.send_wisdom_command(raw_args, source=source)

    @staticmethod
    def _wisdom_candidate_resolved_state(
        state: str,
        *,
        already_advanced: bool,
        view_action: List[Dict[str, Any]],
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Present authoritative draft outcomes without stale mutation controls."""
        if state == "pending_moderation":
            prefix = "Already sent" if already_advanced else "Sent"
            return (
                f"{prefix} to your collective administrator for approval.",
                view_action,
            )
        if state == "published":
            return (
                "This skill is already published to your collective."
                if already_advanced
                else "Published to your collective.",
                view_action,
            )
        if state == "changes_requested":
            return (
                "Your collective administrator requested changes. Open the draft "
                "to review and revise it.",
                view_action,
            )
        if state == "declined":
            return (
                "This exact draft was declined. Change the skill before suggesting "
                "it again.",
                view_action,
            )
        if state == "invalidated":
            return (
                "This draft was replaced or changed and can no longer be approved. "
                "Open the current draft to continue.",
                view_action,
            )
        return (
            "The private draft is still being prepared. Open it to check its "
            "current state.",
            view_action,
        )

    @staticmethod
    def _wisdom_candidate_html(
        *,
        skill_name: str,
        skill_description: str = "",
        qualification_reason: str,
        status: str,
        actions: List[Dict[str, Any]],
        professionalism_review: Optional[Dict[str, Any]] = None,
    ) -> str:
        controls: list[str] = []
        for action in actions:
            url = action.get("url")
            raw_label = str(action["label"])
            if isinstance(url, str):
                raw_label = raw_label.removesuffix(" ↗")
            label = _html.escape(raw_label)
            if isinstance(url, str):
                controls.append(
                    '<tg-button type="url" url="'
                    f'{_html.escape(url, quote=True)}">{label}</tg-button>'
                )
                continue
            callback_data = str(action.get("callback_data") or "")
            style = ' style="primary"' if action.get("primary") else ""
            controls.append(
                f'<tg-button type="callback_data"{style} data="'
                f'{_html.escape(callback_data, quote=True)}">{label}</tg-button>'
            )
        control_html = (
            '<tg-button-row align="left">'
            f"{' '.join(controls)}"
            "</tg-button-row>"
            if controls
            else ""
        )
        from hermes_wisdom.professionalism import review_text

        review_html = (
            "<br/><br/>"
            + _html.escape(
                review_text(professionalism_review, include_checks=True)
            ).replace("\n", "<br/>")
            if professionalism_review is not None
            else ""
        )
        status_html = (
            _html.escape(status)
            .replace("Hermes detected another", "Hermes detected <b>another</b>")
            .replace("\n", "<br/>")
        )
        description_html = (
            f"What it does: {_html.escape(skill_description)}<br/>"
            if skill_description
            else ""
        )
        share_question = (
            "<br/><br/><b>Would you like to share it?</b>"
            if any(str(action.get("callback_data") or "").startswith("wi:publish:") for action in actions)
            else ""
        )
        return (
            "<h3>Hermes Collective Wisdom</h3>"
            f"<p>{status_html}<br/><br/>"
            f"Skill name: <b>{_html.escape(skill_name)}</b><br/>"
            f"{description_html}"
            f"<b>Why others might benefit:</b> {_html.escape(qualification_reason)}<br/>"
            f"{review_html}{share_question}</p>{control_html}"
        )

    @staticmethod
    def _wisdom_candidate_qualification_reason(qualification: str) -> str:
        """Explain the local threshold without exposing its evidence ledger."""
        if qualification == "high_usage":
            return "You used this skill consistently across many days."
        if qualification == "refinement":
            return "You've really refined this skill."
        return "This skill met your local Collective Wisdom qualification rules."

    @staticmethod
    def _wisdom_candidate_keyboard(
        actions: List[Dict[str, Any]],
    ) -> Optional["InlineKeyboardMarkup"]:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        buttons = []
        for action in actions:
            url = action.get("url")
            label = str(action["label"]).removesuffix(" ↗")
            if isinstance(url, str):
                buttons.append(InlineKeyboardButton(label, url=url))
            else:
                buttons.append(
                    InlineKeyboardButton(
                        label, callback_data=str(action.get("callback_data") or "")
                    )
                )
        if not buttons:
            return None
        rows = [buttons[index : index + 3] for index in range(0, len(buttons), 3)]
        return InlineKeyboardMarkup(rows)

    async def _edit_wisdom_candidate_card(
        self,
        query,
        *,
        skill_name: str,
        qualification_reason: str,
        status: str,
        actions: List[Dict[str, Any]],
    ) -> None:
        """Replace a candidate card in place while preserving its context."""
        from telegram.constants import ParseMode
        from plugins.platforms.telegram.adapter import _redact_telegram_error_text

        html = self._wisdom_candidate_html(
            skill_name=skill_name,
            qualification_reason=qualification_reason,
            status=status,
            actions=actions,
        )
        message = getattr(query, "message", None)
        raw_request = getattr(getattr(self, "_bot", None), "do_api_request", None)
        if message is not None and callable(raw_request):
            try:
                await raw_request(
                    "editMessageText",
                    api_kwargs={
                        "chat_id": normalize_telegram_chat_id(message.chat_id),
                        "message_id": int(message.message_id),
                        "rich_message": {"html": html},
                        "link_preview_options": {"is_disabled": True},
                    },
                )
                return
            except Exception as exc:
                logger.debug(
                    "[%s] Candidate rich-card edit failed: %s",
                    self.name,
                    _redact_telegram_error_text(exc),
                )
        try:
            await query.edit_message_text(
                text=(
                    "<b>Hermes Collective Wisdom</b>\n"
                    f"Skill name: <b>{_html.escape(skill_name)}</b>\n{_html.escape(status)}"
                    "\n<b>Why others might benefit:</b> "
                    f"{_html.escape(qualification_reason)}"
                ),
                parse_mode=ParseMode.HTML,
                reply_markup=self._wisdom_candidate_keyboard(actions),
            )
        except Exception:
            pass

    async def send_wisdom_mediation(self, view, *, source):
        """Send already-bound durable controls using the existing rich renderer."""
        from telegram.constants import ParseMode

        from telegram.error import BadRequest
        from hermes_wisdom.delivery import telegram_receipt

        kwargs = self._thread_kwargs_for_send(
            str(source.chat_id), str(getattr(source, "thread_id", None) or "") or None,
            {}, reply_to_mode=self._reply_to_mode,
        )
        raw_request = getattr(getattr(self, "_bot", None), "do_api_request", None)
        # Assessment batches are already bounded. Do not clip their advice or
        # canonical warnings using the catalog's compact preview limits.
        rich_html = self._wisdom_command_html(view, full_details=True)
        if callable(raw_request) and len(rich_html) <= 4096:
            try:
                response = await raw_request("sendRichMessage", api_kwargs={
                    "chat_id": normalize_telegram_chat_id(source.chat_id),
                    "rich_message": {"html": rich_html},
                    "link_preview_options": {"is_disabled": True},
                    **{key: value for key, value in kwargs.items() if value is not None},
                })
                return telegram_receipt(
                    response, chat_id=str(source.chat_id),
                    thread_id=str(getattr(source, "thread_id", None) or ""),
                )
            except BadRequest:
                # Only a definite rejection permits a second delivery attempt.
                pass
        response = await self._send_message_with_thread_fallback(
            chat_id=normalize_telegram_chat_id(source.chat_id),
            text=_html.escape(self._wisdom_command_text(view)), parse_mode=ParseMode.HTML,
            reply_markup=self._wisdom_command_keyboard(view), **kwargs,
        )
        return telegram_receipt(
            response, chat_id=str(source.chat_id),
            thread_id=str(getattr(source, "thread_id", None) or ""),
        )

    async def send_wisdom_candidate_notifications(
        self,
        chat_id: str,
        session_id: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Surface newly qualified local skills in their exact Telegram session."""
        from telegram.constants import ParseMode
        from plugins.platforms.telegram.adapter import _redact_telegram_error_text

        from hermes_wisdom.entitlement import local_work_allowed
        from hermes_wisdom.notice import qualification_notice
        from hermes_wisdom.professionalism import review_text
        from hermes_wisdom.service import WisdomService
        from hermes_wisdom.store import WisdomStore

        events = await self._run_wisdom_profile_operation(
            lambda: WisdomService(store=WisdomStore()).pending_candidate_events(
                session_id=session_id, surface="telegram"
            )
        )
        sent = 0
        for event in events:
            if not await self._run_wisdom_profile_operation(
                lambda: local_work_allowed(WisdomStore())
            ):
                break
            event_id = str(event["id"])
            payload = event.get("payload")
            payload = payload if isinstance(payload, dict) else {}
            skill_name = str(
                payload.get("editorial_name")
                or payload.get("skill_name")
                or "Local skill"
            )
            skill_description = str(payload.get("editorial_description") or "")
            professionalism_review = await self._run_wisdom_profile_operation(
                lambda event=event: (
                    WisdomService().finish_candidate_professionalism_review(
                        skill_id=str(event["skill_id"]),
                        content_hash=str(event["content_hash"]),
                    )
                )
            )
            if not await self._run_wisdom_profile_operation(
                lambda: local_work_allowed(WisdomStore())
            ):
                break
            qualification_reason = self._wisdom_candidate_qualification_reason(
                str(event.get("qualification") or payload.get("qualification") or "")
            )
            notice = qualification_notice(event)
            actions: List[Dict[str, Any]] = [
                {
                    "label": "Not Now",
                    "callback_data": f"wi:defer:{event_id}",
                },
                {
                    "label": "Review first",
                    "callback_data": f"wi:draft:{event_id}",
                },
                {
                    "label": "Yes",
                    "callback_data": f"wi:publish:{event_id}",
                    "primary": True,
                },
            ]
            html = self._wisdom_candidate_html(
                skill_name=skill_name,
                skill_description=skill_description,
                qualification_reason=qualification_reason,
                status=notice,
                actions=actions,
                professionalism_review=professionalism_review,
            )
            delivered = False
            raw_request = getattr(getattr(self, "_bot", None), "do_api_request", None)
            if callable(raw_request):
                rich_payload: Dict[str, Any] = {
                    "chat_id": normalize_telegram_chat_id(chat_id),
                    "rich_message": {"html": html},
                    "link_preview_options": {"is_disabled": True},
                }
                thread_id = self._metadata_thread_id(metadata)
                rich_payload.update(
                    {
                        key: value
                        for key, value in self._thread_kwargs_for_send(
                            chat_id,
                            thread_id,
                            metadata,
                            reply_to_mode=self._reply_to_mode,
                        ).items()
                        if value is not None
                    }
                )
                try:
                    await raw_request("sendRichMessage", api_kwargs=rich_payload)
                    delivered = True
                except Exception as exc:
                    logger.debug(
                        "[%s] Candidate rich-card send failed: %s",
                        self.name,
                        _redact_telegram_error_text(exc),
                    )
            if not delivered and self._bot is not None:
                kwargs: Dict[str, Any] = {
                    "chat_id": normalize_telegram_chat_id(chat_id),
                    "text": (
                        "<b>Hermes Collective Wisdom</b>\n"
                        f"{_html.escape(notice).replace('another', '<b>another</b>')}\n\n"
                        f"Skill name: <code>{_html.escape(skill_name)}</code>\n"
                        + (
                            f"What it does: {_html.escape(skill_description)}\n"
                            if skill_description
                            else ""
                        )
                        + "<b>Why others might benefit:</b> "
                        f"{_html.escape(qualification_reason)}\n"
                        f"{_html.escape(review_text(professionalism_review, include_checks=True))}\n\n"
                        "Would you like to share it?"
                    ),
                    "parse_mode": ParseMode.HTML,
                    "reply_markup": self._wisdom_candidate_keyboard(actions),
                    **self._link_preview_kwargs(),
                }
                try:
                    await self._send_message_with_thread_fallback(**kwargs)
                    delivered = True
                except Exception as exc:
                    logger.warning(
                        "[%s] Candidate notification failed: %s",
                        self.name,
                        _redact_telegram_error_text(exc),
                    )
            if delivered:
                await self._run_wisdom_profile_operation(
                    lambda event_id=event_id: WisdomStore().mark_telegram_delivered(
                        [event_id]
                    )
                )
                sent += 1
        return sent

    @staticmethod
    def _wisdom_api_mapping(value: object) -> Optional[Dict[str, Any]]:
        """Return a JSON-compatible Telegram API object when one is available."""
        if isinstance(value, dict):
            return cast(Dict[str, Any], value)
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            mapped = to_dict()
            if isinstance(mapped, dict):
                return mapped
        return None

    @classmethod
    def _replace_wisdom_action_button(
        cls,
        value: object,
        *,
        callback_data: str,
        completed_label: str,
    ) -> bool:
        """Replace one exact Wisdom action with Telegram's disabled button type.

        Rich-message controls can be inline ``RichTextButton`` values or members
        of a ``RichBlockButtons`` row. The legacy fallback is an inline keyboard.
        All three serialize the actionable button as a mapping containing
        ``callback_data``, so one recursive replacement preserves the rest of
        the notification, including unrelated skills and Portal links.
        """
        changed = False

        def visit(node: object) -> None:
            nonlocal changed
            if isinstance(node, list):
                for child in node:
                    visit(child)
                return
            if not isinstance(node, dict):
                return
            mapped_node = cast(Dict[str, Any], node)
            if mapped_node.get("callback_data") == callback_data:
                mapped_node.pop("callback_data", None)
                mapped_node["disabled"] = {}
                mapped_node["style"] = "success"
                mapped_node["text"] = completed_label
                changed = True
                return
            for child in mapped_node.values():
                visit(child)

        visit(value)
        return changed

    async def _mark_wisdom_action_complete(
        self,
        query,
        *,
        callback_data: str,
        completed_label: str,
    ) -> bool:
        """Keep a Wisdom notification and disable only its completed action."""
        from plugins.platforms.telegram.adapter import _redact_telegram_error_text

        message = getattr(query, "message", None)
        bot = getattr(self, "_bot", None)
        raw_request = getattr(bot, "do_api_request", None)
        if message is None or not callable(raw_request):
            return False

        message_id = getattr(message, "message_id", None)
        chat_id = getattr(message, "chat_id", None)
        if message_id is None or chat_id is None:
            return False

        rich_message = getattr(message, "rich_message", None)
        if rich_message is None:
            api_kwargs = getattr(message, "api_kwargs", None)
            getter = getattr(api_kwargs, "get", None)
            if callable(getter):
                rich_message = getter("rich_message")
        rich_mapping = self._wisdom_api_mapping(rich_message)
        if rich_mapping is not None:
            updated_rich = copy.deepcopy(rich_mapping)
            if self._replace_wisdom_action_button(
                updated_rich,
                callback_data=callback_data,
                completed_label=completed_label,
            ):
                try:
                    await raw_request(
                        "editMessageText",
                        api_kwargs={
                            "chat_id": normalize_telegram_chat_id(chat_id),
                            "message_id": int(message_id),
                            "rich_message": updated_rich,
                            "link_preview_options": {"is_disabled": True},
                        },
                    )
                    return True
                except Exception as exc:
                    logger.warning(
                        "[%s] Could not preserve completed Wisdom rich notification: %s",
                        self.name,
                        _redact_telegram_error_text(exc),
                    )

        reply_markup = getattr(message, "reply_markup", None)
        markup_mapping = self._wisdom_api_mapping(reply_markup)
        if markup_mapping is not None:
            updated_markup = copy.deepcopy(markup_mapping)
            if self._replace_wisdom_action_button(
                updated_markup,
                callback_data=callback_data,
                completed_label=completed_label,
            ):
                try:
                    await raw_request(
                        "editMessageReplyMarkup",
                        api_kwargs={
                            "chat_id": normalize_telegram_chat_id(chat_id),
                            "message_id": int(message_id),
                            "reply_markup": updated_markup,
                        },
                    )
                    return True
                except Exception as exc:
                    logger.warning(
                        "[%s] Could not preserve completed Wisdom fallback notification: %s",
                        self.name,
                        _redact_telegram_error_text(exc),
                    )
        return False
