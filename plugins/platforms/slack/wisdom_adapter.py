"""Profile-bound Wisdom cards and interactions for the Slack adapter."""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

import aiohttp

from gateway.platforms.base import gateway_trust_env
from gateway.wisdom_command_consent import surface_context

try:
    from .block_kit import sanitize_blocks
    from .wisdom_blocks import render_wisdom_blocks, wisdom_fallback_text
except ImportError:  # flat plugin-dir loading
    from block_kit import sanitize_blocks
    from wisdom_blocks import render_wisdom_blocks, wisdom_fallback_text

logger = logging.getLogger("plugins.platforms.slack.adapter")


class SlackWisdomMixin:
    """Wisdom behavior shares the adapter's authorized clients and routing."""

    async def _run_wisdom_profile_operation(
        self, operation, *, profile: Optional[str] = None
    ):
        """Run Wisdom state access inside the Slack control's exact profile."""
        selected = profile or getattr(self, "_owner_profile", None)
        if not isinstance(selected, str) or not selected.strip():
            return await asyncio.to_thread(operation)

        from gateway.run import _profile_runtime_scope
        from hermes_cli.profiles import get_profile_dir

        def scoped():
            with _profile_runtime_scope(get_profile_dir(selected)):
                return operation()

        return await asyncio.to_thread(scoped)

    @staticmethod
    def _wisdom_is_group_channel(channel_id: str) -> bool:
        return not str(channel_id or "").startswith("D")

    def _remember_wisdom_callbacks(
        self,
        view,
        *,
        team_id: str,
        channel_id: str,
        profile: Optional[str],
    ) -> None:
        now = time.monotonic()
        if not hasattr(self, "_wisdom_callback_profiles"):
            self._wisdom_callback_profiles = {}
        callbacks = [
            str(action.callback_data)
            for action in [
                *getattr(view, "navigation_actions", []),
                *view.actions,
                *(action for item in view.items for action in item.actions),
            ]
            if isinstance(getattr(action, "callback_data", None), str)
            and action.callback_data
        ]
        for value in callbacks:
            self._wisdom_callback_profiles[(team_id, channel_id, value)] = (
                profile,
                now,
            )
        stale = [
            key
            for key, (_profile, created) in self._wisdom_callback_profiles.items()
            if now - created > 900
        ]
        for key in stale:
            self._wisdom_callback_profiles.pop(key, None)
        self._trim_oldest_dict_entries(
            self._wisdom_callback_profiles,
            getattr(self, "_WISDOM_CALLBACK_PROFILE_MAX", 2000),
        )

    def _wisdom_callback_profile(
        self, *, team_id: str, channel_id: str, value: str
    ) -> Optional[str]:
        callback_profiles = getattr(self, "_wisdom_callback_profiles", {})
        remembered = callback_profiles.get(
            (team_id, channel_id, value)
        )
        if remembered is None:
            return getattr(self, "_owner_profile", None)
        profile, created = remembered
        if time.monotonic() - created > 900:
            callback_profiles.pop((team_id, channel_id, value), None)
            return getattr(self, "_owner_profile", None)
        return profile

    async def _prepare_wisdom_view(
        self,
        view,
        context,
        *,
        team_id: str,
        channel_id: str,
    ) -> None:
        """Bind controls and turn private group actions into DM continuations."""
        from gateway.wisdom_command import bind_view_callbacks, issue_continuation

        all_actions = [
            *view.actions,
            *(action for item in view.items for action in item.actions),
        ]
        for action in all_actions:
            if action.operation != "continue_dm":
                continue
            token = issue_continuation(
                str(action.arguments.get("raw_args") or ""), context
            )
            action.callback_data = f"wi:continue:{token}"
            action.operation = None
        bind_view_callbacks(view, context)
        self._remember_wisdom_callbacks(
            view,
            team_id=team_id,
            channel_id=channel_id,
            profile=context.profile,
        )

    async def _post_wisdom_response_url(
        self,
        response_url: str,
        *,
        text: str,
        blocks: list[dict[str, Any]],
        replace_original: bool,
    ) -> bool:
        try:
            from .adapter import _read_error_text_limited
        except ImportError:  # flat plugin-dir loading
            from adapter import _read_error_text_limited

        if not response_url:
            return False
        try:
            async with aiohttp.ClientSession(trust_env=gateway_trust_env()) as session:
                async with session.post(
                    response_url,
                    json={
                        "response_type": "ephemeral",
                        "replace_original": replace_original,
                        "text": text,
                        "blocks": blocks,
                    },
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 200:
                        return True
                    body = await _read_error_text_limited(response)
                    logger.warning(
                        "[Slack] Wisdom response_url POST returned %s: %s",
                        response.status,
                        body[:200],
                    )
        except Exception as exc:
            logger.warning("[Slack] Wisdom response_url POST failed: %s", exc)
        return False

    async def _send_wisdom_view(self, view, *, source, proactive: bool = False) -> Any:
        """Send a Wisdom view ephemerally for slashes or persistently otherwise."""
        channel_id = str(source.chat_id)
        team_id = str(getattr(source, "scope_id", None) or "")
        blocks = sanitize_blocks(render_wisdom_blocks(view)) or []
        text = wisdom_fallback_text(view)
        slash_context = None if proactive else self._pop_slash_context(channel_id, team_id)
        if slash_context:
            if await self._post_wisdom_response_url(
                str(slash_context.get("response_url") or ""),
                text=text,
                blocks=blocks,
                replace_original=True,
            ):
                return
            user_id = str(slash_context.get("user_id") or "")
            if user_id:
                try:
                    await self._get_client(
                        channel_id, team_id=team_id or None
                    ).chat_postEphemeral(
                        channel=channel_id,
                        user=user_id,
                        text=text,
                        blocks=blocks,
                    )
                    return
                except Exception as exc:
                    logger.warning(
                        "[Slack] Wisdom ephemeral fallback failed: %s", exc
                    )

        kwargs: Dict[str, Any] = {
            "channel": channel_id,
            "text": text,
            "blocks": blocks,
            "unfurl_links": False,
            "unfurl_media": False,
        }
        thread_id = getattr(source, "thread_id", None)
        if thread_id:
            kwargs["thread_ts"] = str(thread_id)
        return await self._get_client(channel_id, team_id=team_id or None).chat_postMessage(
            **kwargs
        )

    async def send_wisdom_command(self, raw_args: str, *, source) -> None:
        """Execute and render `/wisdom` inside this Slack adapter's profile."""
        from gateway.wisdom_command import (
            WisdomCommandController,
            command_error_text,
        )
        from hermes_wisdom.service import WisdomService

        channel_id = str(source.chat_id)
        team_id = str(getattr(source, "scope_id", None) or "")
        profile = getattr(source, "profile", None) or getattr(
            self, "_owner_profile", None
        )
        user_id = str(getattr(source, "user_id", None) or "")

        def command_action():
            service = WisdomService()
            context = surface_context(self,
                user_id=user_id,
                chat_id=channel_id,
                profile=profile,
                organization_id=service.store.active_org_id(),
                is_group=self._wisdom_is_group_channel(channel_id),
                thread_id=str(getattr(source, "thread_id", None) or ""),
                scope_id=team_id,
            )
            view = WisdomCommandController().execute(raw_args, service, context)
            return view, context

        try:
            view, context = await self._run_wisdom_profile_operation(
                command_action, profile=profile
            )
            await self._prepare_wisdom_view(
                view, context, team_id=team_id, channel_id=channel_id
            )
        except Exception as exc:
            logger.warning(
                "[%s] Collective Wisdom Slack command failed: %s",
                self.name,
                type(exc).__name__,
            )
            from gateway.wisdom_command import WisdomView

            view = WisdomView(
                "Collective Wisdom could not continue", command_error_text(exc)
            )
        await self._send_wisdom_view(view, source=source)

    async def send_wisdom_mediation(self, view, *, source):
        from hermes_wisdom.delivery import slack_receipt

        self._remember_wisdom_callbacks(
            view, team_id=str(getattr(source, "scope_id", None) or ""),
            channel_id=str(source.chat_id),
            profile=getattr(source, "profile", None) or getattr(self, "_owner_profile", None),
        )
        response = await self._send_wisdom_view(view, source=source, proactive=True)
        return slack_receipt(
            response, channel_id=str(source.chat_id),
            thread_id=str(getattr(source, "thread_id", None) or ""),
            scope_id=str(getattr(source, "scope_id", None) or ""),
        )

    async def send_wisdom_candidate_notifications(
        self,
        chat_id: str,
        session_id: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """DM newly qualified local skills after their originating Slack turn."""
        from gateway.wisdom_command import WisdomAction
        from hermes_wisdom.entitlement import local_work_allowed
        from hermes_wisdom.notice import qualification_notice
        from hermes_wisdom.professionalism import review_text
        from hermes_wisdom.service import WisdomService
        from hermes_wisdom.store import WisdomStore

        if self._app is None:
            return 0
        metadata = metadata or {}
        team_id = self._metadata_team_id(metadata)
        user_id = str(metadata.get("user_id") or "")
        if not user_id:
            logger.warning(
                "[Slack] Cannot privately deliver Wisdom candidate without user_id"
            )
            return 0
        profile = metadata.get("profile") or getattr(self, "_owner_profile", None)
        events = await self._run_wisdom_profile_operation(
            lambda: WisdomService(store=WisdomStore()).pending_candidate_events(
                session_id=session_id, surface="slack"
            ),
            profile=profile,
        )
        destination = str(chat_id)
        if self._wisdom_is_group_channel(destination):
            destination = await self._ensure_dm_conversation(
                user_id, team_id=team_id or None
            )

        sent = 0
        for event in events:
            if not await self._run_wisdom_profile_operation(
                lambda: local_work_allowed(WisdomStore()), profile=profile
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
            qualification = str(
                event.get("qualification") or payload.get("qualification") or ""
            )
            professionalism_review = await self._run_wisdom_profile_operation(
                lambda event=event: (
                    WisdomService().finish_candidate_professionalism_review(
                        skill_id=str(event["skill_id"]),
                        content_hash=str(event["content_hash"]),
                    )
                ),
                profile=profile,
            )
            if not await self._run_wisdom_profile_operation(
                lambda: local_work_allowed(WisdomStore()), profile=profile
            ):
                break
            notice = qualification_notice(event)
            view = self._wisdom_candidate_view(
                skill_name=skill_name,
                skill_description=skill_description,
                qualification=qualification,
                status=(
                    f"{notice}\n\n"
                    + review_text(professionalism_review, include_checks=True)
                ),
                actions=[
                    WisdomAction(
                        "Not Now",
                        callback_data=f"wi:defer:{event_id}",
                    ),
                    WisdomAction(
                        "Review first",
                        callback_data=f"wi:draft:{event_id}",
                    ),
                    WisdomAction(
                        "Yes",
                        callback_data=f"wi:publish:{event_id}",
                        primary=True,
                    ),
                ],
            )
            self._remember_wisdom_callbacks(
                view,
                team_id=team_id,
                channel_id=destination,
                profile=profile,
            )
            try:
                await self._get_client(
                    destination, team_id=team_id or None
                ).chat_postMessage(
                    channel=destination,
                    text=wisdom_fallback_text(view),
                    blocks=sanitize_blocks(render_wisdom_blocks(view)) or [],
                    unfurl_links=False,
                    unfurl_media=False,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] Slack Wisdom candidate delivery failed: %s",
                    self.name,
                    exc,
                )
                continue
            await self._run_wisdom_profile_operation(
                lambda event_id=event_id: WisdomStore().mark_surface_delivered(
                    [event_id], surface="slack"
                ),
                profile=profile,
            )
            sent += 1
        return sent

    async def edit_wisdom_publication(self, receipt, view) -> None:
        """Update the original card in its recorded workspace and conversation."""
        await self._get_client(
            receipt["destination"], team_id=receipt.get("scope_id") or None
        ).chat_update(
            channel=receipt["destination"], ts=receipt["message_id"],
            text=wisdom_fallback_text(view),
            blocks=sanitize_blocks(render_wisdom_blocks(view)) or [],
        )

    async def _update_wisdom_interaction(self, body, view) -> None:
        blocks = sanitize_blocks(render_wisdom_blocks(view)) or []
        text = wisdom_fallback_text(view)
        response_url = str(body.get("response_url") or "")
        if response_url and await self._post_wisdom_response_url(
            response_url,
            text=text,
            blocks=blocks,
            replace_original=True,
        ):
            return
        channel_id = str((body.get("channel") or {}).get("id") or "")
        message = body.get("message") or {}
        message_ts = str(message.get("ts") or "")
        team_id = self._event_team_id({}, body)
        if channel_id and message_ts:
            await self._get_client(
                channel_id, team_id=team_id or None
            ).chat_update(
                channel=channel_id,
                ts=message_ts,
                text=text,
                blocks=blocks,
            )
            return
        raise ValueError("This card cannot be updated. Open /wisdom inbox to review its current state.")

    async def _wisdom_interaction_notice(self, body, text: str) -> None:
        response_url = str(body.get("response_url") or "")
        if response_url:
            await self._post_wisdom_response_url(
                response_url,
                text=text,
                blocks=[
                    {
                        "type": "context",
                        "elements": [{"type": "mrkdwn", "text": text}],
                    }
                ],
                replace_original=False,
            )

    @staticmethod
    def _wisdom_candidate_reason(qualification: str) -> str:
        if qualification == "high_usage":
            return "You used this skill consistently across many days."
        if qualification == "refinement":
            return "You've really refined this skill."
        return "This skill met your local Collective Wisdom qualification rules."

    @classmethod
    def _wisdom_candidate_view(
        cls,
        *,
        skill_name: str,
        skill_description: str = "",
        qualification: str,
        status: str,
        actions,
    ):
        from gateway.wisdom_command import WisdomItem, WisdomView

        share_question = (
            "\n\nWould you like to share it?"
            if any((action.callback_data or "").startswith("wi:publish:") for action in actions)
            else ""
        )
        return WisdomView(
            "Hermes Collective Wisdom",
            status.replace(
                "Hermes detected another", "Hermes detected *another*"
            ),
            items=[
                WisdomItem(
                    f"Skill name: {skill_name}",
                    (
                        f"What it does: {skill_description}\n"
                        if skill_description
                        else ""
                    )
                    + f"Why others might benefit: {cls._wisdom_candidate_reason(qualification)}"
                    + share_question,
                    actions=list(actions),
                )
            ],
        )

    async def _mark_wisdom_interaction_complete(
        self, body, *, callback_value: str, completed_label: str
    ) -> bool:
        """Preserve a feed card while replacing one mutation with its result."""
        message = body.get("message") or {}
        original_blocks = message.get("blocks") or []
        if not isinstance(original_blocks, list):
            return False
        updated: list[dict[str, Any]] = []
        changed = False
        inserted_status = False
        for block in original_blocks:
            if not isinstance(block, dict):
                continue
            candidate = dict(block)
            if candidate.get("type") == "actions":
                elements = []
                for element in candidate.get("elements") or []:
                    if (
                        isinstance(element, dict)
                        and str(element.get("value") or "") == callback_value
                    ):
                        changed = True
                        continue
                    elements.append(element)
                if elements:
                    candidate["elements"] = elements
                    updated.append(candidate)
                if changed and not inserted_status:
                    updated.append(
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"✅ *{completed_label}*",
                                }
                            ],
                        }
                    )
                    inserted_status = True
                continue
            updated.append(candidate)
        if not changed:
            return False
        text = str(message.get("text") or completed_label)
        response_url = str(body.get("response_url") or "")
        if response_url and await self._post_wisdom_response_url(
            response_url,
            text=text,
            blocks=sanitize_blocks(updated) or [],
            replace_original=True,
        ):
            return True
        channel_id = str((body.get("channel") or {}).get("id") or "")
        message_ts = str(message.get("ts") or "")
        if channel_id and message_ts:
            team_id = self._event_team_id({}, body)
            await self._get_client(
                channel_id, team_id=team_id or None
            ).chat_update(
                channel=channel_id,
                ts=message_ts,
                text=text,
                blocks=sanitize_blocks(updated) or [],
            )
            return True
        return False

    async def _handle_wisdom_action(self, ack, body, action) -> None:
        """Resolve a Wisdom Block Kit control against authoritative state."""
        await ack()
        value = str(action.get("value") or "")
        if value == "wisdom:portal":
            return
        team_id = self._event_team_id({}, body)
        channel_id = str((body.get("channel") or {}).get("id") or "")
        user = body.get("user") or {}
        user_id = str(user.get("id") or "")
        user_name = str(user.get("name") or "unknown")
        if not self._is_interactive_user_authorized(
            user_id,
            channel_id=channel_id,
            user_name=user_name,
            team_id=team_id,
        ):
            await self._wisdom_interaction_notice(
                body, "You are not authorized to manage Collective Wisdom skills."
            )
            return

        profile = self._wisdom_callback_profile(
            team_id=team_id, channel_id=channel_id, value=value
        )

        try:
            if value.startswith("wa:"):
                def current_controls():
                    from hermes_wisdom.agent_led.actions import current_action_view
                    from hermes_wisdom.service import WisdomService

                    service = WisdomService()
                    context = surface_context(self,
                        user_id=user_id, chat_id=channel_id, profile=profile,
                        organization_id=service.store.active_org_id(),
                        is_group=self._wisdom_is_group_channel(channel_id),
                        thread_id=str((body.get("message") or {}).get("thread_ts") or ""),
                        scope_id=str(team_id or ""),
                    )
                    return current_action_view(value, service, context), context

                view, context = await self._run_wisdom_profile_operation(current_controls, profile=profile)
                await self._prepare_wisdom_view(view, context, team_id=team_id, channel_id=channel_id)
                await self._update_wisdom_interaction(body, view)
                return
            if value.startswith("wi:agent:"):
                def resolve():
                    from hermes_wisdom.mediation_view import resolve_surface_action
                    from hermes_wisdom.service import WisdomService

                    message = body.get("message") or {}
                    return resolve_surface_action(
                        WisdomService(), value, platform="slack", actor_id=user_id,
                        chat_id=channel_id, thread_id=str(message.get("thread_ts") or ""),
                        scope_id=str(team_id or ""),
                    )
                view = await self._run_wisdom_profile_operation(resolve, profile=profile)
                await self._update_wisdom_interaction(body, view)
                return
            if value.startswith("wi:continue:"):
                from gateway.wisdom_command import (
                    WisdomCommandController,
                    resolve_continuation,
                )
                from hermes_wisdom.service import WisdomService

                dm_channel = await self._ensure_dm_conversation(
                    user_id, team_id=team_id or None
                )
                token = value.removeprefix("wi:continue:")

                def continue_action():
                    service = WisdomService()
                    context = surface_context(self,
                        user_id=user_id,
                        chat_id=dm_channel,
                        profile=profile,
                        organization_id=service.store.active_org_id(),
                        is_group=False,
                        scope_id=str(team_id or ""),
                    )
                    raw_args = resolve_continuation(token, context)
                    view = WisdomCommandController().execute(
                        raw_args, service, context
                    )
                    return view, context

                view, context = await self._run_wisdom_profile_operation(
                    continue_action, profile=profile
                )
                await self._prepare_wisdom_view(
                    view, context, team_id=team_id, channel_id=dm_channel
                )
                dm_source = self.build_source(
                    chat_id=dm_channel,
                    chat_type="dm",
                    user_id=user_id,
                    scope_id=team_id or None,
                )
                dm_source.profile = profile
                await self._send_wisdom_view(view, source=dm_source)
                await self._wisdom_interaction_notice(
                    body, "I sent this private Collective Wisdom action to your DM."
                )
                return

            if value.startswith("wi:cmd:"):
                from gateway.wisdom_command import (
                    WisdomCommandController,
                )
                from hermes_wisdom.service import WisdomService

                token = value.removeprefix("wi:cmd:")

                def command_action():
                    service = WisdomService()
                    context = surface_context(self,
                        user_id=user_id,
                        chat_id=channel_id,
                        profile=profile,
                        organization_id=service.store.active_org_id(),
                        is_group=self._wisdom_is_group_channel(channel_id),
                        thread_id=str((body.get("message") or {}).get("thread_ts") or ""),
                        scope_id=str(team_id or ""),
                    )
                    view = WisdomCommandController().execute_token(
                        token, service, context
                    )
                    return view, context

                view, context = await self._run_wisdom_profile_operation(
                    command_action, profile=profile
                )
                await self._prepare_wisdom_view(
                    view, context, team_id=team_id, channel_id=channel_id
                )
                await self._update_wisdom_interaction(body, view)
                return

            if value.startswith(("wi:plan:", "wi:confirm:")):
                def review():
                    from hermes_wisdom.agent_led.actions import current_install_view
                    from hermes_wisdom.service import WisdomService

                    service = WisdomService()
                    context = surface_context(self,
                        user_id=user_id, chat_id=channel_id, profile=profile,
                        organization_id=service.store.active_org_id(),
                        is_group=self._wisdom_is_group_channel(channel_id),
                        thread_id=str((body.get("message") or {}).get("thread_ts") or ""),
                        scope_id=str(team_id or ""),
                    )
                    return current_install_view(value, service, context), context

                view, context = await self._run_wisdom_profile_operation(review, profile=profile)
                await self._prepare_wisdom_view(view, context, team_id=team_id, channel_id=channel_id)
                await self._update_wisdom_interaction(body, view)
                return
            parts = value.split(":", 2)
            if len(parts) != 3 or parts[:2] not in (
                ["wi", "draft"],
                ["wi", "publish"],
                ["wi", "defer"],
                ["wi", "decline"],
            ):
                raise ValueError("Invalid Collective Wisdom action.")
            candidate_action, event_id = parts[1], parts[2]

            def candidate_operation():
                from hermes_wisdom.service import WisdomService

                service = WisdomService()
                if candidate_action == "draft":
                    return service.draft_candidate(event_id)
                if candidate_action == "publish":
                    return service.approve_candidate(event_id)
                if candidate_action == "defer":
                    return service.defer_candidate_prompt(
                        event_id, surface="slack"
                    )
                return service.decline_candidate(event_id)

            result = await self._run_wisdom_profile_operation(
                candidate_operation, profile=profile
            )
            from gateway.wisdom_command import WisdomAction

            skill_name = str(result.get("skill_name") or "Local skill")
            qualification = str(result.get("qualification") or "")
            state = str(
                result.get("publication_state") or result.get("state") or ""
            )
            portal_url = result.get("portal_url")
            view_action = (
                [WisdomAction("View in Portal ↗", url=str(portal_url))]
                if isinstance(portal_url, str) and portal_url
                else []
            )
            if candidate_action == "draft" and state == "ready":
                status = "Private draft created. Nothing is shared until you approve it."
                actions = [
                    WisdomAction(
                        "Not Now", callback_data=f"wi:defer:{event_id}"
                    ),
                    *view_action,
                    WisdomAction(
                        "Yes",
                        callback_data=f"wi:publish:{event_id}",
                        primary=True,
                    ),
                ]
            elif candidate_action == "defer":
                status, actions = (
                    "Not sharing right now. You can revisit this skill in "
                    "Collective Wisdom.",
                    [],
                )
            elif state == "pending_moderation":
                status, actions = (
                    "Sent to your collective administrator for approval.",
                    view_action,
                )
            elif state == "published":
                status, actions = ("Published to your collective.", view_action)
            elif state == "changes_requested":
                status, actions = (
                    "Your collective administrator requested changes.",
                    view_action,
                )
            elif state == "declined":
                status, actions = (
                    "Declined. These exact bytes will not be suggested again.",
                    view_action,
                )
            elif state == "invalidated":
                status, actions = (
                    "This draft changed and must be reviewed again.",
                    view_action,
                )
            else:
                status, actions = (
                    "The private draft is still being prepared.",
                    view_action,
                )
            view = self._wisdom_candidate_view(
                skill_name=skill_name,
                qualification=qualification,
                status=status,
                actions=actions,
            )
            self._remember_wisdom_callbacks(
                view,
                team_id=team_id,
                channel_id=channel_id,
                profile=profile,
            )
            await self._update_wisdom_interaction(body, view)
        except (PermissionError, ValueError) as exc:
            await self._wisdom_interaction_notice(body, str(exc))
        except Exception as exc:
            logger.warning(
                "[%s] Collective Wisdom Slack action failed: %s",
                self.name,
                type(exc).__name__,
                exc_info=True,
            )
            # Preserve the original card and its retryable controls.
            await self._wisdom_interaction_notice(
                body, "Collective Wisdom is temporarily unavailable. Try again."
            )
