"""Matrix exec-approval card lifecycle, mixed into ``MatrixAdapter``.

The card is sent through ``_send_reaction_prompt`` and edited through ``edit_message``; both live on
the facade, as do the reaction registries this mixin reads (``_approval_prompts_by_event``,
``_approval_prompt_by_session``, ``_retired_approval_cards``). Rendering is in ``approval_cards``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from agent.i18n import t
from gateway.approval_actor import approval_refusal, refusal_text
from gateway.platforms.base import ExecApprovalPrompt, SendResult
from plugins.platforms.matrix.adapter_html import _sanitize_matrix_html

logger = logging.getLogger(__name__)


@dataclass
class _MatrixApprovalPrompt:
    """Tracks a pending Matrix reaction-based exec approval prompt."""

    def __init__(
        self,
        session_key: str,
        chat_id: str,
        message_id: str,
        resolved: bool = False,
        requester_user_id: str | None = None,
        expires_at: float | None = None,
        request_id: str | None = None,  # the tools.approval request this prompt answers
        command: str = "",
        description: str = "",
        allow_permanent: bool = True,
        allow_session: bool = True,
        smart_denied: bool = False,
        metadata: dict | None = None,
        offered: tuple[str, ...] = (),
    ):
        self.session_key = session_key
        self.chat_id = chat_id
        self.message_id = message_id
        self.request_id = request_id
        self.resolved = resolved
        self.requester_user_id = requester_user_id
        self.expires_at = expires_at
        self.offered = tuple(offered)  # choices the card seeded; a Deny-only card offers only "deny"
        self.bot_reaction_events: dict[str, str] = {}  # emoji -> event_id
        # Presentation state for compact / summary edits (Matrix-only).
        self.command = command or ""
        self.description = description or ""
        self.allow_permanent = allow_permanent
        self.allow_session = allow_session
        self.smart_denied = smart_denied
        self.metadata = dict(metadata or {})
        self.generation: int = 0  # bumps on each presentation edit
        self.state: str = "pending_expanded"  # pending_expanded|pending_summarized|terminal
        self.summary: str = ""
        self.summary_task: object | None = None
        self.presentation_lock = asyncio.Lock()
        self.terminal_visible = False
        self.terminal_failure_notified = False
        self.terminal_choice: str | None = None
        self.terminal_actor = ""


class MatrixApprovalCardsMixin:
    """Approval-card send, reaction resolve, settle observation and terminal ``m.replace`` for ``MatrixAdapter``."""

    def _apply_pre_rendered_html(self, content: dict, metadata: dict) -> str | None:
        """One authoritative card or a definite failure; NEVER truncate its audit fallback."""
        safe_html = _sanitize_matrix_html(str(metadata.get("matrix_formatted_body") or ""),
                                          allow_collapsible=bool(metadata.get("matrix_html_collapsible")))
        if safe_html.strip():
            content.update(format="org.matrix.custom.html", formatted_body=safe_html)
        if max(len(str(content.get(key) or "")) for key in ("body", "formatted_body")) > self.max_message_length:
            return f"Matrix pre-rendered message exceeds the configured {self.max_message_length}-character transport limit"
        return None

    async def _send_exec_approval_prompt(self, prompt: ExecApprovalPrompt) -> SendResult:
        """Reaction-driven approval: the bot seeds one reaction per offered choice."""
        request_id = str(prompt.request_id or "").strip()
        if not request_id:
            # resolve_gateway_approval without an id falls through to the oldest
            # waiter. A reaction card that cannot name its request must not be sent.
            return SendResult(success=False, error="Matrix reaction card requires a request_id")
        if not self._client:
            return SendResult(success=False, error="Not connected")
        from plugins.platforms.matrix.approval_cards import (
            force_redact_command, format_pending_expanded, load_matrix_approval_summary_config,
        )
        choices = prompt.choices
        allow_session = "session" in choices
        allow_permanent = "always" in choices
        # Base send_exec_approval offers only Deny when it could not show the whole command.
        deny_only = "once" not in choices
        session_key, chat_id = prompt.session_key, prompt.chat_id
        redacted_command = force_redact_command(prompt.command)
        text, html_body = format_pending_expanded(
            command=redacted_command, description=prompt.description,
            allow_permanent=allow_permanent, allow_session=allow_session,
            smart_denied=prompt.smart_denied, deny_only=deny_only,
        )
        send_meta = {**(prompt.metadata or {}), "matrix_formatted_body": html_body, "matrix_html_collapsible": True}

        def _make(message_id, requester, expires_at):
            self._approval_prompt_by_session.setdefault(session_key, set()).add(message_id)
            return _MatrixApprovalPrompt(
                session_key=session_key, chat_id=chat_id, message_id=message_id, requester_user_id=requester,
                expires_at=expires_at, request_id=request_id,
                command=redacted_command, description=prompt.description or "dangerous command",
                allow_permanent=allow_permanent, allow_session=allow_session,
                smart_denied=prompt.smart_denied, metadata=send_meta, offered=tuple(choices),
            )
        reactions = tuple(self._EA_REACTIONS[c] for c in choices)
        result = await self._send_reaction_prompt(
            chat_id, text, send_meta, _make, self._approval_prompts_by_event, reactions, "approval")
        # _send_reaction_prompt registers the card; an override that does not leaves nothing to observe.
        if result.success and (stored := self._approval_prompts_by_event.get(result.message_id or "")):
            summary_cfg = load_matrix_approval_summary_config()
            if summary_cfg.enabled:
                self._schedule_approval_summary(stored, summary_cfg)
            if not self._observe_approval_settle(stored):
                # Settled before the card landed (an answer typed while it was in flight).
                await self._settle_matrix_approval_prompt(stored, "unknown")
        return result

    async def _handle_approval_reaction(self, room_id: str, reacts_to: str, key: str, sender: str) -> bool:
        """Resolve a pending exec-approval prompt from a reaction. True if it was the target."""
        handled, prompt, choice = await self._claim_reaction_prompt(
            self._approval_prompts_by_event, room_id, reacts_to, key, sender, "approval",
            t("platform.matrix.approval.invalid_reaction"), self._expire_matrix_approval_prompt,
            choices=self._approval_reaction_map)
        if not handled and reacts_to in self._retired_approval_cards:
            await self._send_invalid_reaction_feedback(room_id, reacts_to, t("platform.matrix.approval.expired"))
            return True
        if choice is None:
            return handled
        if not prompt.request_id:
            # Without the core request_id a reaction could only hit the session's oldest waiter,
            # which may be a different command than the card shows. Resolve nothing.
            logger.warning("Matrix: approval card %s carries no request id; reaction ignored", reacts_to)
            return True
        if prompt.offered and choice not in prompt.offered:  # a hand-added ✅ on a Deny-only card: keep it live
            await self._send_invalid_reaction_feedback(room_id, reacts_to, t("platform.matrix.approval.invalid_reaction"))
            return True
        extra = getattr(getattr(self, "config", None), "extra", None)  # object.__new__ test doubles lack config
        if requester := approval_refusal(prompt.session_key, sender, extra=extra, request_id=prompt.request_id,
                                         choice=choice):
            await self._send_invalid_reaction_feedback(room_id, reacts_to, refusal_text(requester))
            return True
        try:
            from tools.approval import resolve_gateway_approval_card
            # Claim the card before resolving: the settle observer fires synchronously inside
            # resolve and must see that this reaction owns the terminal edit.
            prior = (prompt.resolved, prompt.state, prompt.terminal_choice, prompt.terminal_actor)
            prompt.resolved, prompt.state = True, "resolved_core_delivered"
            prompt.terminal_choice, prompt.terminal_actor = choice, sender
            count = resolve_gateway_approval_card(prompt.session_key, choice, prompt.request_id,
                                                  actor_id=sender, extra=extra)
            if not count:
                prompt.resolved, prompt.state, prompt.terminal_choice, prompt.terminal_actor = prior
            if count:
                logger.info(
                    "Matrix reaction resolved %d approval(s) for session %s (choice=%s, user=%s)",
                    count, prompt.session_key, choice, sender)
                await self._redact_bot_approval_reactions(room_id, prompt)
                await self._finalize_matrix_approval_prompt(room_id, reacts_to, prompt, choice=choice, actor=sender)
            else:
                # Already timed out or answered elsewhere: the settle observer writes the real outcome,
                # so this reaction only tells the reactor the card no longer takes answers.
                await self._send_invalid_reaction_feedback(
                    room_id, reacts_to, t("platform.matrix.approval.expired"))
        except Exception as exc:
            logger.error("Failed to resolve gateway approval from Matrix reaction: %s", exc, exc_info=True)
        return True

    def _forget_matrix_approval_prompt(
        self,
        target_event_id: str,
        prompt: _MatrixApprovalPrompt,
    ) -> None:
        """Remove one approval card without disturbing concurrent cards."""
        self._approval_prompts_by_event.pop(target_event_id, None)
        self._retired_approval_cards[target_event_id] = None  # a later reaction is told it expired
        while len(self._retired_approval_cards) > 256:
            self._retired_approval_cards.pop(next(iter(self._retired_approval_cards)))
        events = self._approval_prompt_by_session.get(prompt.session_key)
        if events is None:
            return
        events.discard(target_event_id)
        if not events:
            self._approval_prompt_by_session.pop(prompt.session_key, None)

    def _cancel_approval_summary_task(self, prompt: _MatrixApprovalPrompt) -> None:
        task = getattr(prompt, "summary_task", None)
        if task is None:
            return
        if hasattr(task, "done") and not task.done():
            task.cancel()
        prompt.summary_task = None

    def _schedule_approval_summary(self, prompt: _MatrixApprovalPrompt, summary_cfg) -> None:
        """Fire-and-forget summary generation for a pending approval card."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def _runner() -> None:
            from plugins.platforms.matrix.approval_cards import (
                format_pending_summarized,
                generate_command_summary,
            )

            expected_gen = prompt.generation
            summary = await asyncio.to_thread(
                generate_command_summary,
                command=prompt.command,
                description=prompt.description,
                provider_policy=summary_cfg.provider_policy,
                timeout_seconds=summary_cfg.effective_timeout_seconds,
                max_chars=summary_cfg.max_chars,
            )
            if not summary or prompt.resolved or prompt.generation != expected_gen:
                return
            if prompt.state != "pending_expanded":
                return
            body, html_body = format_pending_summarized(
                command=prompt.command,
                description=prompt.description,
                summary=summary,
                allow_permanent=prompt.allow_permanent,
                allow_session=prompt.allow_session,
                smart_denied=prompt.smart_denied,
                deny_only=bool(prompt.offered) and "once" not in prompt.offered,
            )
            edit_meta = {"matrix_formatted_body": html_body, "matrix_html_collapsible": True} if html_body else None
            async with prompt.presentation_lock:
                if (
                    prompt.resolved
                    or prompt.generation != expected_gen
                    or prompt.state != "pending_expanded"
                ):
                    return
                try:
                    result = await self.edit_message(
                        prompt.chat_id,
                        prompt.message_id,
                        body,
                        metadata=edit_meta,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("Matrix: approval summary edit failed: %s", exc, exc_info=True)
                    return
                if not getattr(result, "success", False):
                    logger.warning(
                        "Matrix: approval summary edit failed: %s",
                        getattr(result, "error", None) or "unknown edit failure",
                    )
                    return
                # Resolution may have started while the homeserver edit was in flight.
                # The finalizer waits on this lock and will reassert terminal content.
                if prompt.resolved or prompt.generation != expected_gen:
                    return
                prompt.summary = summary
                prompt.state = "pending_summarized"
                prompt.generation += 1

        prompt.summary_task = loop.create_task(_runner())

    async def _finalize_matrix_approval_prompt(
        self,
        room_id: str,
        target_event_id: str,
        prompt: _MatrixApprovalPrompt,
        *,
        choice: str,
        actor: str = "",
        max_attempts: int = 3,
    ) -> None:
        """Terminal card compaction after resolve/expire.

        Core resolution and UI terminalization are separate. The registry entry
        is retained until a terminal m.replace succeeds.
        Each attempt is bounded; failed delivery remains retryable.
        """
        task = getattr(prompt, "summary_task", None)
        self._cancel_approval_summary_task(prompt)
        # Await cancellation explicitly so CancelledError cannot skip terminal
        # compaction and no summary replacement can land after the terminal one.
        if task is not None and task is not asyncio.current_task():
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug("Matrix: approval summary task failed during finalize: %s", exc, exc_info=True)
        from plugins.platforms.matrix.approval_cards import format_terminal_compact

        async with prompt.presentation_lock:
            if prompt.terminal_visible:
                # Already compacted (e.g. watcher + reaction both fired).
                self._forget_matrix_approval_prompt(target_event_id, prompt)
                return
            if prompt.terminal_choice is None:
                prompt.terminal_choice, prompt.terminal_actor = choice, actor
            choice, actor = prompt.terminal_choice, prompt.terminal_actor
            prompt.state = "terminal_edit_pending"
            body, html_body = format_terminal_compact(
                choice=choice,
                command=prompt.command,
                description=prompt.description,
                actor=actor or "",
                summary=getattr(prompt, "summary", "") or "",
            )
            edit_meta = {"matrix_formatted_body": html_body, "matrix_html_collapsible": True} if html_body else None
            last_error = None
            for attempt in range(1, max(1, int(max_attempts)) + 1):
                try:
                    result = await self.edit_message(
                        room_id,
                        target_event_id,
                        body,
                        metadata=edit_meta,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "Matrix: terminal approval edit failed (attempt %d/%d): %s",
                        attempt,
                        max_attempts,
                        exc,
                        exc_info=True,
                    )
                    result = None
                if result is not None and getattr(result, "success", False):
                    prompt.terminal_visible = True
                    prompt.state = f"terminal_{choice}"
                    prompt.generation += 1
                    self._forget_matrix_approval_prompt(target_event_id, prompt)
                    return
                if result is not None:
                    last_error = getattr(result, "error", None) or "unknown edit failure"
                    logger.warning(
                        "Matrix: terminal approval edit failed (attempt %d/%d): %s",
                        attempt,
                        max_attempts,
                        last_error,
                    )
                if attempt < max_attempts:
                    await asyncio.sleep(min(0.5 * attempt, 2.0))
            # A separate notice cannot replace the authoritative card. Keep
            # retrying the replacement without flooding the room with notices.
            logger.error(
                "Matrix: terminal approval edit exhausted retries for %s: %s",
                target_event_id,
                last_error,
            )
            if prompt.terminal_failure_notified:
                return
            try:
                prompt.terminal_failure_notified = await self._send_invalid_reaction_feedback(
                    room_id,
                    target_event_id,
                    f"Approval outcome: {choice}. Updating the Matrix card failed. "
                    "This prompt is no longer actionable.",
                )
            except Exception:
                logger.warning("Matrix: approval failure notice could not be sent", exc_info=True)
                prompt.terminal_failure_notified = False

    def _observe_approval_settle(self, prompt: _MatrixApprovalPrompt) -> bool:
        """Compact the card when its core request settles by any path other than this card's reaction:
        typed ``/approve``/``/deny``, the session ending, or the timeout. Event-driven through
        ``observe_gateway_settle`` (no polling). On ``timeout`` the shared timeout notice (foreground turn
        or ``/background`` task) edits the card to the timed-out body, so only the reactions are withdrawn."""
        from tools.approval import observe_gateway_settle
        loop = asyncio.get_running_loop()

        def settle(reason: str) -> None:
            def schedule() -> None:
                if prompt.terminal_visible or prompt.terminal_choice is not None:
                    return  # this card's own reaction resolved it and owns the terminal edit
                task = loop.create_task(self._settle_matrix_approval_prompt(prompt, reason))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            loop.call_soon_threadsafe(schedule)

        return observe_gateway_settle(prompt.session_key, prompt.request_id, settle)

    async def _settle_matrix_approval_prompt(self, prompt: _MatrixApprovalPrompt, reason: str) -> None:
        prompt.resolved = True
        prompt.state = "resolved_core_delivered"
        await self._redact_bot_approval_reactions(prompt.chat_id, prompt)
        if reason not in ("resolved", "session_closed", "interrupted"):
            # timeout: the shared timeout notice rewrites the card; unknown: claim nothing about the outcome.
            self._cancel_approval_summary_task(prompt)
            self._forget_matrix_approval_prompt(prompt.message_id, prompt)
            return
        choice = "resolved" if reason == "resolved" else "withdrawn"
        await self._finalize_matrix_approval_prompt(prompt.chat_id, prompt.message_id, prompt, choice=choice)

    async def _expire_matrix_approval_prompt(
        self,
        room_id: str,
        target_event_id: str,
        prompt: _MatrixApprovalPrompt,
    ) -> None:
        prompt.resolved = True
        if prompt.terminal_choice is None:
            prompt.terminal_choice = "expired"
        await self._redact_bot_approval_reactions(room_id, prompt)
        await self._finalize_matrix_approval_prompt(
            room_id, target_event_id, prompt, choice=prompt.terminal_choice, actor=prompt.terminal_actor,
        )
        if prompt.terminal_choice == "expired":
            await self._send_invalid_reaction_feedback(
                room_id, target_event_id,
                t("platform.matrix.approval.expired"))
