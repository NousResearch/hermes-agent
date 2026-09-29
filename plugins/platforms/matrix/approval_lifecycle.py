"""Matrix approval controls and card lifecycle."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from gateway.platforms.base import ExecApprovalPrompt, SendResult
from gateway.platforms.base_exec_approval import EA_HEADER_TEXT

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
        approval_id: str | None = None,
        command: str = "",
        description: str = "",
        allow_permanent: bool = True,
        allow_session: bool = True,
        smart_denied: bool = False,
        metadata: dict | None = None,
    ):
        self.session_key = session_key
        self.chat_id = chat_id
        self.message_id = message_id
        self.approval_id = approval_id
        self.resolved = resolved
        self.requester_user_id = requester_user_id
        self.expires_at = expires_at
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


class MatrixApprovalMixin:
    """Present exact core approval requests through Matrix cards."""

    # Template attrs for the shared _format_exec_approval core (header + fence + reason only;
    # the smart-deny/scope wording lives in the reaction legend below).
    _EA_HEADER = f"⚠️ **{EA_HEADER_TEXT}**\n"

    _EA_CMD_BUDGET = 2000

    _EA_REACTIONS = {"once": "✅", "session": "🌀", "always": "♾️", "deny": "❌"}

    _EA_LEGEND = {"once": "✅ = approve once", "session": "🌀 = approve for this session",
                  "always": "♾️ = approve always", "deny": "❎ = deny"}

    _EA_TYPED_HINT = {"session": "Reply `!approve session` to approve this pattern for the session, ",
                      "always": "`!approve always` to approve permanently, "}

    async def _send_exec_approval_prompt(self, prompt: ExecApprovalPrompt) -> SendResult:
        """Reaction-driven approval: the bot seeds one reaction per offered choice."""
        if not self._client:
            return SendResult(success=False, error="Not connected")
        from plugins.platforms.matrix.approval_cards import (
            force_redact_command, format_pending_expanded, load_matrix_approval_summary_config,
        )
        choices = prompt.choices
        allow_session = "session" in choices
        allow_permanent = "always" in choices
        session_key, chat_id = prompt.session_key, prompt.chat_id
        redacted_command = force_redact_command(prompt.command)
        text, html_body = format_pending_expanded(
            command=redacted_command, description=prompt.description,
            allow_permanent=allow_permanent, allow_session=allow_session,
            smart_denied=prompt.smart_denied,
        )
        send_meta = {**(prompt.metadata or {}), "matrix_formatted_body": html_body}

        def _make(message_id, requester, expires_at):
            self._approval_prompt_by_session.setdefault(session_key, set()).add(message_id)
            return _MatrixApprovalPrompt(
                session_key=session_key, chat_id=chat_id, message_id=message_id, requester_user_id=requester,
                expires_at=expires_at, approval_id=str(send_meta.get("approval_id") or "") or None,
                command=redacted_command, description=prompt.description or "dangerous command",
                allow_permanent=allow_permanent, allow_session=allow_session,
                smart_denied=prompt.smart_denied, metadata=send_meta,
            )
        reactions = tuple(self._EA_REACTIONS[c] for c in choices)
        result = await self._send_reaction_prompt(
            chat_id, text, send_meta, _make, self._approval_prompts_by_event, reactions, "approval")
        if result.success and result.message_id:
            stored = self._approval_prompts_by_event[result.message_id]
            summary_cfg = load_matrix_approval_summary_config()
            if summary_cfg.enabled:
                self._schedule_approval_summary(stored, summary_cfg)
            self._schedule_approval_resolution_watch(stored)
        return result

    async def _handle_approval_reaction(self, room_id: str, reacts_to: str, key: str, sender: str) -> bool:
        """Resolve a pending exec-approval prompt from a reaction. True if it was the target."""
        handled, prompt, choice = await self._claim_reaction_prompt(
            self._approval_prompts_by_event, room_id, reacts_to, key, sender, "approval",
            "That reaction is not valid for this approval prompt.", self._expire_matrix_approval_prompt,
            choices=self._approval_reaction_map)
        if choice is None:
            return handled
        try:
            from tools.approval import consume_gateway_approval_outcome, resolve_gateway_approval
            count = resolve_gateway_approval(
                prompt.session_key, choice,
                **({"approval_id": prompt.approval_id} if prompt.approval_id else {}),
            )
            if count:
                prompt.resolved = True
                prompt.state = "resolved_core_delivered"
                prompt.terminal_choice, prompt.terminal_actor = choice, sender
                consume_gateway_approval_outcome(prompt.session_key, prompt.approval_id)
                logger.info(
                    "Matrix reaction resolved %d approval(s) for session %s (choice=%s, user=%s)",
                    count, prompt.session_key, choice, sender)
                await self._redact_bot_approval_reactions(room_id, prompt)
                await self._finalize_matrix_approval_prompt(room_id, reacts_to, prompt, choice=choice, actor=sender)
        except Exception as exc:
            logger.error("Failed to resolve gateway approval from Matrix reaction: %s", exc)
        return True

    def _forget_matrix_approval_prompt(
        self,
        target_event_id: str,
        prompt: "_MatrixApprovalPrompt",
    ) -> None:
        """Remove one approval card without disturbing concurrent cards."""
        self._approval_prompts_by_event.pop(target_event_id, None)
        events = self._approval_prompt_by_session.get(prompt.session_key)
        if events is None:
            return
        if isinstance(events, set):
            events.discard(target_event_id)
            empty = not events
        else:
            # Legacy in-memory representation stored one event id directly.
            empty = events == target_event_id
        if empty:
            self._approval_prompt_by_session.pop(prompt.session_key, None)

    def _cancel_approval_summary_task(self, prompt: "_MatrixApprovalPrompt") -> None:
        task = getattr(prompt, "summary_task", None)
        if task is None:
            return
        try:
            if hasattr(task, "done") and not task.done():
                task.cancel()
        except Exception:
            pass
        prompt.summary_task = None

    def _schedule_approval_summary(self, prompt: "_MatrixApprovalPrompt", summary_cfg) -> None:
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
            )
            edit_meta = {"matrix_formatted_body": html_body} if html_body else None
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
                    logger.warning("Matrix: approval summary edit failed: %s", exc)
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
        prompt: "_MatrixApprovalPrompt",
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
                logger.debug("Matrix: approval summary task failed during finalize: %s", exc)
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
            edit_meta = {"matrix_formatted_body": html_body} if html_body else None
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
                prompt.terminal_failure_notified = False

    def _schedule_approval_resolution_watch(self, prompt: "_MatrixApprovalPrompt") -> None:
        """Compact the card when the core queue resolves without a reaction."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def _watch() -> None:
            from tools.approval import (
                consume_gateway_approval_outcome,
                has_blocking_approval,
            )

            # Single authoritative deadline was stored on the prompt at
            # registration (prompt.expires_at). Watch until that deadline plus
            # a small grace — do not recompute/extend from config each loop.
            grace_seconds = 5.0
            poll = 0.5
            while True:
                if prompt.terminal_visible:
                    return
                now = time.monotonic()
                expires_at = getattr(prompt, "expires_at", None)
                past_deadline = expires_at is not None and now > float(expires_at) + grace_seconds
                try:
                    pending = has_blocking_approval(
                        prompt.session_key,
                        approval_id=prompt.approval_id,
                    )
                except Exception:
                    pending = True
                if not pending or past_deadline:
                    if prompt.terminal_visible:
                        return
                    choice = prompt.terminal_choice or consume_gateway_approval_outcome(
                        prompt.session_key,
                        prompt.approval_id,
                    )
                    if choice is None:
                        # Reaction/typed path already consumed the outcome and
                        # owns terminalization. Do not stamp Expired over it.
                        if getattr(prompt, "resolved", False):
                            await asyncio.sleep(poll)
                            continue
                        if pending and not past_deadline:
                            await asyncio.sleep(poll)
                            continue
                        choice = "expired"
                    if prompt.terminal_choice is None:
                        prompt.terminal_choice = choice
                    if not prompt.resolved:
                        prompt.resolved = True
                        prompt.state = "resolved_core_delivered"
                    try:
                        await self._redact_bot_approval_reactions(prompt.chat_id, prompt)
                    except Exception:
                        pass
                    await self._finalize_matrix_approval_prompt(
                        prompt.chat_id,
                        prompt.message_id,
                        prompt,
                        choice=choice,
                        actor=prompt.terminal_actor,
                    )
                    if prompt.terminal_visible:
                        return
                    # Retry delivery without forgetting the already-recorded decision.
                    await asyncio.sleep(5.0)
                    continue
                await asyncio.sleep(poll)

        loop.create_task(_watch())

    async def _expire_matrix_approval_prompt(
        self,
        room_id: str,
        target_event_id: str,
        prompt: "_MatrixApprovalPrompt",
    ) -> None:
        from tools.approval import consume_gateway_approval_outcome

        prompt.resolved = True
        if prompt.terminal_choice is None:
            # Typed consent may have won before the deadline while the watcher
            # was asleep. Expiry must not overwrite that exact core decision.
            prompt.terminal_choice = consume_gateway_approval_outcome(
                prompt.session_key, prompt.approval_id,
            ) or "expired"
        await self._redact_bot_approval_reactions(room_id, prompt)
        await self._finalize_matrix_approval_prompt(
            room_id, target_event_id, prompt, choice=prompt.terminal_choice, actor=prompt.terminal_actor,
        )
        if prompt.terminal_choice == "expired":
            await self._send_invalid_reaction_feedback(
                room_id,
                target_event_id,
                "This approval prompt has expired. Run the command again if you still want to approve it.",
            )

    async def _redact_bot_approval_reactions(self, room_id: str, prompt: Any) -> None:
        """Redact the bot's seeded approval reactions (delayed), leaving only the user's reaction."""
        for emoji, evt_id in prompt.bot_reaction_events.items():
            self._schedule_reaction_redaction(room_id, evt_id, "approval resolved")
            logger.debug("Matrix: scheduled bot reaction redaction %s (%s)", emoji, evt_id)
