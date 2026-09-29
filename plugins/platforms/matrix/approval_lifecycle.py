"""Matrix approval controls and card lifecycle."""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from contextvars import Context, copy_context
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, TypeVar

from gateway.platforms.base import ExecApprovalPrompt, SendResult
from agent.i18n import t

if TYPE_CHECKING:
    from plugins.platforms.matrix.approval_cards import MatrixApprovalSummaryConfig

logger = logging.getLogger(__name__)

_Result = TypeVar("_Result")

# Edit failures that repeat identically on every attempt: the bot cannot write to the room, the card
# is gone, or the terminal card is larger than the transport allows.
_PERMANENT_EDIT_ERROR_KINDS = frozenset({"forbidden", "not_found", "too_long"})


@dataclass(frozen=True)
class _TerminalEditRetry:
    """Backoff for terminal card edits that fail with a transient error."""

    first_delay: float = 2.0
    max_delay: float = 60.0
    horizon: float = 600.0

    def delays(self) -> Iterator[float]:
        delay = self.first_delay
        while True:
            yield delay
            delay = min(delay * 2, self.max_delay)


_TERMINAL_EDIT_RETRY = _TerminalEditRetry()


class _TerminalEdit(enum.Enum):
    """Result of one attempt to replace a card with its outcome."""

    VISIBLE = "visible"
    RETRY = "retry"
    PERMANENT = "permanent"


@dataclass
class _MatrixApprovalPrompt:
    """One card for an exact approval, with the profile context from its request."""

    session_key: str
    chat_id: str
    message_id: str
    resolved: bool = False
    requester_user_id: str | None = None
    expires_at: float | None = None
    approval_id: str | None = None
    command: str = ""
    description: str = ""
    allow_permanent: bool = True
    allow_session: bool = True
    smart_denied: bool = False
    metadata: dict = field(default_factory=dict)
    owner_context: Context = field(default_factory=copy_context, repr=False)
    bot_reaction_events: dict[str, str] = field(default_factory=dict)
    generation: int = 0
    state: str = "pending_expanded"
    summary: str = ""
    summary_task: asyncio.Task | None = None
    lifecycle_task: asyncio.Task | None = None
    presentation_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    terminal_visible: bool = False
    terminal_failure_notified: bool = False
    terminal_choice: str | None = None
    terminal_actor: str = ""


class MatrixApprovalMixin:
    """Present exact core approval requests through Matrix cards."""

    if TYPE_CHECKING:
        _client: Any
        _closing: bool
        _approval_reaction_map: dict[str, str]
        _approval_prompts_by_event: dict[str, _MatrixApprovalPrompt]
        _approval_prompt_by_session: dict[str, set[str]]
        _approval_tasks: set[asyncio.Task[Any]]

        async def _send_reaction_prompt(
            self, chat_id: str, text: str, metadata: dict | None, make_prompt: Callable,
            registry: dict, emojis: tuple[str, ...], label: str,
        ) -> SendResult: ...

        async def _claim_reaction_prompt(
            self, registry: dict, room_id: str, reacts_to: str, key: str, sender: str,
            label: str, invalid_text: str, on_expired: Callable, choices: dict | None = None,
        ) -> tuple[bool, Any, Any]: ...

        async def edit_message(
            self, chat_id: str, message_id: str, content: str, *, finalize: bool = False,
            metadata: dict[str, Any] | None = None,
        ) -> SendResult: ...

        async def _send_invalid_reaction_feedback(self, room_id: str, target_event_id: str, text: str) -> bool: ...
        def _matrix_prompt_expired(self, prompt: Any) -> bool: ...
        def _schedule_reaction_redaction(self, room_id: str, reaction_event_id: str, reason: str = "") -> None: ...
        async def _redact_reaction(self, room_id: str, reaction_event_id: str, reason: str = "") -> bool: ...

    manages_exec_approval_lifecycle = True

    # The gateway gives each adapter's disconnect 5 seconds by default, and the rest of disconnect needs time too.
    _approval_close_timeout: float = 2.0
    _approval_clock: Callable[[], float] = staticmethod(time.monotonic)
    _approval_sleep: Callable[[float], Awaitable[None]] = staticmethod(asyncio.sleep)

    # Template attrs for the shared _format_exec_approval core (header + fence + reason only;
    # the smart-deny/scope wording lives in the reaction legend below).
    @property
    def _EA_HEADER(self) -> str:
        return f"⚠️ **{t('gateway.exec_approval.header')}**\n"

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
        owner_context = copy_context()
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
                smart_denied=prompt.smart_denied, metadata=send_meta, owner_context=owner_context,
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
        from .adapter import t

        prompt = self._approval_prompts_by_event.get(reacts_to)
        if prompt is None:
            return False
        if prompt.resolved:
            # A decided card keeps its reactions until its terminal edit lands, so no other
            # reaction handler acts on an approval card.
            return True
        task = self._create_approval_task(
            self._resolve_matrix_approval_reaction(room_id, reacts_to, key, sender, prompt),
            prompt,
        )
        return await task

    async def _resolve_matrix_approval_reaction(
        self, room_id: str, reacts_to: str, key: str, sender: str, prompt: _MatrixApprovalPrompt,
    ) -> bool:
        choices = {
            emoji: choice for emoji, choice in self._approval_reaction_map.items()
            if choice in {"once", "deny"}
            or (choice == "session" and prompt.allow_session and not prompt.smart_denied)
            or (choice == "always" and prompt.allow_session and prompt.allow_permanent and not prompt.smart_denied)
        }
        handled, prompt, choice = await self._claim_reaction_prompt(
            self._approval_prompts_by_event, room_id, reacts_to, key, sender, "approval",
            t("platform.matrix.approval.invalid_reaction"), self._expire_matrix_approval_prompt,
            choices=choices,
        )
        if choice is None:
            return handled
        from tools.approval import resolve_gateway_approval

        count = resolve_gateway_approval(
            prompt.session_key, choice,
            **({"approval_id": prompt.approval_id} if prompt.approval_id else {}),
        )
        if not count:
            return True
        prompt.resolved = True
        prompt.terminal_choice, prompt.terminal_actor = choice, sender
        await self._redact_bot_approval_reactions(room_id, prompt)
        await self._finalize_matrix_approval_prompt(room_id, reacts_to, prompt, choice=choice, actor=sender)
        return True

    def _create_approval_task(self, coroutine: Coroutine[Any, Any, _Result], prompt: _MatrixApprovalPrompt) -> asyncio.Task[_Result]:
        tasks = getattr(self, "_approval_tasks", None)
        if tasks is None:
            tasks = self._approval_tasks = set()
        task = asyncio.get_running_loop().create_task(coroutine, context=prompt.owner_context.copy())
        tasks.add(task)
        task.add_done_callback(tasks.discard, context=prompt.owner_context.copy())
        from agent.async_utils import consume_detached_task_result
        task.add_done_callback(consume_detached_task_result, context=prompt.owner_context.copy())
        return task

    async def _close_matrix_approvals(self) -> None:
        """Withdraw unanswered requests and make one bounded attempt to show each card's outcome.

        Disconnect calls this while the client is still open. A card whose edit fails or runs
        out of time, or any card after a crash, still looks pending after a restart.
        """
        from tools.approval import consume_gateway_approval_outcome, withdraw_gateway_approval

        prompts = list(self._approval_prompts_by_event.values())
        for prompt in prompts:
            if not prompt.resolved and prompt.approval_id:
                prompt.owner_context.run(
                    withdraw_gateway_approval, prompt.session_key, prompt.approval_id,
                    "the Matrix connection closed before the prompt was answered",
                )
            prompt.resolved = True
            if prompt.terminal_choice is None:
                prompt.terminal_choice = consume_gateway_approval_outcome(
                    prompt.session_key, prompt.approval_id,
                ) or "session_closed"
        tasks = list(getattr(self, "_approval_tasks", set()))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        edits = [
            self._create_approval_task(self._show_closed_approval(prompt), prompt)
            for prompt in prompts
            if not prompt.terminal_visible and prompt.state != "terminal_undeliverable"
        ]
        if edits:
            _done, unfinished = await asyncio.wait(edits, timeout=self._approval_close_timeout)
            for task in unfinished:
                task.cancel()
            await asyncio.gather(*edits, return_exceptions=True)
            if unfinished:
                logger.warning("Matrix: %d approval card(s) were not updated before disconnect", len(unfinished))
        for prompt in prompts:
            self._forget_matrix_approval_prompt(prompt.message_id, prompt)

    async def _show_closed_approval(self, prompt: _MatrixApprovalPrompt) -> None:
        seeded, prompt.bot_reaction_events = prompt.bot_reaction_events, {}
        await asyncio.gather(
            self._finalize_matrix_approval_prompt(
                prompt.chat_id, prompt.message_id, prompt,
                choice=prompt.terminal_choice or "session_closed", actor=prompt.terminal_actor,
            ),
            *(self._redact_reaction(prompt.chat_id, event_id, "approval closed") for event_id in seeded.values()),
            return_exceptions=True,
        )

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
        if not task.done():
            task.cancel()
        prompt.summary_task = None

    def _schedule_approval_summary(self, prompt: "_MatrixApprovalPrompt", summary_cfg: MatrixApprovalSummaryConfig) -> None:
        """Fire-and-forget summary generation for a pending approval card."""
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
                remote_timeout_seconds=summary_cfg.remote_timeout_seconds,
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

        prompt.summary_task = self._create_approval_task(_runner(), prompt)

    async def _finalize_matrix_approval_prompt(
        self,
        room_id: str,
        target_event_id: str,
        prompt: "_MatrixApprovalPrompt",
        *,
        choice: str,
        actor: str = "",
    ) -> _TerminalEdit:
        """Make one attempt to replace the card with its outcome.

        Core resolution and UI terminalization are separate. The registry entry
        is retained until a terminal m.replace succeeds.
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
                return _TerminalEdit.VISIBLE
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
            try:
                result = await self.edit_message(room_id, target_event_id, body, metadata=edit_meta)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                result = SendResult(success=False, error=str(exc))
            if getattr(result, "success", False):
                prompt.terminal_visible = True
                prompt.state = f"terminal_{choice}"
                prompt.generation += 1
                self._forget_matrix_approval_prompt(target_event_id, prompt)
                return _TerminalEdit.VISIBLE
            logger.warning(
                "Matrix: terminal approval edit failed for %s: %s",
                target_event_id,
                getattr(result, "error", None) or "unknown edit failure",
            )
            if not prompt.terminal_failure_notified:
                # A separate notice cannot replace the authoritative card, so it is sent once.
                try:
                    prompt.terminal_failure_notified = await self._send_invalid_reaction_feedback(
                        room_id,
                        target_event_id,
                        f"Approval outcome: {choice}. Updating the Matrix card failed. "
                        "This prompt is no longer actionable.",
                    )
                except Exception:
                    prompt.terminal_failure_notified = False
            if getattr(result, "error_kind", None) in _PERMANENT_EDIT_ERROR_KINDS:
                return _TerminalEdit.PERMANENT
            return _TerminalEdit.RETRY

    def _schedule_approval_resolution_watch(self, prompt: _MatrixApprovalPrompt) -> None:
        """Compact the card when the core queue resolves without a reaction."""
        from tools.approval import register_gateway_settle

        loop = asyncio.get_running_loop()

        def completed(reason: str) -> None:
            if getattr(self, "_closing", False):
                return
            loop.call_soon_threadsafe(
                self._start_matrix_approval_completion, prompt, reason,
                context=prompt.owner_context.copy(),
            )

        if prompt.approval_id and register_gateway_settle(prompt.session_key, prompt.approval_id, completed):
            return

        async def observe() -> None:
            from tools.approval import has_blocking_approval

            while not prompt.terminal_visible:
                pending = has_blocking_approval(prompt.session_key, approval_id=prompt.approval_id)
                if not pending or self._matrix_prompt_expired(prompt):
                    await self._complete_matrix_approval(prompt, "timeout" if pending else "resolved")
                    return
                await asyncio.sleep(0.5)

        prompt.lifecycle_task = self._create_approval_task(observe(), prompt)

    def _start_matrix_approval_completion(self, prompt: _MatrixApprovalPrompt, reason: str) -> None:
        prompt.lifecycle_task = self._create_approval_task(self._complete_matrix_approval(prompt, reason), prompt)

    async def _complete_matrix_approval(self, prompt: _MatrixApprovalPrompt, reason: str) -> None:
        from tools.approval import consume_gateway_approval_outcome

        outcome = consume_gateway_approval_outcome(prompt.session_key, prompt.approval_id)
        if getattr(self, "_closing", False) or prompt.terminal_visible:
            return
        if prompt.terminal_choice is None:
            if reason == "session_closed" and outcome == "expired" and self._matrix_prompt_expired(prompt):
                reason = "timeout"
            prompt.terminal_choice = (
                reason if reason in {"interrupted", "session_closed", "notify_failed"}
                else outcome or {"timeout": "expired"}.get(reason, reason)
            )
        prompt.resolved = True
        await self._redact_bot_approval_reactions(prompt.chat_id, prompt)
        started = self._approval_clock()
        delays = _TERMINAL_EDIT_RETRY.delays()
        while not getattr(self, "_closing", False) and not prompt.terminal_visible:
            edit = await self._finalize_matrix_approval_prompt(
                prompt.chat_id, prompt.message_id, prompt,
                choice=prompt.terminal_choice, actor=prompt.terminal_actor,
            )
            if prompt.terminal_visible or getattr(self, "_closing", False):
                return
            if edit is _TerminalEdit.PERMANENT:
                prompt.state = "terminal_undeliverable"
                logger.error(
                    "Matrix: gave up replacing approval card %s with outcome %s: "
                    "the edit cannot succeed",
                    prompt.message_id, prompt.terminal_choice,
                )
                return
            delay = next(delays)
            elapsed = self._approval_clock() - started
            if elapsed + delay > _TERMINAL_EDIT_RETRY.horizon:
                logger.error(
                    "Matrix: gave up replacing approval card %s with outcome %s after %.0f seconds",
                    prompt.message_id, prompt.terminal_choice, elapsed,
                )
                return
            await self._approval_sleep(delay)

    async def _expire_matrix_approval_prompt(
        self,
        room_id: str,
        target_event_id: str,
        prompt: "_MatrixApprovalPrompt",
    ) -> None:
        from .adapter import t

        from tools.approval import consume_gateway_approval_outcome, resolve_gateway_approval

        resolve_gateway_approval(prompt.session_key, "deny", approval_id=prompt.approval_id)
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
                t("platform.matrix.approval.expired"),
            )

    async def _redact_bot_approval_reactions(self, room_id: str, prompt: _MatrixApprovalPrompt) -> None:
        """Redact the bot's seeded approval reactions (delayed), leaving only the user's reaction."""
        seeded, prompt.bot_reaction_events = prompt.bot_reaction_events, {}
        for emoji, evt_id in seeded.items():
            self._schedule_reaction_redaction(room_id, evt_id, "approval resolved")
            logger.debug("Matrix: scheduled bot reaction redaction %s (%s)", emoji, evt_id)
