"""Stateful delivery helpers for editable tool-progress bubbles.

The gateway turn loop owns queueing, throttling, and typing indicators.  This
module owns the mutable message ledger and the rollover/final-drain operations
that must agree about which content is confirmed visible, still pending, or
ACK-ambiguous.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

from gateway.platforms.base import (
    BasePlatformAdapter,
    _custom_unit_to_cp,
    classify_send_error,
)

logger = logging.getLogger(__name__)


class ProgressDeliveryState:
    """Own one turn's editable progress-bubble ledger and transitions."""

    def __init__(self, ctx: Any, adapter: Any) -> None:
        self.ctx = ctx
        self.adapter = adapter
        self.progress_lines: list[Any] = []
        self.delivered_progress_lines: list[Any] = []
        self.progress_msg_id: str | None = None
        self.pending_ambiguous_edit: tuple[str, str, list[Any]] | None = None
        self.can_edit = ctx.progress_grouping != "separate"
        self.last_edit_ts = 0.0

        self.progress_len_fn = (
            adapter.message_len_fn if isinstance(adapter, BasePlatformAdapter) else len
        )
        try:
            raw_progress_limit = int(
                getattr(adapter, "MAX_MESSAGE_LENGTH", 4000) or 4000
            )
        except Exception:
            raw_progress_limit = 4000
        if isinstance(adapter, BasePlatformAdapter):
            try:
                raw_progress_limit = int(
                    adapter.max_message_length_for_chat(ctx.source.chat_id) or 4000
                )
                self.progress_len_fn = adapter.message_len_fn_for_chat(
                    ctx.source.chat_id
                )
            except Exception:
                pass
        self.text_limit = max(
            1,
            raw_progress_limit - (64 if raw_progress_limit > 128 else 0),
        )

        self.edit_accepts_metadata = False
        if ctx._progress_metadata:
            try:
                edit_params = inspect.signature(adapter.edit_message).parameters
                self.edit_accepts_metadata = "metadata" in edit_params or any(
                    param.kind is inspect.Parameter.VAR_KEYWORD
                    for param in edit_params.values()
                )
            except (TypeError, ValueError):
                self.edit_accepts_metadata = False

    @property
    def adapter_name(self) -> str:
        return str(getattr(self.adapter, "name", "unknown"))

    @staticmethod
    def progress_text(lines: list[Any]) -> str:
        return "\n".join(str(line) for line in lines)

    def split_line(self, line: Any) -> list[str]:
        """Split one logical line without changing its visible content."""
        remaining = str(line)
        chunks: list[str] = []
        while self.progress_len_fn(remaining) > self.text_limit:
            split_at = (
                self.text_limit
                if self.progress_len_fn is len
                else _custom_unit_to_cp(
                    remaining,
                    self.text_limit,
                    self.progress_len_fn,
                )
            )
            split_at = max(1, split_at)
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:]
        chunks.append(remaining)
        return chunks

    def split_groups(self, lines: list[Any]) -> list[list[Any]]:
        """Partition progress lines into platform-sized editable bubbles."""
        groups: list[list[Any]] = []
        current: list[Any] = []
        for raw_line in lines:
            parts = self.split_line(raw_line)
            if len(parts) > 1:
                if current:
                    groups.append(current)
                groups.extend([[part] for part in parts[:-1]])
                current = [parts[-1]]
                continue
            line = parts[0]
            candidate = current + [line]
            if (
                current
                and self.progress_len_fn(self.progress_text(candidate))
                > self.text_limit
            ):
                groups.append(current)
                current = [line]
            else:
                current = candidate
        if current:
            groups.append(current)
        return groups

    def track_result(self, result: Any) -> None:
        if (
            self.ctx._cleanup_progress
            and getattr(result, "success", False)
            and getattr(result, "message_id", None)
        ):
            self.ctx._cleanup_msg_ids.append(str(result.message_id))

    async def send_text(self, text: str) -> Any:
        result = await self.adapter.send(
            chat_id=self.ctx.source.chat_id,
            content=text,
            reply_to=self.ctx._progress_reply_to,
            metadata=self.ctx._progress_metadata,
        )
        self.track_result(result)
        return result

    async def edit_message(self, message_id: str, content: str) -> Any:
        kwargs: dict[str, Any] = {
            "chat_id": self.ctx.source.chat_id,
            "message_id": message_id,
            "content": content,
        }
        if getattr(self.adapter, "REQUIRES_EDIT_FINALIZE", False):
            kwargs["finalize"] = True
        if self.edit_accepts_metadata:
            kwargs["metadata"] = self.ctx._progress_metadata
        return await self.adapter.edit_message(**kwargs)

    @staticmethod
    def edit_was_too_long(result: Any) -> bool:
        error_kind = getattr(result, "error_kind", None)
        if error_kind is not None:
            return error_kind == "too_long"
        return (
            classify_send_error(
                None,
                getattr(result, "error", "") or "",
            )
            == "too_long"
        )

    def undelivered_lines(self) -> list[Any]:
        """Return content added or changed since the last successful edit."""
        shared = 0
        shared_limit = min(len(self.progress_lines), len(self.delivered_progress_lines))
        while (
            shared < shared_limit
            and self.progress_lines[shared] == self.delivered_progress_lines[shared]
        ):
            shared += 1
        return self.progress_lines[shared:]

    def reset_dedup(self) -> None:
        self.ctx.last_progress_msg[0] = None
        self.ctx.repeat_count[0] = 0

    def clear_bubble(self, *, disable_edit: bool = False) -> None:
        self.progress_msg_id = None
        self.progress_lines = []
        self.delivered_progress_lines = []
        self.pending_ambiguous_edit = None
        if disable_edit:
            self.can_edit = False
        self.reset_dedup()

    def drop_attempted_send(
        self,
        *,
        sent_all: bool,
        attempted_line: Any = None,
    ) -> None:
        """Remove a non-idempotent send once its outcome can be ambiguous."""
        self.progress_msg_id = None
        if sent_all:
            self.progress_lines = []
            self.delivered_progress_lines = []
        elif self.progress_lines and self.progress_lines[-1] == attempted_line:
            self.progress_lines.pop()
            shared = 0
            shared_limit = min(
                len(self.progress_lines), len(self.delivered_progress_lines)
            )
            while (
                shared < shared_limit
                and self.progress_lines[shared] == self.delivered_progress_lines[shared]
            ):
                shared += 1
            self.delivered_progress_lines = self.delivered_progress_lines[:shared]
        if not self.progress_lines:
            self.reset_dedup()

    async def start_fresh_bubbles(self, lines: list[Any]) -> bool:
        """Send pending lines as new bubbles and keep the newest editable."""
        groups = self.split_groups(lines)
        if not groups:
            return False

        self.progress_msg_id = None
        self.delivered_progress_lines = []
        for index, group in enumerate(groups):
            self.progress_lines = [
                line for remaining_group in groups[index:] for line in remaining_group
            ]
            self.progress_msg_id = None
            self.delivered_progress_lines = []
            try:
                result = await self.send_text(self.progress_text(group))
            except asyncio.CancelledError:
                self.progress_lines = [
                    line
                    for remaining_group in groups[index + 1 :]
                    for line in remaining_group
                ]
                self.progress_msg_id = None
                self.delivered_progress_lines = []
                if not self.progress_lines:
                    self.reset_dedup()
                raise
            except Exception:
                logger.warning(
                    "[%s] Progress continuation send raised with unknown "
                    "outcome; suppressing duplicate retry",
                    self.adapter_name,
                    exc_info=True,
                )
                self.progress_lines = [
                    line
                    for remaining_group in groups[index + 1 :]
                    for line in remaining_group
                ]
                self.progress_msg_id = None
                self.delivered_progress_lines = []
                if not self.progress_lines:
                    self.reset_dedup()
                return False
            if not result.success:
                if getattr(result, "ambiguous", False):
                    logger.warning(
                        "[%s] Progress continuation outcome is ambiguous; "
                        "suppressing duplicate retry",
                        self.adapter_name,
                    )
                    self.progress_lines = [
                        line
                        for remaining_group in groups[index + 1 :]
                        for line in remaining_group
                    ]
                    self.progress_msg_id = None
                    self.delivered_progress_lines = []
                    if not self.progress_lines:
                        self.reset_dedup()
                    return False
                self.progress_lines = [
                    line
                    for remaining_group in groups[index:]
                    for line in remaining_group
                ]
                self.progress_msg_id = None
                self.delivered_progress_lines = []
                return False
            if not result.message_id:
                self.progress_msg_id = None
                self.progress_lines = []
                self.delivered_progress_lines = []
                if index == len(groups) - 1:
                    self.reset_dedup()
                continue
            self.progress_msg_id = result.message_id
            self.progress_lines = list(group)
            self.delivered_progress_lines = list(group)
        return True

    async def flush_fresh_with_retry(
        self,
        lines: list[Any] | None = None,
    ) -> bool:
        """Give a final fresh-send flush one bounded retry."""
        initial_lines = list(self.progress_lines if lines is None else lines)
        if not initial_lines:
            return True
        sent = await self.start_fresh_bubbles(initial_lines)
        if sent or not self.progress_lines:
            return sent
        return await self.start_fresh_bubbles(list(self.progress_lines))

    async def continue_after_too_long(
        self,
        *,
        retry_failed_send: bool = False,
    ) -> bool:
        """Freeze the full bubble and move only unseen content to a fresh one."""
        pending = self.undelivered_lines()
        if not pending:
            self.clear_bubble()
            return True
        if retry_failed_send:
            return await self.flush_fresh_with_retry(pending)
        return await self.start_fresh_bubbles(pending)

    async def settle_pending_ambiguous_edit(self) -> bool:
        """Retry an ACK-lost edit without changing its identity or payload."""
        if self.pending_ambiguous_edit is None:
            return True
        message_id, payload, tentative_lines = self.pending_ambiguous_edit
        try:
            result = await self.edit_message(message_id, payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "[%s] Pending progress edit retry raised with unknown outcome",
                self.adapter_name,
                exc_info=True,
            )
            return False
        if result.success:
            self.delivered_progress_lines = list(tentative_lines)
            self.pending_ambiguous_edit = None
            return True
        logger.warning(
            "[%s] Pending progress edit remains unresolved; retaining exact payload",
            self.adapter_name,
        )
        return False

    async def settle_edit_during_drain(self) -> bool:
        """Finish one pending edit safely before a cancellation boundary."""
        if not (self.can_edit and self.progress_lines and self.progress_msg_id):
            return True
        if self.pending_ambiguous_edit is not None:
            tentative_lines = list(self.pending_ambiguous_edit[2])
            if not await self.settle_pending_ambiguous_edit():
                # The exact edit may have landed, so never replay its payload as
                # a fresh bubble.  Later lines were not part of that attempt and
                # remain safe to deliver as a fresh suffix.
                self.delivered_progress_lines = tentative_lines
                self.pending_ambiguous_edit = None
                pending = self.undelivered_lines()
                if pending:
                    await self.flush_fresh_with_retry(pending)
                return not self.undelivered_lines()

        full_text = self.progress_text(self.progress_lines)
        saw_ambiguous = False
        result = None
        try:
            result = await self.edit_message(self.progress_msg_id, full_text)
        except asyncio.CancelledError:
            raise
        except Exception:
            saw_ambiguous = True
            logger.warning(
                "[%s] Progress drain edit raised with unknown outcome",
                self.adapter_name,
                exc_info=True,
            )

        saw_ambiguous = saw_ambiguous or bool(getattr(result, "ambiguous", False))
        if result is None or (
            not result.success
            and (saw_ambiguous or getattr(result, "retryable", False))
        ):
            try:
                result = await self.edit_message(self.progress_msg_id, full_text)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "[%s] Progress drain edit retry raised with unknown outcome",
                    self.adapter_name,
                    exc_info=True,
                )
                return True
            saw_ambiguous = saw_ambiguous or bool(getattr(result, "ambiguous", False))

        if result.success:
            self.delivered_progress_lines = list(self.progress_lines)
            return True
        if saw_ambiguous:
            logger.warning(
                "[%s] Progress drain edit remained ambiguous; "
                "suppressing fresh fallback",
                self.adapter_name,
            )
            return True
        if self.edit_was_too_long(result):
            await self.continue_after_too_long(retry_failed_send=True)
            return not self.undelivered_lines()

        pending = self.undelivered_lines()
        if pending:
            await self.flush_fresh_with_retry(pending)
        return not self.undelivered_lines()

    async def roll_overflow_if_needed(self) -> bool:
        """Split an overflowing buffer while preserving delivery boundaries."""
        if self.pending_ambiguous_edit is not None:
            if not await self.settle_pending_ambiguous_edit():
                return True
        if not self.progress_lines or not self.can_edit:
            return False
        groups = self.split_groups(self.progress_lines)
        if len(groups) <= 1:
            return False

        first_text = self.progress_text(groups[0])
        if self.progress_msg_id is not None:
            result = await self.edit_message(self.progress_msg_id, first_text)
            if not result.success:
                if getattr(result, "ambiguous", False):
                    self.pending_ambiguous_edit = (
                        self.progress_msg_id,
                        first_text,
                        list(groups[0]),
                    )
                    self.delivered_progress_lines = list(groups[0])
                    logger.warning(
                        "[%s] Progress overflow edit outcome is ambiguous; "
                        "retaining exact edit payload",
                        self.adapter_name,
                    )
                    return True
                if self.edit_was_too_long(result):
                    logger.info(
                        "[%s] Progress bubble hit the platform edit limit; "
                        "starting a fresh editable bubble",
                        self.adapter_name,
                    )
                    return await self.continue_after_too_long()
                if getattr(result, "retryable", False):
                    logger.debug(
                        "[%s] Transient overflow edit failure — keeping can_edit=True",
                        self.adapter_name,
                    )
                    return True
                self.can_edit = False
                return False
            self.delivered_progress_lines = list(groups[0])
        else:
            await self.start_fresh_bubbles([line for group in groups for line in group])
            return True

        remaining_lines = [line for group in groups[1:] for line in group]
        if remaining_lines:
            await self.start_fresh_bubbles(remaining_lines)
        return True
