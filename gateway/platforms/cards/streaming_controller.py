"""Streaming card state machine for Feishu/Lark.

Manages the full lifecycle of a streaming card reply:

    idle → creating → streaming → completed / aborted / terminated

Four independent accumulator blocks:
    - text      : main answer text
    - reasoning : think/reasoning text (shown while generating)
    - toolUse   : active tool-call tracking
    - cardKit   : card identity / sequence counter (mirrors JS cardKit block)

The controller accepts LLM token chunks via ``add_text_chunk`` /
``add_reasoning_chunk`` / ``start_tool_use`` / ``complete_tool_use`` and
pushes updates to Feishu via a throttled ``flush()`` that calls
``im.v1.message.update`` with TENANT token (the update_message API is
TENANT-only).

Port of openclaw-lark ``src/card/streaming-card-controller.js`` (~1045 lines).
The JS implementation includes CardKit v2 streaming (``streamCardContent``,
``setCardStreamingMode``, ``updateCardKitCard``); this Python port targets the
simpler IM-patch path only (``im.v1.message.update``) which is the fallback
path used when CardKit is unavailable.  The 6-phase state machine and 4-block
accumulator logic are equivalent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

class Phase(str, Enum):
    """Explicit state machine phases (mirrors JS PHASE_TRANSITIONS)."""

    IDLE = "idle"
    CREATING = "creating"
    STREAMING = "streaming"
    COMPLETED = "completed"
    ABORTED = "aborted"
    TERMINATED = "terminated"


# Valid forward transitions (same logic as JS PHASE_TRANSITIONS map).
_ALLOWED_TRANSITIONS: Dict[Phase, frozenset] = {
    Phase.IDLE: frozenset({Phase.CREATING, Phase.ABORTED, Phase.TERMINATED}),
    Phase.CREATING: frozenset({Phase.STREAMING, Phase.ABORTED, Phase.TERMINATED}),
    Phase.STREAMING: frozenset({Phase.COMPLETED, Phase.ABORTED, Phase.TERMINATED}),
    Phase.COMPLETED: frozenset(),
    Phase.ABORTED: frozenset(),
    Phase.TERMINATED: frozenset(),
}

TERMINAL_PHASES: frozenset = frozenset({Phase.COMPLETED, Phase.ABORTED, Phase.TERMINATED})

# Throttle: minimum ms between IM-patch flushes.
_FLUSH_THROTTLE_MS: int = 500


# ---------------------------------------------------------------------------
# Accumulator dataclasses (mirrors JS structured state blocks)
# ---------------------------------------------------------------------------

@dataclass
class _TextBlock:
    """Accumulates answer text across streaming tokens."""

    accumulated_text: str = ""
    completed_text: str = ""
    last_flushed_text: str = ""


@dataclass
class _ReasoningBlock:
    """Accumulates reasoning / think-tag text."""

    accumulated_reasoning_text: str = ""
    reasoning_start_time: Optional[float] = None   # epoch seconds
    reasoning_elapsed_ms: float = 0.0
    is_reasoning_phase: bool = False


@dataclass
class _ToolUseBlock:
    """Tracks the currently active tool call."""

    tool_name: Optional[str] = None
    tool_args: Optional[Any] = None
    tool_result: Optional[Any] = None
    started_at: Optional[float] = None             # epoch seconds
    elapsed_ms: float = 0.0
    is_active: bool = False
    steps: list = field(default_factory=list)       # completed steps


@dataclass
class _CardKitBlock:
    """Feishu card / message identity (IM-patch path only in this port)."""

    card_message_id: Optional[str] = None
    card_kit_sequence: int = 0


# ---------------------------------------------------------------------------
# Card JSON helpers
# ---------------------------------------------------------------------------

def _build_streaming_card(
    text: str = "",
    reasoning_text: Optional[str] = None,
    tool_steps: Optional[list] = None,
    is_streaming: bool = True,
) -> Dict[str, Any]:
    """Return a Feishu interactive card payload dict.

    Produces a minimal card with:
    - Optional reasoning section (collapsed details block)
    - Main text body (markdown)
    - Optional tool-use steps list
    - Footer tag indicating streaming / complete state

    Args:
        text: Main answer text (markdown).
        reasoning_text: Think/reasoning content; omitted if None.
        tool_steps: List of ``{"name": str, "elapsed_ms": float}`` dicts.
        is_streaming: True while reply is in progress; False for final card.

    Returns:
        Card JSON-serialisable dict compatible with Feishu ``interactive`` type.
    """
    elements: list = []

    # Reasoning section
    if reasoning_text:
        elements.append({
            "tag": "markdown",
            "content": f"**Thinking...**\n\n{reasoning_text}",
            "text_align": "left",
        })
        elements.append({"tag": "hr"})

    # Tool-use steps
    if tool_steps:
        steps_md = "\n".join(
            f"- {s.get('name', 'tool')} ({s.get('elapsed_ms', 0):.0f} ms)"
            for s in tool_steps
        )
        elements.append({
            "tag": "markdown",
            "content": f"**Tool calls:**\n{steps_md}",
        })
        elements.append({"tag": "hr"})

    # Main text
    if text:
        elements.append({
            "tag": "markdown",
            "content": text,
        })
    elif is_streaming:
        # Show a placeholder cursor while no text yet
        elements.append({
            "tag": "markdown",
            "content": "▌",
        })

    # Footer status note
    status_label = "Generating..." if is_streaming else "Done"
    elements.append({
        "tag": "note",
        "elements": [{"tag": "plain_text", "content": status_label}],
    })

    return {
        "schema": "2.0",
        "body": {"elements": elements},
    }


def _build_final_card(
    text: str,
    reasoning_text: Optional[str] = None,
    reasoning_elapsed_ms: float = 0.0,
    tool_steps: Optional[list] = None,
    elapsed_ms: float = 0.0,
    is_aborted: bool = False,
    is_error: bool = False,
) -> Dict[str, Any]:
    """Return a completed (non-streaming) card payload.

    Args:
        text: Final answer text (markdown).
        reasoning_text: Accumulated reasoning content; omitted if None/empty.
        reasoning_elapsed_ms: Duration the model spent reasoning (ms).
        tool_steps: Completed tool-call steps list.
        elapsed_ms: Total reply elapsed time (ms).
        is_aborted: True if the reply was cancelled by the user.
        is_error: True if the reply ended with an error.

    Returns:
        Card JSON-serialisable dict.
    """
    elements: list = []

    # Reasoning section (collapsed)
    if reasoning_text:
        elapsed_s = reasoning_elapsed_ms / 1000
        elements.append({
            "tag": "collapsible_panel",
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": f"Reasoning ({elapsed_s:.1f}s)",
                },
            },
            "elements": [{"tag": "markdown", "content": reasoning_text}],
        })

    # Tool-use steps
    if tool_steps:
        steps_md = "\n".join(
            f"- {s.get('name', 'tool')} ({s.get('elapsed_ms', 0):.0f} ms)"
            for s in tool_steps
        )
        elements.append({
            "tag": "markdown",
            "content": f"**Tool calls:**\n{steps_md}",
        })

    # Main answer
    display_text = text or ("Aborted." if is_aborted else "An error occurred." if is_error else "")
    elements.append({"tag": "markdown", "content": display_text})

    # Footer
    total_s = elapsed_ms / 1000
    if is_aborted:
        status = f"Aborted ({total_s:.1f}s)"
    elif is_error:
        status = f"Error ({total_s:.1f}s)"
    else:
        status = f"Done ({total_s:.1f}s)"

    elements.append({
        "tag": "note",
        "elements": [{"tag": "plain_text", "content": status}],
    })

    return {
        "schema": "2.0",
        "body": {"elements": elements},
    }


# ---------------------------------------------------------------------------
# StreamingCardController
# ---------------------------------------------------------------------------

class StreamingCardController:
    """Streaming card state machine for Feishu/Lark IM replies.

    Manages the full lifecycle of one reply from ``idle`` through
    ``streaming`` to a terminal phase (``completed`` / ``aborted`` /
    ``terminated``).

    The controller is **not** thread-safe; use it from a single asyncio task
    or protect external access with a lock.

    Args:
        message_id: The Feishu message ID to update (``om_xxx``).  Used as
            the target for ``im.v1.message.update`` (PATCH).  If None the
            controller operates in "accumulate-only" mode and ``flush()``
            is a no-op.
        client: A ``lark_oapi.Client`` instance initialised with
            ``TENANT`` token type.  Required for the ``flush()`` call.
            May be None for unit-testing without network access.

    Example::

        ctrl = StreamingCardController(message_id="om_xxx", client=lark_client)
        await ctrl.add_text_chunk("Hello ")
        await ctrl.add_text_chunk("world!")
        await ctrl.flush()
        await ctrl.mark_completed()
    """

    def __init__(
        self,
        message_id: Optional[str],
        client: Any,
    ) -> None:
        self._message_id = message_id
        self._client = client

        # ---- State machine ----
        self._phase: Phase = Phase.IDLE
        self._terminal_reason: Optional[str] = None

        # ---- Accumulator blocks ----
        self.text = _TextBlock()
        self.reasoning = _ReasoningBlock()
        self.tool_use = _ToolUseBlock()
        self.card_kit = _CardKitBlock(card_message_id=message_id)

        # ---- Lifecycle ----
        self._dispatch_start_time: float = time.time()
        self._last_flush_time: float = 0.0
        self._flush_lock: asyncio.Lock = asyncio.Lock()
        self._pending_flush: bool = False

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def phase(self) -> Phase:
        """Current state machine phase."""
        return self._phase

    @property
    def is_terminal_phase(self) -> bool:
        """True if the controller has reached a terminal phase."""
        return self._phase in TERMINAL_PHASES

    @property
    def is_aborted(self) -> bool:
        """True if the reply was explicitly aborted."""
        return self._phase == Phase.ABORTED

    @property
    def terminal_reason(self) -> Optional[str]:
        """Human-readable reason for entering a terminal phase, or None."""
        return self._terminal_reason

    def elapsed_ms(self) -> float:
        """Elapsed milliseconds since the controller was created."""
        return (time.time() - self._dispatch_start_time) * 1000

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _transition(self, to: Phase, source: str, reason: Optional[str] = None) -> bool:
        """Attempt a phase transition.

        Args:
            to: Target phase.
            source: Caller label used for log messages.
            reason: Optional human-readable reason recorded in
                ``terminal_reason`` when entering a terminal phase.

        Returns:
            True if the transition was accepted; False if rejected.
        """
        from_phase = self._phase
        if from_phase == to:
            return False
        allowed = _ALLOWED_TRANSITIONS.get(from_phase, frozenset())
        if to not in allowed:
            logger.warning(
                "streaming_card: phase transition rejected from=%s to=%s source=%s",
                from_phase.value, to.value, source,
            )
            return False
        self._phase = to
        logger.info(
            "streaming_card: phase transition from=%s to=%s source=%s reason=%s",
            from_phase.value, to.value, source, reason,
        )
        if to in TERMINAL_PHASES:
            self._terminal_reason = reason
            self._on_enter_terminal_phase()
        return True

    def _on_enter_terminal_phase(self) -> None:
        """Hook called immediately upon entering any terminal phase."""
        self._pending_flush = False

    # ------------------------------------------------------------------
    # Tool-use helpers
    # ------------------------------------------------------------------

    def _mark_tool_use_activity(self) -> None:
        if not self.tool_use.started_at:
            self.tool_use.started_at = time.time()
        self.tool_use.elapsed_ms = (time.time() - self.tool_use.started_at) * 1000
        self.tool_use.is_active = True

    def _capture_tool_use_elapsed(self) -> None:
        if not self.tool_use.started_at:
            return
        self.tool_use.elapsed_ms = (time.time() - self.tool_use.started_at) * 1000
        self.tool_use.is_active = False

    # ------------------------------------------------------------------
    # Public API: chunk ingestion
    # ------------------------------------------------------------------

    async def add_text_chunk(self, chunk: str) -> None:
        """Append a text token to the answer accumulator.

        Transitions from ``idle`` to ``creating`` → ``streaming`` on first
        call if the controller has not yet started.  Schedules a throttled
        flush after accumulating.

        Args:
            chunk: Raw text token from the LLM stream.
        """
        if not chunk:
            return
        if self.is_terminal_phase:
            return

        self._capture_tool_use_elapsed()

        # Exit reasoning phase if we were in it
        if self.reasoning.is_reasoning_phase:
            self.reasoning.is_reasoning_phase = False
            if self.reasoning.reasoning_start_time:
                self.reasoning.reasoning_elapsed_ms = (
                    (time.time() - self.reasoning.reasoning_start_time) * 1000
                )

        self.text.accumulated_text += chunk

        await self._ensure_streaming()
        if not self.is_terminal_phase:
            await self._schedule_flush()

    async def add_reasoning_chunk(self, chunk: str) -> None:
        """Append a reasoning/think token to the reasoning accumulator.

        Args:
            chunk: Reasoning text token (e.g. content inside ``<think>`` tags).
        """
        if not chunk:
            return
        if self.is_terminal_phase:
            return

        if not self.reasoning.reasoning_start_time:
            self.reasoning.reasoning_start_time = time.time()

        self.reasoning.is_reasoning_phase = True
        self.reasoning.accumulated_reasoning_text += chunk

        await self._ensure_streaming()
        if not self.is_terminal_phase:
            await self._schedule_flush()

    def start_tool_use(self, tool_name: str, args: Any = None) -> None:
        """Record the start of a tool call.

        Args:
            tool_name: Name of the tool being invoked.
            args: Tool input arguments (arbitrary JSON-serialisable value).
        """
        if self.is_terminal_phase:
            return
        self.tool_use.tool_name = tool_name
        self.tool_use.tool_args = args
        self.tool_use.tool_result = None
        self._mark_tool_use_activity()
        logger.debug("streaming_card: tool_use started name=%s", tool_name)

    def complete_tool_use(self, result: Any = None) -> None:
        """Record the completion of the current tool call.

        Appends a step to the completed steps list and resets the active
        tool-use block.

        Args:
            result: Tool output (arbitrary JSON-serialisable value).
        """
        self._capture_tool_use_elapsed()
        step = {
            "name": self.tool_use.tool_name or "tool",
            "args": self.tool_use.tool_args,
            "result": result,
            "elapsed_ms": self.tool_use.elapsed_ms,
        }
        self.tool_use.steps.append(step)
        self.tool_use.tool_name = None
        self.tool_use.tool_args = None
        self.tool_use.tool_result = None
        self.tool_use.is_active = False
        logger.debug("streaming_card: tool_use completed step=%s", step.get("name"))

    # ------------------------------------------------------------------
    # Public API: lifecycle control
    # ------------------------------------------------------------------

    async def mark_completed(self) -> None:
        """Finalize the reply with a completed-state card.

        Flushes accumulated state, transitions to ``completed``, and
        sends a final non-streaming card update to Feishu.
        """
        if self.is_terminal_phase:
            return
        self._capture_tool_use_elapsed()
        self.text.completed_text = self.text.accumulated_text
        self._transition(Phase.COMPLETED, "mark_completed", "normal")
        await self._flush_final(is_aborted=False, is_error=False)

    async def abort(self) -> None:
        """Abort the reply and send a terminal aborted card.

        Safe to call even if the controller is already in a terminal phase
        (becomes a no-op).
        """
        if self.is_terminal_phase:
            return
        self._capture_tool_use_elapsed()
        if not self._transition(Phase.ABORTED, "abort", "abort"):
            return
        await self._flush_final(is_aborted=True, is_error=False)

    async def terminate(self, reason: str = "unavailable") -> None:
        """Terminate the pipeline (e.g. message was deleted/recalled).

        Args:
            reason: Human-readable termination reason.
        """
        self._transition(Phase.TERMINATED, "terminate", reason)

    # ------------------------------------------------------------------
    # Public API: card JSON snapshot
    # ------------------------------------------------------------------

    def to_card_json(self) -> Dict[str, Any]:
        """Return the current card state as a Feishu card payload dict.

        Generates either a streaming in-progress card or a final completed
        card depending on the current phase.

        Returns:
            JSON-serialisable dict ready to pass as the ``content`` field of
            a Feishu ``interactive`` message.
        """
        in_terminal = self.is_terminal_phase

        if in_terminal:
            return _build_final_card(
                text=self.text.completed_text or self.text.accumulated_text,
                reasoning_text=self.reasoning.accumulated_reasoning_text or None,
                reasoning_elapsed_ms=self.reasoning.reasoning_elapsed_ms,
                tool_steps=self.tool_use.steps or None,
                elapsed_ms=self.elapsed_ms(),
                is_aborted=self.is_aborted,
                is_error=False,
            )

        return _build_streaming_card(
            text=self.text.accumulated_text,
            reasoning_text=(
                self.reasoning.accumulated_reasoning_text
                if self.reasoning.is_reasoning_phase
                else None
            ),
            tool_steps=self.tool_use.steps or None,
            is_streaming=True,
        )

    # ------------------------------------------------------------------
    # Public API: explicit flush
    # ------------------------------------------------------------------

    async def flush(self) -> None:
        """Immediately push the current card state to Feishu.

        Calls ``im.v1.message.update`` (PATCH) with TENANT token.
        Safe to call at any time; becomes a no-op if no message ID or
        client is configured.
        """
        await self._perform_flush()

    # ------------------------------------------------------------------
    # Internal: streaming lifecycle
    # ------------------------------------------------------------------

    async def _ensure_streaming(self) -> None:
        """Transition from idle → creating → streaming if not already there."""
        if self._phase == Phase.STREAMING:
            return
        if self._phase == Phase.IDLE:
            self._transition(Phase.CREATING, "_ensure_streaming")
        if self._phase == Phase.CREATING:
            self._transition(Phase.STREAMING, "_ensure_streaming")

    # ------------------------------------------------------------------
    # Internal: throttled flush scheduling
    # ------------------------------------------------------------------

    async def _schedule_flush(self) -> None:
        """Schedule a flush respecting the throttle window.

        If a flush was performed recently, waits for the remainder of the
        throttle window before flushing.  Concurrent calls are coalesced —
        only one pending flush runs at a time.
        """
        if self.is_terminal_phase:
            return
        if self._pending_flush:
            return  # already scheduled

        now_ms = time.time() * 1000
        elapsed_since_flush = now_ms - self._last_flush_time
        delay_ms = max(0.0, _FLUSH_THROTTLE_MS - elapsed_since_flush)

        self._pending_flush = True
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000)

        self._pending_flush = False
        if not self.is_terminal_phase:
            await self._perform_flush()

    # ------------------------------------------------------------------
    # Internal: actual Feishu PATCH call
    # ------------------------------------------------------------------

    async def _perform_flush(self) -> None:
        """Push current card content to Feishu via ``im.v1.message.update``.

        Uses TENANT token (update_message is TENANT-only in Feishu API).
        Skips silently if ``message_id`` or ``client`` is not set.
        """
        if not self.card_kit.card_message_id or not self._client:
            return
        if self.is_terminal_phase:
            return

        card_payload = self.to_card_json()
        card_json = json.dumps(card_payload, ensure_ascii=False)

        # Skip if nothing changed since last flush
        if card_json == self.text.last_flushed_text:
            return

        async with self._flush_lock:
            try:
                await self._patch_message(card_json)
                self.text.last_flushed_text = card_json
                self._last_flush_time = time.time() * 1000
                logger.debug(
                    "streaming_card: flushed message_id=%s phase=%s",
                    self.card_kit.card_message_id, self._phase.value,
                )
            except Exception as exc:
                logger.warning(
                    "streaming_card: flush failed message_id=%s error=%s",
                    self.card_kit.card_message_id, exc,
                )

    async def _flush_final(
        self, is_aborted: bool = False, is_error: bool = False
    ) -> None:
        """Push the final terminal card to Feishu.

        Args:
            is_aborted: Pass True when the reply was cancelled.
            is_error: Pass True when the reply ended with an error.
        """
        if not self.card_kit.card_message_id or not self._client:
            return

        card_payload = _build_final_card(
            text=self.text.completed_text or self.text.accumulated_text,
            reasoning_text=self.reasoning.accumulated_reasoning_text or None,
            reasoning_elapsed_ms=self.reasoning.reasoning_elapsed_ms,
            tool_steps=self.tool_use.steps or None,
            elapsed_ms=self.elapsed_ms(),
            is_aborted=is_aborted,
            is_error=is_error,
        )
        card_json = json.dumps(card_payload, ensure_ascii=False)

        try:
            await self._patch_message(card_json)
            logger.info(
                "streaming_card: final card sent message_id=%s aborted=%s error=%s",
                self.card_kit.card_message_id, is_aborted, is_error,
            )
        except Exception as exc:
            logger.warning(
                "streaming_card: final flush failed message_id=%s error=%s",
                self.card_kit.card_message_id, exc,
            )

    async def _patch_message(self, card_json: str) -> None:
        """Call Feishu ``im.v1.message.update`` (PATCH) with TENANT token.

        Wraps the synchronous lark_oapi SDK call in a thread executor so it
        does not block the asyncio event loop.

        Args:
            card_json: JSON string of the card payload.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync_patch_message, card_json)

    def _sync_patch_message(self, card_json: str) -> None:
        """Synchronous Feishu message patch via lark_oapi BaseRequest.

        Uses AccessTokenType.TENANT (TENANT-only API requirement for
        ``im.v1.message.update``).

        Args:
            card_json: JSON string of the card payload.
        """
        try:
            from lark_oapi import AccessTokenType
            from lark_oapi.core.enum import HttpMethod
            from lark_oapi.core.model.base_request import BaseRequest
        except ImportError as exc:
            logger.error("streaming_card: lark_oapi not available: %s", exc)
            return

        message_id = self.card_kit.card_message_id
        uri = f"/open-apis/im/v1/messages/{message_id}"

        body = {
            "msg_type": "interactive",
            "content": card_json,
        }

        request = (
            BaseRequest.builder()
            .http_method(HttpMethod.PATCH)
            .uri(uri)
            .token_types({AccessTokenType.TENANT})
            .body(body)
            .build()
        )

        response = self._client.request(request)
        code = getattr(response, "code", None)
        msg = getattr(response, "msg", "")

        if code not in (None, 0):
            logger.warning(
                "streaming_card: PATCH failed code=%s msg=%s message_id=%s",
                code, msg, message_id,
            )

    # ------------------------------------------------------------------
    # Async iterator integration
    # ------------------------------------------------------------------

    async def consume_stream(
        self,
        stream: AsyncIterator[str],
        *,
        flush_on_complete: bool = True,
    ) -> None:
        """Consume an async LLM token stream and accumulate to text block.

        Convenience method for feeding a raw text stream directly into the
        controller.  Each yielded string is passed to ``add_text_chunk``.

        Args:
            stream: Async iterator yielding raw text tokens.
            flush_on_complete: If True, calls ``mark_completed()`` after the
                stream is exhausted.  Set to False when you want to manage
                the lifecycle manually (e.g. to send a final card with
                additional metadata before completing).

        Example::

            async def my_llm_stream():
                yield "Hello "
                yield "world!"

            await ctrl.consume_stream(my_llm_stream())
        """
        async for chunk in stream:
            if self.is_terminal_phase:
                break
            await self.add_text_chunk(chunk)

        if flush_on_complete and not self.is_terminal_phase:
            await self.mark_completed()

    def __repr__(self) -> str:
        return (
            f"StreamingCardController("
            f"message_id={self.card_kit.card_message_id!r}, "
            f"phase={self._phase.value!r}, "
            f"text_len={len(self.text.accumulated_text)}, "
            f"reasoning_len={len(self.reasoning.accumulated_reasoning_text)}"
            f")"
        )
