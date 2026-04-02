"""Gateway streaming consumer — bridges sync agent callbacks to async platform delivery.

The agent fires stream_delta_callback(text) synchronously from its worker thread.
GatewayStreamConsumer:
  1. Receives deltas via on_delta() (thread-safe, sync)
  2. Queues them to an asyncio task via queue.Queue
  3. The async run() task buffers, rate-limits, and progressively edits
     a single message on the target platform

Design: Uses the edit transport (send initial message, then editMessageText).
This is universally supported across Telegram, Discord, and Slack.

Credit: jobless0x (#774, #1312), OutThisLife (#798), clicksingh (#697).
"""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("gateway.stream_consumer")

# Sentinel to signal the stream is complete
_DONE = object()

# Patterns for stripping thinking/reasoning blocks from streamed content.
# Matches complete blocks and also unclosed trailing blocks (still being generated).
_THINK_TAGS = ("think", "thinking", "reasoning", "REASONING_SCRATCHPAD")
_COMPLETE_THINK_RE = re.compile(
    r'<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)>.*?</(?:think|thinking|reasoning|REASONING_SCRATCHPAD)>',
    re.DOTALL | re.IGNORECASE,
)
_UNCLOSED_THINK_RE = re.compile(
    r'<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)>(?:(?!</(?:think|thinking|reasoning|REASONING_SCRATCHPAD)>).)*$',
    re.DOTALL | re.IGNORECASE,
)


def _strip_think_blocks(text: str) -> str:
    """Remove complete and unclosed thinking/reasoning blocks from streamed text.

    Handles both fully closed blocks like ``<think>...</think>`` and
    partially streamed blocks where the closing tag hasn't arrived yet.
    """
    if not text or "<" not in text:
        return text
    # Strip complete blocks first
    text = _COMPLETE_THINK_RE.sub("", text)
    # Strip any trailing unclosed block (still being streamed)
    text = _UNCLOSED_THINK_RE.sub("", text)
    # Collapse excessive blank lines left behind by removed blocks
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


@dataclass
class StreamConsumerConfig:
    """Runtime config for a single stream consumer instance."""
    edit_interval: float = 0.3
    buffer_threshold: int = 40
    cursor: str = " ▉"


class GatewayStreamConsumer:
    """Async consumer that progressively edits a platform message with streamed tokens.

    Usage::

        consumer = GatewayStreamConsumer(adapter, chat_id, config, metadata=metadata)
        # Pass consumer.on_delta as stream_delta_callback to AIAgent
        agent = AIAgent(..., stream_delta_callback=consumer.on_delta)
        # Start the consumer as an asyncio task
        task = asyncio.create_task(consumer.run())
        # ... run agent in thread pool ...
        consumer.finish()  # signal completion
        await task         # wait for final edit
    """

    def __init__(
        self,
        adapter: Any,
        chat_id: str,
        config: Optional[StreamConsumerConfig] = None,
        metadata: Optional[dict] = None,
    ):
        self.adapter = adapter
        self.chat_id = chat_id
        self.cfg = config or StreamConsumerConfig()
        self.metadata = metadata
        self._queue: queue.Queue = queue.Queue()
        self._accumulated = ""
        self._message_id: Optional[str] = None
        self._already_sent = False
        self._edit_supported = True  # Disabled on first edit failure (Signal/Email/HA)
        self._last_edit_time = 0.0
        self._last_sent_text = ""   # Track last-sent text to skip redundant edits

    @property
    def already_sent(self) -> bool:
        """True if at least one message was sent/edited — signals the base
        adapter to skip re-sending the final response."""
        return self._already_sent

    def on_delta(self, text: str) -> None:
        """Thread-safe callback — called from the agent's worker thread."""
        if text:
            self._queue.put(text)

    def finish(self) -> None:
        """Signal that the stream is complete."""
        self._queue.put(_DONE)

    async def run(self) -> None:
        """Async task that drains the queue and edits the platform message."""
        # Platform message length limit — leave room for cursor + formatting
        _raw_limit = getattr(self.adapter, "MAX_MESSAGE_LENGTH", 4096)
        _safe_limit = max(500, _raw_limit - len(self.cfg.cursor) - 100)

        try:
            while True:
                # Drain all available items from the queue
                got_done = False
                while True:
                    try:
                        item = self._queue.get_nowait()
                        if item is _DONE:
                            got_done = True
                            break
                        self._accumulated += item
                    except queue.Empty:
                        break

                # Decide whether to flush an edit
                now = time.monotonic()
                elapsed = now - self._last_edit_time
                should_edit = (
                    got_done
                    or (elapsed >= self.cfg.edit_interval
                        and len(self._accumulated) > 0)
                    or len(self._accumulated) >= self.cfg.buffer_threshold
                )

                if should_edit and self._accumulated:
                    # Strip thinking/reasoning blocks before display
                    display_text = _strip_think_blocks(self._accumulated)
                    if not display_text:
                        # All content so far is inside a thinking block —
                        # skip this edit cycle and wait for real content.
                        if got_done:
                            return
                        await asyncio.sleep(0.05)
                        continue

                    # Split overflow: if clean text exceeds the platform
                    # limit, finalize the current message and start a new one.
                    while (
                        len(display_text) > _safe_limit
                        and self._message_id is not None
                    ):
                        split_at = display_text.rfind("\n", 0, _safe_limit)
                        if split_at < _safe_limit // 2:
                            split_at = _safe_limit
                        chunk = display_text[:split_at]
                        await self._send_or_edit(chunk)
                        display_text = display_text[split_at:].lstrip("\n")
                        self._message_id = None
                        self._last_sent_text = ""

                    if not got_done:
                        display_text += self.cfg.cursor

                    await self._send_or_edit(display_text)
                    self._last_edit_time = time.monotonic()

                if got_done:
                    # Final edit without cursor — strip thinking blocks
                    final_text = _strip_think_blocks(self._accumulated)
                    if final_text and self._message_id:
                        await self._send_or_edit(final_text)
                    return

                await asyncio.sleep(0.05)  # Small yield to not busy-loop

        except asyncio.CancelledError:
            # Best-effort final edit on cancellation
            if self._accumulated and self._message_id:
                try:
                    await self._send_or_edit(_strip_think_blocks(self._accumulated) or self._last_sent_text)
                except Exception:
                    pass
        except Exception as e:
            logger.error("Stream consumer error: %s", e)

    async def _send_or_edit(self, text: str) -> None:
        """Send or edit the streaming message."""
        try:
            if self._message_id is not None:
                if self._edit_supported:
                    # Skip if text is identical to what we last sent
                    if text == self._last_sent_text:
                        return
                    # Edit existing message
                    result = await self.adapter.edit_message(
                        chat_id=self.chat_id,
                        message_id=self._message_id,
                        content=text,
                    )
                    if result.success:
                        self._already_sent = True
                        self._last_sent_text = text
                    else:
                        # Edit not supported by this adapter — stop streaming,
                        # let the normal send path handle the final response.
                        # Without this guard, adapters like Signal/Email would
                        # flood the chat with a new message every edit_interval.
                        logger.debug("Edit failed, disabling streaming for this adapter")
                        self._edit_supported = False
                else:
                    # Editing not supported — skip intermediate updates.
                    # The final response will be sent by the normal path.
                    pass
            else:
                # First message — send new
                result = await self.adapter.send(
                    chat_id=self.chat_id,
                    content=text,
                    metadata=self.metadata,
                )
                if result.success and result.message_id:
                    self._message_id = result.message_id
                    self._already_sent = True
                    self._last_sent_text = text
                else:
                    # Initial send failed — disable streaming for this session
                    self._edit_supported = False
        except Exception as e:
            logger.error("Stream send/edit error: %s", e)
