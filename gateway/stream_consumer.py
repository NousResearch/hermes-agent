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

# Sentinel to signal a tool boundary — finalize current message and start a
# new one so that subsequent text appears below tool progress messages.
_NEW_SEGMENT = object()


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
        self._last_edit_had_cursor = False  # True when last edit appended the cursor
        self._fallback_final_send = False
        self._fallback_prefix = ""

    @property
    def already_sent(self) -> bool:
        """True if at least one message was sent/edited — signals the base
        adapter to skip re-sending the final response."""
        return self._already_sent

    def on_delta(self, text: str) -> None:
        """Thread-safe callback — called from the agent's worker thread.

        When *text* is ``None``, signals a tool boundary: the current message
        is finalized and subsequent text will be sent as a new message so it
        appears below any tool-progress messages the gateway sent in between.
        """
        if text:
            self._queue.put(text)
        elif text is None:
            self._queue.put(_NEW_SEGMENT)

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
                got_segment_break = False
                while True:
                    try:
                        item = self._queue.get_nowait()
                        if item is _DONE:
                            got_done = True
                            break
                        if item is _NEW_SEGMENT:
                            got_segment_break = True
                            break
                        self._accumulated += item
                    except queue.Empty:
                        break

                # Decide whether to flush an edit
                now = time.monotonic()
                elapsed = now - self._last_edit_time
                should_edit = (
                    got_done
                    or got_segment_break
                    or (elapsed >= self.cfg.edit_interval
                        and self._accumulated)
                    or len(self._accumulated) >= self.cfg.buffer_threshold
                )

                if should_edit and self._accumulated:
                    # Split overflow: if accumulated text exceeds the platform
                    # limit, split into properly sized chunks.
                    if (
                        len(self._accumulated) > _safe_limit
                        and self._message_id is None
                    ):
                        # No existing message to edit (first message or after a
                        # segment break).  Use truncate_message — the same
                        # helper the non-streaming path uses — to split with
                        # proper word/code-fence boundaries and chunk
                        # indicators like "(1/2)".
                        chunks = self.adapter.truncate_message(
                            self._accumulated, _safe_limit
                        )
                        for chunk in chunks:
                            await self._send_new_chunk(chunk, self._message_id)
                        self._accumulated = ""
                        self._last_sent_text = ""
                        self._last_edit_time = time.monotonic()
                        if got_done:
                            return
                        if got_segment_break:
                            self._message_id = None
                            self._fallback_final_send = False
                            self._fallback_prefix = ""
                        continue

                    # Existing message: edit it with the first chunk, then
                    # start a new message for the overflow remainder.
                    while (
                        len(self._accumulated) > _safe_limit
                        and self._message_id is not None
                        and self._edit_supported
                    ):
                        split_at = self._accumulated.rfind("\n", 0, _safe_limit)
                        if split_at < _safe_limit // 2:
                            split_at = _safe_limit
                        chunk = self._accumulated[:split_at]
                        await self._send_or_edit(chunk)
                        if self._fallback_final_send:
                            # Edit failed while attempting to split an oversized
                            # message. Keep the full accumulated text intact so
                            # the fallback final-send path can deliver the
                            # remaining continuation without dropping content.
                            break
                        self._accumulated = self._accumulated[split_at:].lstrip("\n")
                        self._message_id = None
                        self._last_sent_text = ""

                    display_text = self._accumulated
                    if not got_done and not got_segment_break:
                        display_text += self.cfg.cursor

                    await self._send_or_edit(display_text)
                    self._last_edit_time = time.monotonic()

                if got_done:
                    # Final edit — always finalize the message, even when
                    # _accumulated is empty.  The cursor may be stuck on the last
                    # progressive edit with no new text arriving since.
                    if self._accumulated:
                        if self._fallback_final_send:
                            await self._send_fallback_final(self._accumulated)
                        elif self._message_id:
                            await self._send_or_edit(self._accumulated)
                        elif not self._already_sent:
                            await self._send_or_edit(self._accumulated)
                    elif self._last_edit_had_cursor and self._message_id:
                        # Cursor is stuck — strip it by editing with clean text.
                        cursor = self.cfg.cursor
                        clean = (
                            self._last_sent_text[:-len(cursor)]
                            if self._last_sent_text.endswith(cursor)
                            else self._last_sent_text
                        )
                        if clean.strip():
                            await self._send_or_edit(clean)
                    return

                # Tool boundary: reset message state so the next text chunk
                # creates a fresh message below any tool-progress messages.
                #
                # Exception: when _message_id is "__no_edit__" the platform
                # never returned a real message ID (e.g. Signal, webhook with
                # github_comment delivery).  Resetting to None would re-enter
                # the "first send" path on every tool boundary and post one
                # platform message per tool call — that is what caused 155
                # comments under a single PR.  Instead, keep all state so the
                # full continuation is delivered once via _send_fallback_final.
                # (When editing fails mid-stream due to flood control the id is
                # a real string like "msg_1", not "__no_edit__", so that case
                # still resets and creates a fresh segment as intended.)
                if got_segment_break and self._message_id != "__no_edit__":
                    self._message_id = None
                    self._accumulated = ""
                    self._last_sent_text = ""
                    self._fallback_final_send = False
                    self._fallback_prefix = ""

                await asyncio.sleep(0.05)  # Small yield to not busy-loop

        except asyncio.CancelledError:
            # Best-effort final edit on cancellation
            if self._accumulated and self._message_id:
                try:
                    await self._send_or_edit(self._accumulated)
                except Exception:
                    pass
        except Exception as e:
            logger.error("Stream consumer error: %s", e)

    # Pattern to strip MEDIA:<path> tags (including optional surrounding quotes).
    # Matches the simple cleanup regex used by the non-streaming path in
    # gateway/platforms/base.py for post-processing.
    _MEDIA_RE = re.compile(r'''[`"']?MEDIA:\s*\S+[`"']?''')

    # Block characters used in terminal progress bars and spinner animations.
    # These render as black/grey rectangles in Telegram and other chat apps.
    # Also includes the streaming cursor char (▉) and ANSI escape sequences.
    _BLOCK_CHARS_RE = re.compile(
        '['
        '\u2588\u2589\u258A\u258B\u258C\u258D\u258E\u258F'  # Full/quarter blocks: █▉▊▋▌▍▎▏
        '\u2591\u2592\u2593\u2594\u2595'                   # Light/shade: ░▒▓▔▕
        '\u2500\u2501\u2502\u2503\u2504\u2505\u2506\u2507\u2508\u2509\u250A\u250B\u250C\u250D\u250E\u250F'  # Box-drawing
        '\u2510\u2511\u2512\u2513\u2514\u2515\u2516\u2517\u2518\u2519\u251A\u251B\u251C\u251D\u251E\u251F\u2520'
        '\u2521\u2522\u2523\u2524\u2525\u2526\u2527\u2528\u2529\u252A\u252B\u252C\u252D\u252E\u252F\u2530'
        '\u2531\u2532\u2533\u2534\u2535\u2536\u2537\u2538\u2539\u253A\u253B\u253C\u253D\u253E\u253F'
        '\u2540\u2541\u2542\u2543\u2544\u2545\u2546\u2547\u2548\u2549\u254A\u254B\u254C\u254D\u254E\u254F'
        '\u2550\u2551\u2552\u2553\u2554\u2555\u2556\u2557\u2558\u2559\u255A\u255B\u255C\u255D\u255E\u255F'
        '\u2560\u2561\u2562\u2563\u2564\u2565\u2566\u2567\u2568\u2569\u256A\u256B\u256C\u256D\u256E\u256F'
        '\u2570\u2571\u2572\u2573\u2574\u2575\u2576\u2577\u2578\u2579\u257A\u257B\u257C\u257D\u257E\u257F'
        '\u2580\u2581\u2582\u2583\u2584\u2585\u2586\u2587'  # More block forms
        '\u01C0\u01C1\u01C2\u01C3\u0180\u01C4\u01C5\u01C6\u01C7\u01C8\u01C9\u01CA\u01CB\u01CC'  # Daggers, bullets
        '\u2302\u2303\u2304\u2305\u2306\u2307'  # House, up-arrow-head, etc.
        '\u25B6\u25B7\u25B8\u25B9\u25BA\u25BB\u25BC\u25BD\u25BE\u25BF'  # Play/triangle arrows ▶▷
        '\u25C0\u25C1\u25C2\u25C3\u25C4\u25C5\u25C6\u25C7\u25C8\u25C9\u25CA\u25CB\u25CC\u25CD\u25CE\u25CF'  # Triangles
        '\u2601\u2602\u2603\u2604\u2605\u2606\u2607\u2608\u2609\u260A\u260B\u260C\u260D\u260E\u260F'  # Stars, etc.
        '\u2660\u2661\u2662\u2663\u2664\u2665\u2666\u2667\u2668\u2669\u266A\u266B\u266C\u266D\u266E\u266F'  # Card suits, music
        '\u27F0\u27F1\u27F2\u27F3\u27F4\u27F5\u27F6\u27F7\u27F8\u27F9\u27FA\u27FB\u27FC\u27FD\u27FE\u27FF'  # Arrows
        '\u2800-\u28FF'  # Braille patterns (block dots)
        ']'
    )

    # ANSI escape sequences (color, cursor movement, clear screen, etc.)
    _ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b[>=]|\x1b[^\x1b]*\x1b')

    @staticmethod
    def _clean_for_display(text: str) -> str:
        """Strip MEDIA: directives, terminal glyphs, and control chars.

        The streaming path delivers raw text chunks that may include
        ``MEDIA:<path>`` tags, ``[[audio_as_voice]]`` directives, terminal
        spinner/progress characters (e.g. ▍ ▉ █ ░), and ANSI escape sequences.
        These are all stripped before the text reaches the chat platform.
        """
        cleaned = text

        # Strip terminal block/progress/spinner characters
        cleaned = GatewayStreamConsumer._BLOCK_CHARS_RE.sub('', cleaned)

        # Strip ANSI escape sequences (colors, cursor moves, clears, etc.)
        cleaned = GatewayStreamConsumer._ANSI_ESCAPE_RE.sub('', cleaned)

        # Strip MEDIA: and [[audio_as_voice]] directives
        if "MEDIA:" in cleaned or "[[audio_as_voice]]" in cleaned:
            cleaned = cleaned.replace("[[audio_as_voice]]", "")
            cleaned = GatewayStreamConsumer._MEDIA_RE.sub("", cleaned)
            # Collapse excessive blank lines left behind by removed tags
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        # Strip zero-width and other invisible junk characters
        cleaned = re.sub(r'[\u200B\u200C\u200D\u200E\u200F\uFEFF\u180E\u00AD]', '', cleaned)

        # Strip trailing whitespace/newlines but preserve leading content
        return cleaned.rstrip()

    async def _send_new_chunk(self, text: str, reply_to_id: Optional[str]) -> Optional[str]:
        """Send a new message chunk, optionally threaded to a previous message.

        Returns the message_id so callers can thread subsequent chunks.
        """
        text = self._clean_for_display(text)
        if not text.strip():
            return reply_to_id
        try:
            meta = dict(self.metadata) if self.metadata else {}
            result = await self.adapter.send(
                chat_id=self.chat_id,
                content=text,
                reply_to=reply_to_id,
                metadata=meta,
            )
            if result.success and result.message_id:
                self._message_id = str(result.message_id)
                self._already_sent = True
                self._last_sent_text = text
                return str(result.message_id)
            else:
                self._edit_supported = False
                return reply_to_id
        except Exception as e:
            logger.error("Stream send chunk error: %s", e)
            return reply_to_id

    def _visible_prefix(self) -> str:
        """Return the visible text already shown in the streamed message."""
        prefix = self._last_sent_text or ""
        if self.cfg.cursor and prefix.endswith(self.cfg.cursor):
            prefix = prefix[:-len(self.cfg.cursor)]
        return self._clean_for_display(prefix)

    def _continuation_text(self, final_text: str) -> str:
        """Return only the part of final_text the user has not already seen."""
        prefix = self._fallback_prefix or self._visible_prefix()
        if prefix and final_text.startswith(prefix):
            return final_text[len(prefix):].lstrip()
        return final_text

    @staticmethod
    def _split_text_chunks(text: str, limit: int) -> list[str]:
        """Split text into reasonably sized chunks for fallback sends."""
        if len(text) <= limit:
            return [text]
        chunks: list[str] = []
        remaining = text
        while len(remaining) > limit:
            split_at = remaining.rfind("\n", 0, limit)
            if split_at < limit // 2:
                split_at = limit
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip("\n")
        if remaining:
            chunks.append(remaining)
        return chunks

    async def _send_fallback_final(self, text: str) -> None:
        """Send the final continuation after streaming edits stop working."""
        final_text = self._clean_for_display(text)
        continuation = self._continuation_text(final_text)
        self._fallback_final_send = False
        if not continuation.strip():
            # Nothing new to send — the visible partial already matches final text.
            self._already_sent = True
            return

        raw_limit = getattr(self.adapter, "MAX_MESSAGE_LENGTH", 4096)
        safe_limit = max(500, raw_limit - 100)
        chunks = self._split_text_chunks(continuation, safe_limit)

        last_message_id: Optional[str] = None
        last_successful_chunk = ""
        sent_any_chunk = False
        for chunk in chunks:
            result = await self.adapter.send(
                chat_id=self.chat_id,
                content=chunk,
                metadata=self.metadata,
            )
            if not result.success:
                if sent_any_chunk:
                    # Some continuation text already reached the user. Suppress
                    # the base gateway final-send path so we don't resend the
                    # full response and create another duplicate.
                    self._already_sent = True
                    self._message_id = last_message_id
                    self._last_sent_text = last_successful_chunk
                    self._fallback_prefix = ""
                    return
                # No fallback chunk reached the user — allow the normal gateway
                # final-send path to try one more time.
                self._already_sent = False
                self._message_id = None
                self._last_sent_text = ""
                self._fallback_prefix = ""
                return
            sent_any_chunk = True
            last_successful_chunk = chunk
            last_message_id = result.message_id or last_message_id

        self._message_id = last_message_id
        self._already_sent = True
        self._last_sent_text = chunks[-1]
        self._fallback_prefix = ""

    async def _send_or_edit(self, text: str) -> None:
        """Send or edit the streaming message."""
        # Strip MEDIA: directives so they don't appear as visible text.
        # Media files are delivered as native attachments after the stream
        # finishes (via _deliver_media_from_response in gateway/run.py).
        text = self._clean_for_display(text)
        if not text.strip():
            return
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
                        self._last_edit_had_cursor = text.endswith(self.cfg.cursor)
                    else:
                        # If an edit fails mid-stream (especially Telegram flood control),
                        # stop progressive edits and send only the missing tail once the
                        # final response is available.
                        logger.debug("Edit failed, disabling streaming for this adapter")
                        self._fallback_prefix = self._visible_prefix()
                        self._fallback_final_send = True
                        self._edit_supported = False
                        self._already_sent = True
                else:
                    # Editing not supported — skip intermediate updates.
                    # The final response will be sent by the fallback path.
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
                    self._last_edit_had_cursor = text.endswith(self.cfg.cursor)
                elif result.success:
                    # Platform accepted the message but returned no message_id
                    # (e.g. Signal).  Can't edit without an ID — switch to
                    # fallback mode: suppress intermediate deltas, send only
                    # the missing tail once the final response is ready.
                    self._already_sent = True
                    self._edit_supported = False
                    self._fallback_prefix = self._clean_for_display(text)
                    self._fallback_final_send = True
                    # Sentinel prevents re-entering this branch on every delta
                    self._message_id = "__no_edit__"
                else:
                    # Initial send failed — disable streaming for this session
                    self._edit_supported = False
        except Exception as e:
            logger.error("Stream send/edit error: %s", e)
