"""Telegram text formatting and parse fallbacks."""

from __future__ import annotations

from typing import Any

from gateway.stt_echo import (
    TELEGRAM_STT_ECHO_METADATA_KEY, chunk_telegram_stt_echo_html, telegram_stt_echo_plain,
)


class TelegramTextSendMixin:
    """Select text presentation while the adapter owns routing, pacing and retries."""

    def _prepare_text_chunks(self, content: str, metadata: dict[str, Any] | None) -> list[str]:
        """Keep each STT quote expandable; retain MarkdownV2 chunking for other text."""
        from plugins.platforms.telegram import adapter as telegram

        if (metadata or {}).get(TELEGRAM_STT_ECHO_METADATA_KEY):
            return chunk_telegram_stt_echo_html(content, self.MAX_MESSAGE_LENGTH, len_fn=telegram.utf16_len)
        chunks = self.truncate_message(self.format_message(content), self.MAX_MESSAGE_LENGTH, len_fn=telegram.utf16_len)
        if len(chunks) > 1:
            # * Pagination parentheses need escaping outside MarkdownV2 code fences.
            chunks = [
                telegram._separate_chunk_indicator_from_fence(
                    telegram.re.sub(r" \((\d+)/(\d+)\)$", r" \\(\1/\2\\)", chunk))
                for chunk in chunks
            ]
        return chunks

    async def _send_formatted_text_chunk(
        self, chunk: str, send_kwargs: dict[str, Any], metadata: dict[str, Any] | None,
    ):
        """Select the wire parse mode without bypassing the adapter's delivery loop."""
        if (metadata or {}).get(TELEGRAM_STT_ECHO_METADATA_KEY):
            return await self._send_chunk_html_or_plain(chunk, send_kwargs)
        return await self._send_chunk_markdown_or_plain(chunk, send_kwargs)

    async def _send_chunk_markdown_or_plain(self, chunk: str, send_kwargs: dict[str, Any]):
        """MarkdownV2 first; on a parse/markdown rejection resend as stripped plain text."""
        from plugins.platforms.telegram import adapter as telegram

        try:
            return await telegram._await_with_thread_deadline(
                self._bot.send_message(text=chunk, parse_mode=telegram.ParseMode.MARKDOWN_V2, **send_kwargs),
                timeout=telegram._TEXT_SEND_DEADLINE, label="telegram-send", dump_on_blocked_loop=False)
        except Exception as md_error:
            if "parse" in str(md_error).lower() or "markdown" in str(md_error).lower():
                telegram.logger.warning("[%s] MarkdownV2 parse failed, falling back to plain text: %s", self.name, md_error)
                return await telegram._await_with_thread_deadline(
                    self._bot.send_message(text=telegram._strip_mdv2(chunk), parse_mode=None, **send_kwargs),
                    timeout=telegram._TEXT_SEND_DEADLINE, label="telegram-send", dump_on_blocked_loop=False)
            raise

    async def _send_chunk_html_or_plain(self, chunk: str, send_kwargs: dict[str, Any]):
        """Use plain text only after a definitive Bot API HTML entity rejection."""
        from plugins.platforms.telegram import adapter as telegram

        try:
            return await telegram._await_with_thread_deadline(
                self._bot.send_message(text=chunk, parse_mode=telegram.ParseMode.HTML, **send_kwargs),
                timeout=telegram._TEXT_SEND_DEADLINE, label="telegram-send", dump_on_blocked_loop=False)
        except Exception as html_error:
            bad_request = self._telegram_error_types()[1]
            parse_error = any(marker in str(html_error).lower() for marker in (
                "can't parse entities", "cannot parse entities", "unsupported start tag", "unsupported end tag"))
            if not bad_request or not isinstance(html_error, bad_request) or not parse_error:
                raise
            telegram.logger.warning(
                "[%s] STT HTML parse failed, falling back to plain text: %s",
                self.name, telegram._redact_telegram_error_text(html_error))
            return await telegram._await_with_thread_deadline(
                self._bot.send_message(text=telegram_stt_echo_plain(chunk), parse_mode=None, **send_kwargs),
                timeout=telegram._TEXT_SEND_DEADLINE, label="telegram-send", dump_on_blocked_loop=False)
