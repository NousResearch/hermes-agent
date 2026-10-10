"""Paragraph bubbles for completed conversational replies, never arbitrary sends."""
import re

from gateway.platforms.helpers import split_markdown_atoms


def reply_chunks(content: str) -> list[str]:
    # Loose lists, reference links, indented code and non-backtick fences can span
    # paragraphs. Keep that reply intact rather than detach their continuation.
    if re.search(r"(?m)^(?:[ \t]*(?:[-+*]|\d+[.)])\s|[ \t]*>|[ \t]*\[[^\]]+\]:| {4}|\t|[ \t]+```|`{4,}|[ \t]*~~~)", content):
        return [content]
    return split_markdown_atoms(content) or [content]


class SlackReplyChunksMixin:
    def _format_chunks(self, content: str) -> list[str]:
        """mrkdwn-format ``content`` and split to ``MAX_MESSAGE_LENGTH`` (never empty)."""
        formatted = self.format_message(content)
        return self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH) or [formatted]

    def prefers_buffered_reply(self, chat_id: str) -> bool:
        return self.config.extra.get("dm_reply_bursts") is True and chat_id.startswith("D")

    def reply_chunks(self, content: str, chat_id: str) -> list[str]:
        if not self.prefers_buffered_reply(chat_id):
            return [content]
        return reply_chunks(content)

