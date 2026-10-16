"""Webex Messages API text limits, shared by live and standalone delivery."""

from gateway.platforms.base import BasePlatformAdapter


# https://github.com/webex/webex-openapi-specs/blob/main/public-spec/webex-messaging.json
# Both text and markdown are limited to bytes, rather than Unicode codepoints.
MAX_MESSAGE_BYTES = 7439


def utf8_length(text: str) -> int:
    return len(text.encode("utf-8"))


def split_message(text: str, max_bytes: int = MAX_MESSAGE_BYTES) -> list[str]:
    """Keep code fences balanced and account for split indicators in the byte budget."""
    return BasePlatformAdapter.truncate_message(text, max_bytes, len_fn=utf8_length)
