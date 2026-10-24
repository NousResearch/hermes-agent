"""Printable, typed message content with legacy SQLite read compatibility."""

import json
from typing import Any

from agent.message_sanitization import _sanitize_surrogates
from hermes_state_common import _json_or


class SessionContentCodecMixin:
    _CONTENT_V2_PREFIX = "__HERMES_CONTENT_V2__:"

    _CONTENT_V2_JSON_PREFIX = _CONTENT_V2_PREFIX + "j:"

    _CONTENT_V2_STRING_PREFIX = _CONTENT_V2_PREFIX + "s:"

    _CONTENT_V2_BYTES_PREFIX = _CONTENT_V2_PREFIX + "b:"

    @classmethod
    def _encode_content(cls, content: Any) -> Any:
        """Serialize message content with printable, versioned type tags."""
        if isinstance(content, str):
            sanitized = _sanitize_surrogates(content.replace("\x00", "\ufffd"))
            if "\x00" in content or content.startswith(
                (cls._CONTENT_V2_PREFIX, cls._CONTENT_JSON_PREFIX)
            ):
                return cls._CONTENT_V2_STRING_PREFIX + json.dumps(
                    sanitized, separators=(",", ":")
                )
            return sanitized
        if isinstance(content, bytes):
            return cls._CONTENT_V2_BYTES_PREFIX + content.hex()
        if content is None or isinstance(content, (int, float)):
            return content
        return cls._CONTENT_V2_JSON_PREFIX + json.dumps(content, separators=(",", ":"))

    @classmethod
    def _decode_content(cls, content: Any) -> Any:
        """Reverse :meth:`_encode_content`; malformed or unknown data remains raw."""
        if not isinstance(content, str):
            return content
        if content.startswith(cls._CONTENT_V2_JSON_PREFIX):
            decoded = _json_or(
                content[len(cls._CONTENT_V2_JSON_PREFIX):], None,
                "Failed to decode V2 structured message content; returning raw string",
            )
            return decoded if isinstance(decoded, (list, dict)) else content
        if content.startswith(cls._CONTENT_V2_STRING_PREFIX):
            decoded = _json_or(
                content[len(cls._CONTENT_V2_STRING_PREFIX):], None,
                "Failed to decode V2 string message content; returning raw string",
            )
            return decoded if isinstance(decoded, str) else content
        if content.startswith(cls._CONTENT_V2_BYTES_PREFIX):
            payload = content[len(cls._CONTENT_V2_BYTES_PREFIX):]
            try:
                decoded = bytes.fromhex(payload)
            except ValueError:
                return content
            return decoded if decoded.hex() == payload else content
        if content.startswith(cls._CONTENT_JSON_PREFIX):
            return _json_or(
                content[len(cls._CONTENT_JSON_PREFIX):], content,
                "Failed to decode JSON-encoded message content; returning raw string",
            )
        return content

