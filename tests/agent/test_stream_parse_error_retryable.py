"""Regression guard: every native streaming-parser failure is retryable.

The OpenAI and Anthropic SDKs parse streaming payloads with ``jiter`` (a Rust
extension). When a proxy or provider emits malformed bytes, jiter raises a bare
``ValueError`` whose message ends with ``at line N column M`` — the shape
observed in production was ``key must be a string at line 1 column 338``.

Hermes only recognised ONE of jiter's messages (``expected ident at line``) as
transient wire damage. Every other message fell through to
``_is_local_validation_error``, whose ``ValueError`` branch assumes a local
programming bug — so the turn was aborted as ``Non-retryable`` instead of
retrying. The user-visible effect was a truncated reply followed by a silent
stall until they sent another message.

The invariant this file protects, stated as a relationship rather than a
snapshot of jiter's wording:

    any ValueError carrying jiter's position suffix  → retryable (not local)
    a ValueError without it                          → local validation error

That form is deliberate: matching the suffix keeps the guard correct for jiter
messages that do not exist yet, which is what let the original bug through.
"""
from __future__ import annotations

import json
import ssl

import pytest

from agent.errors import is_provider_stream_parse_error
from agent.turn_api_error import _is_local_validation_error


class TestJiterStreamParseErrorsAreRetryable:
    """Every jiter message shape must be treated as provider wire damage."""

    # Each entry is (error, why) — messages taken from real jiter output, not invented.
    JITER_MESSAGES = [
        "key must be a string at line 1 column 338",   # the production stall
        "key must be a string at line 1 column 566",
        "expected ident at line 1 column 149",         # the one already handled
        "expected value at line 1 column 6",
        "expected `:` at line 1 column 6",
        "trailing comma at line 1 column 8",
        "invalid number at line 1 column 7",
        "EOF while parsing an object at line 1 column 1",
        "EOF while parsing a list at line 1 column 4",
        "EOF while parsing a value at line 1 column 3",
        "EOF while parsing a string at line 1 column 8",
        "trailing characters at line 1 column 8",
    ]

    @pytest.mark.parametrize("message", JITER_MESSAGES)
    def test_recognised_as_provider_stream_parse_error(self, message: str) -> None:
        assert is_provider_stream_parse_error(ValueError(message)), (
            f"{message!r} is jiter wire damage and must be retryable"
        )

    @pytest.mark.parametrize("message", JITER_MESSAGES)
    def test_not_classified_as_local_validation(self, message: str) -> None:
        """The non-retryable gate must let these through to the retry path."""
        assert not _is_local_validation_error(ValueError(message)), (
            f"{message!r} must not abort the turn as a local bug"
        )


class TestLocalValidationStillAborts:
    """The predicate must not be loosened into 'all ValueErrors are retryable'."""

    @pytest.mark.parametrize("error", [
        ValueError("invalid local request shape"),
        ValueError("tools must be a list"),
        ValueError("bad arg"),
        TypeError("unsupported operand type(s) for +: 'int' and 'str'"),
    ])
    def test_plain_errors_remain_local_validation(self, error: BaseException) -> None:
        assert not is_provider_stream_parse_error(error)
        assert _is_local_validation_error(error), (
            f"{error!r} is a local programming bug and must stay non-retryable"
        )

    def test_error_without_position_suffix_is_not_parse_error(self) -> None:
        """A message that *mentions* line/column but has no suffix is not jiter."""
        assert not is_provider_stream_parse_error(
            ValueError("expected ident at line 1 column 149 raised earlier")
        )

    def test_position_suffix_without_line_keyword_does_not_match(self) -> None:
        assert not is_provider_stream_parse_error(ValueError("at line 1 column 5"))


class TestExistingCarveOutsPreserved:
    """Carve-outs that predate this fix must keep working."""

    def test_json_decode_error_is_not_parse_error(self) -> None:
        """JSONDecodeError is a ValueError subclass with its own position format."""
        try:
            json.loads("{not valid json")
        except json.JSONDecodeError as exc:
            assert not is_provider_stream_parse_error(exc)
            assert not _is_local_validation_error(exc)
        else:
            raise AssertionError("json.loads should have raised")

    def test_unicode_encode_error_is_not_parse_error(self) -> None:
        exc = UnicodeEncodeError("ascii", "\ud800", 0, 1, "surrogates not allowed")
        assert not is_provider_stream_parse_error(exc)
        assert not _is_local_validation_error(exc)

    def test_ssl_error_is_not_parse_error(self) -> None:
        exc = ssl.SSLError("certificate verify failed")
        assert not is_provider_stream_parse_error(exc)
        assert not _is_local_validation_error(exc)

    def test_nonetype_not_iterable_type_error_unaffected(self) -> None:
        assert not _is_local_validation_error(
            TypeError("'NoneType' object is not iterable")
        )

    def test_non_value_error_never_matches(self) -> None:
        assert not is_provider_stream_parse_error(RuntimeError("key must be a string at line 1 column 2"))
