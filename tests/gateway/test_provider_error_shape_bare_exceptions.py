"""Bare python exception strings must never reach chat as assistant prose.

Observed live (2026-08-01, Telegram gateway): a MoA aggregator turn failed
with ``TypeError: 'types.SimpleNamespace' object is not iterable`` and the
raw ``str(exc)`` became the turn's reply body. ``_looks_like_gateway_
provider_error`` matched none of its envelope shapes (``api ... failed``,
``http NNN``, ``error code:`` ...), so the sanitizer passed the exception
text through to the user verbatim instead of rewriting it to the safe
canned provider-failure reply.

The shape regex now catches the two common bare-exception shapes at message
start:

- ``'X' object is/has/does ...`` (dunder-protocol TypeErrors and friends)
- ``SomethingError: ...`` / ``SomethingException: ...``

Both heuristics of ``_looks_like_gateway_provider_error`` (short body,
start-anchored marker) still gate the rewrite, so assistant prose that
merely *mentions* an exception mid-sentence keeps flowing through.
"""

import pytest

from gateway.run import (
    _gateway_provider_error_reply,
    _looks_like_gateway_provider_error,
)


@pytest.mark.parametrize(
    "text",
    [
        "'types.SimpleNamespace' object is not iterable",  # the live incident
        "'NoneType' object has no attribute 'choices'",
        "'dict' object does not support indexing",
        "TypeError: 'NoneType' object is not subscriptable",
        "KeyError: 'choices'",
        "ConnectionError: pool timeout",
        "⚠️ RuntimeError: stream closed before completion",  # symbol prefix
        "  ValueError: invalid literal for int()",  # leading whitespace
    ],
)
def test_bare_exception_strings_are_provider_errors(text):
    assert _looks_like_gateway_provider_error(text)


@pytest.mark.parametrize(
    "text",
    [
        # Normal assistant prose stays untouched.
        "here are the tweets you asked for",
        # Mid-sentence mentions are not start-anchored markers.
        "the 'agent' object has a long history in AI research",
        "In python, a TypeError: usually means the types don't match — "
        "for example when you add a str to an int.",
        # The long-body heuristic still vetoes envelope-shaped starts.
        "TypeError: this looks like an envelope but is actually a long "
        "assistant explanation of exception handling that keeps going "
        + "and elaborates further with many details. " * 12,
        "",
    ],
)
def test_assistant_prose_is_not_rewritten(text):
    assert not _looks_like_gateway_provider_error(text)


def test_rewrite_produces_safe_reply_without_raw_exception_text():
    leak = "'types.SimpleNamespace' object is not iterable"
    reply = _gateway_provider_error_reply(leak)
    assert "SimpleNamespace" not in reply
    assert "gateway logs" in reply
