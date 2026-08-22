"""Tests for the non-auth HTTP 403 (proxy/WAF block) guidance branch in
``agent.conversation_loop``.

Regression context (#53099): a custom OpenAI-compatible relay sitting behind a
WAF blocked the OpenAI SDK's default ``User-Agent`` and answered with a short
plain-text 403 body (``Your request was blocked.``). Hermes classified that as
auth and printed "Your API key was rejected by the provider", sending the user
down the wrong debugging path even though the key, model, and endpoint were all
valid.

The fix adds a conservative branch — ``_summarize_403_proxy_block`` — that fires
only when a 403 body is clearly *not* auth-shaped, and prints proxy/WAF +
``model.default_headers`` guidance instead of the key-rejection message.
"""
from __future__ import annotations

import inspect

from agent import conversation_loop
from agent.conversation_loop import _summarize_403_proxy_block


class _FakeAPIError(Exception):
    """Stand-in for an OpenAI SDK status error: carries ``status_code`` and an
    optional structured ``body`` like the real exceptions do."""

    def __init__(self, message: str, status_code: int, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def test_plain_text_403_block_is_detected():
    """A plain-text 'Your request was blocked.' 403 is recognized as a
    proxy/WAF block and its body is echoed back as a short snippet."""
    err = _FakeAPIError("Your request was blocked.", status_code=403, body=None)
    snippet = _summarize_403_proxy_block(err, 403)
    assert snippet is not None
    assert "blocked" in snippet.lower()


def test_string_body_403_block_is_detected():
    """A 403 whose ``body`` is the plain-text block string is detected too."""
    err = _FakeAPIError(
        "Error code: 403 - Your request was blocked.",
        status_code=403,
        body="Your request was blocked.",
    )
    snippet = _summarize_403_proxy_block(err, 403)
    assert snippet == "Your request was blocked."


def test_auth_shaped_json_403_is_not_a_block():
    """A standard auth-shaped JSON 403 body must NOT be treated as a WAF block —
    callers keep the existing key-rejection guidance."""
    err = _FakeAPIError(
        "permission denied",
        status_code=403,
        body={"error": {"message": "You do not have access to model gpt-5.5",
                        "type": "invalid_request_error"}},
    )
    assert _summarize_403_proxy_block(err, 403) is None


def test_auth_shaped_plain_text_403_is_not_a_block():
    """Even a plain-text 403 that mentions an API-key/permission failure must
    fall back to the key-rejection guidance (markers win)."""
    err = _FakeAPIError("Incorrect API key provided.", status_code=403, body=None)
    assert _summarize_403_proxy_block(err, 403) is None

    err2 = _FakeAPIError("You do not have access to this model.", status_code=403, body=None)
    assert _summarize_403_proxy_block(err2, 403) is None


def test_401_is_never_a_block():
    """The heuristic only applies to 403 — a 401 is left to the OAuth/key
    guidance regardless of body shape."""
    err = _FakeAPIError("Your request was blocked.", status_code=401, body=None)
    assert _summarize_403_proxy_block(err, 401) is None


def test_block_snippet_is_truncated():
    """A long plain-text block body is collapsed to a single short line."""
    long_body = "Your request was blocked. " * 50
    err = _FakeAPIError(long_body, status_code=403, body=long_body)
    snippet = _summarize_403_proxy_block(err, 403)
    assert snippet is not None
    assert len(snippet) <= 201
    assert "\n" not in snippet


def test_waf_guidance_strings_present_in_source():
    """The user-facing proxy/WAF guidance — including the ``default_headers``
    workaround — must exist in the run_conversation body so a future refactor
    can't silently drop it."""
    source = inspect.getsource(conversation_loop.run_conversation)

    assert "_summarize_403_proxy_block" in source
    assert "proxy/WAF block" in source
    assert "default_headers" in source
    assert "User-Agent: curl/8.7.1" in source
    # And the existing key-rejection guidance must still be present for the
    # genuinely auth-shaped case.
    assert "Your API key was rejected by the provider" in source
