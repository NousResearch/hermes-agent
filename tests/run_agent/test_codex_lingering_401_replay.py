"""Lingering Codex 401 token_expired after a live-token refresh (#88510).

A persisted openai-codex session can keep returning HTTP 401
``token_expired`` even though the same bearer works for a new session and
for a minimal Responses request. Hermes already strips cached
``codex_reasoning_items`` on HTTP 400 ``invalid_encrypted_content``.
401 is classified as auth (and must stay that way) so the first pass
still runs the Codex OAuth refresh. Only after that refresh has already
been attempted, and only when the transcript still holds cached
encrypted reasoning, do we reuse the existing one-shot strip+retry.

These tests call the production helpers the conversation loop will use.
They must fail on origin/main (helpers missing / recovery not wired).
"""

from __future__ import annotations

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_retry_state import TurnRetryState


class _Fake401(Exception):
    def __init__(self, message, status_code=401, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}


def _cached_messages():
    return [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "ok",
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": "blob-1"},
                {"type": "reasoning", "encrypted_content": "blob-2"},
            ],
        },
    ]


def _token_expired_error():
    return _Fake401(
        "HTTP 401: Provided authentication token is expired. "
        "Please try signing in again. error code: token_expired",
        status_code=401,
        body={"error": {"code": "token_expired", "message": "Provided authentication token is expired. Please try signing in again."}},
    )


def test_token_expired_401_stays_auth_not_invalid_encrypted_content():
    """Classifier must not treat 401 token_expired as replay rejection."""
    result = classify_api_error(_token_expired_error(), provider="openai-codex", model="gpt-5.4")
    assert result.reason == FailoverReason.auth
    assert result.reason != FailoverReason.invalid_encrypted_content


def test_invalid_encrypted_content_400_still_classifies_as_replay():
    err = _Fake401(
        "invalid_encrypted_content",
        status_code=400,
        body={"error": {"code": "invalid_encrypted_content", "message": "could not decrypt the provided encrypted_content"}},
    )
    result = classify_api_error(err, provider="openai-codex", model="gpt-5.4")
    assert result.reason == FailoverReason.invalid_encrypted_content


def test_should_not_strip_before_codex_auth_refresh():
    from agent.conversation_loop import should_strip_codex_replay_after_lingering_401

    assert (
        should_strip_codex_replay_after_lingering_401(
            api_mode="codex_responses",
            provider="openai-codex",
            status_code=401,
            error=_token_expired_error(),
            auth_retry_attempted=False,
            replay_strip_attempted=False,
            replay_enabled=True,
            messages=_cached_messages(),
        )
        is False
    )


def test_should_not_strip_when_transcript_has_no_cached_reasoning():
    from agent.conversation_loop import should_strip_codex_replay_after_lingering_401

    assert (
        should_strip_codex_replay_after_lingering_401(
            api_mode="codex_responses",
            provider="openai-codex",
            status_code=401,
            error=_token_expired_error(),
            auth_retry_attempted=True,
            replay_strip_attempted=False,
            replay_enabled=True,
            messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        )
        is False
    )


def test_should_not_strip_generic_401_without_token_expired():
    from agent.conversation_loop import should_strip_codex_replay_after_lingering_401

    err = _Fake401(
        "Incorrect API key provided",
        status_code=401,
        body={"error": {"code": "invalid_api_key", "message": "Incorrect API key provided"}},
    )
    assert (
        should_strip_codex_replay_after_lingering_401(
            api_mode="codex_responses",
            provider="openai-codex",
            status_code=401,
            error=err,
            auth_retry_attempted=True,
            replay_strip_attempted=False,
            replay_enabled=True,
            messages=_cached_messages(),
        )
        is False
    )


def test_should_strip_after_auth_retry_when_cache_and_token_expired():
    from agent.conversation_loop import should_strip_codex_replay_after_lingering_401

    assert (
        should_strip_codex_replay_after_lingering_401(
            api_mode="codex_responses",
            provider="openai-codex",
            status_code=401,
            error=_token_expired_error(),
            auth_retry_attempted=True,
            replay_strip_attempted=False,
            replay_enabled=True,
            messages=_cached_messages(),
        )
        is True
    )


def test_recover_strips_cache_once_and_reuses_400_oneshot():
    from agent.conversation_loop import recover_codex_replay_after_lingering_401
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.api_mode = "codex_responses"
    agent.provider = "openai-codex"
    agent._codex_reasoning_replay_enabled = True
    retry = TurnRetryState()
    retry.codex_auth_retry_attempted = True
    messages = _cached_messages()

    recovered = recover_codex_replay_after_lingering_401(
        agent,
        retry,
        messages,
        status_code=401,
        error=_token_expired_error(),
    )
    assert recovered is True
    assert agent._codex_reasoning_replay_enabled is False
    assert "codex_reasoning_items" not in messages[1]
    assert retry.invalid_encrypted_content_retry_attempted is True

    # Second 401 in the same attempt must not fire again.
    messages[1]["codex_reasoning_items"] = [{"encrypted_content": "again"}]
    agent._codex_reasoning_replay_enabled = True
    recovered_again = recover_codex_replay_after_lingering_401(
        agent,
        retry,
        messages,
        status_code=401,
        error=_token_expired_error(),
    )
    assert recovered_again is False
    assert messages[1]["codex_reasoning_items"] == [{"encrypted_content": "again"}]
