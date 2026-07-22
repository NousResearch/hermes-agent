"""Regression guard for #14782: json.JSONDecodeError must not be classified
as a local validation error by the main agent loop.

`json.JSONDecodeError` inherits from `ValueError`. The agent loop's
non-retryable classifier at run_agent.py treats `ValueError` / `TypeError`
as local programming bugs and skips retry. Without an explicit carve-out,
a transient provider hiccup (malformed response body, truncated stream,
routing-layer corruption) that surfaces as a JSONDecodeError would bypass
the retry path and fail the turn immediately.

This test mirrors the exact predicate shape used in run_agent.py so that
any future refactor of that predicate must preserve the invariant:

    JSONDecodeError     → NOT local validation error (retryable)
    UnicodeEncodeError  → NOT local validation error (surrogate path)
    bare ValueError     → IS local validation error (programming bug)
    bare TypeError      → IS local validation error (programming bug)
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent

# Kept in lock-step with agent/conversation_loop.py's _JITER_PARSE_ERROR_RE —
# jiter/serde_json's small, stable error-message vocabulary, verified
# directly against the installed jiter package, not just the line/column
# suffix (an unrelated ValueError could coincidentally share that suffix).
_JITER_PARSE_ERROR_RE = re.compile(
    r"^(?:eof while parsing|trailing (?:characters|comma)|key must be a string|"
    r"expected value|invalid (?:type|escape|unicode|length)|control character|"
    r"number out of range|recursion limit exceeded|duplicate field|unknown field)"
    r".* at line \d+ column \d+$",
    re.IGNORECASE,
)


def _mirror_agent_predicate(err: BaseException) -> bool:
    """Exact shape of run_agent.py's is_local_validation_error check.

    Kept in lock-step with the source. If you change one, change both —
    or, better, refactor the check into a shared helper and have both
    sites import it.
    """
    import ssl

    return (
        isinstance(err, (ValueError, TypeError))
        and not isinstance(err, (UnicodeEncodeError, json.JSONDecodeError))
        and not isinstance(err, ssl.SSLError)
        # NoneType-is-not-iterable shape errors come from upstream SDK /
        # provider response mismatches, not local programming bugs. See
        # the agent/conversation_loop.py inline comment for #33136.
        and not (
            isinstance(err, TypeError)
            and "nonetype" in str(err).lower()
            and "not iterable" in str(err).lower()
        )
        # jiter (openai SDK's Rust JSON parser) raises a plain ValueError
        # on a truncated/corrupted SSE chunk — a transient provider/network
        # issue, not a local bug. See #65147.
        and not (
            type(err) is ValueError
            and _JITER_PARSE_ERROR_RE.search(str(err))
        )
    )


class TestJSONDecodeErrorIsRetryable:

    def test_json_decode_error_is_not_local_validation(self):
        """Provider returning malformed JSON surfaces as JSONDecodeError —
        must be treated as transient so the retry path runs."""
        try:
            json.loads("{not valid json")
        except json.JSONDecodeError as exc:
            assert not _mirror_agent_predicate(exc), (
                "json.JSONDecodeError must be excluded from the "
                "ValueError/TypeError local-validation classification."
            )
        else:
            raise AssertionError("json.loads should have raised")

    def test_unicode_encode_error_is_not_local_validation(self):
        """Existing carve-out — surrogate sanitization handles this separately."""
        try:
            "\ud800".encode("utf-8")
        except UnicodeEncodeError as exc:
            assert not _mirror_agent_predicate(exc)
        else:
            raise AssertionError("encoding lone surrogate should raise")

    def test_bare_value_error_is_local_validation(self):
        """Programming bugs that raise bare ValueError must still be
        classified as local validation errors (non-retryable)."""
        assert _mirror_agent_predicate(ValueError("bad arg"))

    def test_bare_type_error_is_local_validation(self):
        assert _mirror_agent_predicate(TypeError("wrong type"))


class TestAgentLoopSourceStillHasCarveOut:
    """Belt-and-suspenders: the production source must actually include
    the json.JSONDecodeError carve-out. Protects against an accidental
    revert that happens to leave the test file intact."""

    def test_run_agent_excludes_jsondecodeerror_from_local_validation(self):
        import inspect
        from agent import conversation_loop
        # The agent loop body lives in agent/conversation_loop.py after
        # the run_agent.py refactor.  Assert the carve-out is present in
        # the extracted module specifically — if it ever moves back or
        # disappears, this fails loudly rather than silently passing
        # against a non-existent inline replica.
        src = inspect.getsource(conversation_loop)
        # The predicate we care about must reference json.JSONDecodeError
        # in its exclusion tuple. We check for the specific co-occurrence
        # rather than the literal string so harmless reformatting doesn't
        # break us.
        assert "is_local_validation_error" in src
        assert "JSONDecodeError" in src, (
            "agent/conversation_loop.py must carve out json.JSONDecodeError "
            "from the is_local_validation_error classification — see #14782."
        )



class TestNoneTypeNotIterableIsRetryable:
    """Regression for #33136 / closes lingering Telegram \"Non-retryable error (HTTP None)\".

    The chatgpt.com Codex backend (and any other upstream SDK / provider shim)
    can surface ``TypeError: 'NoneType' object is not iterable`` as a wire-shape
    mismatch, not a local programming bug. Even after #33042 made our own
    consumer immune, third-party paths and mocked clients can still produce
    this shape. The classifier should treat it as retryable so the normal
    retry/fallback chain runs.
    """

    def test_nonetype_not_iterable_is_retryable(self):
        err = TypeError("'NoneType' object is not iterable")
        assert not _mirror_agent_predicate(err), (
            "TypeError('NoneType ... not iterable') must be excluded from "
            "is_local_validation_error — it is a provider/SDK shape mismatch, "
            "not a local bug. See #33136."
        )

    def test_nonetype_not_iterable_uppercase_variants_still_retryable(self):
        # The carve-out is case-insensitive; SDK message phrasing can vary.
        for msg in [
            "'NoneType' object is not iterable",
            "NoneType object is not iterable",
            "argument of type 'NoneType' is not iterable",
        ]:
            err = TypeError(msg)
            assert not _mirror_agent_predicate(err), (
                f"Variant {msg!r} should be classified as retryable provider shape error."
            )

    def test_unrelated_type_error_remains_local_validation(self):
        """TypeError without the NoneType-not-iterable pattern still aborts (programming bug)."""
        assert _mirror_agent_predicate(TypeError("tools must be a list"))
        assert _mirror_agent_predicate(TypeError("expected str, got int"))


class TestAgentLoopSourceHasNoneTypeCarveOut:
    """Belt-and-suspenders: the production source must include the carve-out."""

    def test_conversation_loop_excludes_nonetype_not_iterable_from_local_validation(self):
        import inspect
        from agent import conversation_loop
        src = inspect.getsource(conversation_loop)
        assert "is_local_validation_error" in src
        # The specific check must be present.
        assert "nonetype" in src.lower() and "not iterable" in src.lower(), (
            "agent/conversation_loop.py must carve out 'NoneType is not iterable' "
            "TypeErrors from the is_local_validation_error classification — see #33136."
        )


class TestJiterParseErrorIsRetryable:
    """Regression for #65147: jiter (openai SDK's Rust SSE JSON parser)
    raises a plain ValueError on a truncated/corrupted stream chunk — not a
    json.JSONDecodeError subclass, so the #14782 carve-out alone doesn't
    catch it. Its messages consistently end in "at line N column N"."""

    def test_jiter_style_value_error_is_not_local_validation(self):
        err = ValueError("expected value at line 1 column 223")
        assert not _mirror_agent_predicate(err), (
            "A ValueError shaped like jiter's parse-failure message must be "
            "excluded from is_local_validation_error — it is a transient "
            "provider/network stream corruption, not a local bug. See #65147."
        )

    def test_unrelated_value_error_remains_local_validation(self):
        """A bare ValueError without jiter's message shape still aborts."""
        assert _mirror_agent_predicate(ValueError("bad arg"))
        assert _mirror_agent_predicate(ValueError("invalid literal for int()"))

    def test_value_error_subclass_is_not_matched(self):
        """Only a bare ValueError (type(err) is ValueError) matches — a
        subclass like json.JSONDecodeError is already excluded above via
        isinstance, so this carve-out must not double-match/broaden scope."""
        try:
            json.loads("{not valid json at line 1 column 5")
        except json.JSONDecodeError as exc:
            # Already excluded by the JSONDecodeError isinstance check —
            # confirm the jiter carve-out doesn't need to fire for it.
            assert not _mirror_agent_predicate(exc)

    def test_unrelated_valueerror_with_same_line_column_suffix_is_not_matched(self):
        """A same-shaped SUFFIX alone must not be enough to match — only
        jiter/serde_json's actual message vocabulary should. An app-level
        validation error that happens to end in "at line N column N" (e.g.
        a config-file parser reporting its own location) must still abort
        as a local programming bug, not silently retry."""
        assert _mirror_agent_predicate(
            ValueError("custom field validation failed at line 1 column 223")
        )
        assert _mirror_agent_predicate(
            ValueError("unexpected indentation at line 4 column 10")
        )


_TEST_AGENT_KWARGS = {
    "api_key": "-".join(["not", "a", "real", "credential", "placeholder"]),
    "base_url": "https://openrouter.ai/api/v1",
    "quiet_mode": True,
    "skip_context_files": True,
    "skip_memory": True,
}


def _agent_with_mocked_client():
    """Minimal AIAgent with a mocked OpenAI client, ready for run_conversation."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(**_TEST_AGENT_KWARGS)
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a.tool_delay = 0
    a.compression_enabled = False
    a.save_trajectories = False

    return a


def _mock_success_response(content="Done"):
    msg = SimpleNamespace(
        content=content, tool_calls=None, reasoning=None,
        reasoning_content=None, reasoning_details=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    resp = SimpleNamespace(choices=[choice], model="test/model", usage=None)

    return resp


class TestJiterParseErrorRealLoopRetry:
    """Real end-to-end coverage through agent.run_conversation() — not a
    source scan. A source-presence check (inspect.getsource) passes when
    the implementation is subtly broken and fails on a pure refactor with
    identical behavior; it also can't run against a built/bundled artifact.
    Drive the actual retry classifier instead, mirroring the pattern in
    tests/run_agent/test_streaming.py's failing-first-call/succeeding-
    second-call tests."""

    def test_jiter_style_valueerror_retries_then_succeeds(self):
        agent = _agent_with_mocked_client()
        agent.client.chat.completions.create.side_effect = [
            ValueError("expected value at line 1 column 223"),
            _mock_success_response("Done"),
        ]

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.time.sleep"),
        ):
            result = agent.run_conversation("hello")

        assert agent.client.chat.completions.create.call_count == 2, (
            "A jiter-shaped ValueError on the first call must be retried, not "
            "aborted as a local programming bug — see #65147."
        )
        assert result.get("completed") is True
        assert result.get("failed") is not True

    def test_unrelated_valueerror_with_same_suffix_aborts_without_retry(self):
        """Sanity check for the negative case: a same-suffix but non-jiter
        ValueError must still abort immediately as a local validation
        error, proving the carve-out is jiter-specific and not just
        matching the line/column suffix shape."""
        agent = _agent_with_mocked_client()
        agent.client.chat.completions.create.side_effect = ValueError(
            "custom field validation failed at line 1 column 223"
        )

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.time.sleep"),
        ):
            result = agent.run_conversation("hello")

        assert agent.client.chat.completions.create.call_count == 1, (
            "A non-jiter ValueError sharing only the line/column suffix must "
            "abort immediately, not retry."
        )
        assert result.get("failed") is True
