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
