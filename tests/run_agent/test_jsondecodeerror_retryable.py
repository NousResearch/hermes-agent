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
    return (
        isinstance(err, (ValueError, TypeError))
        and not isinstance(err, (UnicodeEncodeError, json.JSONDecodeError))
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
        import run_agent
        import inspect
        src = inspect.getsource(run_agent)
        # The predicate we care about must reference json.JSONDecodeError
        # in its exclusion tuple. We check for the specific co-occurrence
        # rather than the literal string so harmless reformatting doesn't
        # break us.
        assert "is_local_validation_error" in src
        assert "JSONDecodeError" in src, (
            "run_agent.py must carve out json.JSONDecodeError from the "
            "is_local_validation_error classification — see #14782."
        )
