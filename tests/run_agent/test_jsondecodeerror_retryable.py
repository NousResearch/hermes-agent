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


    def test_bare_value_error_is_local_validation(self):
        """Programming bugs that raise bare ValueError must still be
        classified as local validation errors (non-retryable)."""
        assert _mirror_agent_predicate(ValueError("bad arg"))






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


    def test_unrelated_type_error_remains_local_validation(self):
        """TypeError without the NoneType-not-iterable pattern still aborts (programming bug)."""
        assert _mirror_agent_predicate(TypeError("tools must be a list"))
        assert _mirror_agent_predicate(TypeError("expected str, got int"))


