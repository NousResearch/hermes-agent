"""Regression tests for #41805 — the terminal non-retryable return in
``agent.conversation_loop.run_conversation`` must surface ``failure_reason``.

A hard usage-limit / billing 429 (HTTP 402/429 classified as
``FailoverReason.billing``) is treated as a non-retryable *client* error: once
credential-pool rotation and fallback are exhausted it returns from the
``is_client_error`` branch, NOT from the retry-exhaustion branch. Only the
latter used to stamp ``failure_reason``, so a kanban worker that hit a hard
quota wall before any tool work exited a generic ``1`` instead of the
rate-limit sentinel — and the dispatcher then re-spawned it indefinitely.

These tests follow the source-inspection pattern already used for this
function (see ``test_nous_oauth_401_guidance.py``); ``run_conversation`` needs a
fully-constructed agent + live API client to drive end-to-end, so we assert on
the structure of the terminal returns instead.
"""

from __future__ import annotations

import inspect

from agent import conversation_loop


def test_terminal_returns_stamp_failure_reason():
    """Both terminal failure returns surface ``failure_reason``.

    There are two terminal ``"failed": True`` returns that a hard quota/billing
    error can reach: the non-retryable ``is_client_error`` abort and the
    retry-exhaustion abort. Both must carry ``failure_reason`` so the kanban
    worker can pick the rate-limit sentinel exit code (#41805).
    """
    source = inspect.getsource(conversation_loop.run_conversation)
    occurrences = source.count('"failure_reason": classified.reason.value')
    assert occurrences >= 2, (
        "Expected the non-retryable client-error return AND the retry-"
        "exhaustion return to both stamp failure_reason; found "
        f"{occurrences}. A hard billing/usage-limit 429 exits via the "
        "non-retryable path, so it must surface failure_reason too (#41805)."
    )


def test_non_retryable_client_error_return_carries_failure_reason():
    """The non-retryable terminal return block includes ``failure_reason``.

    Pin the fix to the specific return that hard billing errors take: the
    terminal non-retryable client-error return whose error payload is the
    summarized ``_nonretryable_summary`` (as opposed to the retry-exhaustion
    path below).
    """
    source = inspect.getsource(conversation_loop.run_conversation)
    marker = '"error": _nonretryable_summary,'
    idx = source.find(marker)
    assert idx != -1, "non-retryable client-error return not found"
    # failure_reason should appear within the same return dict, i.e. after the
    # error line and before the NEXT return statement opens. (A comment block
    # may sit between the two lines, so scope to the dict, not a fixed offset.)
    next_return = source.find("return {", idx + len(marker))
    window = source[idx : next_return if next_return != -1 else idx + 1500]
    assert '"failure_reason": classified.reason.value' in window, (
        "non-retryable client-error return must stamp failure_reason so a "
        "hard billing/usage-limit 429 exits with the rate-limit sentinel "
        "rather than a generic crash (#41805)."
    )


def test_proactive_nous_rate_limit_guard_carries_failure_reason():
    """The pre-API Nous guard reports a transient rate-limit failure.

    This return bypasses ``classify_api_error`` entirely, so it must stamp the
    reason directly for Kanban's worker-exit contract.
    """
    source = inspect.getsource(conversation_loop.run_conversation)
    marker = '"error": _nous_msg,'
    idx = source.find(marker)
    assert idx != -1, "proactive Nous rate-limit return not found"
    next_return = source.find("return {", idx + len(marker))
    window = source[idx : next_return if next_return != -1 else idx + 1000]
    assert '"failure_reason": "rate_limit"' in window, (
        "proactive Nous rate-limit guard must use the transient quota exit "
        "contract instead of counting a generic task failure (#41805)"
    )
