"""Regression: the curator's LLM fork must not poison process-global stdout.

``run_curator_review()`` runs its LLM pass on a daemon thread by default.  The
old implementation wrapped ``review_agent.run_conversation()`` in a
process-global ``contextlib.redirect_stdout(open(os.devnull))``.  Because
``redirect_stdout`` rebinds ``sys.stdout`` for the WHOLE process, two overlapping
curator/background-review passes restore in the wrong order and leave
``sys.stdout`` pointing at an already-closed devnull handle.  Every subsequent
bare ``print`` anywhere in the process then raises::

    ValueError: I/O operation on closed file.

Observed 2026-07-27 on a live gateway: nine cron jobs failed inside a single
scheduler tick, all with that error, because one curator pass had poisoned
``sys.stdout`` for the entire process.

The contract asserted here is behavioural, not implementation-shaped: after any
number of concurrent curator review passes, a print from an unrelated thread
must still reach the real stream.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import time

import pytest


def _drain(fn):
    """Bind a StringIO as the real stdout, run fn, return what reached it."""
    real_out = io.StringIO()
    orig = sys.stdout
    sys.stdout = real_out
    try:
        fn()
    finally:
        sys.stdout = orig
    return real_out.getvalue()


def test_global_redirect_on_overlapping_threads_poisons_stdout():
    """Characterize the bug we are protecting against.

    This documents WHY the curator may not use ``contextlib.redirect_stdout``
    from a worker thread. If CPython ever makes redirect_stdout thread-local
    this test will fail loudly and the guard below can be revisited.
    """

    def body():
        def worker(hold: float):
            with open(os.devnull, "w", encoding="utf-8") as devnull, \
                    contextlib.redirect_stdout(devnull):
                time.sleep(hold)

        # Overlapping enter/exit: A exits first and restores sys.stdout to the
        # devnull handle B installed; B then exits and closes that handle.
        a = threading.Thread(target=worker, args=(0.05,))
        b = threading.Thread(target=worker, args=(0.30,))
        a.start()
        time.sleep(0.02)
        b.start()
        a.join()
        b.join()

        with pytest.raises(ValueError, match="closed file"):
            sys.stdout.write("this must fail")

    _drain(body)


def test_thread_scoped_silence_leaves_stdout_usable():
    """The replacement primitive survives the same overlapping pattern."""
    from agent.thread_scoped_output import thread_scoped_silence

    def body():
        def worker(hold: float):
            with thread_scoped_silence():
                print("silenced chatter")
                time.sleep(hold)

        a = threading.Thread(target=worker, args=(0.05,))
        b = threading.Thread(target=worker, args=(0.30,))
        a.start()
        time.sleep(0.02)
        b.start()
        a.join()
        b.join()

        # No poisoning: the main thread can still print.
        print("survivor")

    captured = _drain(body)
    assert "survivor" in captured
    assert "silenced chatter" not in captured


def test_run_llm_review_does_not_poison_other_threads_stdout():
    """Behavioral guard on the exact call site that caused the outage.

    Runs ``_run_llm_review()`` on a worker thread with a stubbed review agent
    that blocks mid-conversation, and asserts an unrelated thread's print
    still reaches the real stream while the review is executing — and that
    stdout is still usable afterwards.  Fails on the pre-fix implementation,
    which wrapped the conversation in a process-global ``redirect_stdout``.
    """
    from unittest.mock import patch

    import agent.curator as curator

    in_review = threading.Event()
    unrelated_done = threading.Event()

    class _StubReviewAgent:
        """Stands in for the forked AIAgent; blocks inside the review pass."""

        def __init__(self, *args, **kwargs):
            self._session_messages = []

        def run_conversation(self, user_message=""):
            # Chatter the real agent would emit — must be silenced.
            print("review agent chatter")
            in_review.set()
            assert unrelated_done.wait(timeout=5)
            return {"final_response": "review done"}

    def body():
        results = {}

        def run_review():
            results["meta"] = curator._run_llm_review("review prompt")

        def unrelated_writer():
            assert in_review.wait(timeout=5)
            print("unrelated write during review")
            unrelated_done.set()

        reviewer = threading.Thread(target=run_review)
        writer = threading.Thread(target=unrelated_writer)
        with patch("run_agent.AIAgent", _StubReviewAgent):
            reviewer.start()
            writer.start()
            reviewer.join(timeout=10)
            writer.join(timeout=10)

        meta = results.get("meta") or {}
        assert meta.get("error") is None, meta
        assert meta.get("final") == "review done"

        # stdout must not be poisoned once the review pass finishes.
        print("survivor")

    captured = _drain(body)
    assert "unrelated write during review" in captured
    assert "survivor" in captured
    assert "review agent chatter" not in captured
