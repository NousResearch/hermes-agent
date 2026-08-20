"""Regression guard: retry/backoff deadlines must be monotonic.

The agent loop's two retry-backoff blocks in ``agent/conversation_loop.py``
(invalid-response retry and API-error retry) build a deadline
``sleep_end = ... + wait_time`` and spin on ``while ... < sleep_end:``
with a short ``time.sleep(0.2)`` so the interrupt flag stays responsive.

If that deadline is built from ``time.time()`` (wall-clock), a clock
adjustment during the wait — NTP correction, DST transition on systems
that do not use UTC monotonic clocks, manual ``date`` change, VM resume
into a different host clock — can either extend the wait far past the
configured backoff (clock jumps backward) or end it almost immediately
(clock jumps forward). Either case breaks the retry contract: backoff
windows are meant to space out provider calls deterministically, not
to drift with the host clock.

``time.monotonic()`` is the contract-correct primitive here: it never
goes backwards, never DST-jumps, and is unaffected by ``settimeofday``.

These tests assert the invariant via source inspection (the established
pattern in this directory — see ``TestDeadRetryCode`` and
``test_counters_not_reset_in_preamble`` in ``test_run_agent.py``).
Behaviourally driving ``run_conversation`` through both retry paths
just to observe a 0.2 s sleep loop adds heavyweight setup with no
extra coverage — the bug is structural, so the guard is structural.
"""
from __future__ import annotations

import inspect
import re


def _loop_source() -> str:
    from agent.conversation_loop import run_conversation
    return inspect.getsource(run_conversation)


class TestRetryBackoffUsesMonotonicDeadline:
    """Both retry-backoff blocks must build ``sleep_end`` from
    ``time.monotonic()`` and poll it the same way."""

    def test_sleep_end_assigned_from_monotonic(self):
        """Each retry-backoff block assigns ``sleep_end = time.monotonic() + wait_time``."""
        src = _loop_source()
        count = src.count("sleep_end = time.monotonic() + wait_time")
        assert count == 2, (
            f"Expected 2 monotonic sleep_end assignments (invalid-response retry "
            f"and API-error retry), found {count}. A wall-clock deadline here "
            f"is a clock-drift bug — see test docstring."
        )

    def test_no_wall_clock_sleep_end_assignment(self):
        """No backoff block may assign ``sleep_end`` from ``time.time()``.

        Catches regressions where someone "simplifies" the monotonic call
        back to wall-clock during a refactor.
        """
        src = _loop_source()
        assert "sleep_end = time.time()" not in src, (
            "sleep_end must be derived from time.monotonic(), not time.time(). "
            "Wall-clock deadlines drift with NTP/DST/manual clock changes."
        )

    def test_loop_condition_polls_monotonic(self):
        """The interrupt-responsive spin loop must poll ``time.monotonic()``.

        Mixing the wall-clock and monotonic clocks across the assignment
        and the loop check would produce a deadline in one time domain
        compared against a "now" in another — a silently broken wait.
        """
        src = _loop_source()
        count = src.count("while time.monotonic() < sleep_end:")
        assert count == 2, (
            f"Expected 2 monotonic loop conditions, found {count}."
        )
        # Negative side: no wall-clock variant slipped in.
        assert "while time.time() < sleep_end:" not in src

    def test_remaining_seconds_uses_monotonic(self):
        """The ``Ns remaining`` activity-touch message must compute its
        delta against the same clock the deadline was built from.

        ``int(sleep_end - time.time())`` after ``sleep_end = time.monotonic() + N``
        is meaningless — the subtraction crosses time domains and yields
        a huge positive or negative integer depending on epoch offset.
        """
        src = _loop_source()
        count = src.count("int(sleep_end - time.monotonic())")
        assert count == 2, (
            f"Expected 2 monotonic remaining-seconds calculations, found {count}."
        )
        assert "int(sleep_end - time.time())" not in src

    def test_both_backoff_blocks_are_internally_consistent(self):
        """Within each retry-backoff block, the assignment, loop condition,
        and remaining-seconds calculation must all use the same clock.

        This catches the class of regression where someone fixes one of
        the three calls but leaves the others on wall-clock — the deadline
        and the "now" comparison would then live in different time domains.
        """
        src = _loop_source()
        # Find each ``sleep_end = ...`` assignment and its trailing block
        # up to the next blank-line separator at the same indent.  The block
        # always contains the loop condition and one ``Ns remaining`` line.
        blocks = re.findall(
            r"sleep_end = time\.\w+\(\) \+ wait_time"
            r"(?:.*?)"
            r"s remaining",
            src,
            re.DOTALL,
        )
        assert len(blocks) == 2, (
            f"Expected 2 backoff blocks, found {len(blocks)}."
        )
        for i, block in enumerate(blocks, 1):
            assert "time.time()" not in block, (
                f"Backoff block #{i} mixes wall-clock and monotonic — "
                "every clock reference inside one block must use the same "
                "primitive."
            )
            assert block.count("time.monotonic()") == 3, (
                f"Backoff block #{i} should reference time.monotonic() "
                f"exactly 3 times (assignment, loop check, remaining), "
                f"found {block.count('time.monotonic()')}."
            )
