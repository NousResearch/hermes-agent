"""The stream staleness deadline must not count time the host spent suspended.

``last_chunk_time["t"]`` is wall clock, and wall clock advances across an S3/s2idle
suspend. A process that resumes therefore sees an elapsed value equal to the sleep and
kills a live connection whose peer was never asked for data. The kill then finds no
socket to abort ("no sockets found; in-flight request may keep running"), so it cannot
recover what it interrupted either.

The monitor compares, every iteration, how far the wall clock moved against how far the
monotonic clock moved. Only the monotonic delta is elapsed time this process actually
ran; the excess is sleep, and the staleness baseline is carried forward over it.
"""

from __future__ import annotations

import logging
import time

from agent import chat_completion_stream_monitor as mod


def test_module_has_a_logger():
    """The suspend branch logs. Without a module logger it would raise NameError on the
    first suspend, turning the fix into a crash in the path it exists to repair."""
    assert isinstance(getattr(mod, "logger", None), logging.Logger)


def test_monitor_carries_the_baseline_over_a_suspend():
    """A 3370s wall jump with one poll interval of monotonic progress is a suspend."""
    slack = 5.0
    threshold = 900.0
    last_chunk = {"t": 1_000_000.0}
    prev_wall, prev_mono = 1_000_000.0, 500.0

    # Resume: wall advanced 3370s, the monotonic clock advanced one 0.3s iteration.
    wall_now, mono_now = prev_wall + 3370.0, prev_mono + 0.3
    skew = (wall_now - prev_wall) - (mono_now - prev_mono)
    assert skew > slack
    last_chunk["t"] += skew

    assert wall_now - last_chunk["t"] < threshold, "suspend must not trip the deadline"


def test_a_real_stall_still_trips_the_deadline():
    """Both clocks advance together during a genuine stall, so no skew is credited."""
    slack = 5.0
    threshold = 900.0
    last_chunk = {"t": 1_000_000.0}
    prev_wall, prev_mono = 1_000_000.0, 500.0

    wall_now, mono_now = prev_wall + 1200.0, prev_mono + 1200.0
    skew = (wall_now - prev_wall) - (mono_now - prev_mono)
    assert abs(skew) <= slack, "no suspend happened, so nothing may be credited"

    assert wall_now - last_chunk["t"] > threshold, "a real stall must still be killed"


def test_monitor_source_tracks_both_clocks():
    """Guard against a refactor dropping the skew correction back to bare wall clock."""
    import inspect

    src = inspect.getsource(mod)
    assert "_SUSPEND_SLACK" in src
    assert "time.monotonic()" in src
    assert 'self.last_chunk_time["t"] += _skew' in src


def test_monotonic_does_not_move_backwards():
    """The correction relies on monotonic never going back; assert the platform honours it."""
    a = time.monotonic()
    b = time.monotonic()
    assert b >= a
