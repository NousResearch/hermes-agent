"""Unit tests for the session-list liveness flag (``_session_is_active``).

The client session list used to have no way to tell an in-flight turn from an
un-ended idle session: ``is_active`` was never sent, so clients fell back to
``ended_at`` — which stays null for sessions that finished but were never
explicitly ended, leaving every old chat showing as running. Liveness now
derives from the durable activity projection: a non-empty
``last_activity_description`` with a recent stamp means a turn is in flight,
and the label is cleared when the turn exits.
"""

from __future__ import annotations

import time

from gateway.platforms.api_server import _session_is_active


def _row(**overrides):
    row = {
        "ended_at": None,
        "last_activity_at": None,
        "last_activity_description": None,
    }
    row.update(overrides)
    return row


def test_live_turn_is_active():
    now = time.time()
    assert _session_is_active(
        _row(last_activity_description="receiving stream response", last_activity_at=now - 30),
        now=now,
    )


def test_cleared_label_after_turn_end_is_idle():
    now = time.time()
    assert not _session_is_active(
        _row(last_activity_description="", last_activity_at=now - 60), now=now
    )


def test_long_idle_session_is_not_active():
    now = time.time()
    assert not _session_is_active(_row(last_activity_at=now - 172_800), now=now)


def test_stranded_label_ages_out():
    now = time.time()
    assert not _session_is_active(
        _row(last_activity_description="tool running", last_activity_at=now - 100_000),
        now=now,
    )


def test_ended_session_is_never_active():
    now = time.time()
    assert not _session_is_active(
        _row(
            ended_at=now - 10,
            last_activity_description="x",
            last_activity_at=now - 5,
        ),
        now=now,
    )


def test_missing_activity_timestamp_is_not_active():
    assert not _session_is_active(_row(last_activity_description="x"))
