"""Invariants for the kanban dispatcher's per-tick log line.

The dispatcher logs one INFO line per board tick. Counts alone tell an operator
that something failed but not *which* card, so the line also names the crashed,
timed-out and auto-blocked task ids. These tests pin the two contracts that make
that addition safe: the failures are named (bounded), and a tick with nothing
failing logs exactly the line it logged before ids were appended.
"""

from __future__ import annotations

import logging
import types

from gateway.kanban_watchers_common import logger as _dispatcher_logger
from gateway.kanban_watchers_dispatcher import _log_spawn_results

# The segment monitors and other readers parse. It must be byte-identical.
_COUNTS = (
    "kanban dispatcher [demo]: spawned=1 reclaimed=0 "
    "crashed=2 timed_out=1 promoted=0 auto_blocked=1"
)
_CLEAN_COUNTS = (
    "kanban dispatcher [demo]: spawned=1 reclaimed=0 "
    "crashed=0 timed_out=0 promoted=0 auto_blocked=0"
)
# Documented cap on ids named per bucket.
_MAX_NAMED = 8


def _result(*, crashed=(), timed_out=(), auto_blocked=()):
    return types.SimpleNamespace(
        spawned=[("t_bad0000", "lane", "/tmp/ws")],
        reclaimed=0,
        promoted=0,
        crashed=list(crashed),
        timed_out=list(timed_out),
        auto_blocked=list(auto_blocked),
    )


def _tick_lines(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("kanban dispatcher [")
    ]


def test_tick_line_names_failing_cards(caplog):
    """A populated failure bucket is named on the line, after the counts."""
    crashed = ["t_badcafe", "t_badbeef"]
    auto_blocked = ["t_badfeed"]
    res = _result(crashed=crashed, timed_out=["t_baddead"], auto_blocked=auto_blocked)
    with caplog.at_level(logging.INFO, logger=_dispatcher_logger.name):
        assert _log_spawn_results([("demo", res)])

    lines = _tick_lines(caplog)
    assert len(lines) == 1, lines
    line = lines[0]

    assert line.startswith(_COUNTS), line
    assert f"crashed_ids={','.join(crashed)}" in line, line
    assert "timed_out_ids=t_baddead" in line, line
    assert f"auto_blocked_ids={','.join(auto_blocked)}" in line, line


def test_mass_failure_tick_caps_named_ids(caplog):
    """A mass auto-block names at most the cap, then accounts for the rest."""
    extra = 3
    auto_blocked = [f"t_bad{i:04d}" for i in range(_MAX_NAMED + extra)]
    with caplog.at_level(logging.INFO, logger=_dispatcher_logger.name):
        _log_spawn_results([("demo", _result(auto_blocked=auto_blocked))])

    lines = _tick_lines(caplog)
    assert len(lines) == 1, lines
    line = lines[0]

    assert "auto_blocked_ids=" in line, line
    named = line.split("auto_blocked_ids=", 1)[1].split(" +", 1)[0].split(",")
    assert len(named) == _MAX_NAMED, line
    assert named == auto_blocked[:_MAX_NAMED], line
    # Nothing is silently dropped: named + remaining accounts for every id.
    assert f"+{extra} more" in line, line
    assert len(named) + extra == len(auto_blocked)
    assert auto_blocked[-1] not in line, line


def test_clean_tick_line_is_unchanged(caplog):
    """No failures means no ids suffix: the line is what it always was."""
    with caplog.at_level(logging.INFO, logger=_dispatcher_logger.name):
        assert _log_spawn_results([("demo", _result())])

    lines = _tick_lines(caplog)
    assert len(lines) == 1, lines
    assert lines[0] == _CLEAN_COUNTS
