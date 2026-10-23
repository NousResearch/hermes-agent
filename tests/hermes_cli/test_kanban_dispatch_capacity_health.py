"""Regression tests for the dispatcher-stuck health telemetry's capacity
suppression (#DRE-223 / #DRE-292).

``any_capacity_full`` + a single bad_ticks increment used to collapse every
dispatched board's ``capacity_full`` flag into one ``any()`` check. That is
correct for a single board, but wrong across multiple boards sharing a tick:
one board saturated on its OWN ``max_spawn`` cap would excuse a DIFFERENT,
uncapped board's genuine zero-spawn stall (broken PATH/venv/credentials),
hiding a real operator-actionable failure. ``any_genuine_stall`` judges each
board against its own ready/spawn/capacity state instead.
"""
from hermes_cli.kanban_db_dispatch import DispatchResult, any_capacity_full, any_genuine_stall


def _result(spawned=False, capacity_full=False):
    res = DispatchResult()
    if spawned:
        res.spawned = [("t1", "some-profile", "/workspace")]
    res.capacity_full = capacity_full
    return res


def test_single_full_board_is_not_a_stall():
    """A lone board saturated on its concurrency cap is healthy load, not a stall."""
    assert any_genuine_stall([(True, _result(capacity_full=True))]) is False


def test_single_free_board_with_real_error_is_a_stall():
    """Ready work pending, nothing spawned, no capacity excuse -> genuine stall."""
    assert any_genuine_stall([(True, _result(capacity_full=False))]) is True


def test_mixed_full_and_errored_board_stays_visible():
    """The #DRE-292 bug: one full board must not hide another board's real stall."""
    entries = [
        (True, _result(capacity_full=True)),   # board A: at its own max_spawn, healthy
        (True, _result(capacity_full=False)),  # board B: ready work, zero spawn, no excuse
    ]
    # The OLD any()-of-capacity_full logic would wrongly suppress this tick.
    assert any_capacity_full(res for _ready, res in entries) is True
    # The fixed per-board logic must still flag it as a genuine stall.
    assert any_genuine_stall(entries) is True


def test_mixed_full_and_genuinely_idle_board_is_not_a_stall():
    """A full board plus a board with NO ready work at all stays suppressed."""
    entries = [
        (True, _result(capacity_full=True)),    # board A: full, has ready work waiting
        (False, _result(capacity_full=False)),  # board B: nothing ready, nothing to excuse
    ]
    assert any_genuine_stall(entries) is False


def test_empty_board_result_is_not_a_stall():
    """A board with no ready work never counts, regardless of its result."""
    assert any_genuine_stall([(False, _result())]) is False


def test_quarantined_or_errored_board_result_is_excused_here():
    """``result is None`` (corrupt-DB quarantine, tick exception) already logs its
    own specific warning elsewhere; it must not also drive the generic alarm."""
    assert any_genuine_stall([(True, None)]) is False


def test_successful_spawn_resets():
    """A board that spawned something this tick is never a stall."""
    assert any_genuine_stall([(True, _result(spawned=True))]) is False


def test_no_boards_is_not_a_stall():
    assert any_genuine_stall([]) is False


def test_two_free_boards_one_errored_one_healthy_idle():
    """Multiple non-full boards: only the one with unexplained zero-spawn counts."""
    entries = [
        (False, _result()),              # idle, nothing ready
        (True, _result(capacity_full=False)),  # ready, zero spawn, no excuse -> stall
    ]
    assert any_genuine_stall(entries) is True
