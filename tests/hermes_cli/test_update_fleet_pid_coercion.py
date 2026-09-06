"""Regression for PR #102733 review — pre-restart PID coercion.

The settling-stale fix (b7e57cc0) compares each fleet row's pid against a
snapshot of pre-restart gateway PIDs, only treating a "stale" row as
still-settling when its pid provably belonged to the outgoing process.

The original patch built that snapshot with:

    {int(p) for p in (pre_restart_pids or []) if isinstance(p, int)}

``isinstance(p, int)`` FILTERS the input instead of converting it: any PID
that arrives as a numeric string (e.g. "1234") is silently dropped rather
than coerced, so it can never match its int counterpart in a later ``in``
check. That would silently defeat the whole settling-race fix for that
gateway, re-introducing the false-positive failure it was meant to solve.

``_coerce_pid_set()`` fixes this by converting with ``int()`` instead of
filtering. These tests pin that behavior directly.
"""

from __future__ import annotations

from hermes_cli.update_cmd import _coerce_pid_set


class TestCoercePidSet:
    def test_plain_ints_pass_through(self):
        assert _coerce_pid_set([1234, 5678]) == {1234, 5678}

    def test_numeric_strings_are_coerced_not_dropped(self):
        # This is the exact scenario the reviewer on #102733 called out:
        # a string PID must still end up in the set, not be silently lost.
        assert _coerce_pid_set(["1234", 5678]) == {1234, 5678}

    def test_all_numeric_strings(self):
        assert _coerce_pid_set(["1234", "5678"]) == {1234, 5678}

    def test_none_and_empty_are_safe(self):
        assert _coerce_pid_set(None) == set()
        assert _coerce_pid_set([]) == set()

    def test_garbage_values_are_dropped_not_raised(self):
        assert _coerce_pid_set([1234, "not-a-pid", None, "5678"]) == {1234, 5678}

    def test_mixed_set_matches_regardless_of_original_type(self):
        # The membership check this feeds must succeed whether either side
        # started out as an int or a numeric string.
        pre_restart = _coerce_pid_set(["1234"])
        row_pid_as_string = "1234"
        assert _coerce_pid_set([row_pid_as_string]) & pre_restart
