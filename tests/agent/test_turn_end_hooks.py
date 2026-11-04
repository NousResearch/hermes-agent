"""Unit tests for the turn-end policy hook (agent/turn_end_hooks.py).

The `pre_turn_end` user-hook aggregation lives in `hermes_cli.plugins`
(`get_pre_turn_end_continue_message`) and is tested in
`tests/hermes_cli/test_plugins.py`, alongside `get_pre_verify_continue_message`.

Motivation (real incident): an agent wrote 'Ejecuto C en secuencia' as a
text response and ended the turn without emitting any tool_call. The
user had to point it out. `pre_verify` cannot catch this because it only
fires after code edits; `pre_turn_end` fires on every text-only stop.
"""

from __future__ import annotations

from agent import turn_end_hooks


class TestMaxTurnEndNudges:
    def test_default_when_unset(self):
        assert (
            turn_end_hooks.max_turn_end_nudges({})
            == turn_end_hooks.DEFAULT_MAX_TURN_END_NUDGES
        )
        assert (
            turn_end_hooks.max_turn_end_nudges({"agent": {}})
            == turn_end_hooks.DEFAULT_MAX_TURN_END_NUDGES
        )

    def test_default_is_two(self):
        # More conservative than pre_verify's 3 — mode-error checks are binary.
        assert turn_end_hooks.DEFAULT_MAX_TURN_END_NUDGES == 2

    def test_reads_and_coerces(self):
        assert turn_end_hooks.max_turn_end_nudges({"agent": {"max_turn_end_nudges": 5}}) == 5
        assert turn_end_hooks.max_turn_end_nudges({"agent": {"max_turn_end_nudges": "2"}}) == 2
        assert turn_end_hooks.max_turn_end_nudges({"agent": {"max_turn_end_nudges": -1}}) == 0
        assert turn_end_hooks.max_turn_end_nudges({"agent": {"max_turn_end_nudges": 0}}) == 0

    def test_bad_value_falls_back(self):
        assert (
            turn_end_hooks.max_turn_end_nudges({"agent": {"max_turn_end_nudges": "x"}})
            == turn_end_hooks.DEFAULT_MAX_TURN_END_NUDGES
        )

    def test_none_value_falls_back(self):
        # Defensive: config.get() returning None must not crash int().
        assert (
            turn_end_hooks.max_turn_end_nudges({"agent": {"max_turn_end_nudges": None}})
            == turn_end_hooks.DEFAULT_MAX_TURN_END_NUDGES
        )


class TestTurnEndHookEnabled:
    def test_enabled_by_default(self):
        assert turn_end_hooks.turn_end_hook_enabled({}) is True
        assert turn_end_hooks.turn_end_hook_enabled({"agent": {}}) is True

    def test_opt_out_via_config(self):
        off = {"agent": {"pre_turn_end_hook": False}}
        assert turn_end_hooks.turn_end_hook_enabled(off) is False
        off_str = {"agent": {"pre_turn_end_hook": "false"}}
        assert turn_end_hooks.turn_end_hook_enabled(off_str) is False

    def test_explicit_on(self):
        on = {"agent": {"pre_turn_end_hook": True}}
        assert turn_end_hooks.turn_end_hook_enabled(on) is True
