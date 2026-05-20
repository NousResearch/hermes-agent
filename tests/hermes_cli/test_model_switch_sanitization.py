"""Regression tests for ANSI/control-char poisoning of sessions.model.

Background: in May 2026, four discord sessions persisted with the raw escape
sequence ``\\x1b[B\\x1b[A`` as their ``model`` value, polluting /insights
aggregations with ~26M tokens of unattributable activity. Kanban: t_6d086799.

Two layers of defense were added:

1. ``hermes_cli.model_switch.parse_model_flags`` strips ANSI CSI escapes and
   C0/DEL control characters before flag parsing — the user-facing path.
2. ``hermes_state.SessionDB._insert_session_row`` re-sanitizes its ``model``
   parameter on every INSERT so any code path that bypasses (1) still can't
   poison the column.

These tests guard layer (1). Layer (2) is exercised indirectly through
``test_hermes_state`` integration tests when they touch session inserts.
"""

from __future__ import annotations

from hermes_cli.model_switch import parse_model_flags


class TestParseModelFlagsSanitization:
    def test_pure_arrow_key_garbage_yields_empty_model(self):
        # Down-arrow then up-arrow — exact bytes seen in production poisoning.
        assert parse_model_flags("\x1b[B\x1b[A") == ("", "", False)

    def test_leading_ansi_strips_but_keeps_model(self):
        assert parse_model_flags("\x1b[Bsonnet") == ("sonnet", "", False)

    def test_trailing_ansi_strips_but_keeps_model(self):
        assert parse_model_flags("sonnet\x1b[2J") == ("sonnet", "", False)

    def test_ansi_with_provider_flag_strips_garbage_only(self):
        got = parse_model_flags("\x1b[A\x1b[B --provider anthropic")
        assert got == ("", "anthropic", False)

    def test_c0_control_chars_stripped(self):
        # NUL, SOH, STX prefix + DEL suffix.
        assert parse_model_flags("\x00\x01\x02opus\x7f") == ("opus", "", False)

    def test_tab_in_input_does_not_break_split(self):
        # \t is in the C0 range but should be safely treated as whitespace by
        # raw_args.split(); we explicitly do NOT strip it. The split eats it.
        assert parse_model_flags("opus\tmini") == ("opus mini", "", False)

    def test_plain_model_unaffected(self):
        assert parse_model_flags("opus") == ("opus", "", False)

    def test_global_flag_still_works_after_sanitizer(self):
        assert parse_model_flags("sonnet --global") == ("sonnet", "", True)

    def test_unicode_dash_normalization_still_works_after_sanitizer(self):
        # em-dash before "provider" should still be normalized to "--provider"
        assert parse_model_flags("claude-sonnet-4-6 \u2014provider anthropic") == (
            "claude-sonnet-4-6",
            "anthropic",
            False,
        )

    def test_ansi_with_global_and_provider_combined(self):
        got = parse_model_flags("\x1b[Bopus\x1b[A --provider anthropic --global")
        assert got == ("opus", "anthropic", True)
