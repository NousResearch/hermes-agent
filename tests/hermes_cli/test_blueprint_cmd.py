"""Behavioral tests for the pure helpers in hermes_cli.blueprint_cmd.

The catalog-dependent functions (match_blueprint, handle_blueprint_command,
build_blueprint_seed) require cron.blueprint_catalog; only the stdlib-pure
helpers are covered here to keep the suite fast and import-free.
"""

import pytest

from hermes_cli.blueprint_cmd import (
    BlueprintCommandResult,
    _fmt_candidates,
    _manage_hint,
    _parse_kv,
)


# ── _parse_kv ─────────────────────────────────────────────────────────────────

class TestParseKv:
    def test_splits_slot_equals_value(self):
        values, leftovers = _parse_kv(["time=08:00", "days=weekdays"])
        assert values == {"time": "08:00", "days": "weekdays"}
        assert leftovers == []

    def test_bare_tokens_go_to_leftovers(self):
        values, leftovers = _parse_kv(["morning-brief", "extra"])
        assert values == {}
        assert leftovers == ["morning-brief", "extra"]

    def test_mixed_tokens(self):
        values, leftovers = _parse_kv(["morning-brief", "time=09:00", "extra"])
        assert values == {"time": "09:00"}
        assert leftovers == ["morning-brief", "extra"]

    def test_empty_key_goes_to_leftovers(self):
        # "=value" has an empty key — treat as leftover
        values, leftovers = _parse_kv(["=something"])
        assert values == {}
        assert leftovers == ["=something"]

    def test_value_may_contain_equals(self):
        # Only the first "=" is the partition point
        values, leftovers = _parse_kv(["criteria=from=boss"])
        assert values["criteria"] == "from=boss"

    def test_empty_input(self):
        values, leftovers = _parse_kv([])
        assert values == {}
        assert leftovers == []

    def test_whitespace_stripped_from_key_and_value(self):
        values, _ = _parse_kv(["  key  =  val  "])
        assert values.get("key") == "val"

    def test_multiple_same_key_last_wins(self):
        values, _ = _parse_kv(["k=first", "k=second"])
        assert values["k"] == "second"


# ── _manage_hint ──────────────────────────────────────────────────────────────

class TestManageHint:
    def test_cli_surface_mentions_cron(self):
        hint = _manage_hint("cli")
        assert "/cron" in hint

    def test_gateway_surface_no_cron_command(self):
        hint = _manage_hint("gateway")
        assert "/cron" not in hint

    def test_gateway_hint_non_empty(self):
        assert _manage_hint("gateway").strip() != ""

    def test_unknown_surface_falls_through(self):
        # Anything that isn't "cli" gets the gateway hint
        hint = _manage_hint("telegram")
        assert "/cron" not in hint


# ── _fmt_candidates ───────────────────────────────────────────────────────────

class _FakeBlueprint:
    def __init__(self, key, title):
        self.key = key
        self.title = title


class TestFmtCandidates:
    def test_contains_query(self):
        result = _fmt_candidates("morn", [_FakeBlueprint("morning-brief", "Morning Brief")])
        assert "morn" in result

    def test_lists_all_candidate_keys(self):
        candidates = [
            _FakeBlueprint("morning-brief", "Morning Brief"),
            _FakeBlueprint("morning-standup", "Morning Standup"),
        ]
        result = _fmt_candidates("morning", candidates)
        assert "morning-brief" in result
        assert "morning-standup" in result

    def test_includes_blueprint_instruction(self):
        result = _fmt_candidates("x", [_FakeBlueprint("x-thing", "X Thing")])
        assert "/blueprint" in result


# ── BlueprintCommandResult ────────────────────────────────────────────────────

def test_blueprint_command_result_defaults():
    r = BlueprintCommandResult("some text")
    assert r.text == "some text"
    assert r.agent_seed is None


def test_blueprint_command_result_with_seed():
    r = BlueprintCommandResult("text", agent_seed="seed content")
    assert r.agent_seed == "seed content"
