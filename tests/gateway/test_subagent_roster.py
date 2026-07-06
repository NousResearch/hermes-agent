"""Tests for the live delegate_task subagent roster bubble.

Covers: the subagent_roster display knob (resolution + normalisation), the pure
SubagentRosterState fold + formatter, the pipeline gate, and gateway-level
wiring (callback attach, sentinel relay, seed/edit/collapse, races).
"""

from gateway.display_config import OVERRIDEABLE_KEYS, resolve_display_setting


# ── display_config: resolution + normalisation ──────────────────────────────
class TestSubagentRosterConfig:
    def test_default_is_off_everywhere(self):
        assert resolve_display_setting({}, "telegram", "subagent_roster") == "off"
        assert resolve_display_setting({}, "discord", "subagent_roster") == "off"
        assert resolve_display_setting({}, "unknown", "subagent_roster") == "off"

    def test_global_on_applies(self):
        cfg = {"display": {"subagent_roster": "on"}}
        assert resolve_display_setting(cfg, "telegram", "subagent_roster") == "on"

    def test_platform_override_wins(self):
        cfg = {
            "display": {
                "subagent_roster": "off",
                "platforms": {"telegram": {"subagent_roster": "on"}},
            }
        }
        assert resolve_display_setting(cfg, "telegram", "subagent_roster") == "on"
        assert resolve_display_setting(cfg, "discord", "subagent_roster") == "off"

    def test_unknown_string_fails_safe_to_off(self):
        cfg = {"display": {"platforms": {"telegram": {"subagent_roster": "loud"}}}}
        assert resolve_display_setting(cfg, "telegram", "subagent_roster") == "off"

    def test_legacy_booleans(self):
        on = {"display": {"platforms": {"telegram": {"subagent_roster": True}}}}
        off = {"display": {"platforms": {"telegram": {"subagent_roster": False}}}}
        assert resolve_display_setting(on, "telegram", "subagent_roster") == "on"
        assert resolve_display_setting(off, "telegram", "subagent_roster") == "off"

    def test_case_insensitive(self):
        cfg = {"display": {"platforms": {"telegram": {"subagent_roster": "ON"}}}}
        assert resolve_display_setting(cfg, "telegram", "subagent_roster") == "on"

    def test_flag_is_in_overrideable_keys(self):
        assert "subagent_roster" in OVERRIDEABLE_KEYS


# ── shared duration formatter ───────────────────────────────────────────────
class TestDurationFormat:
    def test_shapes(self):
        from gateway.duration_format import format_duration

        assert format_duration(0) == "0:00"
        assert format_duration(45) == "0:45"
        assert format_duration(83) == "1:23"
        assert format_duration(3665) == "1:01:05"

    def test_clamps_negative_and_bad(self):
        from gateway.duration_format import format_duration

        assert format_duration(-5) == "0:00"
        assert format_duration(None) == "0:00"


# ── roster-only human elapsed formatter (3m 9s, not 3:09) ───────────────────
class TestFormatElapsed:
    def test_shapes(self):
        from gateway.subagent_roster import format_elapsed

        assert format_elapsed(0) == "0s"
        assert format_elapsed(9) == "9s"
        assert format_elapsed(45) == "45s"
        assert format_elapsed(60) == "1m"          # trailing 0s dropped
        assert format_elapsed(83) == "1m 23s"
        assert format_elapsed(189) == "3m 9s"       # the 3:09 -> 3m 9s case
        assert format_elapsed(3600) == "1h 0m"
        assert format_elapsed(3665) == "1h 1m 5s"
        assert format_elapsed(3720) == "1h 2m"      # trailing 0s dropped

    def test_clamps_negative_and_bad(self):
        from gateway.subagent_roster import format_elapsed

        assert format_elapsed(-5) == "0s"
        assert format_elapsed(None) == "0s"


# ── label is rendered as an inline code span (backticks) ────────────────────
class TestRosterLabelBackticks:
    def test_running_row_label_in_backticks(self):
        from gateway.subagent_roster import format_subagent_roster

        rows = [{"glyph": "▶", "label": "verify php", "elapsed": 5.0, "running": True, "tools": 0}]
        text = format_subagent_roster(rows)
        assert "▶ `verify php` · `5s`" in text

    def test_backticks_in_goal_are_stripped(self):
        # A goal containing backticks must not break the inline code span.
        from gateway.subagent_roster import roster_label

        assert roster_label("run `pytest` now") == "run pytest now"


# ── pure roster state: fold ─────────────────────────────────────────────────
class TestFoldSubagentRoster:
    def _state(self):
        from gateway.subagent_roster import SubagentRosterState

        return SubagentRosterState()

    def test_running_row_from_active_registry(self):
        s = self._state()
        s.start("a", goal="verify php", task_index=0, started_at=900.0)
        active = {"a": {"started_at": 940.0, "tool_count": 7}}
        rows = s.fold(active, now=1000.0)
        assert rows == [{
            "glyph": "▶", "label": "verify php",
            "elapsed": 60.0, "running": True, "tools": 7,
            "model": "", "reasoning": None,
        }]

    def test_terminal_row_from_complete_map(self):
        s = self._state()
        s.start("a", goal="run tests", task_index=0, started_at=900.0)
        s.complete("a", status="completed", duration=45.0)
        rows = s.fold({}, now=1000.0)
        assert rows[0]["glyph"] == "✓" and rows[0]["running"] is False
        assert rows[0]["elapsed"] == 45.0

    def test_terminal_row_keeps_final_tool_count(self):
        # A finished child must KEEP its tool count, not drop it to 0. The
        # registry deletes the child on completion, so the count is carried on
        # the complete() event and stored in terminal state.
        s = self._state()
        s.start("a", goal="run tests", task_index=0, started_at=900.0)
        s.complete("a", status="completed", duration=45.0, tools=56)
        rows = s.fold({}, now=1000.0)
        assert rows[0]["tools"] == 56

    def test_complete_sentinel_carries_tool_count(self):
        s = self._state()
        s.apply_event(("__roster_start__", "a", "goal a", 0, 100.0))
        s.apply_event(("__roster_complete__", "a", "completed", 3.0, 12))
        rows = s.fold({}, now=200.0)
        assert rows[0]["glyph"] == "✓" and rows[0]["tools"] == 12

    def test_start_sentinel_preserves_context_metadata_when_present(self):
        s = self._state()
        s.apply_event(("__roster_start__", "a", "goal a", 0, 100.0, "m", None, True, "child-session"))
        rows = s.fold({}, now=200.0)
        assert rows[0]["context_available"] is True
        assert rows[0]["context_child_session_id"] == "child-session"

    def test_complete_sentinel_without_tool_count_is_zero(self):
        # Older producers / replayed queues omit the tool tail; must not crash.
        s = self._state()
        s.apply_event(("__roster_start__", "a", "goal a", 0, 100.0))
        s.apply_event(("__roster_complete__", "a", "completed", 3.0))
        rows = s.fold({}, now=200.0)
        assert rows[0]["tools"] == 0

    def test_status_glyph_mapping(self):
        s = self._state()
        for i, st in enumerate(["completed", "failed", "timeout", "interrupted"]):
            s.start(st, goal=st, task_index=i, started_at=0.0)
            s.complete(st, status=st, duration=1.0)
        rows = s.fold({}, now=10.0)
        assert [r["glyph"] for r in rows] == ["✓", "✗", "⏱", "⏹"]

    def test_unknown_status_fails_closed_to_errored(self):
        s = self._state()
        s.start("a", goal="x", task_index=0, started_at=0.0)
        s.complete("a", status="weird_new_state", duration=1.0)
        rows = s.fold({}, now=10.0)
        assert rows[0]["glyph"] == "?"  # never ✓

    def test_order_follows_seen_order_no_reorder(self):
        s = self._state()
        s.start("a", goal="a", task_index=0, started_at=0.0)
        s.start("b", goal="b", task_index=1, started_at=0.0)
        s.complete("a", status="completed", duration=5.0)  # a done, b running
        active = {"b": {"started_at": 0.0, "tool_count": 0}}
        rows = s.fold(active, now=10.0)
        assert [r["label"] for r in rows] == ["a", "b"]  # done row stays put

    def test_missing_from_both_falls_back_to_meta_started(self):
        s = self._state()
        s.start("a", goal="x", task_index=0, started_at=950.0)
        rows = s.fold({}, now=1000.0)
        assert rows[0]["running"] is True and rows[0]["elapsed"] == 50.0

    def test_complete_without_start_synthesizes_row(self):
        s = self._state()
        s.complete("a", status="completed", duration=6.0)
        rows = s.fold({}, now=10.0)
        assert len(rows) == 1 and rows[0]["glyph"] == "✓"

    def test_apply_event_dispatch(self):
        s = self._state()
        s.apply_event(("__roster_start__", "a", "goal a", 0, 100.0))
        s.apply_event(("__roster_complete__", "a", "completed", 3.0))
        rows = s.fold({}, now=200.0)
        assert rows[0]["glyph"] == "✓" and rows[0]["elapsed"] == 3.0

    def test_has_records(self):
        s = self._state()
        assert s.has_records() is False
        s.start("a", goal="x")
        assert s.has_records() is True


# ── pure roster: label + formatter ──────────────────────────────────────────
class TestRosterLabel:
    def test_cap_and_whitespace(self):
        from gateway.subagent_roster import roster_label

        assert roster_label("") == "subagent"
        assert roster_label("multi\nline   goal") == "multi line goal"
        long = "x" * 200
        out = roster_label(long)
        assert len(out) == _LABEL_CAP and out.endswith("…")


class TestFormatSubagentRoster:
    def test_empty_returns_none(self):
        from gateway.subagent_roster import format_subagent_roster

        assert format_subagent_roster([]) is None

    def test_full_table_header_and_rows(self):
        from gateway.subagent_roster import format_subagent_roster

        rows = [
            {"glyph": "▶", "label": "verify php", "elapsed": 83.0, "running": True, "tools": 8},
            {"glyph": "▶", "label": "verify fe", "elapsed": 80.0, "running": True, "tools": 0},
            {"glyph": "✓", "label": "run tests", "elapsed": 45.0, "running": False, "tools": 0},
            {"glyph": "✗", "label": "check types", "elapsed": 30.0, "running": False, "tools": 0},
        ]
        text = format_subagent_roster(rows)
        lines = text.split("\n")
        # Live header carries the WALL-CLOCK fallback = slowest row's elapsed
        # (max(83,80,45,30)=83), NOT the sum: children run in parallel.
        assert lines[0] == "🤖 Subagents — 2 running, 1 done, 1 failed · `1m 23s`"
        assert lines[1] == "▶ `verify php` · `1m 23s` · 8 tools"
        assert lines[2] == "▶ `verify fe` · `1m 20s`"
        assert lines[3] == "✓ `run tests` · `45s`"
        assert lines[4] == "✗ `check types` · `30s`"

    def test_terminal_rows_keep_tool_count(self):
        # Tool count must persist on a DONE row, not vanish when running flips
        # to False. Alfredo asked to keep the count after the agent finishes.
        from gateway.subagent_roster import format_subagent_roster

        rows = [
            {"glyph": "✓", "label": "review", "elapsed": 949.0, "running": False, "tools": 56},
            {"glyph": "✗", "label": "audit", "elapsed": 30.0, "running": False, "tools": 1},
            {"glyph": "✓", "label": "noop", "elapsed": 5.0, "running": False, "tools": 0},
        ]
        text = format_subagent_roster(rows)
        lines = text.split("\n")
        assert lines[1] == "✓ `review` · `15m 49s` · 56 tools"
        assert lines[2] == "✗ `audit` · `30s` · 1 tool"  # singular
        assert lines[3] == "✓ `noop` · `5s`"  # 0 tools -> omit suffix

    def test_collapsed_keeps_breakdown_with_summary_header(self):
        # On finish the roster keeps the per-child breakdown (each marked with
        # its terminal glyph) under a summary header — it does NOT collapse to a
        # bare one-liner. Alfredo wants to see WHICH children did what.
        from gateway.subagent_roster import format_subagent_roster

        rows = [
            {"glyph": "✓", "label": "a", "elapsed": 45.0, "running": False, "tools": 0},
            {"glyph": "✓", "label": "b", "elapsed": 60.0, "running": False, "tools": 0},
            {"glyph": "✗", "label": "c", "elapsed": 134.0, "running": False, "tools": 0},
        ]
        out = format_subagent_roster(rows, collapsed=True)
        lines = out.split("\n")
        # A failure is present -> ⚠️ leads (a green check there would lie).
        # Header elapsed is the WALL-CLOCK fallback = slowest child
        # (max(45,60,134)=134), NOT the sum: children run in parallel.
        assert lines[0] == "⚠️ 3 subagents · 2 ✓ · 1 ✗ · `2m 14s`"
        assert lines[1] == "✓ `a` · `45s`"
        assert lines[2] == "✓ `b` · `1m`"
        assert lines[3] == "✗ `c` · `2m 14s`"

    def test_collapsed_all_success_leads_with_green_check(self):
        # Clear "all done" indicator: when every child finished successfully the
        # header LEADS with ✅ instead of the 🤖 robot, so finished is obvious.
        from gateway.subagent_roster import format_subagent_roster

        rows = [
            {"glyph": "✓", "label": "a", "elapsed": 45.0, "running": False, "tools": 0},
            {"glyph": "✓", "label": "b", "elapsed": 60.0, "running": False, "tools": 0},
        ]
        out = format_subagent_roster(rows, collapsed=True)
        lines = out.split("\n")
        # Header elapsed is the WALL-CLOCK fallback = slowest child
        # (max(45,60)=60), NOT the sum: children run in parallel.
        assert lines[0] == "✅ 2 subagents · 2 ✓ · `1m`"

    def test_collapsed_single_success_leads_with_green_check(self):
        from gateway.subagent_roster import format_subagent_roster

        rows = [{"glyph": "✓", "label": "a", "elapsed": 5.0, "running": False, "tools": 0}]
        out = format_subagent_roster(rows, collapsed=True)
        assert out.split("\n")[0] == "✅ 1 subagent · 1 ✓ · `5s`"

    def test_collapsed_with_running_row_keeps_robot(self):
        # Defensive: if a running row somehow reaches the collapsed render, do
        # NOT claim finished — keep the 🤖 robot.
        from gateway.subagent_roster import format_subagent_roster

        rows = [
            {"glyph": "▶", "label": "a", "elapsed": 5.0, "running": True, "tools": 1},
            {"glyph": "✓", "label": "b", "elapsed": 6.0, "running": False, "tools": 0},
        ]
        out = format_subagent_roster(rows, collapsed=True)
        assert out.split("\n")[0].startswith("🤖 ")

    def test_collapsed_timeout_only_leads_with_warning(self):
        # timed-out / interrupted count as not-all-clean -> ⚠️, never ✅.
        from gateway.subagent_roster import format_subagent_roster

        rows = [{"glyph": "⏱", "label": "a", "elapsed": 900.0, "running": False, "tools": 3}]
        out = format_subagent_roster(rows, collapsed=True)
        assert out.split("\n")[0].startswith("⚠️ ")

    def test_collapsed_keeps_tool_count_on_done_rows(self):
        from gateway.subagent_roster import format_subagent_roster

        rows = [
            {"glyph": "✓", "label": "a", "elapsed": 45.0, "running": False, "tools": 9},
            {"glyph": "✗", "label": "b", "elapsed": 60.0, "running": False, "tools": 0},
        ]
        out = format_subagent_roster(rows, collapsed=True)
        lines = out.split("\n")
        assert lines[1] == "✓ `a` · `45s` · 9 tools"
        assert lines[2] == "✗ `b` · `1m`"

    def test_collapsed_renders_model_and_reasoning(self):
        from gateway.subagent_roster import format_subagent_roster

        rows = [
            {
                "glyph": "✓", "label": "review", "elapsed": 322.0,
                "running": False, "tools": 0,
                "model": "us.anthropic.claude-opus-4-8",
                "reasoning": {"enabled": True, "effort": "high"},
            },
        ]
        out = format_subagent_roster(rows, collapsed=True)
        assert "✓ `review` · opus-4-8 high · `5m 22s`" in out

    def test_row_cap_overflow(self):
        from gateway.subagent_roster import format_subagent_roster

        rows = [
            {"glyph": "▶", "label": f"g{i}", "elapsed": 1.0, "running": True, "tools": 0}
            for i in range(13)
        ]
        text = format_subagent_roster(rows)
        assert text.split("\n")[-1] == "… +3 more"  # 13 rows, cap 10

    def test_header_wall_clock_overrides_sum(self):
        # The header total is the delegate_task WALL-CLOCK when provided, NOT a
        # sum of child elapsed. Two children [6,10] but wall_clock=4.0 -> "4s".
        from gateway.subagent_roster import format_subagent_roster

        rows = [
            {"glyph": "✓", "label": "a", "elapsed": 6.0, "running": False, "tools": 0},
            {"glyph": "✓", "label": "b", "elapsed": 10.0, "running": False, "tools": 0},
        ]
        out = format_subagent_roster(rows, collapsed=True, wall_clock=4.0)
        assert out.split("\n")[0] == "✅ 2 subagents · 2 ✓ · `4s`"

    def test_header_fallback_is_max_not_sum(self):
        # With no wall_clock, the header falls back to the SLOWEST child
        # (max(6,10)=10 -> "10s"), never the sum (16 -> "16s").
        from gateway.subagent_roster import format_subagent_roster

        rows = [
            {"glyph": "✓", "label": "a", "elapsed": 6.0, "running": False, "tools": 0},
            {"glyph": "✓", "label": "b", "elapsed": 10.0, "running": False, "tools": 0},
        ]
        out = format_subagent_roster(rows, collapsed=True)
        assert out.split("\n")[0] == "✅ 2 subagents · 2 ✓ · `10s`"

    def test_header_ignores_bad_wall_clock(self):
        # A malformed/non-finite wall_clock must NOT crash the render and must
        # route to the max(child) fallback. inf in particular must not reach
        # format_elapsed's int(round(inf)) (OverflowError). All -> "10s".
        from gateway.subagent_roster import format_subagent_roster

        rows = [
            {"glyph": "✓", "label": "a", "elapsed": 6.0, "running": False, "tools": 0},
            {"glyph": "✓", "label": "b", "elapsed": 10.0, "running": False, "tools": 0},
        ]
        for bad in ("x", -5, float("inf"), float("-inf"), float("nan"), None, 10**400):
            out = format_subagent_roster(rows, collapsed=True, wall_clock=bad)
            assert out.split("\n")[0] == "✅ 2 subagents · 2 ✓ · `10s`", bad


class TestShortenModel:
    def test_strips_region_provider_prefix(self):
        from gateway.subagent_roster import shorten_model

        assert shorten_model("us.anthropic.claude-opus-4-8") == "opus-4-8"
        assert shorten_model("claude-sonnet-4-6") == "sonnet-4-6"

    def test_preserves_version_dot(self):
        from gateway.subagent_roster import shorten_model

        # A version dot must NOT be treated as a provider-prefix separator.
        assert shorten_model("gpt-5.5") == "gpt-5.5"
        assert shorten_model("gpt-4.1-mini") == "gpt-4.1-mini"

    def test_empty(self):
        from gateway.subagent_roster import shorten_model

        assert shorten_model("") == ""
        assert shorten_model(None) == ""

    def test_reasoning_tag(self):
        from gateway.subagent_roster import reasoning_tag

        assert reasoning_tag({"enabled": True, "effort": "high"}) == "high"
        assert reasoning_tag({"enabled": False}) == ""
        assert reasoning_tag("max") == "max"
        assert reasoning_tag(None) == ""


class TestResolveRosterInterval:
    def test_default_when_unset(self):
        from gateway.subagent_roster import (
            ROSTER_EDIT_INTERVAL,
            resolve_roster_interval,
        )

        assert resolve_roster_interval({}, "telegram") == ROSTER_EDIT_INTERVAL

    def test_per_platform_override_wins(self):
        from gateway.subagent_roster import resolve_roster_interval

        cfg = {"display": {"platforms": {"telegram": {"subagent_roster_interval": 15}}}}
        assert resolve_roster_interval(cfg, "telegram") == 15.0

    def test_global_setting_applies(self):
        from gateway.subagent_roster import resolve_roster_interval

        cfg = {"display": {"subagent_roster_interval": 7}}
        assert resolve_roster_interval(cfg, "discord") == 7.0

    def test_clamped_to_floor(self):
        from gateway.subagent_roster import (
            ROSTER_EDIT_INTERVAL_FLOOR,
            resolve_roster_interval,
        )

        # A too-small / zero value must clamp up to the floor, never flood.
        cfg = {"display": {"platforms": {"telegram": {"subagent_roster_interval": 0}}}}
        assert resolve_roster_interval(cfg, "telegram") == ROSTER_EDIT_INTERVAL_FLOOR

    def test_garbage_falls_back_to_default(self):
        from gateway.subagent_roster import (
            ROSTER_EDIT_INTERVAL,
            resolve_roster_interval,
        )

        cfg = {"display": {"platforms": {"telegram": {"subagent_roster_interval": "nonsense"}}}}
        # _normalise returns 10.0 for unparseable, then clamp keeps it.
        assert resolve_roster_interval(cfg, "telegram") == ROSTER_EDIT_INTERVAL


# ── pipeline gate: roster must keep the queue/consumer alive ────────────────
class TestRosterPipelineGate:
    def test_roster_keeps_pipeline_alive_with_everything_else_off(self):
        from gateway.run import _tool_progress_pipeline_enabled

        assert _tool_progress_pipeline_enabled(
            is_webhook=False,
            progress_mode="off",
            tool_completion_durations_enabled=False,
            subagent_progress_enabled=False,
            delegate_task_args_enabled=False,
            subagent_roster_enabled=True,
        ) is True

    def test_everything_off_is_disabled(self):
        from gateway.run import _tool_progress_pipeline_enabled

        assert _tool_progress_pipeline_enabled(
            is_webhook=False,
            progress_mode="off",
            tool_completion_durations_enabled=False,
            subagent_progress_enabled=False,
            delegate_task_args_enabled=False,
            subagent_roster_enabled=False,
        ) is False

    def test_webhook_always_disabled_even_with_roster(self):
        from gateway.run import _tool_progress_pipeline_enabled

        assert _tool_progress_pipeline_enabled(
            is_webhook=True,
            progress_mode="all",
            tool_completion_durations_enabled=True,
            subagent_progress_enabled=True,
            delegate_task_args_enabled=True,
            subagent_roster_enabled=True,
        ) is False

    def test_default_param_back_compat(self):
        # Existing callers that omit subagent_roster_enabled still work.
        from gateway.run import _tool_progress_pipeline_enabled

        assert _tool_progress_pipeline_enabled(
            is_webhook=False,
            progress_mode="off",
            tool_completion_durations_enabled=False,
            subagent_progress_enabled=False,
            delegate_task_args_enabled=False,
        ) is False


from gateway.subagent_roster import _LABEL_CAP


# ── is_flood_error: the retry-vs-latch dividing line for seed failures ──────
class TestIsFloodError:
    """A flood/rate reject is known-not-delivered (safe to re-seed); an
    ambiguous failure might have landed (must latch). is_flood_error draws
    that line so the roster seed path retries floods but not ambiguous fails.
    """

    class _R:
        def __init__(self, success=False, error=None, message_id=None, retryable=False):
            self.success = success
            self.error = error
            self.message_id = message_id
            self.retryable = retryable

    def test_none_is_not_flood(self):
        from gateway.subagent_roster import is_flood_error
        assert is_flood_error(None) is False

    def test_retryable_flag_is_flood(self):
        # Telegram short floods (<=5s) come back retryable=True with no error str.
        from gateway.subagent_roster import is_flood_error
        assert is_flood_error(self._R(retryable=True)) is True

    def test_flood_control_error_string(self):
        # Long floods (>5s) return error="flood_control:{wait}", retryable=False.
        from gateway.subagent_roster import is_flood_error
        assert is_flood_error(self._R(error="flood_control:18")) is True

    def test_retry_after_phrasing(self):
        from gateway.subagent_roster import is_flood_error
        assert is_flood_error(self._R(error="Too Many Requests: retry after 12")) is True

    def test_rate_limit_phrasing(self):
        from gateway.subagent_roster import is_flood_error
        assert is_flood_error(self._R(error="rate limit exceeded")) is True

    def test_case_insensitive(self):
        from gateway.subagent_roster import is_flood_error
        assert is_flood_error(self._R(error="FLOOD CONTROL EXCEEDED")) is True

    def test_ambiguous_failure_is_not_flood(self):
        # A network drop / unknown error MIGHT have delivered → not flood → latch.
        from gateway.subagent_roster import is_flood_error
        assert is_flood_error(self._R(error="Bad Gateway")) is False
        assert is_flood_error(self._R(error="connection reset")) is False
        assert is_flood_error(self._R(error=None)) is False

    def test_success_result_with_no_error(self):
        # Defensive: a success result is never treated as a flood.
        from gateway.subagent_roster import is_flood_error
        assert is_flood_error(self._R(success=True, message_id="42")) is False
