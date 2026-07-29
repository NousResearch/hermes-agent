"""Tests for per-turn accounting: summary formatter, collector, spinner flow, gating.

The formatter and collector are pure (no terminal, no agent, no network), so
they're exercised directly. The gating test drives the real CLI methods on a
stub object with only the attributes the gate reads, so quiet-mode /
config-false behaviour is verified against the shipped code path rather than
against a re-implementation.
"""

import pytest

from agent.turn_summary import (
    TurnSummaryCollector,
    TurnTally,
    format_elapsed,
    format_token_flow,
    format_turn_summary,
)


# ── format_elapsed ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.0, "0.0s"),
        (12.44, "12.4s"),
        (59.9, "59.9s"),
        (60.0, "1m00s"),
        (125.0, "2m05s"),
        (-3.0, "0.0s"),
    ],
)
def test_format_elapsed(seconds, expected):
    assert format_elapsed(seconds) == expected


# ── format_turn_summary: pure formatter ─────────────────────────────────────


def test_zero_tools_fast_turn_renders_nothing():
    """A quick chat reply with no tool calls has nothing to summarise."""
    assert format_turn_summary(0.8, TurnTally()) == ""


def test_zero_tools_slow_turn_still_reports_wall_time():
    """A long toolless turn (big model, no tools) is worth timing."""
    assert format_turn_summary(31.2, TurnTally()) == "⋯ 31.2s"


def test_single_edit_with_line_deltas():
    collector = TurnSummaryCollector()
    collector.begin()
    collector.record_tool(
        "patch",
        result='{"success": true, "diff": "--- a/x.py\\n+++ b/x.py\\n@@\\n+new\\n-old\\n"}',
    )
    assert collector.render(6.0) == "⋯ 6.0s · edited 1 file +1 -1"


def test_mixed_verbs_render_in_priority_order():
    collector = TurnSummaryCollector()
    collector.begin()
    for _ in range(4):
        collector.record_tool("read_file", result="contents")
    for _ in range(3):
        collector.record_tool("terminal", result="ok")
    collector.record_tool("write_file", result='{"bytes_written": 10}')
    collector.record_tool("write_file", result='{"bytes_written": 12}')

    line = collector.render(12.4)
    # Edits first, then reads, then commands — regardless of call order.
    assert line == "⋯ 12.4s · edited 2 files · read 4 files · ran 3 commands"


def test_pluralization_singular_and_plural():
    one = TurnTally(verbs={"read": {"files": 1}})
    many = TurnTally(verbs={"read": {"files": 3}})
    assert format_turn_summary(1.0, one) == "⋯ 1.0s · read 1 file"
    assert format_turn_summary(1.0, many) == "⋯ 1.0s · read 3 files"


def test_pluralization_irregular_nouns():
    """The singulariser handles -ies / -ses without producing 'memorie'."""
    mem = TurnTally(verbs={"updated": {"memories": 1}})
    assert format_turn_summary(1.0, mem) == "⋯ 1.0s · updated 1 memory"
    times = TurnTally(verbs={"searched the web": {"times": 1}})
    assert format_turn_summary(1.0, times) == "⋯ 1.0s · searched the web 1 time"


def test_missing_line_deltas_omits_plus_minus():
    """write_file reports no diff, so we count the edit and skip +/-."""
    collector = TurnSummaryCollector()
    collector.begin()
    collector.record_tool("write_file", result='{"bytes_written": 42}')
    line = collector.render(3.0)
    assert line == "⋯ 3.0s · edited 1 file"
    assert "+" not in line and " -" not in line


def test_patch_result_without_diff_field_omits_deltas():
    collector = TurnSummaryCollector()
    collector.begin()
    collector.record_tool("patch", result='{"success": true}')
    assert collector.render(2.0) == "⋯ 2.0s · edited 1 file"


def test_diff_headers_not_counted_as_line_changes():
    collector = TurnSummaryCollector()
    collector.begin()
    diff = "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,3 @@\n ctx\n+a\n+b\n-c\n"
    collector.record_tool("patch", result={"success": True, "diff": diff})
    assert collector.render(1.0) == "⋯ 1.0s · edited 1 file +2 -1"


def test_json_string_result_with_raw_newlines_still_parses():
    """Some serialisers emit literal newlines inside the diff string."""
    collector = TurnSummaryCollector()
    collector.begin()
    collector.record_tool("patch", result='{"success": true, "diff": "@@\n+a\n+b\n-c\n"}')
    assert collector.render(1.0) == "⋯ 1.0s · edited 1 file +2 -1"


def test_long_tallies_truncate_to_more_tail():
    tally = TurnTally(
        verbs={
            "edited": {"files": 1},
            "read": {"files": 2},
            "ran": {"commands": 3},
            "searched": {"paths": 4},
            "browsed": {"pages": 5},
            "delegated": {"tasks": 6},
        }
    )
    line = format_turn_summary(9.0, tally)
    assert line == "⋯ 9.0s · edited 1 file · read 2 files · ran 3 commands · searched 4 paths · +2 more"
    assert line.count("·") == 5


def test_max_segments_configurable():
    tally = TurnTally(verbs={"edited": {"files": 1}, "read": {"files": 2}, "ran": {"commands": 1}})
    assert format_turn_summary(5.0, tally, max_segments=1) == "⋯ 5.0s · edited 1 file · +2 more"


def test_none_tally_is_safe():
    assert format_turn_summary(0.1, None) == ""


# ── collector semantics ────────────────────────────────────────────────────


def test_failed_tools_are_not_counted():
    """A denied write must not be summarised as a successful edit."""
    collector = TurnSummaryCollector()
    collector.begin()
    collector.record_tool("write_file", result='{"error": "denied"}', is_error=True)
    assert collector.tally.total_tools == 0
    assert collector.render(4.0) == "⋯ 4.0s"


def test_internal_and_empty_tool_names_ignored():
    collector = TurnSummaryCollector()
    collector.begin()
    collector.record_tool("_thinking")
    collector.record_tool(None)
    collector.record_tool("")
    assert collector.tally.total_tools == 0


def test_unknown_tools_bucket_into_generic_count():
    collector = TurnSummaryCollector()
    collector.begin()
    collector.record_tool("some_mcp__weird_tool", result="{}")
    collector.record_tool("another_plugin_tool", result="{}")
    assert collector.render(2.0) == "⋯ 2.0s · called 2 tools"


def test_begin_resets_previous_turn():
    collector = TurnSummaryCollector()
    collector.begin()
    collector.record_tool("read_file", result="x")
    collector.begin()
    assert collector.tally.total_tools == 0
    assert collector.render(0.5) == ""


def test_line_deltas_aggregate_across_edits():
    collector = TurnSummaryCollector()
    collector.begin()
    collector.record_tool("patch", result={"success": True, "diff": "@@\n+a\n+b\n-c\n"})
    collector.record_tool("patch", result={"success": True, "diff": "@@\n+d\n-e\n-f\n"})
    assert collector.render(7.5) == "⋯ 7.5s · edited 2 files +3 -3"


def test_malformed_result_payloads_do_not_raise():
    collector = TurnSummaryCollector()
    collector.begin()
    for bad in ("not json at all", "", None, 42, [], {"diff": None}):
        collector.record_tool("patch", result=bad)
    assert collector.tally.verbs["edited"]["files"] == 6
    assert collector.tally.has_line_deltas is False


# ── spinner token flow (PART B) ────────────────────────────────────────────


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (0, ""),
        (-5, ""),
        (32, "↓ 32 tok"),
        (999, "↓ 999 tok"),
        (1200, "↓ 1.2k tok"),
        (25_400, "↓ 25.4k tok"),
        (2_500_000, "↓ 2.5M tok"),
    ],
)
def test_format_token_flow(tokens, expected):
    assert format_token_flow(tokens) == expected


def test_format_token_flow_bad_input_is_empty():
    assert format_token_flow(None) == ""
    assert format_token_flow("lots") == ""


# ── gating: quiet mode / config false / non-interactive ────────────────────


class _StubAgent:
    def __init__(self, quiet_mode=False, session_output_tokens=0):
        self.quiet_mode = quiet_mode
        self.session_output_tokens = session_output_tokens


def _make_cli(**overrides):
    """Bind the real CLI accounting methods onto a minimal stub object.

    Avoids constructing HermesCLI (which loads config, sessions, and a
    prompt_toolkit app) while still exercising the shipped gate + emit code.
    """
    import cli as cli_module

    class _Stub:
        _turn_summary_enabled = True
        _spinner_token_flow_enabled = True
        tool_progress_mode = "all"
        _interactive_turn = True
        _agent_running = True
        agent = None
        _turn_summary_collector = None
        _turn_summary_start = 0.0
        _turn_token_baseline = 0
        _spinner_text = "⚡ reading file"
        _tool_start_time = 0

        _turn_summary_is_active = cli_module.HermesCLI._turn_summary_is_active
        _turn_summary_begin = cli_module.HermesCLI._turn_summary_begin
        _turn_summary_record = cli_module.HermesCLI._turn_summary_record
        _turn_summary_emit = cli_module.HermesCLI._turn_summary_emit
        _spinner_token_flow = cli_module.HermesCLI._spinner_token_flow
        _render_spinner_text = cli_module.HermesCLI._render_spinner_text

    stub = _Stub()
    for key, value in overrides.items():
        setattr(stub, key, value)
    return stub


def _emit_and_capture(stub, monkeypatch):
    printed = []
    import cli as cli_module

    monkeypatch.setattr(cli_module, "_cprint", lambda text: printed.append(text))
    stub._turn_summary_begin()
    stub._turn_summary_record("read_file", "contents", False)
    stub._turn_summary_emit()
    return printed


def test_gating_enabled_prints_summary(monkeypatch):
    stub = _make_cli()
    printed = _emit_and_capture(stub, monkeypatch)
    assert len(printed) == 1
    assert "read 1 file" in printed[0]


def test_gating_quiet_mode_prints_nothing(monkeypatch):
    stub = _make_cli(agent=_StubAgent(quiet_mode=True))
    assert _emit_and_capture(stub, monkeypatch) == []


def test_gating_config_false_prints_nothing(monkeypatch):
    stub = _make_cli(_turn_summary_enabled=False)
    assert _emit_and_capture(stub, monkeypatch) == []


def test_gating_tool_progress_off_prints_nothing(monkeypatch):
    stub = _make_cli(tool_progress_mode="off")
    assert _emit_and_capture(stub, monkeypatch) == []


def test_gating_non_interactive_prints_nothing(monkeypatch):
    """Single-query / -Q / gateway paths never set _interactive_turn."""
    stub = _make_cli(_interactive_turn=False)
    assert _emit_and_capture(stub, monkeypatch) == []


def test_spinner_token_flow_appears_when_enabled():
    stub = _make_cli(agent=_StubAgent(session_output_tokens=1200))
    assert stub._spinner_token_flow() == "↓ 1.2k tok"
    assert "↓ 1.2k tok" in stub._render_spinner_text()


def test_spinner_token_flow_uses_per_turn_baseline():
    stub = _make_cli(
        agent=_StubAgent(session_output_tokens=5200), _turn_token_baseline=5000
    )
    assert stub._spinner_token_flow() == "↓ 200 tok"


def test_spinner_token_flow_config_false_is_silent():
    stub = _make_cli(
        _spinner_token_flow_enabled=False, agent=_StubAgent(session_output_tokens=9000)
    )
    assert stub._spinner_token_flow() == ""
    assert "tok" not in stub._render_spinner_text()


def test_spinner_token_flow_silent_without_agent_or_idle():
    assert _make_cli(agent=None)._spinner_token_flow() == ""
    assert (
        _make_cli(agent=_StubAgent(session_output_tokens=500), _agent_running=False)
        ._spinner_token_flow()
        == ""
    )


def test_turn_summary_config_defaults_present():
    from hermes_cli.config import DEFAULT_CONFIG

    display = DEFAULT_CONFIG["display"]
    assert display["turn_summary"] is True
    assert display["spinner_token_flow"] is True


def test_content_free_diff_reports_unknown_not_zero_zero():
    """A diff with no +/- content lines (bare hunk header) must render as an
    edit with UNKNOWN deltas, never a misleading '+0 -0'.

    Found by E2E-rendering the collector against realistic tool payloads: the
    unit suite only fed diffs that had real content lines.
    """
    from agent.turn_summary import TurnSummaryCollector

    c = TurnSummaryCollector()
    c.begin()
    c.record_tool("patch", result={"success": True, "diff": "@@ -1,3 +1,15 @@"})
    line = c.render(3.0)
    assert "edited 1 file" in line
    assert "+0 -0" not in line

    real = TurnSummaryCollector()
    real.begin()
    real.record_tool(
        "patch",
        result={"success": True, "diff": "--- a/x\n+++ b/x\n@@ -1 +1,2 @@\n-old\n+new\n+extra\n"},
    )
    assert "+2 -1" in real.render(1.0)
