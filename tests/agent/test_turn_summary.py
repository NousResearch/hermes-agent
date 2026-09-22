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






# ── collector semantics ────────────────────────────────────────────────────








def test_begin_resets_previous_turn():
    collector = TurnSummaryCollector()
    collector.begin()
    collector.record_tool("read_file", result="x")
    collector.begin()
    assert collector.tally.total_tools == 0
    assert collector.render(0.5) == ""






# ── spinner token flow (PART B) ────────────────────────────────────────────




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










def test_spinner_token_flow_appears_when_enabled():
    stub = _make_cli(agent=_StubAgent(session_output_tokens=1200))
    assert stub._spinner_token_flow() == "↓ 1.2k tok"
    assert "↓ 1.2k tok" in stub._render_spinner_text()










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
