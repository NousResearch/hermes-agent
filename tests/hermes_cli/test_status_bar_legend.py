"""Status bar: every tier keeps its glyph forms, counters the live-work footer already lists are not
repeated, and ``/statusbar legend`` explains every glyph."""
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import cli as cli_mod
from agent.i18n import t
from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "anthropic/claude-sonnet-4"
    cli_obj.session_start = datetime.now() - timedelta(minutes=2)
    cli_obj.agent = None
    cli_obj._background_tasks = {}
    return cli_obj


def _with_counts(monkeypatch, *, shells=0, agents=0):
    import tools.async_delegation as ad_mod
    import tools.process_registry as pr_mod
    monkeypatch.setattr(ad_mod, "active_count", lambda: agents)
    monkeypatch.setattr(pr_mod, "process_registry", SimpleNamespace(count_running=lambda: shells))


def _attach_footer(cli_obj, *, entries=(), processes=(), goal=""):
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    monitor = cli_obj._subagent_monitor = SubagentMonitor(cli_obj)
    monitor.entries, monitor.processes, monitor.goal = list(entries), list(processes), goal
    return monitor


_AGENT = dict(subagent_id="sa-1", key="sa-1", goal="Worker", elapsed=1, status="running")
_SHELL = dict(kind="process", key="p1", id="p1", command="npm run dev", status="running", elapsed=2,
              detail="", watching=False)


def test_every_tier_keeps_the_compact_glyph_forms():
    cli_obj = _make_cli()
    cli_obj._prompt_start_time = time.time() - 12
    wide = cli_obj._build_status_bar_text(width=140)
    assert "⏱ " + t("cli.shared.duration_s", seconds=12) in wide
    assert " │ 2m │ " in wide  # session duration stays a bare value

    cli_obj._prompt_start_time = None
    cli_obj._prompt_duration = 10.0
    cli_obj._last_turn_finished_at = time.time() - 240
    wide = cli_obj._build_status_bar_text(width=140)
    assert "⏲ " + t("cli.shared.duration_s", seconds=10) in wide and "✓ 4m" in wide


def test_footer_listing_shells_and_agents_hides_their_status_bar_counters(monkeypatch):
    _with_counts(monkeypatch, shells=1, agents=1)
    cli_obj = _make_cli()
    shells, agents = "⚙ 1", "⛓ 1"
    text = cli_obj._build_status_bar_text(width=140)
    assert shells in text and agents in text  # no footer: the bar is the only place they show

    monitor = _attach_footer(cli_obj, entries=[_AGENT], processes=[_SHELL])
    text = cli_obj._build_status_bar_text(width=140)
    assert "⚙" not in text and "⛓" not in text
    compact = cli_obj._build_status_bar_text(width=60)
    assert "⚙" not in compact and "⛓" not in compact

    monitor.collapsed = True  # the collapsed pill still counts them
    assert shells not in cli_obj._build_status_bar_text(width=140)

    cli_obj._approval_state = {"pending": True}  # a prompt hides the footer; the bar keeps them
    text = cli_obj._build_status_bar_text(width=140)
    assert shells in text and agents in text


def test_footer_hides_only_the_kinds_it_lists_and_never_bg_tasks(monkeypatch):
    import threading
    _with_counts(monkeypatch, shells=2, agents=1)
    cli_obj = _make_cli()
    cli_obj._background_tasks = {"bg": threading.Thread(target=lambda: None)}
    _attach_footer(cli_obj, processes=[_SHELL])
    text = cli_obj._build_status_bar_text(width=140)
    assert "⚙" not in text
    assert "⛓ 1" in text  # the footer lists no agent rows
    assert "▶ 1" in text

    _attach_footer(cli_obj, goal="⊙ ship it")  # a goal-only footer lists no shells or agents
    assert "⚙ 2" in cli_obj._build_status_bar_text(width=140)


class _RecordingFields(frozenset):
    """A ``display.status_bar.fields`` stand-in that allows everything and records each name the
    renderer asks about."""
    seen: set

    def __contains__(self, name):
        self.seen.add(name)
        return True


def test_legend_has_a_row_for_every_field_the_status_bar_consults(monkeypatch):
    from hermes_cli.cli_status_bar_legend import legend_rows
    cli_obj = _make_cli()
    cli_obj._status_bar_visible = True
    snapshot = {
        "model_short": "m", "model_name": "m", "duration": "2m", "session_title": "title",
        "prompt_elapsed": "turn 1s", "idle_since": "idle 1s", "context_tokens": 1, "context_length": 10,
        "context_percent": 10, "cache_hit_pct": 50.0, "avg_latency_label": "1s", "avg_velocity_label": "2",
        "compressions": 1, "active_background_tasks": 1, "active_background_processes": 1,
        "active_background_subagents": 1, "goal_active": True, "goal_turns_used": 1, "goal_max_turns": 2,
        "git_branch": "main", "focus_label": "focus", "session_total_tokens": 5, "battery_label": "82%",
        "battery_category": "good"}
    fields = _RecordingFields()
    fields.seen = set()
    monkeypatch.setattr(cli_obj, "_get_status_bar_snapshot", lambda: snapshot)
    monkeypatch.setattr(cli_obj, "_get_status_bar_field_set", lambda: fields)
    monkeypatch.setattr(cli_obj, "_is_session_yolo_active", lambda: True)
    monkeypatch.setattr(cli_obj, "_get_tui_terminal_width", lambda: 400)
    cli_obj._prompt_stash = SimpleNamespace(indicator=lambda: "📌 1")
    cli_obj._get_status_bar_fragments()
    rows = legend_rows()
    legend_fields = [field for _, _, field in rows]
    assert fields.seen and fields.seen <= set(legend_fields), fields.seen - set(legend_fields)
    assert len(legend_fields) == len(set(legend_fields))
    for sample, meaning, field in rows:
        assert sample and meaning and meaning != f"cli.status_bar.legend_{field}", field


def test_statusbar_legend_prints_the_table_and_keeps_the_bar_visible(capsys):
    cli_obj = _make_cli()
    cli_obj._status_bar_visible = True
    printed = []
    cli_obj._console_print = lambda *args, **kwargs: printed.append(args[0])
    cli_obj.process_command("/statusbar legend")
    assert cli_obj._status_bar_visible is True
    from rich.console import Console
    console = Console(width=200, record=True)
    for item in printed:
        console.print(item)
    out = console.export_text()
    assert t("cli.status_bar.legend_cache_hit") in out and "cache_hit" in out and "◎" in out
    assert t("cli.status_bar.legend_idle_since") in out and "idle_since" in out
    cli_obj.process_command("/statusbar")
    assert cli_obj._status_bar_visible is False


def test_legend_samples_use_the_glyphs_the_bar_paints():
    """Each sample the legend shows for a timer or counter is the form the bar renders."""
    from hermes_cli.cli_status_bar_legend import legend_rows
    samples = {field: sample for sample, _, field in legend_rows()}
    live = HermesCLI._format_prompt_elapsed(time.time() - 5, 0.0, live=True)
    frozen = HermesCLI._format_prompt_elapsed(None, 5.0)
    assert live.split()[0] in samples["prompt_elapsed"] and frozen.split()[0] in samples["prompt_elapsed"]
    assert HermesCLI._format_idle_since(time.time() - 5, turn_live=False).split()[0] in samples["idle_since"]
