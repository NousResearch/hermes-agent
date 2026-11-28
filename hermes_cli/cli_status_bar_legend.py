"""Status-bar helpers for the classic CLI: the background counters the live-work footer already
shows, and the ``/statusbar legend`` table that explains each glyph.

The bar itself stays compact (``◎ 48%``, ``◷ 4.8s``, ``↑ 55 t/s``, ``⏲ 12s``); the legend is
where a user looks up what an icon means. Field names are the ``display.status_bar.fields``
keys that hide or show each segment.
"""
from __future__ import annotations

import time

from agent.i18n import t

# Bar order; the meaning of each is ``cli.status_bar.legend_<field>``.
_LEGEND_FIELDS = (
    "model", "context_detail", "context_pct", "cache_hit", "latency", "tps", "compressions",
    "bg_tasks", "bg_processes", "bg_subagents", "goal", "git_branch", "duration", "prompt_elapsed",
    "idle_since", "focus", "yolo", "total_tokens", "stash", "battery", "title")


def footer_covered_fields(cli) -> frozenset:
    """Status-bar counters the live-work footer already shows: its pill counts running shells and
    agents (collapsed or not), so ``⚙`` / ``⛓`` would only repeat it. ``/bg`` tasks (``▶``) are not
    footer rows and stay. A goal- or queue-only footer covers nothing."""
    monitor = getattr(cli, "_subagent_monitor", None)
    if monitor is None or not monitor.has_rows:
        return frozenset()
    from hermes_cli.cli_subagent_monitor import modal_prompt_active
    if modal_prompt_active(cli):  # the footer hides behind a prompt; the bar keeps the counts
        return frozenset()
    covered = set()
    if any(row.get("status") == "running" for row in monitor.processes):
        covered.add("bg_processes")
    if monitor.entries:
        covered.add("bg_subagents")
    return frozenset(covered)


def _samples() -> dict[str, str]:
    """One example of each segment in the form the bar paints it (timers through the bar's own
    formatters, so the legend shows the same glyphs)."""
    from hermes_cli.cli_status_bar_mixin import CLIStatusBarMixin as bar
    from hermes_cli.focus_view import focus_statusbar_segment
    timers = f"{bar._format_prompt_elapsed(None, 12.0, live=True)} · {bar._format_prompt_elapsed(None, 63.0)}"
    return {
        "model": "☤ claude-sonnet-4", "context_detail": "~15.7K/1M", "context_pct": "[██░░░░░░░░] 21%",
        "cache_hit": "◎ 48.4%", "latency": "◷ " + t("cli.status_bar.latency_label", value="4.8"),
        "tps": "↑ " + t("cli.status_bar.velocity_label", value="55"), "compressions": "🗜️ 2",
        "bg_tasks": "▶ 1", "bg_processes": "⚙ 1", "bg_subagents": "⛓ 2",
        "goal": t("cli.status_bar.goal_turns", used=3, max=20), "git_branch": "⎇ main", "duration": "2m",
        "prompt_elapsed": timers, "idle_since": bar._format_idle_since(time.time() - 240, turn_live=False),
        "focus": focus_statusbar_segment(True), "yolo": t("cli.status_bar.yolo_badge"),
        "total_tokens": "Σ1.2M", "stash": "📌 2", "battery": "🔋 82%", "title": "─ weekly-digest"}


def legend_rows() -> list[tuple[str, str, str]]:
    """``(sample, meaning, field)`` for every status-bar segment, in bar order."""
    samples = _samples()
    return [(samples[field], t(f"cli.status_bar.legend_{field}"), field) for field in _LEGEND_FIELDS]


def print_legend(cli) -> None:
    """``/statusbar legend``: each glyph, what it means and the field name that hides it."""
    from rich.table import Table
    table = Table(box=None, show_edge=False, pad_edge=False, padding=(0, 2, 0, 0), title_justify="left",
                  title=t("cli.status_bar.legend_heading"))
    for header in ("legend_col_shows", "legend_col_meaning", "legend_col_field"):
        table.add_column(t(f"cli.status_bar.{header}"), style="dim" if header == "legend_col_field" else None)
    for sample, meaning, field in legend_rows():
        table.add_row(sample, meaning, field)
    cli._console_print(table)
    cli._console_print(f"  {t('cli.status_bar.legend_note')}", highlight=False, markup=False)
