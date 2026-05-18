"""Tests for the /background indicator in the CLI status bar.

The classic prompt_toolkit status bar shows `▶ N` when N tasks launched via
`/background` are still running. Source of truth is `self._background_tasks`
(a Dict[str, threading.Thread]); entries are removed in the task thread's
finally block, so len() reflects truly-running tasks.
"""

import threading
from datetime import datetime

from cli import HermesCLI


def _stub_thread() -> threading.Thread:
    """Return a Thread instance that's never started — pure dict-value stand-in."""
    return threading.Thread(target=lambda: None)


def _make_cli():
    """Bare-metal HermesCLI for snapshot/build tests (no __init__ side effects)."""
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "anthropic/claude-opus-4.6"
    cli_obj.agent = None
    cli_obj._background_tasks = {}
    # The snapshot reads session_start to compute duration; supply a stub.
    cli_obj.session_start = datetime.now()
    return cli_obj


def test_snapshot_reports_zero_when_no_background_tasks():
    cli_obj = _make_cli()
    snap = cli_obj._get_status_bar_snapshot()
    assert snap["active_background_tasks"] == 0


def test_snapshot_counts_live_background_tasks():
    cli_obj = _make_cli()
    cli_obj._background_tasks = {"bg_a": _stub_thread(), "bg_b": _stub_thread()}
    snap = cli_obj._get_status_bar_snapshot()
    assert snap["active_background_tasks"] == 2


def test_snapshot_safe_when_background_tasks_attr_missing():
    """Older HermesCLI instances (tests with __new__, etc.) may lack the attr."""
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "x"
    cli_obj.agent = None
    cli_obj.session_start = datetime.now()
    # No _background_tasks at all — must not raise.
    snap = cli_obj._get_status_bar_snapshot()
    assert snap["active_background_tasks"] == 0


def test_plain_text_status_omits_indicator_when_idle():
    cli_obj = _make_cli()
    text = cli_obj._build_status_bar_text(width=80)
    assert "▶" not in text


def test_plain_text_status_shows_indicator_when_active():
    cli_obj = _make_cli()
    cli_obj._background_tasks = {"bg_a": _stub_thread()}
    text = cli_obj._build_status_bar_text(width=80)
    assert "▶ 1" in text


def test_plain_text_status_shows_higher_count():
    cli_obj = _make_cli()
    cli_obj._background_tasks = {
        "a": _stub_thread(),
        "b": _stub_thread(),
        "c": _stub_thread(),
    }
    text = cli_obj._build_status_bar_text(width=80)
    assert "▶ 3" in text


def test_narrow_width_omits_bg_indicator():
    """The narrow tier (<52) is already cramped — bg is secondary, drop it."""
    cli_obj = _make_cli()
    cli_obj._background_tasks = {"bg_a": _stub_thread()}
    text = cli_obj._build_status_bar_text(width=40)
    assert "▶" not in text


def test_fragments_include_bg_segment_when_active():
    cli_obj = _make_cli()
    cli_obj._background_tasks = {"a": _stub_thread(), "b": _stub_thread()}
    cli_obj._status_bar_visible = True
    # _get_status_bar_fragments asks _get_tui_terminal_width(); stub it wide.
    cli_obj._get_tui_terminal_width = lambda: 120  # type: ignore[method-assign]
    frags = cli_obj._get_status_bar_fragments()
    rendered = "".join(text for _style, text in frags)
    assert "▶ 2" in rendered


def test_fragments_omit_bg_segment_when_idle():
    cli_obj = _make_cli()
    cli_obj._status_bar_visible = True
    cli_obj._get_tui_terminal_width = lambda: 120  # type: ignore[method-assign]
    frags = cli_obj._get_status_bar_fragments()
    rendered = "".join(text for _style, text in frags)
    assert "▶" not in rendered
