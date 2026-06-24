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


# ── Background terminal-process indicator (⚙ N) ───────────────────────────
# Source of truth is tools.process_registry.process_registry._running (a dict
# of currently-running shell processes spawned by terminal(background=true)).
# Distinct from /background tasks above: ▶ counts agent threads, ⚙ counts
# shell processes. Both can be active simultaneously.


class _FakeRunningRegistry:
    """Minimal stand-in for process_registry; exposes count_running()."""

    def __init__(self, count: int) -> None:
        self._count = count

    def count_running(self) -> int:
        return self._count


def _patch_process_registry(monkeypatch, count: int) -> None:
    import tools.process_registry as pr_mod
    monkeypatch.setattr(pr_mod, "process_registry", _FakeRunningRegistry(count))


def test_snapshot_reports_zero_when_no_background_processes(monkeypatch):
    cli_obj = _make_cli()
    _patch_process_registry(monkeypatch, 0)
    snap = cli_obj._get_status_bar_snapshot()
    assert snap["active_background_processes"] == 0


def test_snapshot_counts_live_background_processes(monkeypatch):
    cli_obj = _make_cli()
    _patch_process_registry(monkeypatch, 3)
    snap = cli_obj._get_status_bar_snapshot()
    assert snap["active_background_processes"] == 3


def test_snapshot_safe_when_process_registry_raises(monkeypatch):
    """If count_running() raises the snapshot stays at 0; no propagate."""
    cli_obj = _make_cli()
    import tools.process_registry as pr_mod

    class _BoomRegistry:
        def count_running(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(pr_mod, "process_registry", _BoomRegistry())
    snap = cli_obj._get_status_bar_snapshot()
    assert snap["active_background_processes"] == 0


def test_plain_text_status_shows_proc_indicator_when_active(monkeypatch):
    cli_obj = _make_cli()
    _patch_process_registry(monkeypatch, 2)
    text = cli_obj._build_status_bar_text(width=80)
    assert "⚙ 2" in text


def test_plain_text_status_omits_proc_indicator_when_idle(monkeypatch):
    cli_obj = _make_cli()
    _patch_process_registry(monkeypatch, 0)
    text = cli_obj._build_status_bar_text(width=80)
    assert "⚙" not in text


def test_fragments_include_proc_segment_when_active(monkeypatch):
    cli_obj = _make_cli()
    _patch_process_registry(monkeypatch, 1)
    cli_obj._status_bar_visible = True
    cli_obj._get_tui_terminal_width = lambda: 120  # type: ignore[method-assign]
    frags = cli_obj._get_status_bar_fragments()
    rendered = "".join(text for _style, text in frags)
    assert "⚙ 1" in rendered


def test_indicators_independent_agents_and_processes(monkeypatch):
    """▶ (agent tasks) and ⚙ (shell processes) render side-by-side."""
    cli_obj = _make_cli()
    cli_obj._background_tasks = {"bg_a": _stub_thread()}
    _patch_process_registry(monkeypatch, 2)
    cli_obj._status_bar_visible = True
    cli_obj._get_tui_terminal_width = lambda: 120  # type: ignore[method-assign]
    frags = cli_obj._get_status_bar_fragments()
    rendered = "".join(text for _style, text in frags)
    assert "▶ 1" in rendered
    assert "⚙ 2" in rendered


# ── Background/async subagent indicator (⛓ N) ─────────────────────────────
# Source of truth is tools.async_delegation.active_count() — the count of
# delegate_task delegations (batch + background single) still in the
# "running" state. Distinct from ▶ (/background agent threads) and ⚙ (shell
# processes); all three can be active at once.


def _patch_async_active(monkeypatch, count: int) -> None:
    import tools.async_delegation as ad_mod
    monkeypatch.setattr(ad_mod, "active_count", lambda: count)


def test_snapshot_reports_zero_when_no_background_subagents(monkeypatch):
    cli_obj = _make_cli()
    _patch_async_active(monkeypatch, 0)
    snap = cli_obj._get_status_bar_snapshot()
    assert snap["active_background_subagents"] == 0


def test_snapshot_counts_live_background_subagents(monkeypatch):
    cli_obj = _make_cli()
    _patch_async_active(monkeypatch, 4)
    snap = cli_obj._get_status_bar_snapshot()
    assert snap["active_background_subagents"] == 4


def test_snapshot_safe_when_async_active_count_raises(monkeypatch):
    """If active_count() raises the snapshot stays at 0; no propagate."""
    cli_obj = _make_cli()
    import tools.async_delegation as ad_mod

    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(ad_mod, "active_count", _boom)
    snap = cli_obj._get_status_bar_snapshot()
    assert snap["active_background_subagents"] == 0


def test_plain_text_status_shows_subagent_indicator_when_active(monkeypatch):
    cli_obj = _make_cli()
    _patch_async_active(monkeypatch, 3)
    text = cli_obj._build_status_bar_text(width=80)
    assert "⛓ 3" in text


def test_plain_text_status_omits_subagent_indicator_when_idle(monkeypatch):
    cli_obj = _make_cli()
    _patch_async_active(monkeypatch, 0)
    text = cli_obj._build_status_bar_text(width=80)
    assert "⛓" not in text


def test_fragments_include_subagent_segment_when_active(monkeypatch):
    cli_obj = _make_cli()
    _patch_async_active(monkeypatch, 2)
    cli_obj._status_bar_visible = True
    cli_obj._get_tui_terminal_width = lambda: 120  # type: ignore[method-assign]
    frags = cli_obj._get_status_bar_fragments()
    rendered = "".join(text for _style, text in frags)
    assert "⛓ 2" in rendered


def test_all_three_background_indicators_independent(monkeypatch):
    """▶ (agent tasks), ⚙ (shell processes), ⛓ (subagents) all coexist."""
    cli_obj = _make_cli()
    cli_obj._background_tasks = {"bg_a": _stub_thread()}
    _patch_process_registry(monkeypatch, 2)
    _patch_async_active(monkeypatch, 5)
    cli_obj._status_bar_visible = True
    cli_obj._get_tui_terminal_width = lambda: 120  # type: ignore[method-assign]
    frags = cli_obj._get_status_bar_fragments()
    rendered = "".join(text for _style, text in frags)
    assert "▶ 1" in rendered
    assert "⚙ 2" in rendered
    assert "⛓ 5" in rendered
