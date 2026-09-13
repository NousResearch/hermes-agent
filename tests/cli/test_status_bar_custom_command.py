"""Invariant tests for the opt-in custom-command status-bar field."""

from datetime import datetime, timedelta

from hermes_cli import status_bar_custom
from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "anthropic/claude-sonnet-4-20250514"
    cli_obj.session_start = datetime.now() - timedelta(minutes=3)
    cli_obj.conversation_history = [{"role": "user", "content": "hi"}]
    cli_obj.agent = None
    return cli_obj


def _reset_state():
    with status_bar_custom._lock:
        status_bar_custom._state.update(command=None, at=0.0, value="", running=False)


def test_custom_segment_never_runs_on_render_path(monkeypatch):
    """A repaint returns the cached value instantly; the command itself only executes on
    the background refresh thread. A cold cache yields '' without blocking, and once the
    refresh lands the shaped first line is served from cache with no further execution."""
    _reset_state()
    calls = []

    def fake_run(command):
        calls.append(command)
        return status_bar_custom._shape_output("  branch-x  \nsecond line ignored\n")

    monkeypatch.setattr(status_bar_custom, "_run_command", fake_run)

    pending = []

    class DeferredThread:
        def __init__(self, target=None, args=(), daemon=None, **kwargs):
            self._target, self._args = target, args

        def start(self):
            pending.append(self)

        def run_now(self):
            self._target(*self._args)

    monkeypatch.setattr(status_bar_custom.threading, "Thread", DeferredThread)

    # Cold cache: instant "" and exactly one background refresh dispatched, none executed.
    assert status_bar_custom.custom_segment("echo branch-x") == ""
    assert calls == []  # nothing ran synchronously
    assert len(pending) == 1
    pending[0].run_now()
    assert calls == ["echo branch-x"]

    # Warm cache: the shaped value, still exactly one execution.
    assert status_bar_custom.custom_segment("echo branch-x") == "branch-x"
    assert calls == ["echo branch-x"]
    _reset_state()


def test_custom_field_is_doubly_opt_in(monkeypatch):
    """The segment renders only when BOTH 'custom' is in the configured field list AND a
    custom_command is set; the default field set (None) never consults the runner."""
    _reset_state()

    def boom(command):
        raise AssertionError("custom_segment must not be called")

    monkeypatch.setattr(status_bar_custom, "custom_segment", boom)

    # Default field set: runner untouched, no segment.
    cli_obj = _make_cli()
    cli_obj._status_bar_field_set_cache = None
    cli_obj._status_bar_custom_command_cache = "echo hi"
    snapshot = cli_obj._get_status_bar_snapshot()
    assert snapshot["custom_segment"] == ""

    # Field requested but no command configured: still nothing.
    cli_obj2 = _make_cli()
    cli_obj2._status_bar_field_set_cache = frozenset({"model", "custom"})
    cli_obj2._status_bar_custom_command_cache = ""
    snapshot2 = cli_obj2._get_status_bar_snapshot()
    assert snapshot2["custom_segment"] == ""

    # Both present: the cached segment value renders in the bar.
    monkeypatch.setattr(status_bar_custom, "custom_segment", lambda cmd: "⎈ prod-cluster")
    cli_obj3 = _make_cli()
    fields = frozenset({"model", "custom"})
    cli_obj3._status_bar_field_set_cache = fields
    cli_obj3._status_bar_custom_command_cache = "kubectl config current-context"
    snapshot3 = cli_obj3._get_status_bar_snapshot()
    assert snapshot3["custom_segment"] == "⎈ prod-cluster"
    text = "".join(
        t for seg in cli_obj3._status_bar_segments(
            snapshot3, 120, fields, False, styled=False) for _, t in seg)
    assert "⎈ prod-cluster" in text
    _reset_state()
