"""Tests for the machine-wide ``hermes monitor`` command."""

from argparse import ArgumentParser
import json

from hermes_cli.subcommands.monitor import (
    _ViewState,
    _apply_key,
    _render_screen,
    _json_payload,
    _selectable_rows,
    build_monitor_parser,
    cmd_monitor,
)
from hermes_cli.monitor_processes import MonitorSnapshot, ProcessKey, ProcessRow


def _args(*argv):
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_monitor_parser(subparsers)
    return parser.parse_args(["monitor", *argv])


def _snapshot():
    return MonitorSnapshot(
        observed_at=200.0,
        processes=[ProcessRow(
            pid=123, create_time=100.0, kind="kanban", profile="coder",
            cpu_percent=12.5, rss=64 * 1024 * 1024, elapsed=100.0,
            status="sleeping", tty="pts/3", tmux="%7",
            activity="task t_abcd [literal]", confidence="verified",
        )],
        delegations=[{
            "owner_pid": 123, "delegation_id": "deleg_abcd", "task_index": 0,
            "status": "running", "goal": "[notatag] inspect",
            "updated_at": "2026-09-26 10:32:18",
        }],
    )


def test_monitor_parser_and_top_level_registration():
    args = _args("--json", "--sort", "rss", "--only-profile", "coder")
    assert args.json is True
    assert args.sort == "rss"
    assert args.only_profile == "coder"
    assert args.func is cmd_monitor

    from hermes_cli.main import _build_cli_parser
    parser, _subparsers = _build_cli_parser()
    parsed = parser.parse_args(["monitor", "--once"])
    assert parsed.command == "monitor"
    assert parsed.func is cmd_monitor


def test_monitor_rejects_non_finite_interval():
    import pytest

    with pytest.raises(SystemExit):
        _args("--interval", "inf")


def test_monitor_json_is_redacted_and_stable(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        "hermes_cli.subcommands.monitor.MonitorSampler.sample",
        lambda self: calls.append(True) or _snapshot(),
    )
    monkeypatch.setattr("hermes_cli.subcommands.monitor.time.sleep", lambda _seconds: None)

    assert cmd_monitor(_args("--json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["totals"] == {"processes": 1, "delegations": 1, "profiles": 1}
    assert payload["processes"][0]["activity"] == "task t_abcd [literal]"
    assert len(calls) == 2
    assert "cmdline" not in payload["processes"][0]
    assert "environment" not in payload["processes"][0]
    assert "transcript" not in payload["delegations"][0]
    assert "goal" not in payload["delegations"][0]
    assert set(payload["delegations"][0]).isdisjoint({"model", "provider", "last_tool"})


def test_manifest_controlled_activity_fields_never_reach_public_serialization():
    snapshot = _snapshot()
    snapshot.delegations[0].update({
        "model": "PROMPT_SECRET", "provider": "PROMPT_SECRET", "last_tool": "PROMPT_SECRET",
    })

    encoded = json.dumps(_json_payload(snapshot))
    assert "PROMPT_SECRET" not in encoded
    assert set(_json_payload(snapshot)["delegations"][0]).isdisjoint(
        {"model", "provider", "last_tool"},
    )


def test_monitor_screen_is_literal_and_contains_all_sections():
    from io import StringIO
    from rich.console import Console

    out = StringIO()
    Console(file=out, width=140, color_system=None).print(_render_screen(_snapshot(), sort="cpu"))
    rendered = out.getvalue()
    assert "HERMES MONITOR" in rendered
    assert "RUNTIMES" in rendered
    assert "DELEGATED AGENTS" in rendered
    assert "task t_abcd [literal]" in rendered
    assert "[notatag] inspect" not in rendered
    assert "64.0 MiB" in rendered


def test_interactive_keys_sort_select_and_open_safe_details():
    snapshot = _snapshot()
    state = _ViewState(sort="cpu")

    assert _apply_key(state, "s", snapshot) is True
    assert state.sort == "rss"
    assert _apply_key(state, "down", snapshot) is True
    assert state.selected_key == ProcessKey(123, 100.0)
    assert _apply_key(state, "enter", snapshot) is True
    assert state.details is True
    assert _apply_key(state, "a", snapshot) is True
    assert state.show_helpers is True

    from io import StringIO
    from rich.console import Console

    out = StringIO()
    Console(file=out, width=140, color_system=None).print(
        _render_screen(
            snapshot,
            sort=state.sort,
            show_helpers=state.show_helpers,
            selected_key=state.selected_key,
            show_details=state.details,
        )
    )
    rendered = out.getvalue()
    assert "RUNTIME DETAILS" in rendered
    assert "PID 123" in rendered
    assert "[notatag] inspect" not in rendered
    assert _apply_key(state, "q", snapshot) is False


def test_generic_descendant_helpers_follow_the_all_toggle():
    helper = ProcessRow(**{**_snapshot().processes[0].to_dict(), "kind": "helper"})
    snapshot = MonitorSnapshot(200.0, [helper], [])

    assert _selectable_rows(_ViewState(show_helpers=False), snapshot) == []
    assert _selectable_rows(_ViewState(show_helpers=True), snapshot) == [helper]


def test_interactive_selection_tracks_process_incarnation_and_details_are_literal():
    first = _snapshot().processes[0]
    first.status = "[bold]sleeping[/bold]"
    first.activity = "[red]literal activity[/red]"
    replacement = ProcessRow(**{**first.to_dict(), "create_time": 150.0})
    state = _ViewState()

    original = MonitorSnapshot(200.0, [first], [])
    assert _apply_key(state, "down", original) is True
    assert state.selected_key == ProcessKey(123, 100.0)

    reused = MonitorSnapshot(200.0, [replacement], [])
    assert _apply_key(state, "down", reused) is True
    assert state.selected_key == ProcessKey(123, 150.0)
    assert _apply_key(state, "enter", reused) is True

    from io import StringIO
    from rich.console import Console

    out = StringIO()
    Console(file=out, width=140, color_system=None).print(_render_screen(
        reused, selected_key=state.selected_key, show_details=True,
    ))
    rendered = out.getvalue()
    assert "[bold]sleeping[/bold]" in rendered
    assert "[red]literal activity[/red]" in rendered
