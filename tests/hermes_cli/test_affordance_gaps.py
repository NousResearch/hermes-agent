"""Tests for affordance gap logging."""

import json

import pytest

from hermes_cli.affordance_gaps import (
    format_affordance_gaps,
    load_affordance_gaps,
    log_affordance_gap,
)


def test_log_and_load_affordance_gap(tmp_path):
    log_path = tmp_path / "affordance_gaps.jsonl"

    event = log_affordance_gap(
        goal="Summarize a video",
        missing_capability="video transcript fetcher",
        failure_description="No tool could retrieve the transcript.",
        available_tools=["web", "web", "memory"],
        session_id="session-1",
        source="test",
        log_path=log_path,
    )

    assert event["failure_mode"] == "missing_affordance"
    assert event["available_tools"] == ["memory", "web"]

    loaded = load_affordance_gaps(log_path=log_path)
    assert len(loaded) == 1
    assert loaded[0]["goal"] == "Summarize a video"
    assert loaded[0]["missing_capability"] == "video transcript fetcher"


def test_load_skips_invalid_jsonl_rows(tmp_path):
    log_path = tmp_path / "affordance_gaps.jsonl"
    log_path.write_text(
        "\n".join([
            "{bad json",
            json.dumps({"goal": "ok", "missing_capability": "tool"}),
        ]),
        encoding="utf-8",
    )

    loaded = load_affordance_gaps(log_path=log_path)

    assert loaded == [{"goal": "ok", "missing_capability": "tool"}]


def test_format_affordance_gaps_empty():
    assert format_affordance_gaps([]) == "No affordance gaps logged."


def test_format_affordance_gaps_includes_key_fields():
    output = format_affordance_gaps([
        {
            "timestamp": "2026-04-25T00:00:00+00:00",
            "goal": "Do the task",
            "missing_capability": "browser auth",
            "failure_description": "Could not access the site.",
            "available_tools": ["web"],
            "session_id": "abc",
        }
    ])

    assert "browser auth" in output
    assert "Goal: Do the task" in output
    assert "Session: abc" in output


def test_log_requires_goal_and_missing_capability(tmp_path):
    with pytest.raises(ValueError, match="goal is required"):
        log_affordance_gap(goal="", missing_capability="tool", log_path=tmp_path / "gaps.jsonl")

    with pytest.raises(ValueError, match="missing_capability is required"):
        log_affordance_gap(goal="goal", missing_capability="", log_path=tmp_path / "gaps.jsonl")


def _run_hermes_cli(argv, capsys):
    from hermes_cli.main import _build_cli_parser, _parse_cli_args

    parser, subparsers = _build_cli_parser()
    args = _parse_cli_args(parser, subparsers, argv)
    args.func(args)
    return capsys.readouterr().out


def test_gaps_cli_logs_and_lists_under_hermes_home(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))

    out = _run_hermes_cli(
        [
            "gaps", "log",
            "--goal", "Summarize a video",
            "--missing-capability", "video transcript fetcher",
            "--tool", "web",
            "--session-id", "session-1",
        ],
        capsys,
    )
    assert "video transcript fetcher" in out

    rows = [json.loads(line) for line in (home / "affordance_gaps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["source"] == "cli"
    assert rows[0]["session_id"] == "session-1"

    out = _run_hermes_cli(["gaps", "list"], capsys)
    assert "video transcript fetcher" in out
    assert "Session: session-1" in out


def test_gaps_cli_persistence_is_profile_scoped(tmp_path, monkeypatch, capsys):
    profile_a = tmp_path / "a"
    profile_b = tmp_path / "b"

    monkeypatch.setenv("HERMES_HOME", str(profile_a))
    _run_hermes_cli(["gaps", "log", "--goal", "g", "--missing-capability", "only-in-a"], capsys)

    monkeypatch.setenv("HERMES_HOME", str(profile_b))
    assert "No affordance gaps logged." in _run_hermes_cli(["gaps"], capsys)
    assert not (profile_b / "affordance_gaps.jsonl").exists()

    monkeypatch.setenv("HERMES_HOME", str(profile_a))
    assert "only-in-a" in _run_hermes_cli(["gaps", "list"], capsys)


def test_gaps_is_known_builtin_and_skips_plugin_discovery(monkeypatch):
    from hermes_cli.main import _build_cli_parser, _plugin_cli_discovery_needed

    _parser, subparsers = _build_cli_parser()
    assert "gaps" in subparsers.choices

    monkeypatch.setattr("sys.argv", ["hermes", "gaps", "list"])
    assert _plugin_cli_discovery_needed() is False
