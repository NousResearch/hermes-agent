from __future__ import annotations

import json


def test_learn_command_status_start_stop_and_delete(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.learn import commands, runtime

    starts = []
    stops = []
    monkeypatch.setattr(runtime, "ensure_running", lambda: starts.append("start"))
    monkeypatch.setattr(runtime, "stop_runtime", lambda: stops.append("stop"))

    assert "Learn is off" in commands.handle_learn_command("status")
    assert "Learn started" in commands.handle_learn_command("start")
    assert starts == ["start"]
    assert "Learn stopped" in commands.handle_learn_command("stop")
    assert stops == ["stop"]

    events_file = home / "learn" / "events.jsonl"
    events_file.write_text(json.dumps({"category": "development"}) + "\n", encoding="utf-8")
    assert "Learn data deleted" in commands.handle_learn_command("delete-data")
    assert not events_file.exists()


def test_learn_command_review_creates_usage_suggestions(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.learn import analyzer, commands

    monkeypatch.setattr(
        analyzer,
        "create_usage_suggestions",
        lambda: [{"id": "learn1", "title": "Daily communication follow-up summary", "status": "pending"}],
    )

    output = commands.handle_learn_command("review")

    assert "Created 1 Learn suggestion" in output
    assert "Run /suggestions to review" in output


def test_learn_command_rejects_future_modes_until_implemented(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))

    from hermes_cli.learn import commands, runtime

    starts = []
    monkeypatch.setattr(runtime, "ensure_running", lambda: starts.append("start"))

    output = commands.handle_learn_command("start teach")

    assert "Learn start failed" in output
    assert starts == []
