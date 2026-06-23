from __future__ import annotations

import json
from pathlib import Path


def test_initial_status_is_off_and_profile_local(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.learn import state

    status = state.get_status()

    assert status["mode"] == "off"
    assert status["state"] == "stopped"
    assert status["enabled"] is False
    assert status["running"] is False
    assert status["paused"] is False
    assert status["hermes_home"] == str(home)
    assert status["storage_path"] == str(home / "learn")
    assert not (home / "learn" / "state.json").exists()


def test_start_pause_resume_stop_persists_state(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.learn import state

    started = state.start(mode="learn")
    assert started["mode"] == "learn"
    assert started["state"] == "running"
    assert started["enabled"] is True
    assert started["running"] is True
    assert started["paused"] is False
    assert started["started_at"]

    paused = state.pause()
    assert paused["state"] == "paused"
    assert paused["running"] is False
    assert paused["paused"] is True
    assert paused["paused_at"]

    resumed = state.resume()
    assert resumed["state"] == "running"
    assert resumed["running"] is True
    assert resumed["paused"] is False

    stopped = state.stop()
    assert stopped["state"] == "stopped"
    assert stopped["running"] is False
    assert stopped["paused"] is False
    assert stopped["stopped_at"]

    on_disk = json.loads((home / "learn" / "state.json").read_text(encoding="utf-8"))
    assert on_disk["mode"] == "learn"
    assert on_disk["state"] == "stopped"


def test_invalid_mode_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))

    from hermes_cli.learn import state

    try:
        state.start(mode="screenshots")
    except ValueError as exc:
        assert "mode" in str(exc)
    else:
        raise AssertionError("invalid mode should raise ValueError")


def test_future_modes_are_not_startable_in_mvp(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))

    from hermes_cli.learn import state

    for mode in ("ask_first", "auto_draft", "teach"):
        try:
            state.start(mode=mode)
        except ValueError as exc:
            assert "learn" in str(exc)
        else:
            raise AssertionError(f"{mode} should not start before it has distinct behavior")


def test_delete_data_clears_events_but_keeps_selected_mode(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.learn import state

    state.start(mode="learn")
    events_file = home / "learn" / "events.jsonl"
    events_file.write_text('{"kind":"app","title":"redacted"}\n', encoding="utf-8")

    deleted = state.delete_data()

    assert deleted["mode"] == "learn"
    assert deleted["state"] == "stopped"
    assert deleted["collected_event_count"] == 0
    assert deleted["data_deleted_at"]
    assert not events_file.exists()


def test_status_counts_events_without_reading_payloads(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.learn import state

    learn_dir = home / "learn"
    learn_dir.mkdir(parents=True)
    (learn_dir / "events.jsonl").write_text(
        '{"window_title":"Secret quarterly plan"}\n{"window_title":"Private inbox"}\n',
        encoding="utf-8",
    )

    status = state.get_status()

    assert status["collected_event_count"] == 2
    assert "Secret quarterly plan" not in json.dumps(status)
