from __future__ import annotations

import json


def _write_events(home, events):
    learn_dir = home / "learn"
    learn_dir.mkdir(parents=True, exist_ok=True)
    (learn_dir / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def test_analyzer_creates_usage_suggestion_for_repeated_pattern(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_time import now as hermes_now
    from hermes_cli.learn import analyzer

    _write_events(
        home,
        [
            {
                "timestamp": hermes_now().isoformat(),
                "process_name": "outlook.exe",
                "window_title": "Inbox",
                "domain": None,
                "category": "communication",
                "idle": False,
                "duration_seconds": 240,
            },
            {
                "timestamp": hermes_now().isoformat(),
                "process_name": "outlook.exe",
                "window_title": "Inbox",
                "domain": None,
                "category": "communication",
                "idle": False,
                "duration_seconds": 240,
            },
            {
                "timestamp": hermes_now().isoformat(),
                "process_name": "teams.exe",
                "window_title": "Chat",
                "domain": None,
                "category": "communication",
                "idle": False,
                "duration_seconds": 240,
            },
        ],
    )

    created = []

    def add_suggestion(**kwargs):
        created.append(kwargs)
        return {"id": "learn1", "status": "pending", **kwargs}

    suggestions = analyzer.create_usage_suggestions(add_fn=add_suggestion)

    assert len(suggestions) == 1
    assert created[0]["source"] == "usage"
    assert created[0]["dedup_key"] == "learn:usage:communication"
    assert created[0]["job_spec"]["deliver"] == "origin"
    assert created[0]["job_spec"]["schedule"] == "0 16 * * 1-5"
    assert "communication" in created[0]["description"].lower()


def test_analyzer_writes_profile_local_opportunity_report_and_dedupes(tmp_path, monkeypatch):
    home = tmp_path / "profile-a"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from cron import suggestions as cron_suggestions
    from hermes_time import now as hermes_now
    from hermes_cli.learn import analyzer

    _write_events(
        home,
        [
            {
                "timestamp": hermes_now().isoformat(),
                "process_name": "code.exe",
                "window_title": "hermes-agent",
                "domain": None,
                "category": "development",
                "idle": False,
                "duration_seconds": 300,
            },
            {
                "timestamp": hermes_now().isoformat(),
                "process_name": "pwsh.exe",
                "window_title": "pytest",
                "domain": None,
                "category": "development",
                "idle": False,
                "duration_seconds": 300,
            },
            {
                "timestamp": hermes_now().isoformat(),
                "process_name": "code.exe",
                "window_title": "learn",
                "domain": None,
                "category": "development",
                "idle": False,
                "duration_seconds": 300,
            },
        ],
    )

    first = analyzer.create_usage_suggestions()
    second = analyzer.create_usage_suggestions()

    assert len(first) == 1
    assert second == []
    assert (home / "learn" / "opportunities.json").exists()
    assert (home / "cron" / "suggestions.json").exists()

    old_cron_dir = cron_suggestions.CRON_DIR
    old_suggestions_file = cron_suggestions.SUGGESTIONS_FILE
    cron_suggestions.CRON_DIR = home / "cron"
    cron_suggestions.SUGGESTIONS_FILE = cron_suggestions.CRON_DIR / "suggestions.json"
    try:
      pending = cron_suggestions.list_pending()
    finally:
      cron_suggestions.CRON_DIR = old_cron_dir
      cron_suggestions.SUGGESTIONS_FILE = old_suggestions_file

    assert len(pending) == 1
    assert pending[0]["source"] == "usage"
    assert pending[0]["dedup_key"] == "learn:usage:development"
