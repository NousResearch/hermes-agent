from argparse import Namespace


def test_evolution_enable_sets_config_and_creates_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.evolution import evolution_command

    rc = evolution_command(Namespace(evolution_command="enable"))

    assert rc == 0
    assert (tmp_path / "evolution").is_dir()
    assert not (tmp_path / "evolution" / "events.jsonl").exists()
    assert "enabled" in capsys.readouterr().out.lower()

    from hermes_cli.config import load_config

    assert load_config()["evolution"]["enabled"] is True


def test_evolution_disable_preserves_existing_events(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.evolution import evolution_command

    evolution_command(Namespace(evolution_command="enable"))
    events_path = tmp_path / "evolution" / "events.jsonl"
    events_path.write_text('{"id":"evt_20260608_073000_a1b2c3"}\n', encoding="utf-8")

    rc = evolution_command(Namespace(evolution_command="disable"))

    assert rc == 0
    assert events_path.exists()
    assert "preserved" in capsys.readouterr().out.lower()

    from hermes_cli.config import load_config

    assert load_config()["evolution"]["enabled"] is False


def test_evolution_is_registered_as_builtin_subcommand():
    from hermes_cli.main import _BUILTIN_SUBCOMMANDS

    assert "evolution" in _BUILTIN_SUBCOMMANDS


def test_evolution_list_disabled_with_no_events_prints_enable_guidance(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.evolution import evolution_command

    rc = evolution_command(
        Namespace(
            evolution_command="list",
            days=30,
            limit=50,
            event_type=None,
            target=None,
        )
    )

    assert rc == 0
    output = capsys.readouterr().out.lower()
    assert "disabled" in output
    assert "hermes evolution enable" in output
    assert "no evolution events" in output


def test_evolution_list_displays_existing_events_with_short_ids_when_disabled(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.evolution_log import append_event
    from hermes_cli.evolution import evolution_command

    append_event({
        "schema_version": 1,
        "id": "evt_20260608_073000_a1b2c3",
        "timestamp": "2026-06-08T07:30:00Z",
        "type": "memory.add",
        "target_name": "user",
        "target": "memories/USER.md",
        "target_kind": "memory",
        "summary": "Recorded timezone preference",
    })

    rc = evolution_command(
        Namespace(
            evolution_command="list",
            days=30,
            limit=50,
            event_type=None,
            target=None,
        )
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "disabled" in output.lower()
    assert "a1b2c3" in output
    assert "memory.add" in output
    assert "user" in output
    assert "Recorded timezone preference" in output


def test_evolution_timeline_behaves_like_list(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.evolution_log import append_event
    from hermes_cli.evolution import evolution_command

    append_event({
        "schema_version": 1,
        "id": "evt_20260608_073000_d4e5f6",
        "timestamp": "2026-06-08T07:30:00Z",
        "type": "skill.patch",
        "target_name": "daily-task-assistant",
        "target": "skills/daily-task-assistant/SKILL.md",
        "target_kind": "skill",
        "summary": "Improved reminder inference",
    })

    rc = evolution_command(
        Namespace(
            evolution_command="timeline",
            days=30,
            limit=50,
            event_type=None,
            target=None,
        )
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "d4e5f6" in output
    assert "skill.patch" in output
    assert "Improved reminder inference" in output


def test_evolution_show_accepts_short_id_and_displays_metadata(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.evolution_log import append_event
    from hermes_cli.evolution import evolution_command

    event_id = "evt_20260608_073000_a1b2c3"
    append_event({
        "schema_version": 1,
        "id": event_id,
        "timestamp": "2026-06-08T07:30:00Z",
        "type": "memory.add",
        "actor": "agent",
        "source_tool": "memory",
        "target": "memories/USER.md",
        "target_kind": "memory",
        "target_name": "user",
        "summary": "Recorded preference",
        "reason": "Manual verification",
        "redaction_enabled": True,
        "redaction_applied": False,
        "diff_format": "unified",
        "diff_truncated": False,
        "diff": "--- before\n+++ after\n+pref",
    })

    rc = evolution_command(Namespace(evolution_command="show", event_id="a1b2c3"))

    assert rc == 0
    output = capsys.readouterr().out
    assert event_id in output
    assert "memory.add" in output
    assert "agent" in output
    assert "memory" in output
    assert "memories/USER.md" in output
    assert "Recorded preference" in output
    assert "Manual verification" in output
    assert "Redaction" in output
    assert "Truncated" in output
    assert "--- before" in output
    assert "+pref" in output


def test_evolution_show_reports_ambiguous_short_id(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.evolution_log import append_event
    from hermes_cli.evolution import evolution_command

    append_event({
        "id": "evt_20260608_073000_aaa111",
        "timestamp": "2026-06-08T07:30:00Z",
    })
    append_event({
        "id": "evt_20260608_073001_bbb111",
        "timestamp": "2026-06-08T07:31:00Z",
    })

    rc = evolution_command(Namespace(evolution_command="show", event_id="111"))

    assert rc == 1
    output = capsys.readouterr().out
    assert "Ambiguous" in output
    assert "evt_20260608_073000_aaa111" in output
    assert "evt_20260608_073001_bbb111" in output


def test_evolution_stats_prints_overview_counts_targets_and_activity(
    tmp_path, monkeypatch, capsys
):
    from datetime import datetime, timezone

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.evolution_log import append_event
    from hermes_cli.evolution import evolution_command

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    append_event({
        "id": "evt_20260608_073000_a1b2c3",
        "timestamp": ts,
        "type": "memory.add",
        "target": "memories/USER.md",
        "target_kind": "memory",
        "target_name": "user",
    })
    append_event({
        "id": "evt_20260608_073001_b2c3d4",
        "timestamp": ts,
        "type": "memory.add",
        "target": "memories/MEMORY.md",
        "target_kind": "memory",
        "target_name": "memory",
    })
    append_event({
        "id": "evt_20260608_073002_d4e5f6",
        "timestamp": ts,
        "type": "skill.patch",
        "target": "skills/daily-task-assistant/SKILL.md",
        "target_kind": "skill",
        "target_name": "daily-task-assistant",
    })

    rc = evolution_command(Namespace(evolution_command="stats", days=30))

    assert rc == 0
    output = capsys.readouterr().out
    assert "Total events: 3" in output
    assert "Memory:  2" in output
    assert "Skills:  1" in output
    assert "Curator: 0" in output
    assert "memory.add" in output
    assert "skill.patch" in output
    assert "memories/USER.md" in output
    assert "memories/MEMORY.md" in output
    assert "skills/daily-task-assistant/SKILL.md" in output
    assert "Activity by day" in output


def test_evolution_clear_previews_without_deleting(tmp_path, monkeypatch, capsys):
    from datetime import datetime, timedelta, timezone

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.evolution_log import append_event, read_events
    from hermes_cli.evolution import evolution_command

    old_ts = (datetime.now(timezone.utc) - timedelta(days=100)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    new_ts = (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    append_event({"id": "old", "timestamp": old_ts})
    append_event({"id": "new", "timestamp": new_ts})

    rc = evolution_command(
        Namespace(evolution_command="clear", older_than=90, yes=False)
    )

    assert rc == 0
    output = capsys.readouterr().out.lower()
    assert "would delete 1" in output
    assert "retained 1" in output
    events, warnings = read_events()
    assert warnings == []
    assert [event["id"] for event in events] == ["old", "new"]


def test_evolution_clear_yes_deletes_only_old_events(tmp_path, monkeypatch, capsys):
    from datetime import datetime, timedelta, timezone

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.evolution_log import append_event, read_events
    from hermes_cli.evolution import evolution_command

    old_ts = (datetime.now(timezone.utc) - timedelta(days=100)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    new_ts = (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    append_event({"id": "old", "timestamp": old_ts})
    append_event({"id": "new", "timestamp": new_ts})

    rc = evolution_command(
        Namespace(evolution_command="clear", older_than=90, yes=True)
    )

    assert rc == 0
    output = capsys.readouterr().out.lower()
    assert "deleted 1" in output
    assert "retained 1" in output
    events, warnings = read_events()
    assert warnings == []
    assert [event["id"] for event in events] == ["new"]
