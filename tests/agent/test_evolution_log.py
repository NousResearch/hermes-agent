def test_generate_event_id_has_expected_shape():
    import re
    from datetime import datetime, timezone

    from agent.evolution_log import generate_event_id

    event_id = generate_event_id(datetime(2026, 6, 8, 7, 30, 0, tzinfo=timezone.utc))

    assert re.match(r"^evt_20260608_073000_[0-9a-f]{6}$", event_id)


def test_utc_timestamp_uses_z_suffix():
    from datetime import datetime, timezone

    from agent.evolution_log import utc_timestamp

    assert (
        utc_timestamp(datetime(2026, 6, 8, 7, 30, 0, tzinfo=timezone.utc))
        == "2026-06-08T07:30:00Z"
    )


def test_make_unified_diff_uses_simple_headers():
    from agent.evolution_log import make_unified_diff

    diff = make_unified_diff("old\n", "new\n")

    assert "--- before" in diff
    assert "+++ after" in diff
    assert "-old" in diff
    assert "+new" in diff


def test_truncate_text_marks_truncated():
    from agent.evolution_log import truncate_text

    text, truncated = truncate_text("abcdef", 3)

    assert truncated is True
    assert text.startswith("abc")
    assert "truncated" in text.lower()


def test_redact_text_if_enabled_reports_applied(monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)
    from agent.evolution_log import redact_text_if_enabled

    text, applied = redact_text_if_enabled(
        "OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012", True
    )

    assert applied is True
    assert "abc123def456" not in text


def test_append_event_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.evolution_log import append_event, get_events_path, read_events

    event = {
        "schema_version": 1,
        "id": "evt_20260608_073000_a1b2c3",
        "timestamp": "2026-06-08T07:30:00Z",
        "type": "memory.add",
    }

    append_event(event)

    assert get_events_path().exists()
    events, warnings = read_events()
    assert warnings == []
    assert events == [event]


def test_read_events_skips_bad_json_line(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.evolution_log import get_events_path, read_events

    path = get_events_path()
    path.parent.mkdir(parents=True)
    path.write_text('{"id":"ok"}\nnot-json\n{"id":"ok2"}\n', encoding="utf-8")

    events, warnings = read_events()

    assert [event["id"] for event in events] == ["ok", "ok2"]
    assert warnings


def test_build_event_includes_schema_and_metadata_without_model_provider():
    from datetime import datetime, timezone

    from agent.evolution_log import build_event

    event = build_event(
        event_type="memory.add",
        source_tool="memory",
        target="memories/USER.md",
        target_kind="memory",
        target_name="user",
        before_text="",
        after_text="pref\n",
        summary="Recorded preference",
        reason="User asked Hermes to remember it.",
        now=datetime(2026, 6, 8, 7, 30, 0, tzinfo=timezone.utc),
        config={
            "evolution": {"redact": True, "record_diff": True, "max_diff_chars": 20000}
        },
    )

    assert event["schema_version"] == 1
    assert event["id"].startswith("evt_20260608_073000_")
    assert event["timestamp"] == "2026-06-08T07:30:00Z"
    assert event["actor"] == "agent"
    assert event["source_tool"] == "memory"
    assert event["type"] == "memory.add"
    assert event["target"] == "memories/USER.md"
    assert event["target_kind"] == "memory"
    assert event["target_name"] == "user"
    assert event["summary"] == "Recorded preference"
    assert event["reason"] == "User asked Hermes to remember it."
    assert event["diff_format"] == "unified"
    assert "+pref" in event["diff"]
    assert event["redaction_enabled"] is True
    assert event["redaction_applied"] is False
    assert event["diff_truncated"] is False
    assert event["max_diff_chars"] == 20000
    assert "model" not in event
    assert "provider" not in event


def test_fallback_summary_is_useful():
    from agent.evolution_log import fallback_summary

    summary = fallback_summary("memory.add", "user", "memories/USER.md")

    assert summary
    assert "memory" in summary.lower()
    assert "user" in summary.lower()


def test_resolve_event_id_supports_full_unique_short_and_ambiguous():
    from agent.evolution_log import resolve_event_id

    events = [
        {"id": "evt_20260608_073000_aaa111"},
        {"id": "evt_20260608_073001_bbb111"},
        {"id": "evt_20260608_073002_ccc222"},
    ]

    found, matches = resolve_event_id(events, "evt_20260608_073002_ccc222")
    assert found == events[2]
    assert matches == [events[2]]

    found, matches = resolve_event_id(events, "aaa111")
    assert found == events[0]
    assert matches == [events[0]]

    found, matches = resolve_event_id(events, "111")
    assert found is None
    assert matches == [events[0], events[1]]


def test_filter_events_supports_days_type_target_and_limit():
    from datetime import datetime, timedelta, timezone

    from agent.evolution_log import filter_events

    now = datetime.now(timezone.utc)
    events = [
        {
            "id": "new-skill",
            "timestamp": (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "type": "skill.edit",
            "target": "skills/bar/SKILL.md",
            "target_kind": "skill",
            "target_name": "bar",
        },
        {
            "id": "new-memory",
            "timestamp": (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "type": "memory.add",
            "target": "memories/USER.md",
            "target_kind": "memory",
            "target_name": "user",
        },
        {
            "id": "old-skill",
            "timestamp": (now - timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "type": "skill.patch",
            "target": "skills/foo/SKILL.md",
            "target_kind": "skill",
            "target_name": "foo",
        },
    ]

    assert [e["id"] for e in filter_events(events, days=30)] == [
        "new-skill",
        "new-memory",
    ]
    assert [e["id"] for e in filter_events(events, event_type="skill")] == [
        "new-skill",
        "old-skill",
    ]
    assert [e["id"] for e in filter_events(events, event_type="skill.patch")] == [
        "old-skill"
    ]
    assert [e["id"] for e in filter_events(events, target_query="USER")] == [
        "new-memory"
    ]
    assert [e["id"] for e in filter_events(events, limit=1)] == ["new-skill"]
