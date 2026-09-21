import json
import sqlite3
import time

import pytest


def test_append_event_creates_sqlite_row_and_jsonl_mirror(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "octacon"))

    from hermes_cli.profile_activity_ledger import append_event, ledger_db_path, ledger_jsonl_dir

    event_id = append_event(
        source="test",
        actor_profile="octacon",
        target_profile="default",
        event_type="skill.loaded",
        object_type="skill",
        object_id="kanban-worker",
        payload={"via": "unit-test"},
    )

    db_path = ledger_db_path()
    assert db_path == tmp_path / "governance" / "profile-activity-ledger.sqlite"
    assert db_path.exists()

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT event_id, source, actor_profile, target_profile, event_type, object_type, object_id, payload_json FROM activity_events"
        ).fetchall()

    assert len(rows) == 1
    assert rows[0][:7] == (
        event_id,
        "test",
        "octacon",
        "default",
        "skill.loaded",
        "skill",
        "kanban-worker",
    )
    assert json.loads(rows[0][7]) == {"via": "unit-test"}

    mirror_files = sorted(ledger_jsonl_dir().glob("*.jsonl"))
    assert len(mirror_files) == 1
    mirror = [json.loads(line) for line in mirror_files[0].read_text().splitlines()]
    assert mirror[0]["event_id"] == event_id
    assert mirror[0]["event_type"] == "skill.loaded"


def test_append_event_is_idempotent_by_event_id_and_mirrors_once(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from hermes_cli.profile_activity_ledger import append_event, ledger_db_path, ledger_jsonl_dir

    first = append_event(
        event_id="evt-fixed",
        source="test",
        event_type="kanban.task.created",
        object_type="kanban_task",
        object_id="t_123",
        payload={"n": 1},
    )
    second = append_event(
        event_id="evt-fixed",
        source="test",
        event_type="kanban.task.created",
        object_type="kanban_task",
        object_id="t_123",
        payload={"n": 2},
    )

    assert first == second == "evt-fixed"
    with sqlite3.connect(ledger_db_path()) as conn:
        assert conn.execute("SELECT COUNT(*) FROM activity_events").fetchone()[0] == 1
        payload = conn.execute("SELECT payload_json FROM activity_events").fetchone()[0]
    assert json.loads(payload) == {"n": 1}

    mirror_lines = []
    for mirror_file in ledger_jsonl_dir().glob("*.jsonl"):
        mirror_lines.extend(mirror_file.read_text().splitlines())
    assert len(mirror_lines) == 1


def test_activity_events_are_append_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from hermes_cli.profile_activity_ledger import append_event, ledger_db_path

    append_event(
        event_id="evt-append-only",
        source="test",
        event_type="gateway.message.received",
        object_type="gateway_message",
        object_id="m_1",
    )

    with sqlite3.connect(ledger_db_path()) as conn:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("UPDATE activity_events SET source = 'mutated'")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("DELETE FROM activity_events")


def test_feature_flag_defaults_false_and_reads_nested_config(monkeypatch):
    from hermes_cli import profile_activity_ledger as pal

    monkeypatch.setattr(pal, "load_config", lambda: {})
    assert pal.is_enabled() is False

    monkeypatch.setattr(
        pal,
        "load_config",
        lambda: {"governance": {"profile_activity_ledger": {"enabled": True}}},
    )
    assert pal.is_enabled() is True


def test_record_event_if_enabled_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli import profile_activity_ledger as pal

    monkeypatch.setattr(pal, "is_enabled", lambda cfg=None: False)
    assert pal.record_event_if_enabled(source="test", event_type="noop") is None
    assert not pal.ledger_db_path().exists()


def test_record_event_if_enabled_normalises_producer_metadata(tmp_path, monkeypatch):
    """Legacy producer metadata must reach the append-only ledger safely."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli import profile_activity_ledger as pal

    monkeypatch.setattr(pal, "is_enabled", lambda cfg=None: True)
    event_id = pal.record_event_if_enabled(
        source="test",
        event_type="delivery_error",
        profile="kensei",
        severity="error",
        correlation_id="job-123",
        idempotency_key="event-123",
        payload={"detail": "offline"},
    )

    assert event_id == "event-123"
    events = pal.query_events(event_types=["delivery_error"])
    assert len(events) == 1
    assert events[0]["event_id"] == "event-123"
    assert events[0]["actor_profile"] == "kensei"
    assert events[0]["payload"] == {
        "correlation_id": "job-123",
        "detail": "offline",
        "severity": "error",
    }


# ── P05 Batch 1: synthetic profile event round-trip tests ────────────────────
#
# The tests below extend the original ledger tests (above) with synthetic
# profile events that exercise the full schema shape: event_type,
# actor_profile, target_profile, payload, ts.  These prove the schema
# supports governance events beyond skill/kanban activity.
# ───────────────────────────────────────────────────────────────────────────


class TestSyntheticSchemaFields:
    """The ledger schema must support the full synthetic event shape."""

    def test_event_id_is_returned(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.profile_activity_ledger import append_event

        eid = append_event(
            source="test",
            event_type="profile.test",
            actor_profile="octacon",
            target_profile="denji",
            payload={"key": "value"},
        )
        assert eid.startswith("pal_")

    def test_occurred_at_preserved(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.profile_activity_ledger import append_event, query_events

        ts = int(time.time()) - 3600
        append_event(
            source="test",
            event_type="profile.test",
            occurred_at=ts,
        )
        events = query_events(event_types=["profile.test"])
        assert len(events) == 1
        assert events[0]["occurred_at"] == ts

    def test_actor_and_target_profile_preserved(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.profile_activity_ledger import append_event, query_events

        append_event(
            source="test",
            event_type="profile.grant",
            actor_profile="kensei",
            target_profile="octacon",
        )
        events = query_events(event_types=["profile.grant"])
        assert len(events) == 1
        assert events[0]["actor_profile"] == "kensei"
        assert events[0]["target_profile"] == "octacon"

    def test_payload_round_trips_as_dict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.profile_activity_ledger import append_event, query_events

        payload = {"action": "promote", "skill": "code-review", "tier": 2}
        append_event(
            source="test",
            event_type="profile.test",
            payload=payload,
        )
        events = query_events(event_types=["profile.test"])
        assert len(events) == 1
        assert events[0]["payload"] == payload

    def test_payload_with_nested_structures(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.profile_activity_ledger import append_event, query_events

        payload = {
            "answers": {"q1": 4, "q2": "good"},
            "meta": {"tags": ["a", "b"]},
        }
        append_event(
            source="test",
            event_type="profile.self_eval.submit",
            payload=payload,
        )
        events = query_events(event_types=["profile.self_eval.submit"])
        assert events[0]["payload"] == payload

    def test_empty_payload_defaults_to_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.profile_activity_ledger import append_event, query_events

        append_event(source="test", event_type="profile.test")
        events = query_events(event_types=["profile.test"])
        assert events[0]["payload"] == {}


class TestSyntheticRoundTrip:
    """Full append → query round-trip through the ledger schema."""

    def test_multiple_events_query_all(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.profile_activity_ledger import append_event, query_events

        base = int(time.time())
        for i in range(3):
            append_event(
                source="test",
                event_type="profile.test",
                actor_profile=f"profile_{i}",
                occurred_at=base + i,
            )
        events = query_events(event_types=["profile.test"])
        assert len(events) == 3
        # newest-first ordering
        assert events[0]["occurred_at"] >= events[1]["occurred_at"]

    def test_filter_by_target_profile(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.profile_activity_ledger import append_event, query_events

        append_event(source="test", event_type="profile.test", target_profile="alpha")
        append_event(source="test", event_type="profile.test", target_profile="beta")
        alpha = query_events(target_profile="alpha")
        assert len(alpha) == 1
        assert alpha[0]["target_profile"] == "alpha"

    def test_filter_by_actor_profile(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.profile_activity_ledger import append_event, query_events

        append_event(source="test", event_type="profile.test", actor_profile="kensei")
        append_event(source="test", event_type="profile.test", actor_profile="denji")
        kensei = query_events(actor_profile="kensei")
        assert len(kensei) == 1
        assert kensei[0]["actor_profile"] == "kensei"

    def test_filter_by_since_until(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.profile_activity_ledger import append_event, query_events

        base = int(time.time()) - 10000
        append_event(source="test", event_type="profile.test", occurred_at=base)
        append_event(source="test", event_type="profile.test", occurred_at=base + 5000)
        recent = query_events(since=base + 1000)
        assert len(recent) == 1
        assert recent[0]["occurred_at"] == base + 5000

    def test_limit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.profile_activity_ledger import append_event, query_events

        base = int(time.time())
        for i in range(5):
            append_event(source="test", event_type="profile.test", occurred_at=base + i)
        events = query_events(event_types=["profile.test"], limit=2)
        assert len(events) == 2

    def test_empty_ledger_returns_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.profile_activity_ledger import query_events

        assert query_events(event_types=["profile.test"]) == []


class TestSyntheticIdempotency:
    """The event_id is the idempotency key."""

    def test_duplicate_event_id_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.profile_activity_ledger import append_event, query_events

        eid = append_event(
            source="test", event_type="profile.test", event_id="custom-001"
        )
        assert eid == "custom-001"
        eid2 = append_event(
            source="test", event_type="profile.test", event_id="custom-001"
        )
        assert eid2 == "custom-001"
        events = query_events(event_types=["profile.test"])
        assert len(events) == 1

    def test_auto_generated_event_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.profile_activity_ledger import append_event, query_events

        eid1 = append_event(source="test", event_type="profile.test")
        eid2 = append_event(source="test", event_type="profile.test")
        assert eid1 != eid2
        assert len(query_events(event_types=["profile.test"])) == 2
