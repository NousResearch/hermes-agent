"""Executable Phase 0 invariants for the pure Roadmaps contract."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from roadmaps_contract import (  # noqa: E402
    DuplicateEventConflict,
    EventEnvelope,
    FixtureRoadmapRepository,
    QualifiedNodeKey,
    Relation,
    Scope,
    StaleEventError,
    build_fixture,
    replay_events,
    transition_node,
    transition_plan,
)


@pytest.fixture
def scope() -> Scope:
    return Scope("profile-a", "project-a", "roadmap-a")


def event(scope: Scope, *, version: int = 1, event_id: str | None = None, payload: dict | None = None, event_type: str = "roadmap.node.progressed") -> EventEnvelope:
    return EventEnvelope.create(
        event_id=event_id or f"evt-{version}", scope=scope, aggregate_id="roadmap-a",
        aggregate_version=version, payload=payload or {"state": "ready", "nested": {"items": [1, 2]}},
        event_type=event_type,
        occurred_at=datetime(2026, 8, 14, 12, 0, tzinfo=timezone(timedelta(hours=2))),
        received_at=datetime(2026, 8, 14, 12, 1, tzinfo=timezone.utc),
    )


def test_scope_and_qualified_node_key_preserve_profile_project_roadmap_identity(scope: Scope) -> None:
    assert scope.as_tuple() == ("profile-a", "project-a", "roadmap-a")
    assert QualifiedNodeKey(scope, "node-1").as_tuple() == ("profile-a", "project-a", "roadmap-a", "node-1")


def test_cross_profile_project_and_roadmap_relations_are_rejected(scope: Scope) -> None:
    source = QualifiedNodeKey(scope, "from")
    for other in (Scope("profile-b", "project-a", "roadmap-a"), Scope("profile-a", "project-b", "roadmap-a"), Scope("profile-a", "project-a", "roadmap-b")):
        with pytest.raises(ValueError, match="same scope"):
            Relation(source, QualifiedNodeKey(other, "to"))


def test_relation_kind_and_identifiers_are_strict(scope: Scope) -> None:
    with pytest.raises(ValueError, match="relation kind"):
        Relation(QualifiedNodeKey(scope, "from"), QualifiedNodeKey(scope, "to"), "unknown")
    with pytest.raises(TypeError, match="relation kind"):
        Relation(QualifiedNodeKey(scope, "from"), QualifiedNodeKey(scope, "to"), 3)  # type: ignore[arg-type]
    for bad in ("", " trimmed", "trimmed ", "a/b", "a\\b", "a\x00b", 3):
        with pytest.raises((ValueError, TypeError)):
            Scope(bad, "project-a", "roadmap-a")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        QualifiedNodeKey(scope, "node/id")


def test_event_hash_is_deterministic_and_payload_is_not_retained(scope: Scope) -> None:
    source = {"nested": {"items": [1, 2]}, "state": "ready"}
    first = event(scope, payload=source)
    second = EventEnvelope.create(event_id="other", scope=scope, aggregate_id="roadmap-a", event_type="roadmap.node.progressed", payload={"nested": {"items": [1, 2]}, "state": "ready"}, occurred_at=first.occurred_at, received_at=first.received_at)
    assert first.payload_hash == second.payload_hash
    source["nested"]["items"].append(3)
    source["state"] = "blocked"
    assert first.payload_hash == second.payload_hash
    assert not hasattr(first, "payload")


def test_event_timestamps_are_aware_and_normalized_to_utc(scope: Scope) -> None:
    current = event(scope)
    assert current.occurred_at.tzinfo is timezone.utc
    assert current.occurred_at.hour == 10
    assert current.received_at.tzinfo is timezone.utc


def test_event_constructor_rejects_arbitrary_hash_and_invalid_identity(scope: Scope) -> None:
    current = event(scope)
    with pytest.raises(ValueError, match="payload_hash"):
        EventEnvelope(1, "evt", "roadmap.node.progressed", "roadmap", "roadmap-a", scope, "actor", current.occurred_at, current.received_at, None, None, 1, "arbitrary")
    with pytest.raises(ValueError, match="schema_version"):
        EventEnvelope(2, current.event_id, current.event_type, current.aggregate_type, current.aggregate_id, scope, current.actor, current.occurred_at, current.received_at, None, None, 1, current.payload_hash)
    with pytest.raises(TypeError, match="payload"):
        EventEnvelope.create(event_id="evt-payload", scope=scope, aggregate_id="roadmap-a", event_type="roadmap.node.progressed", payload=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="aggregate_type"):
        EventEnvelope.create(event_id="evt-type", scope=scope, aggregate_id="roadmap-a", event_type="roadmap.node.progressed", aggregate_type=3, payload={})  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_event_type", ("", " roadmap.node.progressed", "roadmap.node.progressed ", "roadmap/node", "roadmap\x00node", 3, None))
def test_event_type_is_required_and_strict(scope: Scope, bad_event_type: object) -> None:
    with pytest.raises((ValueError, TypeError), match="event_type"):
        EventEnvelope.create(event_id="evt-type", scope=scope, aggregate_id="roadmap-a", event_type=bad_event_type, payload={})  # type: ignore[arg-type]


def test_duplicate_event_id_with_different_event_type_is_rejected(scope: Scope) -> None:
    first = event(scope, event_id="evt-1")
    different = event(scope, event_id="evt-1", event_type="roadmap.node.completed")
    with pytest.raises(DuplicateEventConflict):
        replay_events((first, different))


def test_duplicate_event_id_with_different_payload_is_rejected(scope: Scope) -> None:
    first = event(scope, event_id="evt-1")
    different = event(scope, event_id="evt-1", payload={"state": "blocked"})
    with pytest.raises(DuplicateEventConflict):
        first.ensure_same_identity(different)


def test_duplicate_event_id_with_same_payload_is_idempotent(scope: Scope) -> None:
    first = event(scope, event_id="evt-1")
    replay = event(scope, event_id="evt-1")
    assert first.ensure_same_identity(replay) is None


def test_replay_identical_duplicate_event_id_is_idempotent(scope: Scope) -> None:
    first = event(scope, event_id="evt-1")
    assert replay_events((first, first)) == (first,)


def test_duplicate_event_id_with_different_version_is_rejected(scope: Scope) -> None:
    first = event(scope, event_id="evt-1", version=1)
    different = event(scope, event_id="evt-1", version=2)
    with pytest.raises(DuplicateEventConflict):
        replay_events((first, different))


@pytest.mark.parametrize("field", ("actor", "correlation_id"))
def test_duplicate_event_id_with_different_metadata_is_rejected(scope: Scope, field: str) -> None:
    first = event(scope, event_id="evt-1")
    kwargs = {field: "different-actor" if field == "actor" else "corr-2"}
    different = EventEnvelope.create(
        event_id=first.event_id, scope=scope, aggregate_id=first.aggregate_id,
        event_type="roadmap.node.progressed",
        aggregate_version=first.aggregate_version, payload={"state": "ready", "nested": {"items": [1, 2]}},
        actor=kwargs.get("actor", first.actor), correlation_id=kwargs.get("correlation_id"),
        occurred_at=first.occurred_at, received_at=first.received_at,
    )
    with pytest.raises(DuplicateEventConflict):
        replay_events((first, different))


def test_duplicate_event_id_with_different_timestamp_is_rejected(scope: Scope) -> None:
    first = event(scope, event_id="evt-1")
    different = EventEnvelope.create(
        event_id=first.event_id, scope=scope, aggregate_id=first.aggregate_id,
        event_type="roadmap.node.progressed",
        aggregate_version=first.aggregate_version, payload={"state": "ready", "nested": {"items": [1, 2]}},
        occurred_at=first.occurred_at + timedelta(seconds=1), received_at=first.received_at,
    )
    with pytest.raises(DuplicateEventConflict):
        replay_events((first, different))


def test_replay_from_cursor_returns_only_new_events(scope: Scope) -> None:
    stream = (event(scope, version=1), event(scope, version=2), event(scope, version=3))
    assert replay_events(stream, cursor=2) == stream[2:]


def test_replay_deduplicates_duplicates_after_cursor(scope: Scope) -> None:
    first = event(scope, version=1, event_id="evt-1")
    second = event(scope, version=2, event_id="evt-2")
    assert replay_events((first, second, first, second), cursor=0) == (first, second)
    assert replay_events((first, second, first, second), cursor=2) == ()


def test_fixture_repository_deduplicates_duplicate_events(scope: Scope) -> None:
    first = event(scope, version=1, event_id="evt-1")
    second = event(scope, version=2, event_id="evt-2")
    repository = FixtureRoadmapRepository(scope, build_fixture(scope), (first, second, first, second))
    assert repository.get_events_after(scope, 0) == (first, second)


def test_replay_rejects_late_or_regressive_aggregate_version(scope: Scope) -> None:
    with pytest.raises(StaleEventError):
        replay_events((event(scope, version=1), event(scope, version=3), event(scope, version=2)))


def test_fixture_repository_rejects_non_monotonic_stream(scope: Scope) -> None:
    repository = FixtureRoadmapRepository(scope, build_fixture(scope), (event(scope, version=1), event(scope, version=3), event(scope, version=2)))
    with pytest.raises(StaleEventError):
        repository.get_events_after(scope, 0)


def test_fixture_repository_rejects_event_outside_scope(scope: Scope) -> None:
    other_scope = Scope("profile-a", "project-a", "roadmap-b")
    with pytest.raises(ValueError, match="scope"):
        FixtureRoadmapRepository(scope, build_fixture(scope), (event(other_scope),))


def test_fixture_repository_is_injected_read_only_and_does_not_write(tmp_path: Path, scope: Scope) -> None:
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    repository = FixtureRoadmapRepository(scope, build_fixture(scope), (event(scope, version=1), event(scope, version=2)))
    assert repository.list(profile_id="profile-a", project_id="project-a") == (scope,)
    assert repository.get_snapshot(scope)["scope"] == scope
    assert repository.get_snapshot(scope)["todos"] == ()
    assert repository.get_events_after(scope, 1)[0].aggregate_version == 2
    assert not any(name in dir(repository) for name in ("save", "write", "update", "delete", "persist"))
    with pytest.raises(TypeError):
        repository.get(scope)["new"] = "mutation"  # type: ignore[index]
    assert before == {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}


def test_terminal_transitions_are_rejected_and_essential_transitions_are_valid() -> None:
    assert transition_plan("draft", "proposed") == "proposed"
    assert transition_plan("validated", "in_progress") == "in_progress"
    assert transition_node("planned", "ready") == "ready"
    assert transition_node("blocked", "completed") == "completed"
    for transition in (("archived", "draft", transition_plan), ("archived", "ready", transition_node), ("completed", "in_progress", transition_node)):
        with pytest.raises(ValueError):
            transition[2](transition[0], transition[1])
