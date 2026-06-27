from agent.memory_sidecar.models import ContextQuery, IngestSource, Observation, ObservationFile, SessionFact
from agent.memory_sidecar.store import MemorySidecarStore


def test_store_round_trip_query_and_checkpoint(tmp_path):
    store = MemorySidecarStore(tmp_path / "memory_sidecar.db")

    checkpoint = store.upsert_checkpoint(
        IngestSource(
            source_id="src-1",
            path="/tmp/session.jsonl",
            last_offset=128,
            partial_line="{\"partial\": true",
            last_event_at="2026-06-26T18:11:10Z",
            updated_at="2026-06-26T18:12:00Z",
        )
    )
    assert checkpoint.last_offset == 128

    stored = store.insert_observation(
        Observation(
            session_id="session-1",
            message_id="msg-7",
            role="assistant",
            event_ts="2026-06-26T18:11:10Z",
            observation_type="decision",
            title="Adopt sidecar DB",
            summary="Use a Hermes-native SQLite sidecar for compact context retrieval.",
            detail="Decision: keep the memory sidecar inside Hermes with no HTTP worker.",
            concepts=("memory", "privacy"),
            files=(ObservationFile(file_path="agent/memory_sidecar/store.py", change_kind="created"),),
            privacy_status="public",
            confidence=0.91,
        )
    )
    assert stored.id is not None

    store.upsert_session_fact(
        SessionFact(
            session_id="session-1",
            user_goal="Build a safe Hermes-native memory sidecar",
            latest_summary="Models and store landed for the MVP.",
            last_seen_at="2026-06-26T18:13:00Z",
        )
    )

    result = store.query_context(
        ContextQuery(
            query="Hermes sidecar",
            session_id="session-1",
            types=("decision",),
            concepts=("memory",),
            file_path="agent/memory_sidecar/store.py",
            limit=5,
            time_bias="relevant",
        )
    )

    assert len(result.observations) == 1
    assert result.observations[0].title == "Adopt sidecar DB"
    assert result.decisions[0].observation_type == "decision"
    assert result.changed_files == ("agent/memory_sidecar/store.py",)
    assert result.session_fact is not None
    assert result.session_fact.user_goal == "Build a safe Hermes-native memory sidecar"
    assert any("decisions" in line.lower() or "changed files" in line.lower() for line in result.suggested_follow_ups)

    loaded_checkpoint = store.get_checkpoint("src-1")
    assert loaded_checkpoint is not None
    assert loaded_checkpoint.partial_line == "{\"partial\": true"

    store.close()
