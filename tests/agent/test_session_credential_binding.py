"""Real-SQLite coverage for session credential binding persistence."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.conversation_compression import _adopt_live_compression_child
from hermes_state import SessionDB


_BINDING_KEY = "credential_binding"
_CANDIDATE_A = {
    "provider": "openai-codex",
    "entry_id": "entry-alpha",
    "account_id": "account-alpha",
}
_CANDIDATE_B = {
    "provider": "openai-codex",
    "entry_id": "entry-beta",
    "account_id": "account-beta",
}


def _create_session(db: SessionDB, session_id: str = "session-binding") -> None:
    db.create_session(
        session_id,
        "cli",
        model="test-model",
        model_config={"preserved": "value"},
        system_prompt="",
    )


def _race_bind(db_path: str, start, results, candidate: dict[str, str]) -> None:
    """Spawn target: separate SessionDB connection in a separate process."""
    db = SessionDB(Path(db_path))
    try:
        start.wait(timeout=15)
        results.put(db.get_or_bind_session_credential("session-binding", **candidate))
    finally:
        db.close()


def test_get_or_bind_session_credential_stores_nonsecret_identity_once(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    _create_session(db)

    binding = db.get_or_bind_session_credential("session-binding", **_CANDIDATE_A)

    assert binding == _CANDIDATE_A
    persisted_config = json.loads(db.get_session("session-binding")["model_config"])
    assert persisted_config["preserved"] == "value"
    assert persisted_config[_BINDING_KEY] == _CANDIDATE_A
    assert set(persisted_config[_BINDING_KEY]) == {"provider", "entry_id", "account_id"}
    db.close()


def test_get_or_bind_session_credential_repeats_the_existing_binding(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    _create_session(db)

    first = db.get_or_bind_session_credential("session-binding", **_CANDIDATE_A)
    repeated = db.get_or_bind_session_credential("session-binding", **_CANDIDATE_A)

    assert first == _CANDIDATE_A
    assert repeated == _CANDIDATE_A
    db.close()


def test_get_or_bind_session_credential_returns_existing_conflicting_binding(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    _create_session(db)
    db.get_or_bind_session_credential("session-binding", **_CANDIDATE_A)

    binding = db.get_or_bind_session_credential("session-binding", **_CANDIDATE_B)

    assert binding == _CANDIDATE_A
    assert db.get_or_bind_session_credential("session-binding", **_CANDIDATE_A) == _CANDIDATE_A
    db.close()


@pytest.mark.parametrize("field", ["provider", "entry_id", "account_id"])
def test_get_or_bind_session_credential_rejects_whitespace_padded_identifiers(tmp_path, field):
    """Literal API identifiers must be preserved, not silently normalized."""
    db = SessionDB(tmp_path / "state.db")
    _create_session(db)
    candidate = dict(_CANDIDATE_A)
    candidate[field] = f" {candidate[field]} "

    with pytest.raises(ValueError, match="literal"):
        db.get_or_bind_session_credential("session-binding", **candidate)

    assert json.loads(db.get_session("session-binding")["model_config"]) == {"preserved": "value"}
    db.close()


@pytest.mark.parametrize(
    "stored_config",
    ["{not-json", json.dumps({"credential_binding": {"provider": "openai-codex"}})],
)
def test_get_or_bind_session_credential_fails_closed_for_corrupt_persisted_config(
    tmp_path, stored_config
):
    """Tolerant general config parsing must not permit a silent credential rebind."""
    db = SessionDB(tmp_path / "state.db")
    _create_session(db)
    db._conn.execute(
        "UPDATE sessions SET model_config = ? WHERE id = ?", (stored_config, "session-binding")
    )

    with pytest.raises(ValueError, match="Invalid credential binding"):
        db.get_or_bind_session_credential("session-binding", **_CANDIDATE_A)

    assert db.get_session("session-binding")["model_config"] == stored_config
    db.close()


def test_publish_compression_child_inherits_parent_binding_atomically(tmp_path):
    """A child candidate cannot replace the parent's durable identity at publish time."""
    db = SessionDB(tmp_path / "state.db")
    parent_id, child_id = "binding-parent", "binding-child"
    db.create_session(
        parent_id,
        "cli",
        model="test-model",
        model_config={"parent_unrelated": "keep", _BINDING_KEY: _CANDIDATE_A},
        system_prompt="",
    )
    assert db.try_acquire_compression_lock(parent_id, "test-compressor")

    db.publish_compression_child(
        parent_session_id=parent_id,
        child_session_id=child_id,
        source="cli",
        messages=[{"role": "user", "content": "[CONTEXT COMPACTION] summary"}],
        model="child-model",
        model_config={"child_unrelated": "keep", _BINDING_KEY: _CANDIDATE_B},
        system_prompt="",
        compression_lock_holder="test-compressor",
    )

    child_config = json.loads(db.get_session(child_id)["model_config"])
    assert child_config["child_unrelated"] == "keep"
    assert child_config[_BINDING_KEY] == _CANDIDATE_A
    # Recovery/resume targets the child row, which must resolve the inherited pin.
    assert db.get_or_bind_session_credential(child_id, **_CANDIDATE_B) == _CANDIDATE_A
    db.close()


def test_adopt_live_compression_child_recovers_inherited_binding(tmp_path):
    """The stale-contender recovery alias resolves the published child's durable pin."""
    db = SessionDB(tmp_path / "state.db")
    parent_id, child_id = "adoption-parent", "adoption-child"
    db.create_session(
        parent_id,
        "cli",
        model="test-model",
        model_config={_BINDING_KEY: _CANDIDATE_A},
        system_prompt="",
    )
    assert db.try_acquire_compression_lock(parent_id, "test-compressor")
    db.publish_compression_child(
        parent_session_id=parent_id,
        child_session_id=child_id,
        source="cli",
        messages=[{"role": "user", "content": "[CONTEXT COMPACTION] summary"}],
        model="child-model",
        model_config={_BINDING_KEY: _CANDIDATE_B},
        system_prompt="",
        compression_lock_holder="test-compressor",
    )
    agent = SimpleNamespace(
        session_id=parent_id,
        _session_db_created=False,
        _cached_system_prompt=None,
        _last_flushed_db_idx=0,
        _flushed_db_message_session_id=None,
        _flushed_db_message_ids=set(),
        context_compressor=None,
        _memory_manager=None,
        platform="cli",
    )

    recovered = _adopt_live_compression_child(agent, db, parent_id)

    assert recovered and recovered[-1]["content"] == "[CONTEXT COMPACTION] summary"
    assert agent.session_id == child_id
    assert db.get_or_bind_session_credential(child_id, **_CANDIDATE_B) == _CANDIDATE_A
    db.close()


def test_get_or_bind_session_credential_raises_for_missing_session(tmp_path):
    db = SessionDB(tmp_path / "state.db")

    with pytest.raises(ValueError, match="Session not found: missing-session"):
        db.get_or_bind_session_credential("missing-session", **_CANDIDATE_A)

    db.close()


def test_get_or_bind_session_credential_is_atomic_across_processes(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    _create_session(db)
    db.close()

    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    workers = [
        context.Process(target=_race_bind, args=(str(db_path), start, results, candidate))
        for candidate in (_CANDIDATE_A, _CANDIDATE_B)
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0

    returned = [results.get(timeout=5) for _ in workers]
    assert returned[0] == returned[1]
    assert returned[0] in (_CANDIDATE_A, _CANDIDATE_B)

    db = SessionDB(db_path)
    persisted_config = json.loads(db.get_session("session-binding")["model_config"])
    assert persisted_config[_BINDING_KEY] == returned[0]
    db.close()
