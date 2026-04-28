from datetime import UTC, datetime
from pathlib import Path

from agent.runtime.state import AgentRuntimeState, ModelRuntimeState, RuntimeSessionState
from agent.runtime.state_store import RuntimeStateStore


def _state(session_id: str) -> RuntimeSessionState:
    return RuntimeSessionState(
        session_id=session_id,
        platform="telegram",
        agent=AgentRuntimeState(agent_id="moss"),
        model=ModelRuntimeState(provider="openai-codex", model="gpt-5.5"),
        created_at=datetime(2026, 4, 27, 11, 0, tzinfo=UTC),
        updated_at=datetime(2026, 4, 27, 11, 0, tzinfo=UTC),
    )


def test_runtime_state_store_appends_and_reads_latest(tmp_path: Path):
    store = RuntimeStateStore(tmp_path)

    first = store.append(_state("s1"), step_id="step-1", reason="initial")
    second = store.append(_state("s2"), step_id="step-2", reason="updated")

    assert first.checkpoint_id != second.checkpoint_id
    assert store.latest().checkpoint_id == second.checkpoint_id
    assert store.latest().state.session_id == "s2"
    assert [item.step_id for item in store.list()] == ["step-1", "step-2"]


def test_runtime_state_store_reads_checkpoint_by_id(tmp_path: Path):
    store = RuntimeStateStore(tmp_path)
    checkpoint = store.append(_state("s1"), step_id="step-1", reason="initial")

    restored = store.read(checkpoint.checkpoint_id)

    assert restored is not None
    assert restored.checkpoint_id == checkpoint.checkpoint_id
    assert restored.state.session_id == "s1"


def test_runtime_state_store_index_survives_corrupt_trailing_jsonl_line(tmp_path: Path):
    store = RuntimeStateStore(tmp_path)
    checkpoint = store.append(_state("s1"), step_id="step-1", reason="initial")
    (tmp_path / "runtime-state.jsonl").write_text(
        (tmp_path / "runtime-state.jsonl").read_text(encoding="utf-8") + "{not-json\n",
        encoding="utf-8",
    )

    assert store.latest().checkpoint_id == checkpoint.checkpoint_id
    assert store.read(checkpoint.checkpoint_id).state.session_id == "s1"


def test_runtime_state_store_returns_none_when_empty(tmp_path: Path):
    store = RuntimeStateStore(tmp_path)

    assert store.latest() is None
    assert store.read("missing") is None
    assert store.list() == []
