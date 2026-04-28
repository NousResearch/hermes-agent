from datetime import UTC, datetime

from agent.runtime.state import (
    AgentRuntimeState,
    ModelRuntimeState,
    RuntimeMutation,
    RuntimeMutationRisk,
    RuntimeMutationType,
    RuntimeSessionState,
    ToolRuntimeState,
)


def test_runtime_session_state_round_trips_json():
    state = RuntimeSessionState(
        session_id="session-1",
        platform="telegram",
        agent=AgentRuntimeState(agent_id="moss", display_name="Moss"),
        model=ModelRuntimeState(provider="openai-codex", model="gpt-5.5"),
        tools=[ToolRuntimeState(name="read_file", toolset="file", enabled=True)],
        messages=[{"role": "user", "content": "ping"}],
        metadata={"task": "capability-runtime"},
        created_at=datetime(2026, 4, 27, 11, 0, tzinfo=UTC),
        updated_at=datetime(2026, 4, 27, 11, 1, tzinfo=UTC),
    )

    restored = RuntimeSessionState.model_validate_json(state.model_dump_json())

    assert restored.session_id == "session-1"
    assert restored.agent.agent_id == "moss"
    assert restored.model.model == "gpt-5.5"
    assert restored.tools[0].name == "read_file"
    assert restored.metadata["task"] == "capability-runtime"


def test_runtime_mutation_carries_validator_and_rollback_hint():
    mutation = RuntimeMutation(
        mutation_id="mut-1",
        mutation_type=RuntimeMutationType.SKILL_PATCH,
        scope="skills/devops/hermes-core-backup/SKILL.md",
        rationale="Capture repeated backup-check failure mode.",
        risk=RuntimeMutationRisk.LOW,
        validator="skill_view hermes-core-backup succeeds",
        rollback_hint="revert skill patch",
        payload={"old": "x", "new": "y"},
    )

    restored = RuntimeMutation.model_validate_json(mutation.model_dump_json())

    assert restored.mutation_type is RuntimeMutationType.SKILL_PATCH
    assert restored.risk is RuntimeMutationRisk.LOW
    assert restored.validator.startswith("skill_view")
    assert restored.rollback_hint == "revert skill patch"
