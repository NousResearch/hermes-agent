from types import SimpleNamespace

from agent.beta.orchestrator import _recommended_operations


def test_recommended_operations_deduplicate_exact_critical_actions():
    response = SimpleNamespace(
        recommendation=(
            "Restart PostgreSQL on db-1",
            "Restart PostgreSQL on db-1",
            "Delete stale backup on db-1",
        )
    )

    operations = _recommended_operations("maintain db-1", response)

    assert [operation.action for operation in operations] == [
        "Restart PostgreSQL on db-1",
        "Delete stale backup on db-1",
    ]
    assert len({operation.fingerprint for operation in operations}) == 2
