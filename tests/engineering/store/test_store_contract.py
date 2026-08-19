from __future__ import annotations

import inspect

from engineering.store.base import (
    EngineeringStore,
    EngineeringStoreConflict,
    EngineeringStoreError,
    EvidenceAlreadyExists,
    EvidenceNotFound,
    WorkflowAlreadyExists,
    WorkflowNotFound,
)


EXPECTED_OPERATIONS = {
    "append_evidence",
    "create_workflow",
    "get_evidence",
    "get_review",
    "get_verification",
    "get_workflow",
    "list_evidence",
    "save_review",
    "save_verification",
    "save_workflow",
}


def test_engineering_store_is_a_capability_protocol() -> None:
    public_methods = {
        name
        for name, value in inspect.getmembers(
            EngineeringStore, predicate=inspect.isfunction
        )
        if not name.startswith("_")
    }

    assert getattr(EngineeringStore, "_is_protocol", False) is True
    assert public_methods == EXPECTED_OPERATIONS


def test_store_contract_has_no_storage_specific_apis() -> None:
    forbidden = {"query", "execute", "transaction", "cursor", "path"}

    assert forbidden.isdisjoint(vars(EngineeringStore))


def test_store_errors_distinguish_missing_and_duplicate_facts() -> None:
    assert issubclass(WorkflowAlreadyExists, EngineeringStoreError)
    assert issubclass(WorkflowNotFound, EngineeringStoreError)
    assert issubclass(EvidenceNotFound, EngineeringStoreError)
    assert issubclass(EvidenceAlreadyExists, EngineeringStoreConflict)
    assert WorkflowAlreadyExists is not WorkflowNotFound
    assert WorkflowNotFound is not EvidenceNotFound


def test_list_evidence_attempt_filter_is_optional() -> None:
    signature = inspect.signature(EngineeringStore.list_evidence)

    assert signature.parameters["attempt"].default is None
