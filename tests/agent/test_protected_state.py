"""Pure contract tests for deterministic protected state values."""
from __future__ import annotations

import json

import pytest

from agent.protected_state import (
    CaptureStatus,
    ContractValidationError,
    ProtectedBlock,
    ProtectedFact,
    ProvenancePointer,
    SourceIdentity,
    Supersession,
    canonical_json,
    parse_canonical_json,
    sha256_hex,
)


PROVENANCE = {"session_id": "session-1", "message_id": "message-1"}
SOURCE = {"source_type": "runtime", "source_id": "run-1"}


def fact_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "schema_version": "protected-fact-v1",
        "fact_kind": "task_state",
        "capture_status": "CAPTURED",
        "value": "complete",
        "provenance": PROVENANCE,
        "source_identity": SOURCE,
    }
    data.update(overrides)
    return data


def test_canonical_json_is_stable_and_sha256_is_over_canonical_bytes() -> None:
    assert canonical_json({"b": 2, "a": "é"}) == '{"a":"é","b":2}'
    assert canonical_json({"a": "é"}) == canonical_json({"a": "é"})
    assert sha256_hex({"b": 2, "a": "é"}) == sha256_hex({"a": "é", "b": 2})


def test_json_parser_rejects_duplicate_keys_and_non_finite_constants() -> None:
    with pytest.raises(ContractValidationError, match="duplicate key"):
        parse_canonical_json('{"a": 1, "a": 2}')
    with pytest.raises(ContractValidationError, match="non-finite"):
        parse_canonical_json('{"a": NaN}')
    with pytest.raises(ContractValidationError, match="non-finite"):
        canonical_json({"a": float("inf")})


def test_provenance_and_source_identity_are_strict_and_round_trip() -> None:
    pointer = ProvenancePointer.from_dict(
        {
            **PROVENANCE,
            "tool_call_id": "tool-1",
            "tool_name": "terminal",
            "parent_session_id": "root",
        }
    )
    assert pointer.to_dict()["tool_name"] == "terminal"
    assert SourceIdentity.from_dict(SOURCE).to_dict() == SOURCE
    with pytest.raises(ContractValidationError, match="unknown field"):
        ProvenancePointer.from_dict({**PROVENANCE, "extra": True})
    with pytest.raises(ContractValidationError, match="missing field"):
        SourceIdentity.from_dict({"source_type": "runtime"})


def test_protected_fact_rejects_unknown_fields_and_has_stable_identity() -> None:
    fact = ProtectedFact.from_dict(fact_data())
    reordered = ProtectedFact.from_dict(
        {
            "source_identity": SOURCE,
            "value": "complete",
            "capture_status": "CAPTURED",
            "schema_version": "protected-fact-v1",
            "provenance": PROVENANCE,
            "fact_kind": "task_state",
        }
    )
    assert fact.fact_id == reordered.fact_id
    assert fact.to_dict() == fact_data()
    with pytest.raises(ContractValidationError, match="unknown field"):
        ProtectedFact.from_dict({**fact_data(), "fact_id": fact.fact_id})


def test_pointer_only_fact_may_have_null_value_but_status_is_explicit() -> None:
    fact = ProtectedFact.from_dict(
        fact_data(capture_status="POINTER_ONLY", value=None)
    )
    assert fact.capture_status is CaptureStatus.POINTER_ONLY
    assert fact.value is None


def test_protected_fact_rejects_non_json_values_and_invalid_schema() -> None:
    with pytest.raises(ContractValidationError, match="non-finite"):
        ProtectedFact.from_dict(fact_data(value={"bad": float("nan")}))
    with pytest.raises(ContractValidationError, match="schema_version"):
        ProtectedFact.from_dict(fact_data(schema_version="protected-fact-v2"))
    with pytest.raises(ContractValidationError, match="non-empty"):
        ProtectedFact.from_dict(fact_data(fact_kind=""))


def test_direct_contract_construction_is_validated_and_nested_values_are_immutable() -> None:
    pointer = ProvenancePointer.from_dict(PROVENANCE)
    source = SourceIdentity.from_dict(SOURCE)
    with pytest.raises(ContractValidationError, match="non-empty"):
        ProtectedFact("", CaptureStatus.CAPTURED, {}, pointer, source)

    raw_value = {"nested": [1]}
    fact = ProtectedFact(
        "task_state", CaptureStatus.CAPTURED, raw_value, pointer, source
    )
    raw_value["nested"].append(2)  # type: ignore[union-attr]
    assert fact.to_dict()["value"] == {"nested": [1]}
    with pytest.raises(TypeError):
        fact.value["new"] = "mutation"  # type: ignore[index]


def test_supersession_requires_explicit_authority_and_transition_provenance() -> None:
    old = ProtectedFact.from_dict(fact_data(value="pending"))
    new = ProtectedFact.from_dict(
        fact_data(value="complete", provenance={"session_id": "session-1", "message_id": "message-2"})
    )
    supersession = Supersession.from_dict(
        {
            "schema_version": "protected-supersession-v1",
            "old_fact_id": old.fact_id,
            "new_fact_id": new.fact_id,
            "new_provenance": new.provenance.to_dict(),
            "authority_ref": "runtime:completion",
            "ordering": 1,
        }
    )
    assert supersession.new_fact_id == new.fact_id
    with pytest.raises(ContractValidationError, match="self"):
        Supersession.from_dict(
            {
                **supersession.to_dict(),
                "new_fact_id": old.fact_id,
            }
        )
    with pytest.raises(ContractValidationError, match="non-empty"):
        Supersession(
            old.fact_id,
            new.fact_id,
            new.provenance,
            "",
            1,
        )


def test_direct_block_construction_rejects_duplicate_or_dangling_transitions() -> None:
    old = ProtectedFact.from_dict(fact_data(value="pending"))
    new = ProtectedFact.from_dict(
        fact_data(value="complete", provenance={"session_id": "session-1", "message_id": "message-2"})
    )
    transition = Supersession.from_dict(
        {
            "schema_version": "protected-supersession-v1",
            "old_fact_id": old.fact_id,
            "new_fact_id": new.fact_id,
            "new_provenance": new.provenance.to_dict(),
            "authority_ref": "runtime:completion",
            "ordering": 1,
        }
    )
    with pytest.raises(ContractValidationError, match="duplicate fact"):
        ProtectedBlock((new, new))
    with pytest.raises(ContractValidationError, match="target"):
        ProtectedBlock((old,), (transition,))


def test_protected_block_normalizes_sequence_inputs_and_rejects_invalid_elements() -> None:
    fact = ProtectedFact.from_dict(fact_data())
    from_list = ProtectedBlock([fact], [])
    from_tuple = ProtectedBlock((fact,), ())

    assert isinstance(from_list.facts, tuple)
    assert isinstance(from_list.supersessions, tuple)
    assert from_list.to_dict() == from_tuple.to_dict()
    with pytest.raises(ContractValidationError, match="invalid fact"):
        ProtectedBlock([object()])  # type: ignore[list-item]


def test_protected_block_is_deterministic_and_rejects_duplicate_facts() -> None:
    first = ProtectedFact.from_dict(fact_data())
    second = ProtectedFact.from_dict(
        fact_data(
            value="complete",
            provenance={"session_id": "session-1", "message_id": "message-2"},
        )
    )
    block = ProtectedBlock.from_dict(
        {
            "schema_version": "protected-block-v1",
            "facts": [first.to_dict(), second.to_dict()],
            "supersessions": [],
        }
    )
    reversed_block = ProtectedBlock.from_dict(
        {
            "schema_version": "protected-block-v1",
            "facts": [second.to_dict(), first.to_dict()],
            "supersessions": [],
        }
    )
    assert block.block_id == reversed_block.block_id
    with pytest.raises(ContractValidationError, match="duplicate fact"):
        ProtectedBlock.from_dict(
            {
                "schema_version": "protected-block-v1",
                "facts": [first.to_dict(), first.to_dict()],
                "supersessions": [],
            }
        )


def test_protected_block_requires_supersession_target_in_current_block() -> None:
    old = ProtectedFact.from_dict(fact_data(value="pending"))
    new = ProtectedFact.from_dict(
        fact_data(value="complete", provenance={"session_id": "session-1", "message_id": "message-2"})
    )
    supersession = Supersession.from_dict(
        {
            "schema_version": "protected-supersession-v1",
            "old_fact_id": old.fact_id,
            "new_fact_id": new.fact_id,
            "new_provenance": new.provenance.to_dict(),
            "authority_ref": "runtime:completion",
            "ordering": 1,
        }
    )
    block = ProtectedBlock.from_dict(
        {
            "schema_version": "protected-block-v1",
            "facts": [new.to_dict()],
            "supersessions": [supersession.to_dict()],
        }
    )
    assert block.block_id.startswith("pb1_")
    with pytest.raises(ContractValidationError, match="target"):
        ProtectedBlock.from_dict(
            {
                "schema_version": "protected-block-v1",
                "facts": [old.to_dict()],
                "supersessions": [supersession.to_dict()],
            }
        )


def test_contract_dicts_are_json_serializable() -> None:
    fact = ProtectedFact.from_dict(fact_data())
    block = ProtectedBlock.from_dict(
        {
            "schema_version": "protected-block-v1",
            "facts": [fact.to_dict()],
            "supersessions": [],
        }
    )
    json.dumps(block.to_dict(), ensure_ascii=False, sort_keys=True)
