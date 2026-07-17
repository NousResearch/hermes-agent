from __future__ import annotations

import hashlib
import json

import pytest

from gateway import canonical_writer_schema_reconciliation as reconciliation
from gateway import canonical_writer_schema_reconciliation_control as control
from gateway.canonical_writer_db import QueryResult


def _truth() -> reconciliation.CanonicalTruthReceipt:
    return reconciliation.CanonicalTruthReceipt(
        row_count=1,
        canonical14_sha256="a" * 64,
        relation_receipts=tuple(
            reconciliation.CanonicalRelationTruthReceipt(
                relation=relation,
                row_count=1 if index == 0 else 0,
                chunk_count=1 if index == 0 else 0,
                chunk_manifest_sha256=hashlib.sha256(
                    f"relation:{index}".encode()
                ).hexdigest(),
            )
            for index, relation in enumerate(
                reconciliation.CANONICAL_TRUTH_RELATIONS
            )
        ),
        quarantine_anchor_receipts=tuple(
            reconciliation.CanonicalQuarantineAnchorReceipt(
                anchor=anchor,
                object_oid=1_000 + index,
                owner="postgres",
                kind="n" if index == 1 else "r",
                persistence="" if index == 1 else "p",
                acl_sha256=hashlib.sha256(anchor.encode()).hexdigest(),
            )
            for index, anchor in enumerate(
                reconciliation.CANONICAL_QUARANTINE_ANCHORS,
                start=1,
            )
        ),
    )


def _observation_result(*, tamper: str | None = None) -> QueryResult:
    truth = _truth()
    relations = json.dumps(
        truth.value["relation_receipts"],
        sort_keys=True,
        separators=(",", ":"),
    )
    quarantine = json.dumps(
        truth.value["quarantine_anchors"],
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(
        "\n".join(
            (
                "canonical-writer-schema-reconciliation-control-observation-v1",
                "1",
                truth.canonical14_sha256,
                relations,
                quarantine,
            )
        ).encode()
    ).hexdigest()
    row: list[str] = [
        "1",
        truth.canonical14_sha256,
        relations,
        quarantine,
        digest,
    ]
    if tamper == "hash":
        row[-1] = "f" * 64
    elif tamper == "relations":
        row[2] = "[]"
    return QueryResult(
        (
            "row_count",
            "canonical14_sha256",
            "relation_receipts",
            "quarantine_anchor_receipts",
            "observation_sha256",
        ),
        (tuple(row),),
        "SELECT 1",
    )


def _apply_result(
    *,
    plan: str,
    intent: str,
    truth: str,
    observation: str,
    tamper: str | None = None,
) -> QueryResult:
    helper = (
        reconciliation.EXPECTED_MISSING_HELPER_CATALOG_IDENTITY
        .definition_sha256
    )
    receipt = hashlib.sha256(
        "\n".join(
            (
                "canonical-writer-schema-reconciliation-control-apply-v1",
                "true",
                plan,
                intent,
                truth,
                observation,
                helper,
            )
        ).encode()
    ).hexdigest()
    row = ["true", plan, intent, truth, observation, helper, receipt]
    if tamper == "binding":
        row[1] = "9" * 64
    elif tamper == "receipt":
        row[-1] = "8" * 64
    elif tamper == "helper":
        row[5] = "7" * 64
    return QueryResult(
        (
            "applied",
            "plan_sha256",
            "authorized_intent_sha256",
            "canonical_truth_receipt_sha256",
            "observation_sha256",
            "helper_definition_sha256",
            "receipt_sha256",
        ),
        (tuple(row),),
        "SELECT 1",
    )


def test_observation_parser_accepts_exact_fixed_receipt() -> None:
    parsed = control.parse_control_observation(_observation_result())
    assert parsed.truth == _truth()
    assert len(parsed.observation_sha256) == 64


@pytest.mark.parametrize("tamper", ("hash", "relations"))
def test_observation_parser_rejects_hash_or_payload_tamper(tamper: str) -> None:
    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_control_observation_invalid",
    ):
        control.parse_control_observation(_observation_result(tamper=tamper))


def test_apply_parser_accepts_exact_outer_bindings() -> None:
    plan, intent, truth, observation = (character * 64 for character in "abcd")
    parsed = control.parse_control_apply_receipt(
        _apply_result(
            plan=plan,
            intent=intent,
            truth=truth,
            observation=observation,
        ),
        plan_sha256=plan,
        authorized_intent_sha256=intent,
        canonical_truth_receipt_sha256=truth,
        observation_sha256=observation,
    )
    assert parsed.applied is True


@pytest.mark.parametrize("tamper", ("binding", "receipt", "helper"))
def test_apply_parser_rejects_binding_receipt_or_helper_tamper(
    tamper: str,
) -> None:
    plan, intent, truth, observation = (character * 64 for character in "abcd")
    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_control_apply_receipt_invalid",
    ):
        control.parse_control_apply_receipt(
            _apply_result(
                plan=plan,
                intent=intent,
                truth=truth,
                observation=observation,
                tamper=tamper,
            ),
            plan_sha256=plan,
            authorized_intent_sha256=intent,
            canonical_truth_receipt_sha256=truth,
            observation_sha256=observation,
        )


def test_set_local_hash_sql_rejects_unsupported_guc_name() -> None:
    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_control_binding_invalid",
    ):
        control.set_local_hash_sql(
            "muncho.schema_reconciliation_object_name",
            "a" * 64,
        )


@pytest.mark.parametrize(
    "name",
    (control.PLAN_GUC, control.AUTHORIZED_INTENT_GUC, control.TRUTH_RECEIPT_GUC),
)
def test_set_local_hash_sql_accepts_only_fixed_digest_bindings(name: str) -> None:
    assert control.set_local_hash_sql(name, "a" * 64) == (
        f"SET LOCAL {name} = '{'a' * 64}'"
    )
