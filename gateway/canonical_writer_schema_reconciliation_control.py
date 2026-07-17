"""Fixed PostgreSQL control boundary for the one schema reconciliation.

The normal reconciliation login is deliberately powerless.  It owns no
objects and receives no writer or migration-owner membership.  Its complete
database authority is one inert role with CONNECT, control-schema USAGE, and
EXECUTE on the two zero-argument routines named here.

This module contains only mechanical protocol constants and strict receipt
parsers.  It does not choose a migration, interpret intent, or accept SQL,
object identifiers, routine names, or actions from a caller.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from gateway.canonical_writer_db import PostgresProtocolError, QueryResult
from gateway.canonical_writer_schema_reconciliation import (
    CanonicalQuarantineAnchorReceipt,
    CanonicalRelationTruthReceipt,
    CanonicalTruthReceipt,
    EXPECTED_MISSING_HELPER_CATALOG_IDENTITY,
    SchemaReconciliationError,
)


EXECUTOR_ROLE = "canonical_brain_schema_reconciler"
CONTROL_SCHEMA = "canonical_brain_reconciliation"
OBSERVER_ROUTINE = "observe_missing_discord_routeback_helper_v1"
APPLY_ROUTINE = "apply_missing_discord_routeback_helper_v1"
OBSERVER_SIGNATURE = f"{CONTROL_SCHEMA}.{OBSERVER_ROUTINE}()"
APPLY_SIGNATURE = f"{CONTROL_SCHEMA}.{APPLY_ROUTINE}()"
OBSERVER_PROSRC_SHA256 = (
    "47b63aa737d29e1d5b3a54fc824606d91c322a7869118b6f331040e0a3ef96fe"
)
OBSERVER_DEFINITION_SHA256 = (
    "7813ead62d79011f2f2c4e1895405bb35a8edc959e244a14fc22d1ab1be56974"
)
APPLY_PROSRC_SHA256 = (
    "2a28d4700d550bcc8ddc56ea870fc5f669f55a47f9abc7e1993b99b178db1719"
)
APPLY_DEFINITION_SHA256 = (
    "63d6388e50086bf2203bafb7d74291cbec32d04c1f2e05af4f007df4c1e9c8d6"
)
CONTROL_ROUTINE_PROCONFIG = (
    "search_path=pg_catalog, pg_temp",
    "TimeZone=UTC",
    "DateStyle=ISO, YMD",
    "IntervalStyle=postgres",
    "extra_float_digits=3",
    "bytea_output=hex",
    "lock_timeout=15s",
    "statement_timeout=5min",
)

FOUNDATION_RECEIPT_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-control-foundation.v1"
)
OBSERVATION_RECEIPT_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-control-observation.v1"
)
APPLY_RECEIPT_SCHEMA = (
    "muncho-canonical-writer-schema-reconciliation-control-apply.v1"
)

PLAN_GUC = "muncho.schema_reconciliation_plan_sha256"
AUTHORIZED_INTENT_GUC = (
    "muncho.schema_reconciliation_authorized_intent_sha256"
)
TRUTH_RECEIPT_GUC = "muncho.schema_reconciliation_truth_receipt_sha256"
CONTROL_OBSERVATION_GUC = (
    "muncho.schema_reconciliation_control_observation_sha256"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

OBSERVER_CALL_SQL = (
    "SELECT row_count, canonical14_sha256, relation_receipts::text, "
    "quarantine_anchor_receipts::text, observation_sha256 FROM "
    f"{OBSERVER_SIGNATURE}"
)
APPLY_CALL_SQL = (
    "SELECT applied::text, plan_sha256, authorized_intent_sha256, "
    "canonical_truth_receipt_sha256, observation_sha256, "
    f"helper_definition_sha256, receipt_sha256 FROM {APPLY_SIGNATURE}"
)

_OBSERVATION_COLUMNS = (
    "row_count",
    "canonical14_sha256",
    "relation_receipts",
    "quarantine_anchor_receipts",
    "observation_sha256",
)
_APPLY_COLUMNS = (
    "applied",
    "plan_sha256",
    "authorized_intent_sha256",
    "canonical_truth_receipt_sha256",
    "observation_sha256",
    "helper_definition_sha256",
    "receipt_sha256",
)
_OBSERVATION_DOMAIN = (
    "canonical-writer-schema-reconciliation-control-observation-v1"
)
_APPLY_DOMAIN = "canonical-writer-schema-reconciliation-control-apply-v1"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _require_sha256(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SchemaReconciliationError(code)
    return value


@dataclass(frozen=True)
class ControlObservation:
    """Fixed-hash observation returned by the owner-owned observer."""

    truth: CanonicalTruthReceipt
    observation_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.truth, CanonicalTruthReceipt)
            or not isinstance(self.observation_sha256, str)
            or _SHA256.fullmatch(self.observation_sha256) is None
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_control_observation_invalid"
            )

    @property
    def value(self) -> Mapping[str, Any]:
        return {
            "schema": OBSERVATION_RECEIPT_SCHEMA,
            "truth_receipt_sha256": self.truth.sha256,
            "observation_sha256": self.observation_sha256,
        }


@dataclass(frozen=True)
class ControlApplyReceipt:
    """Plan- and authorization-bound receipt for the fixed apply routine."""

    applied: bool
    plan_sha256: str
    authorized_intent_sha256: str
    canonical_truth_receipt_sha256: str
    observation_sha256: str
    helper_definition_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        digests = (
            self.plan_sha256,
            self.authorized_intent_sha256,
            self.canonical_truth_receipt_sha256,
            self.observation_sha256,
            self.helper_definition_sha256,
            self.receipt_sha256,
        )
        if (
            type(self.applied) is not bool
            or any(not isinstance(value, str) for value in digests)
            or any(_SHA256.fullmatch(value) is None for value in digests)
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_control_apply_receipt_invalid"
            )
        expected = _sha256_text(
            "\n".join(
                (
                    _APPLY_DOMAIN,
                    "true" if self.applied else "false",
                    self.plan_sha256,
                    self.authorized_intent_sha256,
                    self.canonical_truth_receipt_sha256,
                    self.observation_sha256,
                    self.helper_definition_sha256,
                )
            )
        )
        if self.receipt_sha256 != expected:
            raise SchemaReconciliationError(
                "schema_reconciliation_control_apply_receipt_invalid"
            )
        if (
            self.helper_definition_sha256
            != EXPECTED_MISSING_HELPER_CATALOG_IDENTITY.definition_sha256
        ):
            raise SchemaReconciliationError(
                "schema_reconciliation_control_apply_receipt_invalid"
            )

    @property
    def value(self) -> Mapping[str, Any]:
        return {
            "schema": APPLY_RECEIPT_SCHEMA,
            "applied": self.applied,
            "plan_sha256": self.plan_sha256,
            "authorized_intent_sha256": self.authorized_intent_sha256,
            "canonical_truth_receipt_sha256": (
                self.canonical_truth_receipt_sha256
            ),
            "observation_sha256": self.observation_sha256,
            "helper_definition_sha256": self.helper_definition_sha256,
            "receipt_sha256": self.receipt_sha256,
        }


def parse_control_observation(result: QueryResult) -> ControlObservation:
    """Validate the exact observer row without accepting canonical data rows."""

    code = "schema_reconciliation_control_observation_invalid"
    if (
        not isinstance(result, QueryResult)
        or result.command_tag.upper() != "SELECT 1"
        or result.columns != _OBSERVATION_COLUMNS
        or len(result.rows) != 1
        or len(result.rows[0]) != len(_OBSERVATION_COLUMNS)
    ):
        raise PostgresProtocolError(code)
    (
        row_count_text,
        canonical14_sha256,
        relations_text,
        quarantine_text,
        observation_sha256,
    ) = result.rows[0]
    if (
        not isinstance(row_count_text, str)
        or not row_count_text.isdigit()
        or not isinstance(relations_text, str)
        or not isinstance(quarantine_text, str)
    ):
        raise SchemaReconciliationError(code)
    _require_sha256(canonical14_sha256, code)
    _require_sha256(observation_sha256, code)
    expected_observation = _sha256_text(
        "\n".join(
            (
                _OBSERVATION_DOMAIN,
                row_count_text,
                canonical14_sha256,
                relations_text,
                quarantine_text,
            )
        )
    )
    if observation_sha256 != expected_observation:
        raise SchemaReconciliationError(code)
    try:
        relations_value = json.loads(relations_text)
        quarantine_value = json.loads(quarantine_text)
        if not isinstance(relations_value, list) or not isinstance(
            quarantine_value, list
        ):
            raise ValueError
        relations = tuple(
            CanonicalRelationTruthReceipt.from_mapping(item)
            for item in relations_value
        )
        quarantine = tuple(
            CanonicalQuarantineAnchorReceipt.from_mapping(item)
            for item in quarantine_value
        )
        truth = CanonicalTruthReceipt(
            row_count=int(row_count_text),
            canonical14_sha256=canonical14_sha256,
            relation_receipts=relations,
            quarantine_anchor_receipts=quarantine,
        )
    except (
        TypeError,
        ValueError,
        json.JSONDecodeError,
        SchemaReconciliationError,
    ) as exc:
        raise SchemaReconciliationError(code) from exc
    return ControlObservation(
        truth=truth,
        observation_sha256=observation_sha256,
    )


def parse_control_apply_receipt(
    result: QueryResult,
    *,
    plan_sha256: str,
    authorized_intent_sha256: str,
    canonical_truth_receipt_sha256: str,
    observation_sha256: str,
) -> ControlApplyReceipt:
    """Validate the exact fixed apply receipt and its outer authorization."""

    code = "schema_reconciliation_control_apply_receipt_invalid"
    expected = tuple(
        _require_sha256(value, code)
        for value in (
            plan_sha256,
            authorized_intent_sha256,
            canonical_truth_receipt_sha256,
            observation_sha256,
        )
    )
    if (
        not isinstance(result, QueryResult)
        or result.command_tag.upper() != "SELECT 1"
        or result.columns != _APPLY_COLUMNS
        or len(result.rows) != 1
        or len(result.rows[0]) != len(_APPLY_COLUMNS)
        or result.rows[0][0] not in {"t", "f", "true", "false"}
    ):
        raise PostgresProtocolError(code)
    row = result.rows[0]
    receipt = ControlApplyReceipt(
        applied=row[0] in {"t", "true"},
        plan_sha256=_require_sha256(row[1], code),
        authorized_intent_sha256=_require_sha256(row[2], code),
        canonical_truth_receipt_sha256=_require_sha256(row[3], code),
        observation_sha256=_require_sha256(row[4], code),
        helper_definition_sha256=_require_sha256(row[5], code),
        receipt_sha256=_require_sha256(row[6], code),
    )
    if (
        receipt.plan_sha256,
        receipt.authorized_intent_sha256,
        receipt.canonical_truth_receipt_sha256,
        receipt.observation_sha256,
    ) != expected:
        raise SchemaReconciliationError(code)
    return receipt


def set_local_hash_sql(name: str, value: str) -> str:
    """Render one of the three caller-set transaction-local digest bindings."""

    if name not in {
        PLAN_GUC,
        AUTHORIZED_INTENT_GUC,
        TRUTH_RECEIPT_GUC,
    }:
        raise SchemaReconciliationError(
            "schema_reconciliation_control_binding_invalid"
        )
    _require_sha256(value, "schema_reconciliation_control_binding_invalid")
    return f"SET LOCAL {name} = '{value}'"


__all__ = [
    "APPLY_CALL_SQL",
    "APPLY_DEFINITION_SHA256",
    "APPLY_PROSRC_SHA256",
    "APPLY_RECEIPT_SCHEMA",
    "APPLY_ROUTINE",
    "APPLY_SIGNATURE",
    "AUTHORIZED_INTENT_GUC",
    "CONTROL_OBSERVATION_GUC",
    "CONTROL_SCHEMA",
    "CONTROL_ROUTINE_PROCONFIG",
    "ControlApplyReceipt",
    "ControlObservation",
    "EXECUTOR_ROLE",
    "FOUNDATION_RECEIPT_SCHEMA",
    "OBSERVER_CALL_SQL",
    "OBSERVER_DEFINITION_SHA256",
    "OBSERVER_PROSRC_SHA256",
    "OBSERVATION_RECEIPT_SCHEMA",
    "OBSERVER_ROUTINE",
    "OBSERVER_SIGNATURE",
    "PLAN_GUC",
    "TRUTH_RECEIPT_GUC",
    "parse_control_apply_receipt",
    "parse_control_observation",
    "set_local_hash_sql",
]
