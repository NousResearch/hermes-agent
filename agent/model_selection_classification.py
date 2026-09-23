"""Host-attested scope, separate from model-proposed routing requirements.

This is same-user governance, not human-presence authentication. Adapters load
records by exact execution identity; model intake cannot supply these records.
"""
from __future__ import annotations

import json

from agent.model_selection_integrity import canonical_json, content_hash
from agent.model_selection_store import _connect
from agent.model_selection_types import RoutingBlocked
from hermes_cli.sqlite_util import transaction

IDENTITY_FIELDS = ("execution_kind", "execution_id", "attempt_id", "slot_id", "role")
RISK_FLAGS = frozenset({"auth", "migration", "concurrency", "persistence", "production", "novel", "ambiguous"})


def scope_hash(requirements: dict) -> str:
    return content_hash({field: requirements.get(field) for field in (
        "task_class", "required_capabilities", "input_tokens", "reserve_tokens", "reasoning", "provenance",
    )})


def execution_identity(requirements: dict) -> dict:
    identity = {field: requirements.get(field, "") for field in IDENTITY_FIELDS}
    if any(not isinstance(value, str) or (not value and field != "slot_id")
           for field, value in identity.items()):
        raise RoutingBlocked("schema_invalid", "classification needs exact execution identity")
    return identity


def _record(row) -> dict | None:
    if row is None:
        return None
    try:
        record = json.loads(row["content_json"])
        valid = (canonical_json(record) == row["content_json"]
                 and content_hash(record) == row["content_hash"]
                 and content_hash(record["execution"]) == row["execution_key"]
                 and record["version"] == row["version"]
                 and record["schema_version"] == 1)
    except (ValueError, TypeError, KeyError) as exc:
        raise RoutingBlocked("stale_or_revoked_decision", "invalid classification record") from exc
    if not valid:
        raise RoutingBlocked("stale_or_revoked_decision", "classification integrity mismatch")
    return record


def get_classification(home, requirements: dict, *, version: int | None = None) -> dict | None:
    key = content_hash(execution_identity(requirements))
    with transaction(_connect(home)) as conn:
        if version is None:
            row = conn.execute(
                "SELECT * FROM routing_classifications WHERE execution_key=? ORDER BY version DESC LIMIT 1",
                (key,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM routing_classifications WHERE execution_key=? AND version=?",
                (key, version),
            ).fetchone()
    record = _record(row)
    if record is not None and record["scope_hash"] != scope_hash(requirements):
        raise RoutingBlocked("stale_or_revoked_decision", "classification scope changed; attest a new version")
    return record


def attest_classification(home, requirements: dict, *, authority: str, attester: dict,
                          complete: bool, risk_flags: list[str], evidence: list[str],
                          expected_version: int) -> dict:
    """Append a CAS-versioned attestation; never rewrite retained execution evidence."""
    identity = execution_identity(requirements)
    fields = {"parent": {"task_id", "run_id", "role"}, "operator": {"approval_ref"}}
    if authority not in fields or not isinstance(attester, dict) or set(attester) != fields[authority]:
        raise RoutingBlocked("schema_invalid", "classification authority provenance is required")
    if any(not isinstance(value, str) or not value.strip() for value in attester.values()):
        raise RoutingBlocked("schema_invalid", "classification attester identity is incomplete")
    if type(complete) is not bool or type(expected_version) is not int or expected_version < 0:
        raise RoutingBlocked("schema_invalid", "classification needs completeness and expected version")
    if not isinstance(risk_flags, list) or any(flag not in RISK_FLAGS for flag in risk_flags):
        raise RoutingBlocked("schema_invalid", "classification risk flags are invalid")
    if not isinstance(evidence, list) or not evidence or any(
        not isinstance(ref, str) or not ref.strip() or len(ref) > 512 for ref in evidence
    ):
        raise RoutingBlocked("schema_invalid", "classification needs bounded scope evidence references")
    key = content_hash(identity)
    with transaction(_connect(home)) as conn:
        previous = _record(conn.execute(
            "SELECT * FROM routing_classifications WHERE execution_key=? ORDER BY version DESC LIMIT 1",
            (key,),
        ).fetchone())
        version = previous["version"] if previous else 0
        if version != expected_version:
            raise RoutingBlocked("stale_or_revoked_decision", "classification version changed")
        if previous and previous["authority"] == "operator" and authority == "parent":
            raise RoutingBlocked("stale_or_revoked_decision", "parent cannot lower operator authority")
        if previous and authority == "parent":
            risk_flags = sorted(set(risk_flags) | set(previous["risk_flags"]))
        record = {
            "schema_version": 1, "version": version + 1, "execution": identity,
            "scope_hash": scope_hash(requirements),
            "authority": authority, "attester": dict(attester), "complete": complete,
            "risk_flags": sorted(set(risk_flags)), "evidence": list(evidence),
        }
        conn.execute(
            "INSERT INTO routing_classifications (execution_key, version, content_json, content_hash) "
            "VALUES (?, ?, ?, ?)",
            (key, record["version"], canonical_json(record), content_hash(record)),
        )
    return record
