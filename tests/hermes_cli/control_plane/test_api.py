from __future__ import annotations

import json

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest
import pytest_asyncio

from hermes_cli._parser import build_top_level_parser
from hermes_cli.control_plane.api import router
from hermes_cli.control_plane.nodes import NodeRegistry
from hermes_cli.subcommands.harness import build_harness_parser


def _parser():
    top, subparsers, _ = build_top_level_parser()
    build_harness_parser(subparsers)
    return top


@pytest_asyncio.fixture
async def client():
    api = FastAPI()
    api.include_router(router)
    async with AsyncClient(
        transport=ASGITransport(app=api), base_url="http://test"
    ) as test_client:
        yield test_client


def _enrollment(**overrides):
    payload = {
        "enrollment_key": "request-1",
        "node_id": "node-1",
        "role": "worker",
        "owner": "ops",
        "actor": "operator:alice",
        "capabilities": {"os": "linux"},
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_api_and_cli_share_the_authoritative_registry(
    tmp_path, monkeypatch, capsys, client
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    enrolled = await client.post("/api/control-plane/v1/nodes", json=_enrollment())
    assert enrolled.status_code == 200
    issuance = enrolled.json()
    assert issuance["node"]["state"] == "enrolled"
    assert issuance["credential"]

    args = _parser().parse_args([
        "harness",
        "nodes",
        "list",
        "--state",
        "enrolled",
    ])
    args.func(args)
    assert json.loads(capsys.readouterr().out) == [issuance["node"]]

    args = _parser().parse_args([
        "harness",
        "nodes",
        "transition",
        "node-1",
        "active",
        "--actor",
        "operator:bob",
        "--expected-revision",
        "1",
        "--reason",
        "ready",
    ])
    args.func(args)
    cli_transition = json.loads(capsys.readouterr().out)

    shown = await client.get("/api/control-plane/v1/nodes/node-1")
    assert shown.status_code == 200
    assert shown.json() == cli_transition

    retired = await client.post(
        "/api/control-plane/v1/nodes/node-1/transitions",
        json={
            "state": "retired",
            "actor": "operator:alice",
            "expected_revision": 2,
            "reason": "decommissioned",
        },
    )
    assert retired.status_code == 200
    assert retired.json()["credential_status"] == "revoked"
    assert NodeRegistry().get("node-1").state == "retired"
    rejected = await client.post(
        "/api/control-plane/v1/nodes/node-1/authenticate",
        json={"credential": issuance["credential"]},
    )
    assert rejected.status_code == 401
    rejected_observation = await client.post(
        "/api/control-plane/v1/nodes/node-1/observations",
        json={
            "credential": issuance["credential"],
            "schema_version": 1,
            "report_sequence": 1,
            "observed_at": 100,
            "health_state": "healthy",
            "capabilities": {},
        },
    )
    assert rejected_observation.status_code == 401

    args = _parser().parse_args([
        "harness",
        "nodes",
        "history",
        "node-1",
    ])
    args.func(args)
    cli_history = json.loads(capsys.readouterr().out)
    api_history = await client.get("/api/control-plane/v1/nodes/node-1/history")
    assert api_history.json()["events"] == cli_history
    audit = await client.get("/api/control-plane/v1/audit")
    assert audit.json() == {"valid": True}


@pytest.mark.asyncio
async def test_audit_api_detects_deleted_tail_and_all_events(
    tmp_path, monkeypatch, client
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    registry = NodeRegistry()
    registry.enroll(**_enrollment())

    with registry.connect() as conn:
        conn.execute(
            "DELETE FROM managed_node_events WHERE sequence = "
            "(SELECT MAX(sequence) FROM managed_node_events)"
        )
    assert (await client.get("/api/control-plane/v1/audit")).json() == {"valid": False}

    with registry.connect() as conn:
        conn.execute("DELETE FROM managed_node_events")
    assert (await client.get("/api/control-plane/v1/audit")).json() == {"valid": False}


@pytest.mark.asyncio
async def test_audit_api_reports_corrupt_details_json_as_invalid(
    tmp_path, monkeypatch, client
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    registry = NodeRegistry()
    registry.enroll(**_enrollment())

    with registry.connect() as conn:
        conn.execute(
            "UPDATE managed_node_events SET details_json = ? WHERE sequence = 1",
            ("{not-json",),
        )

    audit = await client.get("/api/control-plane/v1/audit")

    assert audit.status_code == 200
    assert audit.json() == {"valid": False}


@pytest.mark.asyncio
async def test_api_contracts_and_lifecycle_error_mappings(
    tmp_path, monkeypatch, client
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    first = await client.post("/api/control-plane/v1/nodes", json=_enrollment())
    retry = await client.post("/api/control-plane/v1/nodes", json=_enrollment())
    assert retry.status_code == 200
    assert retry.json()["node"] == first.json()["node"]
    assert retry.json()["credential"] is None

    conflict = await client.post(
        "/api/control-plane/v1/nodes",
        json=_enrollment(owner="another-team"),
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "enrollment_conflict"

    duplicate_id = await client.post(
        "/api/control-plane/v1/nodes",
        json=_enrollment(enrollment_key="request-2"),
    )
    assert duplicate_id.status_code == 409
    assert duplicate_id.json()["error"]["code"] == "enrollment_conflict"

    invalid = await client.post(
        "/api/control-plane/v1/nodes/node-1/transitions",
        json={
            "state": "recovering",
            "actor": "operator:alice",
            "expected_revision": 1,
            "reason": "skip quarantine",
        },
    )
    assert invalid.status_code == 409
    assert invalid.json()["error"]["code"] == "invalid_transition"

    activated = await client.post(
        "/api/control-plane/v1/nodes/node-1/transitions",
        json={
            "state": "active",
            "actor": "operator:alice",
            "expected_revision": 1,
            "reason": "ready",
        },
    )
    assert activated.status_code == 200

    stale = await client.post(
        "/api/control-plane/v1/nodes/node-1/transitions",
        json={
            "state": "retired",
            "actor": "operator:alice",
            "expected_revision": 1,
            "reason": "stale",
        },
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "revision_conflict"

    listing = await client.get(
        "/api/control-plane/v1/nodes", params={"state": "active"}
    )
    assert listing.status_code == 200
    assert listing.json()["nodes"] == [activated.json()]

    bad_filter = await client.get(
        "/api/control-plane/v1/nodes", params={"state": "unknown"}
    )
    assert bad_filter.status_code == 400
    assert bad_filter.json()["error"]["code"] == "invalid_state"

    for path in (
        "/api/control-plane/v1/nodes/missing",
        "/api/control-plane/v1/nodes/missing/history",
    ):
        missing = await client.get(path)
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "node_not_found"


@pytest.mark.asyncio
async def test_api_authentication_rotation_revocation_and_non_disclosure(
    tmp_path, monkeypatch, client
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    enrolled = await client.post("/api/control-plane/v1/nodes", json=_enrollment())
    issuance = enrolled.json()
    raw = issuance["credential"]

    authenticated = await client.post(
        "/api/control-plane/v1/nodes/node-1/authenticate",
        json={"credential": raw},
    )
    assert authenticated.status_code == 200
    assert authenticated.json() == {"authenticated": True}

    rejected = await client.post(
        "/api/control-plane/v1/nodes/node-1/authenticate",
        json={"credential": f"{raw}x"},
    )
    assert rejected.status_code == 401
    assert rejected.json()["error"]["code"] == "invalid_node_credential"

    rotated = await client.post(
        "/api/control-plane/v1/nodes/node-1/credential/rotate",
        json={"actor": "operator:bob", "expected_credential_revision": 1},
    )
    assert rotated.status_code == 200
    replacement = rotated.json()["credential"]
    assert replacement and replacement != raw

    stale = await client.post(
        "/api/control-plane/v1/nodes/node-1/credential/revoke",
        json={"actor": "operator:bob", "expected_credential_revision": 1},
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "credential_revision_conflict"

    revoked = await client.post(
        "/api/control-plane/v1/nodes/node-1/credential/revoke",
        json={"actor": "operator:bob", "expected_credential_revision": 2},
    )
    assert revoked.status_code == 200
    assert revoked.json()["credential_status"] == "revoked"

    rejected = await client.post(
        "/api/control-plane/v1/nodes/node-1/authenticate",
        json={"credential": replacement},
    )
    assert rejected.status_code == 401

    shown = await client.get("/api/control-plane/v1/nodes/node-1")
    history = await client.get("/api/control-plane/v1/nodes/node-1/history")
    audit = await client.get("/api/control-plane/v1/audit")
    public_output = json.dumps([shown.json(), history.json(), audit.json()])
    assert raw not in public_output
    assert replacement not in public_output
    assert audit.json() == {"valid": True}


@pytest.mark.asyncio
async def test_observation_policy_reconciliation_api_cli_consistency(
    tmp_path, monkeypatch, capsys, client
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    enrolled = await client.post("/api/control-plane/v1/nodes", json=_enrollment())
    credential = enrolled.json()["credential"]

    observed = await client.post(
        "/api/control-plane/v1/nodes/node-1/observations",
        json={
            "credential": credential,
            "schema_version": 1,
            "report_sequence": 1,
            "observed_at": 100,
            "health_state": "healthy",
            "capabilities": {"os": "linux", "gpu": False},
        },
    )
    assert observed.status_code == 200
    assert "credential" not in observed.json()

    args = _parser().parse_args(["harness", "nodes", "observation", "node-1"])
    args.func(args)
    assert json.loads(capsys.readouterr().out) == observed.json()

    policy = await client.put(
        "/api/control-plane/v1/nodes/node-1/policy",
        json={
            "schema_version": 1,
            "desired_health_state": "healthy",
            "capabilities": {"os": "linux"},
            "expected_revision": 0,
            "actor": "operator:alice",
        },
    )
    assert policy.status_code == 200
    args = _parser().parse_args(["harness", "nodes", "policy", "show", "node-1"])
    args.func(args)
    assert json.loads(capsys.readouterr().out) == policy.json()

    reconciled = await client.get("/api/control-plane/v1/nodes/node-1/reconciliation")
    assert reconciled.status_code == 200
    assert reconciled.json()["in_sync"] is True
    args = _parser().parse_args(["harness", "nodes", "reconcile", "node-1"])
    args.func(args)
    assert json.loads(capsys.readouterr().out) == reconciled.json()

    history = await client.get("/api/control-plane/v1/nodes/node-1/history")
    public_output = json.dumps([
        observed.json(),
        policy.json(),
        reconciled.json(),
        history.json(),
    ])
    assert credential not in public_output
    assert (await client.get("/api/control-plane/v1/audit")).json() == {"valid": True}


@pytest.mark.asyncio
async def test_observation_and_policy_stable_error_mappings(
    tmp_path, monkeypatch, client
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    enrolled = await client.post("/api/control-plane/v1/nodes", json=_enrollment())
    credential = enrolled.json()["credential"]
    body = {
        "credential": credential,
        "schema_version": 1,
        "report_sequence": 1,
        "observed_at": 100,
        "health_state": "healthy",
        "capabilities": {},
    }
    wrong = await client.post(
        "/api/control-plane/v1/nodes/node-1/observations",
        json={**body, "credential": "wrong"},
    )
    assert wrong.status_code == 401
    assert wrong.json()["error"]["code"] == "invalid_node_credential"
    unknown = await client.post(
        "/api/control-plane/v1/nodes/unknown/observations", json=body
    )
    assert unknown.status_code == wrong.status_code
    assert unknown.json() == wrong.json()

    accepted = await client.post(
        "/api/control-plane/v1/nodes/node-1/observations", json=body
    )
    assert accepted.status_code == 200
    replay = await client.post(
        "/api/control-plane/v1/nodes/node-1/observations", json=body
    )
    assert replay.status_code == 409
    assert replay.json()["error"]["code"] == "report_sequence_conflict"

    invalid = await client.post(
        "/api/control-plane/v1/nodes/node-1/observations",
        json={**body, "schema_version": 2, "report_sequence": 2},
    )
    assert invalid.status_code == 422

    policy_body = {
        "schema_version": 1,
        "desired_health_state": "healthy",
        "capabilities": {},
        "expected_revision": 0,
        "actor": "operator:alice",
    }
    assert (
        await client.put("/api/control-plane/v1/nodes/node-1/policy", json=policy_body)
    ).status_code == 200
    stale = await client.put(
        "/api/control-plane/v1/nodes/node-1/policy", json=policy_body
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "policy_revision_conflict"

    missing = await client.get("/api/control-plane/v1/nodes/missing/reconciliation")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "node_not_found"
