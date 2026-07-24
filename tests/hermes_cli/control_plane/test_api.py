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
    assert enrolled.json()["state"] == "enrolled"

    args = _parser().parse_args([
        "harness",
        "nodes",
        "list",
        "--state",
        "enrolled",
    ])
    args.func(args)
    assert json.loads(capsys.readouterr().out) == [enrolled.json()]

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
    assert NodeRegistry().get("node-1").state == "retired"

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
async def test_api_contracts_and_lifecycle_error_mappings(
    tmp_path, monkeypatch, client
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    first = await client.post("/api/control-plane/v1/nodes", json=_enrollment())
    retry = await client.post("/api/control-plane/v1/nodes", json=_enrollment())
    assert retry.status_code == 200
    assert retry.json() == first.json()

    conflict = await client.post(
        "/api/control-plane/v1/nodes",
        json=_enrollment(owner="another-team"),
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "enrollment_conflict"

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
