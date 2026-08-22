import hashlib
import hmac
import json
from pathlib import Path

import pytest

from plugins.builder_adapter.auth import signed_material
from plugins.builder_adapter.client import BuilderAdapterClient, OperatorKey
from plugins.builder_adapter.errors import AdapterError


def test_start_builds_minimal_signed_intent():
    seen = {}

    def transport(method, target, body, headers):
        seen.update(method=method, target=target, body=body, headers=headers)
        return 200, json.dumps({"status": "QUEUED", "dispatch_id": "d"}).encode()

    key = OperatorKey("operator-key", b"s" * 32)
    client = BuilderAdapterClient(
        Path("/tmp/adapter.sock"), key, clock=lambda: 1234, transport=transport
    )
    dispatch_id = "00000000-0000-0000-0000-000000000001"
    result = client.start(
        "CYCLE_ONE",
        {"contract_id": "CONTRACT_ONE", "repository_id": "hermes_agent", "revision": 7},
        dispatch_id=dispatch_id,
    )

    payload = json.loads(seen["body"])
    assert result["status"] == "QUEUED"
    assert seen["method"] == "POST"
    assert seen["target"] == "/v1/dispatches"
    assert payload["dispatch_id"] == dispatch_id
    assert payload["expected_cycle_revision"] == 7
    digest = hashlib.sha256(seen["body"]).hexdigest()
    expected = hmac.new(
        key.secret,
        signed_material(
            "POST",
            "/v1/dispatches",
            "1234",
            seen["headers"]["X-Hermes-Nonce"],
            digest,
        ),
        hashlib.sha256,
    ).hexdigest()
    assert seen["headers"]["X-Hermes-Signature"] == expected


def test_status_signs_path_without_query_string():
    seen = {}

    def transport(method, target, body, headers):
        seen.update(method=method, target=target, body=body, headers=headers)
        return 200, b'{"status":"RUNNING"}'

    client = BuilderAdapterClient(
        Path("/tmp/adapter.sock"),
        OperatorKey("operator-key", b"s" * 32),
        clock=lambda: 1234,
        transport=transport,
    )
    client.status("dispatch-1", "CYCLE_ONE")
    assert seen["target"] == "/v1/dispatches/dispatch-1?cycle_id=CYCLE_ONE"
    expected = hmac.new(
        b"s" * 32,
        signed_material(
            "GET",
            "/v1/dispatches/dispatch-1",
            "1234",
            seen["headers"]["X-Hermes-Nonce"],
            hashlib.sha256(b"").hexdigest(),
        ),
        hashlib.sha256,
    ).hexdigest()
    assert seen["headers"]["X-Hermes-Signature"] == expected


def test_error_response_is_safe_and_typed():
    def transport(method, target, body, headers):
        return 400, b'{"errors":[{"code":"CONTRACT_MISMATCH","message":"cycle is not registered","retryable":false}]}'

    client = BuilderAdapterClient(
        Path("/tmp/adapter.sock"), OperatorKey("operator-key", b"s" * 32), transport=transport
    )
    with pytest.raises(AdapterError, match="cycle is not registered") as raised:
        client.status("dispatch-1", "CYCLE_ONE")
    assert raised.value.code == "CONTRACT_MISMATCH"

