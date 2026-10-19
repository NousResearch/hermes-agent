"""Canonical API refusals keep their retry semantics on the wire: a retiring profile's
admission is a retryable 503 (Retry-After), a malformed control body is 400, and a
generation/ownership refusal stays 409."""
import pytest

from tests.gateway.test_api_unknown_resolution import _AUTH_HEADERS, _close_fixture, _restarted_api_runs


@pytest.mark.asyncio
async def test_store_refusal_reasons_map_to_retryable_or_client_status(tmp_path, monkeypatch):
    client, adapters, store, owner, unknown, _follower, _executed, tasks = (
        await _restarted_api_runs(tmp_path, monkeypatch))
    path = "/v1/runs/" + unknown["request_id"] + "/resolve-unknown"
    body = {"admission_id": unknown["admission_id"], "execution_generation": unknown["generation"]}
    try:
        cases = (
            ("POST", path, body | {"extra": True}, 400, "invalid_params"),
            ("POST", path, body | {"execution_generation": unknown["generation"] + 1}, 409, "stale_generation"),
            ("POST", "/v1/runs/" + unknown["request_id"] + "/stop", None, 409, "unknown_execution"),
        )
        for method, url, payload, status, code in cases:
            response = await client.request(method, url, json=payload, headers=_AUTH_HEADERS)
            assert (response.status, (await response.json())["error"]["code"]) == (status, code), url
        # A retiring (hot-unserved) profile refuses after the entry drain check: retry later, not a conflict.
        owner.retiring = True
        refused = await client.post("/v1/runs", json={"input": "LATER", "session_id": "api-session"},
                                    headers={**_AUTH_HEADERS, "Idempotency-Key": "retiring-run"})
        assert refused.status == 503, await refused.text()
        assert (await refused.json())["error"]["code"] == "runtime_draining"
        assert refused.headers.get("Retry-After")
    finally:
        owner.retiring = False
        await _close_fixture(client, adapters, store, tasks)
