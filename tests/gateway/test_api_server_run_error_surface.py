"""Durable API clients receive the same classified failure as other Hermes clients."""

import pytest

from gateway.platforms.api_server_runs import terminal_run_status


@pytest.mark.parametrize(
    ("reason", "layer"), [("auth_permanent", "auth"), ("billing", "billing"), ("rate_limit", "provider")]
)
def test_terminal_run_exposes_classified_failure_without_provider_detail(reason, layer):
    status, fields = terminal_run_status({
        "failed": True, "failure_reason": reason, "failure_retryable": False,
        "error": "provider detail contains private data",
    })
    assert status == "failed"
    assert fields["error_surface"] == {"layer": layer, "code": reason, "retryable": False}
    assert "private data" not in str(fields)


def test_success_does_not_publish_stale_failure_metadata():
    status, fields = terminal_run_status({
        "completed": True, "failure_reason": "auth", "failure_retryable": False,
    })
    assert status == "completed"
    assert "error_surface" not in fields


@pytest.mark.asyncio
async def test_classification_survives_poll_events_and_restart(tmp_path):
    from unittest.mock import MagicMock, patch

    from aiohttp.test_utils import TestClient, TestServer

    from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter, _use_idempotency_db
    from tests.waiting import async_wait_until

    adapter = _make_adapter()
    _use_idempotency_db(adapter, tmp_path / "runs.db")
    expected = {"layer": "billing", "code": "billing", "retryable": False}
    agent = MagicMock()
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    agent.run_conversation.return_value = {
        "failed": True, "failure_reason": "billing", "failure_retryable": False,
    }
    async with TestClient(TestServer(_create_runs_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=agent):
            response = await client.post(
                "/v1/runs", json={"input": "hello"}, headers={"Idempotency-Key": "failure-probe"},
            )
            run_id = (await response.json())["run_id"]
            await async_wait_until(lambda: adapter._run_statuses[run_id]["status"] == "failed")
            assert (await (await client.get(f"/v1/runs/{run_id}")).json())["error_surface"] == expected
            events = await (await client.get(f"/v1/runs/{run_id}/events")).text()
            assert '"error_surface"' in events and '"billing"' in events
    adapter._run_idempotency_store.close()
    restarted = _make_adapter()
    _use_idempotency_db(restarted, tmp_path / "runs.db")
    try:
        async with TestClient(TestServer(_create_runs_app(restarted))) as client:
            assert (await (await client.get(f"/v1/runs/{run_id}")).json())["error_surface"] == expected
    finally:
        restarted._run_idempotency_store.close()
