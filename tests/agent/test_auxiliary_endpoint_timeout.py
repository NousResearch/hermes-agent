"""Endpoint-local compression deadlines must not leak across fallback routes."""
from unittest.mock import patch

import pytest

from agent.auxiliary_client import async_call_llm, call_llm
from tests.agent.test_auxiliary_compression_timeout_floor import (
    _client_async, _client_sync, _ok_response, _patches,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("primary_url,fallback_url,task,config,explicit,entry,expected", [
    ("http://localhost:8765/v1", "https://example.invalid/v1", "compression", 60, None, None, (60, 300)),
    ("http://127.0.0.1:8765/v1", "https://example.invalid/v1", "compression", 60, 45, None, (45, 45)),
    ("http://[::1]:8765/v1", "https://example.invalid/v1", "compression", 60, None, 60, (60, 60)),
    ("http://0.0.0.0:8765/v1", "https://example.invalid/v1", "compression", 60, 45, 60, (45, 60)),
    ("https://example.invalid/v1", "http://localhost:8765/v1", "compression", 60, None, None, (300, 60)),
    ("http://localhost:8765/v1", "https://example.invalid/v1", "compression", 450, None, None, (450, 450)),
    ("http://localhost:8765/v1", "https://example.invalid/v1", "title_generation", 60, None, None, (60, 60)),
    ("https://localhost.example.invalid/v1", "https://example.invalid/v1", "compression", 60, None, None, (300, 300)),
])
async def test_fallback_wire_timeout_is_destination_scoped(
    async_mode, primary_url, fallback_url, task, config, explicit, entry, expected,
):
    make_client = _client_async if async_mode else _client_sync
    primary, fallback = make_client(primary_url), make_client(fallback_url)
    primary.chat.completions.create.side_effect = TimeoutError("primary timeout")
    chain_entry = {"provider": "custom", "base_url": fallback_url}
    if entry is not None:
        chain_entry["timeout"] = entry
    p1, p2, p3, p4 = _patches(primary, task_timeout=config)
    with (
        p1, p2, p3, p4,
        patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": [chain_entry]}),
        patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(fallback, "fallback-model", "fallback_chain[0](custom)")),
        patch("agent.auxiliary_client._to_async_client", return_value=(fallback, "fallback-model")),
    ):
        kwargs = dict(task=task, messages=[{"role": "user", "content": "summarise"}], timeout=explicit)
        response = await async_call_llm(**kwargs) if async_mode else call_llm(**kwargs)
    assert response == _ok_response()
    assert (primary.chat.completions.create.call_args.kwargs["timeout"],
            fallback.chat.completions.create.call_args.kwargs["timeout"]) == expected
