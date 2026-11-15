"""/status's next-turn route lookup never touches the runner-wide pre-agent fallback notice.

``_pre_agent_fallback_notice`` is one turn's one-shot slot: the turn runner reads and clears it to
decide whether a LOCAL session keeps its fallback model. /status resolves the route in a worker
thread while other chats' turns run, so clearing or setting that slot, even briefly, can drop
another turn's notice or hand it one the status lookup made.
"""
from unittest.mock import AsyncMock, patch

import pytest

from tests.gateway.test_status_command import _make_event
from tests.gateway.test_status_model_truth import _runner_with_old_usage


@pytest.mark.asyncio
async def test_status_lookup_leaves_another_turns_fallback_notice_untouched(tmp_path, monkeypatch):
    import gateway.run_turn_prepare as prepare
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    runner, db = _runner_with_old_usage(tmp_path)
    runner._resolve_route_context = AsyncMock(return_value=None)
    runner._pre_agent_fallback_notice = "other chat's turn fell back"
    seen = []
    real_channel_runtime = prepare._resolve_channel_runtime

    def observe(r, *a, **k):  # what a concurrent turn reading the slot mid-lookup would see
        seen.append(r._pre_agent_fallback_notice)
        return real_channel_runtime(r, *a, **k)

    try:
        with patch("gateway.run._load_gateway_config",
                   return_value={"model": {"default": "next-model", "provider": "openrouter"}}), \
                patch("gateway.run._resolve_runtime_agent_kwargs",
                      return_value={"provider": "openrouter", "_fallback_notice": "status lookup fell back"}), \
                patch.object(prepare, "_resolve_channel_runtime", observe):
            out = await runner._handle_status_command(_make_event("/status"))
    finally:
        db.close()
    assert "next-model" in out
    assert seen == ["other chat's turn fell back"]
    assert runner._pre_agent_fallback_notice == "other chat's turn fell back"
