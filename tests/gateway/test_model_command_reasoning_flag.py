"""Gateway ``/model <m> --reasoning <level>``: the effort rides with the pick through the same
applier ``/reasoning`` uses (session override by default, ``agent.reasoning_effort`` on --global)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.slash_commands_model import _ModelSwitchContext


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    calls = {}
    runner._switch_cached_agent_model = lambda *_a, **_k: None
    runner._record_model_switch = AsyncMock(return_value=None)  # None = config write succeeded
    runner._model_switch_confirmation = AsyncMock(return_value="switched")
    async def _apply_reasoning_selection(session_key, platform_key, value, persist_global=False, **_kw):
        calls.setdefault("applied", (session_key, platform_key, value, persist_global))
        return "effort set"

    runner._apply_reasoning_selection = _apply_reasoning_selection
    return runner, calls


@pytest.mark.asyncio
async def test_reasoning_flag_applies_after_the_switch_with_the_pick_scope():
    runner, calls = _runner()
    ctx = _ModelSwitchContext(session_key="telegram:c1", source=None, config_path=None,
                              persist_global=True, reasoning_effort="high")
    result = SimpleNamespace(new_model="m", target_provider="nous")
    source = SimpleNamespace(platform=Platform.TELEGRAM)

    reply = await runner._commit_model_switch(result, ctx, source=source)

    assert calls["applied"] == ("telegram:c1", "telegram", "high", True)
    assert reply == "switched\neffort set"


@pytest.mark.asyncio
async def test_no_flag_and_once_leave_reasoning_untouched():
    runner, calls = _runner()
    source = SimpleNamespace(platform=Platform.TELEGRAM)
    result = SimpleNamespace(new_model="m", target_provider="nous")
    await runner._commit_model_switch(
        result, _ModelSwitchContext(session_key="k", source=None, config_path=None, persist_global=False),
        source=source)
    await runner._commit_model_switch(
        result, _ModelSwitchContext(session_key="k", source=None, config_path=None, persist_global=False,
                                    one_turn=True, reasoning_effort="high"),
        source=source)
    assert "applied" not in calls
