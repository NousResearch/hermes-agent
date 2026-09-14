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
    runner._record_model_switch = AsyncMock()
    runner._model_switch_confirmation = AsyncMock(return_value="switched")
    runner._claim_one_turn_reasoning_restore = lambda *_a, **_k: True
    runner._apply_reasoning_selection = (
        lambda session_key, platform_key, value, persist_global=False, temporary=False:
        calls.setdefault(
            "applied", (session_key, platform_key, value, persist_global, temporary)
        ) and "effort set")
    return runner, calls


@pytest.mark.asyncio
async def test_reasoning_flag_applies_after_the_switch_with_the_pick_scope():
    runner, calls = _runner()
    ctx = _ModelSwitchContext(session_key="telegram:c1", source=None, config_path=None,
                              persist_global=True, reasoning_effort="high")
    result = SimpleNamespace(new_model="m", target_provider="nous")
    source = SimpleNamespace(platform=Platform.TELEGRAM)

    reply = await runner._commit_model_switch(result, ctx, source=source)

    assert calls["applied"] == ("telegram:c1", "telegram", "high", True, False)
    assert reply == "switched\neffort set"


@pytest.mark.asyncio
async def test_once_applies_reasoning_for_the_next_turn_without_global_persistence():
    runner, calls = _runner()
    source = SimpleNamespace(platform=Platform.TELEGRAM)
    result = SimpleNamespace(new_model="m", target_provider="nous")
    await runner._commit_model_switch(
        result, _ModelSwitchContext(session_key="k", source=None, config_path=None, persist_global=False),
        source=source)
    assert "applied" not in calls

    await runner._commit_model_switch(
        result, _ModelSwitchContext(session_key="k", source=None, config_path=None, persist_global=False,
                                    one_turn=True, reasoning_effort="high"),
        source=source)

    assert calls["applied"] == ("k", "telegram", "high", False, True)
