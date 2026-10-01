"""Gateway ``/model <m> --reasoning <level>``: the effort rides with the pick through the same
applier ``/reasoning`` uses (session override by default, ``agent.reasoning_effort`` on --global)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import Platform
from gateway.slash_commands_model import _ModelSwitchContext


def _runner(save_error=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    calls = {}
    runner._switch_cached_agent_model = lambda *_a, **_k: None
    runner._record_model_switch = AsyncMock(return_value=save_error)  # None = config write succeeded
    runner._model_switch_confirmation = AsyncMock(return_value="switched")
    runner._set_reasoning_override = Mock(side_effect=lambda key, value: calls.setdefault("override", (key, value)))
    runner._apply_reasoning_selection = (
        lambda session_key, platform_key, value, persist_global=False:
        calls.setdefault("applied", (session_key, platform_key, value, persist_global)) and "effort set")
    return runner, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("persist_global,save_error", [(True, None), (False, None), (True, "disk failure")])
async def test_reasoning_flag_applies_after_the_switch_with_the_pick_scope(persist_global, save_error):
    runner, calls = _runner(save_error)
    ctx = _ModelSwitchContext(session_key="telegram:c1", source=None, config_path=None,
                              persist_global=persist_global, reasoning_effort="high")
    result = SimpleNamespace(new_model="m", target_provider="nous")
    source = SimpleNamespace(platform=Platform.TELEGRAM)

    reply = await runner._commit_model_switch(result, ctx, source=source)

    if persist_global and save_error is None:
        # The paired config write already persisted reasoning: update memory only.
        assert "applied" not in calls
        assert calls["override"] == ("telegram:c1", None)
        assert runner._reasoning_config == {"enabled": True, "effort": "high"}
        assert "high" in reply
    else:
        assert calls["applied"] == ("telegram:c1", "telegram", "high", False)
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
