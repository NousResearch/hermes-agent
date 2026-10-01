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
    runner._apply_reasoning_selection = (
        lambda session_key, platform_key, value, persist_global=False:
        calls.setdefault("applied", (session_key, platform_key, value, persist_global)) and "effort set")
    return runner, calls


@pytest.mark.asyncio
async def test_reasoning_flag_applies_after_the_switch_with_the_pick_scope():
    runner, calls = _runner()
    ctx = _ModelSwitchContext(session_key="telegram:c1", source=None, config_path=None,
                              persist_global=True, reasoning_effort="high")
    result = SimpleNamespace(new_model="m", target_provider="nous", reasoning_config={"enabled": True, "effort": "low"})
    source = SimpleNamespace(platform=Platform.TELEGRAM)

    reply = await runner._commit_model_switch(result, ctx, source=source)

    assert calls["applied"] == ("telegram:c1", "telegram", "high", True)
    assert reply == "switched\neffort set"


@pytest.mark.asyncio
async def test_once_leaves_reasoning_untouched():
    runner, calls = _runner()
    source = SimpleNamespace(platform=Platform.TELEGRAM)
    result = SimpleNamespace(new_model="m", target_provider="nous", reasoning_config={"enabled": True, "effort": "low"})
    await runner._commit_model_switch(
        result, _ModelSwitchContext(session_key="k", source=None, config_path=None, persist_global=False,
                                    one_turn=True, reasoning_effort="high"),
        source=source)
    assert "applied" not in calls
    assert runner._resolve_session_reasoning_config(session_key="k", model="m") == runner._load_reasoning_config("m")


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.DISCORD, Platform.SLACK, Platform.TELEGRAM])
@pytest.mark.parametrize("alias_reasoning", [{"enabled": True, "effort": "medium"}, None, {"enabled": False}])
async def test_normal_switch_replaces_stale_reasoning(platform, alias_reasoning, monkeypatch):
    import gateway.run as gateway_run

    runner, calls = _runner()
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {
        "model": {"default": "old"}, "agent": {"reasoning_effort": "low"}})
    runner._set_session_reasoning_override("k", {"enabled": True, "effort": "xhigh"})
    result = SimpleNamespace(new_model="m", target_provider="nous", reasoning_config=alias_reasoning)
    ctx = _ModelSwitchContext(session_key="k", source=None, config_path=None, persist_global=False)
    await runner._commit_model_switch(result, ctx, source=SimpleNamespace(platform=platform))
    assert runner._resolve_session_reasoning_config(session_key="k", model="m") == (
        alias_reasoning if alias_reasoning is not None else {"enabled": True, "effort": "low"})
    assert "applied" not in calls

