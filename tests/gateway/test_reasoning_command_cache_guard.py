"""Gateway ``/reasoning <level>`` on a large cached context confirms before applying: Anthropic
and OpenAI render the effort setting into the cached prefix, so the change is a one-time full-price
re-read (same guard family as the ``/model`` context-cache confirm)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform


def _runner(agent):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._reasoning_config = {"enabled": True, "effort": "medium"}
    runner._cached_agent_for = lambda session_key: agent
    runner._typed_command_prefix_for = lambda platform: "/"
    runner._request_slash_confirm = AsyncMock(return_value="confirm rendered")
    calls = []
    runner._apply_reasoning_selection = (
        lambda session_key, platform_key, value, persist_global=False:
        calls.append((session_key, platform_key, value, persist_global)) or "applied")
    return runner, calls


_EVENT = SimpleNamespace(source=SimpleNamespace(platform=Platform.TELEGRAM))


@pytest.mark.asyncio
async def test_large_context_effort_change_confirms_then_applies_through_the_same_applier():
    agent = SimpleNamespace(model="claude-fable-5-1",
                            context_compressor=SimpleNamespace(last_prompt_tokens=150_000))
    runner, calls = _runner(agent)
    with patch("hermes_cli.config.load_config", side_effect=FileNotFoundError):
        fired, reply = await runner._reasoning_effort_guard_reply(_EVENT, "telegram:c1", "high", False, "telegram")
    assert (fired, reply) == (True, "confirm rendered")
    assert calls == []  # withheld until the user answers
    handler = runner._request_slash_confirm.call_args.kwargs["handler"]
    assert runner._request_slash_confirm.call_args.kwargs["command"] == "reasoning"
    assert "cancelled" in (await handler("cancel")).lower()
    assert calls == []
    assert await handler("once") == "applied"
    assert calls == [("telegram:c1", "telegram", "high", False)]


@pytest.mark.asyncio
async def test_small_context_unchanged_effort_or_toggle_applies_immediately():
    agent = SimpleNamespace(model="claude-fable-5-1",
                            context_compressor=SimpleNamespace(last_prompt_tokens=150_000))
    runner, _ = _runner(agent)
    with patch("hermes_cli.config.load_config", side_effect=FileNotFoundError):
        assert await runner._reasoning_effort_guard_reply(_EVENT, "k", "medium", False, "telegram") == (False, None)
        assert await runner._reasoning_effort_guard_reply(_EVENT, "k", "show", False, "telegram") == (False, None)
        agent.context_compressor.last_prompt_tokens = 2_000
        assert await runner._reasoning_effort_guard_reply(_EVENT, "k", "high", False, "telegram") == (False, None)
    runner._request_slash_confirm.assert_not_called()
