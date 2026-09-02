"""Every review path hands the fork a snapshot that cannot alias the live transcript.

``AIAgent._spawn_background_review`` is the single chokepoint the automatic
post-turn review, the idle-queue deferral and both explicit ``/refine`` entry
points (CLI mixin + gateway slash command) go through; it clones the snapshot
structurally there. A shallow ``list()`` would share the nested
``tool_calls`` / ``content`` containers with the persisted history, so the
fork's in-place transcript sanitization would rewrite the parent's messages
(#100795). These tests drive the real /refine handlers into the real
chokepoint and capture what reaches the spawn.
"""

import threading
from unittest.mock import MagicMock

import pytest


def _agent_with_real_chokepoint():
    """MagicMock agent whose _spawn_background_review is the REAL method.

    Everything below the chokepoint (thread spawn) is captured at
    ``_spawn_background_review_now`` so no fork actually runs.
    """
    from run_agent import AIAgent

    agent = MagicMock()
    agent.valid_tool_names = {"memory"}
    agent._delegate_depth = 0
    agent._spawn_background_review = AIAgent._spawn_background_review.__get__(agent)
    return agent


def _nested_history():
    return [
        {"role": "user", "content": [{"type": "text", "text": "ask"}]},
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [{
                "id": "call-1",
                "function": {"name": "read_file", "arguments": '{"path":"x"}'},
            }],
        },
    ]


def _assert_isolated(live, snapshot):
    assert snapshot == live  # same shape/bytes …
    for live_msg, snap_msg in zip(live, snapshot):
        assert snap_msg is not live_msg  # … but no shared containers
        for key in ("content", "tool_calls"):
            if isinstance(live_msg.get(key), (dict, list)):
                assert snap_msg[key] is not live_msg[key]
    # Mutating the snapshot the way the fork's sanitizers do must not leak.
    snapshot[0]["content"][0]["text"] = "mutated"
    snapshot[1]["tool_calls"][0]["function"]["arguments"] = "{}"
    assert live[0]["content"][0]["text"] == "ask"
    assert live[1]["tool_calls"][0]["function"]["arguments"] == '{"path":"x"}'


def test_cli_refine_snapshot_does_not_alias_live_history(monkeypatch):
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    monkeypatch.setattr("cli._cprint", lambda *a, **k: None, raising=False)
    agent = _agent_with_real_chokepoint()
    cli = object.__new__(CLICommandsMixin)
    cli.agent = agent
    cli.conversation_history = _nested_history()

    cli._handle_refine_command("/refine")

    agent._spawn_background_review_now.assert_called_once()
    snapshot = agent._spawn_background_review_now.call_args.kwargs["messages_snapshot"]
    _assert_isolated(cli.conversation_history, snapshot)


@pytest.mark.asyncio
async def test_gateway_refine_snapshot_does_not_alias_live_history():
    from gateway.run import GatewayRunner

    key = "agent:main:test:dm:1"
    agent = _agent_with_real_chokepoint()
    agent._session_messages = _nested_history()

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._agent_cache = {key: agent}
    runner._agent_cache_lock = threading.Lock()
    runner._session_key_for_source = lambda source: key

    event = MagicMock()
    event.source = object()
    event.get_command_args.return_value = ""

    out = await runner._handle_refine_command(event)

    assert out.startswith("⚗")
    agent._spawn_background_review_now.assert_called_once()
    snapshot = agent._spawn_background_review_now.call_args.kwargs["messages_snapshot"]
    _assert_isolated(agent._session_messages, snapshot)
