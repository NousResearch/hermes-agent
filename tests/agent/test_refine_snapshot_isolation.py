"""/refine hands the review fork a snapshot that cannot alias the live transcript.

The automatic post-turn review already clones structurally
(``_clone_background_review_messages``); the two explicit ``/refine`` entry
points (CLI mixin + gateway slash command) build their own snapshot and must
use the same clone — a shallow ``list()`` shares the nested ``tool_calls`` /
``content`` containers with the persisted history, so the fork's in-place
transcript sanitization would rewrite the parent's messages (#100795).
"""

import threading
from unittest.mock import MagicMock

import pytest


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
    assert snapshot is not live
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
    agent = MagicMock()
    agent.valid_tool_names = {"memory"}
    cli = object.__new__(CLICommandsMixin)
    cli.agent = agent
    cli.conversation_history = _nested_history()

    cli._handle_refine_command("/refine")

    agent._spawn_background_review.assert_called_once()
    snapshot = agent._spawn_background_review.call_args.kwargs["messages_snapshot"]
    _assert_isolated(cli.conversation_history, snapshot)


@pytest.mark.asyncio
async def test_gateway_refine_snapshot_does_not_alias_live_history():
    from gateway.run import GatewayRunner

    key = "agent:main:test:dm:1"
    agent = MagicMock()
    agent.valid_tool_names = {"memory"}
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
    agent._spawn_background_review.assert_called_once()
    snapshot = agent._spawn_background_review.call_args.kwargs["messages_snapshot"]
    _assert_isolated(agent._session_messages, snapshot)
