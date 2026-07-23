from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from agent.conversation_compression import (
    COMPACTION_DONE_STATUS,
    COMPACTION_STATUS,
    ROUTINE_COMPRESSION_STATUS_SAMPLES,
    _emit_compaction_done,
    buffer_compression_status,
    emit_compression_status,
)
from run_agent import AIAgent


def _make_agent(config):
    with (
        patch("hermes_cli.config.load_config", return_value=config),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, True),
        ({"compression": {"status_messages": False}}, False),
        ({"compression": {"status_messages": "no"}}, False),
    ],
)
def test_compression_status_messages_config(config, expected):
    assert _make_agent(config).compression_status_messages is expected


def test_disabled_compression_status_is_silent_on_chat_platforms():
    agent = _make_agent({"compression": {"status_messages": False}})
    agent.platform = "discord"
    agent._vprint = MagicMock()
    agent.status_callback = MagicMock()

    for message in ROUTINE_COMPRESSION_STATUS_SAMPLES:
        agent._emit_compression_status(message)
        agent._buffer_compression_status(message)

    _emit_compaction_done(agent)

    agent._vprint.assert_not_called()
    agent.status_callback.assert_not_called()
    assert getattr(agent, "_retry_status_buffer", []) == []


def test_disabled_compression_status_preserves_desktop_indicator():
    agent = _make_agent({"compression": {"status_messages": False}})
    agent.platform = "desktop"
    agent._vprint = MagicMock()
    agent.status_callback = MagicMock()

    agent._emit_compression_status(COMPACTION_STATUS)
    _emit_compaction_done(agent)

    agent._vprint.assert_not_called()
    assert agent.status_callback.call_args_list == [
        call("lifecycle", COMPACTION_STATUS),
        call("compacted", COMPACTION_DONE_STATUS),
    ]


def test_compression_status_helpers_support_legacy_duck_typed_agents():
    agent = SimpleNamespace(_emit_status=MagicMock(), _buffer_status=MagicMock())

    emit_compression_status(agent, "emit")
    buffer_compression_status(agent, "buffer")

    agent._emit_status.assert_called_once_with("emit")
    agent._buffer_status.assert_called_once_with("buffer")
