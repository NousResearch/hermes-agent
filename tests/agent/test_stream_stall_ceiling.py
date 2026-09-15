"""Stream-stall continuation ceiling: network stalls are NOT length truncation.

A PARTIAL_STREAM_STUB_ID response means the provider connection dropped
mid-stream (peer closed connection, incomplete chunked read). 2026-09-14: a
7-minute self-hosted provider-gateway drain surfaced to the desktop user as

    "Response remained truncated after 4 continuation attempts"

— the length-truncation ceiling message — sending the user to debug max_tokens
for what was a provider outage. The stall path now gets its own cap (8) and a
ceiling message that names the real failure (provider stream stall).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import PARTIAL_STREAM_STUB_ID, FINISH_REASON_LENGTH


@pytest.fixture()
def agent():
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.compression_enabled = False
        a.save_trajectories = False
        a._vprint_buffer = []
        orig_vprint = a._vprint

        def _capturing_vprint(msg, *args, **kwargs):
            a._vprint_buffer.append(str(msg))
            return orig_vprint(msg, *args, **kwargs)

        a._vprint = _capturing_vprint
        return a


def _stall(content):
    """Mid-stream network stall: the partial-stream stub the streaming layer
    builds when the peer closes the connection before completing the body."""
    from tests.agent.test_run_agent import _mock_assistant_msg
    return SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID,
        model="test/model",
        choices=[SimpleNamespace(
            index=0,
            message=_mock_assistant_msg(content=content),
            finish_reason=FINISH_REASON_LENGTH,
        )],
        usage=None,
    )


def _exhaust_stall_ceiling(agent, parts):
    agent.client.chat.completions.create.side_effect = [_stall(p) for p in parts]
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation("give me a long answer")


class TestStallCeiling:
    def test_stalls_do_not_ceiling_at_four(self, agent):
        """Four stalls used to kill the turn. The stall cap is 8: attempt 5+
        must still continue (a provider outage window usually outlasts 4
        quick retries, and each stall DID recover more text)."""
        from tests.agent.test_run_agent import _mock_response
        agent.client.chat.completions.create.side_effect = [
            _stall("part one. "),
            _stall("part two. "),
            _stall("part three. "),
            _stall("part four. "),
            _stall("part five. "),
            _mock_response(content="the end.", finish_reason="stop"),
        ]
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("give me a long answer")

        assert result["completed"] is True
        assert agent.client.chat.completions.create.call_count == 6
        for fragment in ("part one", "part three", "part five", "the end"):
            assert fragment in result["final_response"]

    def test_stall_ceiling_at_eight_names_the_real_failure(self, agent):
        """At the stall ceiling the error must say the provider stream
        stalled — NOT the length-truncation message, which blames the model's
        output budget and sends the user to debug max_tokens."""
        parts = [f"chunk {i}. " for i in range(8)]
        result = _exhaust_stall_ceiling(agent, parts)

        assert result["completed"] is False
        assert agent.client.chat.completions.create.call_count == 8
        err = result.get("error") or ""
        assert "provider stream stalled" in err
        assert "output" not in err.lower() or "length" not in err.lower()
        assert "truncated after" not in err

    def test_stall_ceiling_keeps_stitched_partial(self, agent):
        parts = [f"chunk {i}. " for i in range(8)]
        result = _exhaust_stall_ceiling(agent, parts)

        assert result["final_response"]
        assert "chunk 0" in result["final_response"]
        assert "chunk 7" in result["final_response"]
        # The user-facing notice names the provider, not the token budget.
        assert "provider" in result["final_response"]
        assert "stalled" in result["final_response"]

    def test_stall_ceiling_vprint_labels_network_not_length(self, agent):
        parts = [f"chunk {i}. " for i in range(8)]
        _exhaust_stall_ceiling(agent, parts)

        printed = "\n".join(agent._vprint_buffer)
        assert "stalling mid-response" in printed

    def test_length_ceiling_still_fires_at_four(self, agent):
        """Genuine finish_reason='length' (normal response id) keeps the
        original cap of 4 and the original message."""
        from tests.agent.test_run_agent import _mock_assistant_msg

        def _length(content):
            return SimpleNamespace(
                id="chatcmpl-length-truncated",
                model="test/model",
                choices=[SimpleNamespace(
                    index=0,
                    message=_mock_assistant_msg(content=content),
                    finish_reason=FINISH_REASON_LENGTH,
                )],
                usage=None,
            )

        agent.client.chat.completions.create.side_effect = [
            _length(f"chunk {i}. ") for i in range(6)
        ]
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("give me a long answer")

        assert result["completed"] is False
        assert agent.client.chat.completions.create.call_count == 4
        assert "truncated after 4 continuation attempts" in (result.get("error") or "")
