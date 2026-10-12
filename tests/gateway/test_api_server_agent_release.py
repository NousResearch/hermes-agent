"""Every api_server request builds a fresh ``AIAgent`` that is dropped after the turn (#137451).

``_run_agent``'s ``finally`` used to drop the agent without ``release_clients()``, so its LLM
clients — the ``codex app-server`` child under ``model.openai_runtime: codex_app_server`` in
particular — stayed alive for the gateway's life: one ~200 MB process per request, and the
leaked writer pinned the Codex thread so the next request could not resume it. The gateway's
agent cache already releases on eviction (``run_agent_cache._release_evicted_agent_soft``);
the api_server lifecycle needs the same call.
"""

from unittest.mock import MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _make_agent() -> MagicMock:
    agent = MagicMock()
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    # Keep checkin() a no-op: the parked-manager registry is another test file's subject.
    agent._memory_manager = None
    return agent


_MODEL_OPTIONS = {"reasoning": {"enabled": False}, "fast": False}


class TestRunAgentReleasesClients:
    @pytest.mark.asyncio
    async def test_release_clients_called_after_turn(self):
        adapter = _make_adapter()
        agent = _make_agent()
        agent.run_conversation.return_value = {"final_response": "ok"}

        with patch.object(adapter, "_create_agent", return_value=agent):
            result, _usage = await adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                session_id="session-release",
                requested_model="MiniMax-M3",
                requested_provider="minimax",
                model_options=_MODEL_OPTIONS,
            )

        assert result["final_response"] == "ok"
        agent.release_clients.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_release_clients_called_when_turn_raises(self):
        adapter = _make_adapter()
        agent = _make_agent()
        agent.run_conversation.side_effect = RuntimeError("boom")

        with patch.object(adapter, "_create_agent", return_value=agent):
            with pytest.raises(RuntimeError, match="boom"):
                await adapter._run_agent(
                    user_message="hello",
                    conversation_history=[],
                    session_id="session-release",
                    requested_model="MiniMax-M3",
                    requested_provider="minimax",
                    model_options=_MODEL_OPTIONS,
                )

        # The finally path must release even on a failed turn — a failed request's
        # codex child leaks exactly the same way a successful one's does.
        agent.release_clients.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_release_clients_failure_does_not_break_the_turn(self):
        adapter = _make_adapter()
        agent = _make_agent()
        agent.run_conversation.return_value = {"final_response": "ok"}
        agent.release_clients.side_effect = RuntimeError("already closed")

        with patch.object(adapter, "_create_agent", return_value=agent):
            result, _usage = await adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                session_id="session-release",
                requested_model="MiniMax-M3",
                requested_provider="minimax",
                model_options=_MODEL_OPTIONS,
            )

        # The release is best-effort cleanup after the reply is complete: a double
        # release or a half-closed client must not fail the request itself.
        assert result["final_response"] == "ok"
