"""The gateway's ``(empty)`` sentinel rewrite reads from the same constant as the CLI explainer
(``agent.turn_explainers.EMPTY_RESPONSE_EXPLANATION``), so a chat user and a terminal user see one
text for "the model produced nothing after retries"."""

from types import SimpleNamespace

import pytest

from agent.turn_explainers import EMPTY_RESPONSE_EXPLANATION
from gateway.run_turn import GatewayTurnMixin


class _Runner(GatewayTurnMixin):
    def __init__(self):
        self.async_session_store = SimpleNamespace(clear_resume_pending=self._noop)

    async def _noop(self, *_a, **_k):
        return None

    async def _clear_restart_failure_count(self, *_a, **_k):
        return None


@pytest.mark.asyncio
async def test_empty_sentinel_rewrite_uses_the_shared_explanation_with_the_model_name():
    runner = _Runner()
    agent_result = {"final_response": "(empty)", "model": "llama3", "messages": [], "api_calls": 2}
    source = SimpleNamespace(chat_id="c1", platform=SimpleNamespace(value="telegram"))
    response, silent, _messages = await runner._hmwa_shape_agent_response(
        agent_result, source, history=[], session_entry=SimpleNamespace(session_id="s"), session_key=None,
        _quick_key=None, run_generation=0, _run_start_session_id="s", _platform_name="telegram",
        _msg_start_time=0.0,
    )
    assert silent is False
    assert EMPTY_RESPONSE_EXPLANATION.format(model="llama3") in response
    assert "after processing tool results" not in response
    assert "/model" in response and "continue" in response
