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


@pytest.mark.asyncio
async def test_response_ready_log_includes_the_session_key(caplog):
    """Per-conversation latency aggregation needs the session key on the completion line:
    ``chat=`` alone merges every Slack thread of one channel (#111931)."""
    runner = _Runner()
    source = SimpleNamespace(chat_id="C1", platform=SimpleNamespace(value="slack"))

    with caplog.at_level("INFO", logger="gateway.run"):
        await runner._hmwa_shape_agent_response(
            {"final_response": "done", "messages": [], "api_calls": 1},
            source, history=[], session_entry=SimpleNamespace(session_id="s"),
            session_key="slack:C1:thread-123", _quick_key=None, run_generation=0,
            _run_start_session_id="s", _platform_name="slack", _msg_start_time=0.0,
        )

    response_log = next(record.getMessage() for record in caplog.records
                        if record.getMessage().startswith("response ready:"))
    assert "session=slack:C1:thread-123" in response_log
