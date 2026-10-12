"""Gateway wiring for undelivered tool-call text: interim messages off leaves no callback."""

from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import strip_think_blocks
from agent.stream_delivery import StreamDeliveryMixin
from agent.turn_tool_round import UNDELIVERED_INTERIM_MARKER, mark_undelivered_interim
from tests.gateway.test_run_progress_topics import _run_with_agent


class UndeliveredToolRoundAgent(StreamDeliveryMixin):
    """Stages one tool-call row through the real ``mark_undelivered_interim``,
    against whatever interim callback the gateway left on the agent for this turn
    (same shape as ``DelayedInterimAgent`` in test_run_progress_topics.py)."""

    runs: list = []

    def __init__(self, **kwargs):
        self.interim_assistant_callback = kwargs.get("interim_assistant_callback")
        self.tools = []

    def _strip_think_blocks(self, content):
        return strip_think_blocks(self, content)

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        quote = "Unit price = $128\n4 x $128 = $512\nTotal = $519"
        row = {
            "role": "assistant",
            "content": quote,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}],
        }
        mark_undelivered_interim(self, assistant_message=SimpleNamespace(content=quote), assistant_msg=row)
        type(self).runs.append({"callback": self.interim_assistant_callback, "row": row})
        return {"final_response": "When do you need them by?", "messages": [], "api_calls": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("interim_on", [False, True])
async def test_disabled_interim_marks_tool_call_text_undelivered(monkeypatch, tmp_path, interim_on):
    """display.interim_assistant_messages=false leaves the agent without an interim callback,
    which is what tells ``mark_undelivered_interim`` that tool-call text never reached the user."""
    monkeypatch.setattr(UndeliveredToolRoundAgent, "runs", [])
    await _run_with_agent(
        monkeypatch,
        tmp_path,
        UndeliveredToolRoundAgent,
        session_id="sess-undelivered-tool-call",
        config_data={"display": {"tool_progress": "off", "interim_assistant_messages": interim_on}},
    )

    (run,) = UndeliveredToolRoundAgent.runs
    row = run["row"]
    if interim_on:
        assert run["callback"] is not None
        assert "api_content" not in row
    else:
        assert run["callback"] is None
        assert row["api_content"] == f"{row['content']}\n\n{UNDELIVERED_INTERIM_MARKER}"
