"""The smart-approval guardian asks for its verdict with thinking off, unless the operator chose.

Its budget is 16 tokens for one word. A reasoning model left to think spends the whole budget on
reasoning and returns nothing: a press reviewer's local Qwen 27B returned 45 empty verdicts out of
45. With thinking off, the managed llama-server answers ``APPROVE`` in 4 tokens. An operator who
set ``auxiliary.approval.reasoning_effort`` keeps that setting.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client
from tools.approval_smart import _smart_approve

LOCAL = "http://127.0.0.1:18434/v1"


def _reply(text):
    return MagicMock(choices=[MagicMock(message=MagicMock(content=text))])


def _verdict_wire(task_config):
    """The request the guardian's verdict call puts on the wire to a local llama-server."""
    with patch("agent.auxiliary_task_config._get_auxiliary_task_config", return_value=task_config), \
            patch.object(auxiliary_client, "_get_auxiliary_task_config", return_value=task_config), \
            patch("agent.auxiliary_client.call_llm", return_value=_reply("APPROVE")) as call_llm:
        assert _smart_approve("python -c \"print('hi')\"", "script execution via -c flag") == "approve"
        sent = call_llm.call_args.kwargs
        return auxiliary_client._build_call_kwargs(
            provider="custom", model="qwen", messages=sent["messages"], temperature=sent["temperature"],
            max_tokens=sent["max_tokens"], extra_body=auxiliary_client._get_task_extra_body("approval"),
            reasoning_config=sent["reasoning_config"], base_url=LOCAL, task="approval")


def test_verdict_goes_out_with_thinking_off():
    assert _verdict_wire({})["reasoning_effort"] == "none"


@pytest.mark.parametrize("task_config", [
    {"reasoning_effort": "high"},
    {"extra_body": {"reasoning": {"enabled": True, "effort": "high"}}},
])
def test_operator_reasoning_setting_wins(task_config):
    assert _verdict_wire(task_config)["reasoning_effort"] == "high"
