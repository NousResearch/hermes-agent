"""A relay connector that cannot render prompts (no ``prompt`` op, e.g. an SMS/iMessage bridge) must not get the
full command as its own message before the approval ask (#157506). The button card can never render there, so the
pre-card full-command send only put a raw block on the person's phone ahead of the plain-text ``/approve`` prompt,
which already carries the command."""

import pytest

from gateway.config import PlatformConfig
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor


class _Connector:
    def __init__(self):
        self.actions, self._identities = [], []

    def set_inbound_handler(self, handler):
        pass

    async def send_outbound(self, action, *, platform=None):
        self.actions.append(action)
        return {"success": True, "message_id": f"m{len(self.actions)}"}


def _relay(connector, ops):
    desc = CapabilityDescriptor(
        contract_version=CONTRACT_VERSION, platform="sms", label="SMS", max_message_length=10000,
        supports_draft_streaming=False, supports_edit=False, supports_threads=False, markdown_dialect="plain",
        len_unit="chars", supported_ops=ops)
    return RelayAdapter(PlatformConfig(), desc, transport=connector)


LONG = "python3 - <<'PY'\n" + "print('x')\n" * 400 + "PY"  # past the 1,500-char relay preview


@pytest.mark.asyncio
async def test_no_prompt_op_sends_nothing_and_fails_so_the_text_prompt_runs():
    connector = _Connector()
    result = await _relay(connector, ("send",)).send_exec_approval(
        chat_id="c", command=LONG, session_key="sk", request_id="r1")
    assert result.success is False and result.error == "relay prompt op unavailable"
    assert connector.actions == [], "nothing reaches the chat ahead of the plain-text approval prompt"


@pytest.mark.asyncio
async def test_prompt_op_keeps_the_full_command_ahead_of_the_card():
    connector = _Connector()
    await _relay(connector, ("send", "prompt")).send_exec_approval(
        chat_id="c", command=LONG, session_key="sk", request_id="r1")
    assert [a["op"] for a in connector.actions] == ["send", "prompt"]
    assert "print('x')" in connector.actions[0]["content"]
