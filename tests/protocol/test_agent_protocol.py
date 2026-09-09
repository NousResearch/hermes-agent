"""
Tests for Common Agent Protocol and JSON Message schemas.
"""

import pytest
from shared.protocol import ProtocolMessage, AgentRequestMessage, AgentResponseMessage, AgentEventMessage, MessageType


def test_protocol_message_serialization():
    req = AgentRequestMessage("sess_100", "msg_1", "run_tool", {"tool_name": "terminal"})
    json_str = req.to_json()
    assert '"type": "agent.request"' in json_str

    deserialized = ProtocolMessage.from_json(json_str)
    assert deserialized.session_id == "sess_100"
    assert deserialized.message_id == "msg_1"
    assert deserialized.payload["action"] == "run_tool"


def test_protocol_response_message():
    resp = AgentResponseMessage("sess_100", "msg_1", True, {"output": "ok"})
    dict_resp = resp.to_dict()
    assert dict_resp["type"] == "agent.response"
    assert dict_resp["payload"]["success"] is True
    assert dict_resp["payload"]["result"]["output"] == "ok"
