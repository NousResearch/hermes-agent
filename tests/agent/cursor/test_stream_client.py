from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.cursor.connect_framing import frame_connect_message, parse_connect_frames
from agent.cursor.proto import agent_pb2
from agent.cursor.stream_client import (
    _consume_connect_stream,
    _encode_client_message,
)


def _server_frame(server_message: agent_pb2.AgentServerMessage) -> bytes:
    return frame_connect_message(server_message.SerializeToString())


def _interaction_frame(**kwargs) -> bytes:
    message = agent_pb2.AgentServerMessage(
        interaction_update=agent_pb2.InteractionUpdate(**kwargs)
    )
    return _server_frame(message)


def _checkpoint_frame(used_tokens: int) -> bytes:
    state = agent_pb2.ConversationStateStructure(
        token_details=agent_pb2.ConversationTokenDetails(used_tokens=used_tokens)
    )
    return _server_frame(agent_pb2.AgentServerMessage(conversation_checkpoint_update=state))


def test_consume_connect_stream_assembles_text_and_tool_calls():
    deltas: list[str] = []
    frames = [
        _interaction_frame(
            text_delta=agent_pb2.TextDeltaUpdate(text="Hello"),
        ),
        _interaction_frame(
            text_delta=agent_pb2.TextDeltaUpdate(text=" world"),
        ),
        _interaction_frame(
            tool_call_started=agent_pb2.ToolCallStartedUpdate(
                call_id="call-1",
                tool_call=agent_pb2.ToolCall(
                    mcp_tool_call=agent_pb2.McpToolCall(
                        args=agent_pb2.McpArgs(
                            name="weather",
                            tool_call_id="call-1",
                            provider_identifier="pi-agent",
                            tool_name="weather",
                        )
                    )
                ),
            )
        ),
        _interaction_frame(
            partial_tool_call=agent_pb2.PartialToolCallUpdate(
                call_id="call-1",
                args_text_delta='{"city":"San',
            )
        ),
        _interaction_frame(
            tool_call_completed=agent_pb2.ToolCallCompletedUpdate(
                call_id="call-1",
                tool_call=agent_pb2.ToolCall(
                    mcp_tool_call=agent_pb2.McpToolCall(
                        args=agent_pb2.McpArgs(
                            name="weather",
                            tool_call_id="call-1",
                            provider_identifier="pi-agent",
                            tool_name="weather",
                            args={"city": b'"San Francisco"', "unit": b'"C"'},
                        )
                    )
                ),
            )
        ),
        _interaction_frame(
            token_delta=agent_pb2.TokenDeltaUpdate(tokens=7),
        ),
        _interaction_frame(
            turn_ended=agent_pb2.TurnEndedUpdate(),
        ),
    ]

    response = _consume_connect_stream(
        frames,
        blob_store={},
        request_context_tools=[],
        on_text_delta=deltas.append,
    )

    choice = response.choices[0]
    assert deltas == ["Hello", " world"]
    assert choice.message.content == "Hello world"
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls is not None
    assert len(choice.message.tool_calls) == 1
    tool_call = choice.message.tool_calls[0]
    assert tool_call.id == "call-1"
    assert tool_call.function.name == "weather"
    assert json.loads(tool_call.function.arguments) == {
        "city": "San Francisco",
        "unit": "C",
    }
    assert response.usage.completion_tokens == 7


def test_consume_connect_stream_uses_checkpoint_tokens_when_no_token_delta():
    frames = [
        _interaction_frame(text_delta=agent_pb2.TextDeltaUpdate(text="Done")),
        _checkpoint_frame(used_tokens=13),
        _interaction_frame(turn_ended=agent_pb2.TurnEndedUpdate()),
    ]

    response = _consume_connect_stream(
        frames,
        blob_store={},
        request_context_tools=[],
    )

    assert response.choices[0].message.content == "Done"
    assert response.usage.completion_tokens == 13
    assert response.usage.total_tokens == 13


def test_consume_connect_stream_rejects_end_stream_error():
    with pytest.raises(Exception, match="permission_denied"):
        _consume_connect_stream(
            [
                frame_connect_message(
                    b'{"error":{"code":"permission_denied","message":"Denied"}}',
                    flags=0b00000010,
                )
            ],
            blob_store={},
            request_context_tools=[],
        )


def test_consume_connect_stream_responds_to_kv_blob_requests():
    blob_id = b"\x01" * 32
    writes: list[bytes] = []
    frames = [
        _server_frame(
            agent_pb2.AgentServerMessage(
                kv_server_message=agent_pb2.KvServerMessage(
                    id=5,
                    get_blob_args=agent_pb2.GetBlobArgs(blob_id=blob_id),
                )
            )
        ),
        _interaction_frame(turn_ended=agent_pb2.TurnEndedUpdate()),
    ]

    _consume_connect_stream(
        frames,
        blob_store={blob_id.hex(): b"blob-data"},
        request_context_tools=[],
        send_client_frame=writes.append,
    )

    assert writes
    # Decode the actual first response frame.
    from agent.cursor.connect_framing import parse_connect_frames

    decoded = list(parse_connect_frames(bytearray(writes[0])))
    assert len(decoded) == 1
    _, payload = decoded[0]
    client_message = agent_pb2.AgentClientMessage()
    client_message.ParseFromString(payload)
    assert client_message.WhichOneof("message") == "kv_client_message"
    assert client_message.kv_client_message.id == 5
    assert client_message.kv_client_message.get_blob_result.blob_data == b"blob-data"


def test_consume_connect_stream_returns_rejected_exec_result_without_handlers():
    writes: list[bytes] = []
    frames = [
        _server_frame(
            agent_pb2.AgentServerMessage(
                exec_server_message=agent_pb2.ExecServerMessage(
                    id=7,
                    exec_id="exec-1",
                    read_args=agent_pb2.ReadArgs(path="/tmp/file.txt"),
                )
            )
        ),
        _interaction_frame(turn_ended=agent_pb2.TurnEndedUpdate()),
    ]

    _consume_connect_stream(
        frames,
        blob_store={},
        request_context_tools=[],
        send_client_frame=writes.append,
    )

    from agent.cursor.connect_framing import parse_connect_frames

    decoded = list(parse_connect_frames(bytearray(writes[0])))
    _, payload = decoded[0]
    client_message = agent_pb2.AgentClientMessage()
    client_message.ParseFromString(payload)
    assert client_message.WhichOneof("message") == "exec_client_message"
    exec_message = client_message.exec_client_message
    assert exec_message.id == 7
    assert exec_message.exec_id == "exec-1"
    assert exec_message.read_result.WhichOneof("result") == "rejected"
    assert exec_message.read_result.rejected.reason == "Tool not available"


def test_consume_connect_stream_sends_kv_and_exec_responses_before_turn_end():
    writes: list[bytes] = []
    frames = [
        _server_frame(
            agent_pb2.AgentServerMessage(
                kv_server_message=agent_pb2.KvServerMessage(
                    id=3,
                    get_blob_args=agent_pb2.GetBlobArgs(blob_id=(b"\x02" * 32)),
                )
            )
        ),
        _server_frame(
            agent_pb2.AgentServerMessage(
                exec_server_message=agent_pb2.ExecServerMessage(
                    id=4,
                    exec_id="exec-read-now",
                    read_args=agent_pb2.ReadArgs(path="/tmp/now.txt"),
                )
            )
        ),
        _interaction_frame(turn_ended=agent_pb2.TurnEndedUpdate()),
    ]

    _consume_connect_stream(
        frames,
        blob_store={(b"\x02" * 32).hex(): b"blob-now"},
        request_context_tools=[],
        send_client_frame=writes.append,
    )

    assert len(writes) == 2

    kv_message = agent_pb2.AgentClientMessage()
    exec_message = agent_pb2.AgentClientMessage()

    _, kv_payload = list(parse_connect_frames(bytearray(writes[0])))[0]
    _, exec_payload = list(parse_connect_frames(bytearray(writes[1])))[0]
    kv_message.ParseFromString(kv_payload)
    exec_message.ParseFromString(exec_payload)

    assert kv_message.WhichOneof("message") == "kv_client_message"
    assert kv_message.kv_client_message.id == 3
    assert kv_message.kv_client_message.get_blob_result.blob_data == b"blob-now"

    assert exec_message.WhichOneof("message") == "exec_client_message"
    assert exec_message.exec_client_message.id == 4
    assert exec_message.exec_client_message.read_result.WhichOneof("result") == "rejected"


def test_encode_client_message_wraps_proto_bytes():
    payload = _encode_client_message(
        agent_pb2.AgentClientMessage(client_heartbeat=agent_pb2.ClientHeartbeat())
    )

    assert isinstance(payload, bytes)
    parsed = agent_pb2.AgentClientMessage()
    parsed.ParseFromString(payload)
    assert parsed.WhichOneof("message") == "client_heartbeat"


def test_consume_connect_stream_responds_to_kv_set_blob_requests():
    blob_id = b"\x03" * 32
    writes: list[bytes] = []
    frames = [
        _server_frame(
            agent_pb2.AgentServerMessage(
                kv_server_message=agent_pb2.KvServerMessage(
                    id=9,
                    set_blob_args=agent_pb2.SetBlobArgs(
                        blob_id=blob_id,
                        blob_data=b"stored-by-server",
                    ),
                )
            )
        ),
        _interaction_frame(turn_ended=agent_pb2.TurnEndedUpdate()),
    ]
    blob_store: dict[str, bytes] = {}

    _consume_connect_stream(
        frames,
        blob_store=blob_store,
        request_context_tools=[],
        send_client_frame=writes.append,
    )

    assert blob_store[blob_id.hex()] == b"stored-by-server"
    assert writes
    _, payload = list(parse_connect_frames(bytearray(writes[0])))[0]
    client_message = agent_pb2.AgentClientMessage()
    client_message.ParseFromString(payload)
    assert client_message.WhichOneof("message") == "kv_client_message"
    assert client_message.kv_client_message.id == 9
    assert client_message.kv_client_message.WhichOneof("message") == "set_blob_result"


def test_consume_connect_stream_responds_to_list_mcp_resources_exec():
    writes: list[bytes] = []
    frames = [
        _server_frame(
            agent_pb2.AgentServerMessage(
                exec_server_message=agent_pb2.ExecServerMessage(
                    id=11,
                    exec_id="exec-mcp-list",
                    list_mcp_resources_exec_args=agent_pb2.ListMcpResourcesExecArgs(),
                )
            )
        ),
        _interaction_frame(turn_ended=agent_pb2.TurnEndedUpdate()),
    ]

    _consume_connect_stream(
        frames,
        blob_store={},
        request_context_tools=[],
        send_client_frame=writes.append,
    )

    _, payload = list(parse_connect_frames(bytearray(writes[0])))[0]
    client_message = agent_pb2.AgentClientMessage()
    client_message.ParseFromString(payload)
    exec_message = client_message.exec_client_message
    assert exec_message.id == 11
    assert (
        exec_message.list_mcp_resources_exec_result.WhichOneof("result") == "success"
    )
