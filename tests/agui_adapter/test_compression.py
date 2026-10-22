"""AG-UI keeps real Hermes compaction authoritative across HTTP legs."""

import copy
import json
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from agui_adapter.server import create_app
from hermes_cli.config import save_config
from tests.agui_adapter.test_conversations import next_user
from tests.agui_adapter.test_modernization_probe import config_for, events
from tests.agui_adapter.test_parked_frontend import (
    frontend_request,
    model_call,
    result_request,
)


@pytest.mark.parametrize("write_state", [False, True, None])
def test_real_compression_is_not_undone_by_browser_history(
    peer, monkeypatch, write_state
):
    from agent.context_compressor import ContextCompressor

    provider_only = write_state is None
    schema_const, schema_reads = 1, []
    provider_schema = {
        "name": "agui_local_context_search",
        "description": "Search local context",
        "parameters": {"type": "object", "properties": {}},
    }

    def schemas(self):
        schema = copy.deepcopy(provider_schema)
        if provider_only:
            schema["parameters"]["properties"] = {"value": {"const": schema_const}}
            schema_reads.append(schema)
        return [schema]

    monkeypatch.setattr(ContextCompressor, "get_tool_schemas", schemas)
    starts = Mock(wraps=ContextCompressor.on_session_start)
    monkeypatch.setattr(
        ContextCompressor,
        "on_session_start",
        lambda self, *args, **kwargs: starts(self, *args, **kwargs),
    )
    request = frontend_request(peer)
    if provider_only:
        request["tools"] = []
    if write_state:
        request["forwardedProps"] = {
            "stateWriterTools": [
                {
                    "name": "agui_compaction_state",
                    "stateKey": "compacted",
                    "arg": "value",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "boolean"}},
                    },
                }
            ]
        }
    endpoint = f"http://127.0.0.1:{peer.server_port}/v1"
    monkeypatch.setenv("OPENAI_API_KEY", "local-compression-probe")
    save_config(
        {
            "model": {
                "default": "gpt-4o",
                "provider": "custom",
                "base_url": endpoint,
                "context_length": 128000,
            },
            "compression": {
                "enabled": True,
                "threshold_tokens": 20000,
                "protect_last_n": 2,
                "tail_mode": "legacy",
                "target_ratio": 0.1,
                "micro_compact": False,
            },
            "auxiliary": {
                "title_generation": {"enabled": False},
                "compression": {
                    "provider": "custom",
                    "model": "summary-probe",
                    "base_url": endpoint,
                    "api_mode": "chat_completions",
                    "context_length": 128000,
                },
            },
        }
    )
    main_requests = []
    summary_requests = []

    def script(incoming, index):
        if incoming["model"] == "summary-probe":
            summary_requests.append(copy.deepcopy(incoming))
            peer.usage = {
                "prompt_tokens": 1000,
                "completion_tokens": 10,
                "total_tokens": 1010,
            }
            return {
                "role": "assistant",
                "content": "COMPRESSION_PROBE_SUMMARY: Continue the user's browser probe.",
            }
        main_requests.append(copy.deepcopy(incoming))
        count = len(main_requests)
        prompt = 24000 if count == 8 else 10
        peer.usage = {
            "prompt_tokens": prompt,
            "completion_tokens": 5,
            "total_tokens": prompt + 5,
        }
        if not provider_only and "HANDOFF_AFTER_COMPRESSION" in str(
            incoming["messages"][-1].get("content", "")
        ):
            calls = [model_call("compressed-front", peer.tool_name, color="blue")]
            if write_state:
                calls.insert(
                    0,
                    model_call(
                        "compressed-writer", "agui_compaction_state", value=True
                    ),
                )
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": calls,
            }
        return {"role": "assistant", "content": f"REPLY_{count}"}

    peer.script = script
    app = create_app(
        config_for(peer, ["context_engine"] if write_state or provider_only else [])
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        # Real ordinary turns, not a synthetic pre-seeded canonical transcript.
        for index in range(8):
            request["messages"][-1]["content"] = (
                f"TURN_{index}: " + "historical context " * 600
            )
            events(client.post("/", json=request))
            if index < 7:
                request = next_user(request, user_id=f"user-{index + 2}")
        record = app.state.executions.conversations.get(request["threadId"])
        agent = record.agent
        assert starts.call_count == 1
        admitted = {tool["function"]["name"]: tool["function"] for tool in agent.tools}
        assert (provider_schema["name"] in admitted) == (write_state or provider_only)
        if write_state:
            assert admitted[provider_schema["name"]] == provider_schema
        assert "terminal" not in admitted
        wire_tools = {tool["function"]["name"] for tool in main_requests[0]["tools"]}
        assert set(admitted) == wire_tools
        before = copy.deepcopy(record.messages)
        schema_const = True
        request = next_user(request, "HANDOFF_AFTER_COMPRESSION", user_id="user-9")
        first = events(client.post("/", json=request))
        assert summary_requests, "the real auxiliary compression route never ran"
        assert agent.context_compressor.compression_count >= 1
        if provider_only:
            assert set(admitted) == {provider_schema["name"]} == agent.valid_tool_names
            assert len(schema_reads) == 2
            assert (
                schema_reads[-1]["parameters"]["properties"]["value"]["const"] is True
            )
            assert json.dumps(main_requests[-1]["tools"]) == json.dumps(
                main_requests[0]["tools"]
            ), "compaction changed the admitted numeric schema to a boolean"
            assert "COMPRESSION_PROBE_SUMMARY" in str(record.messages)
            assert len(str(record.messages)) < len(str(before))
            assert record.agent is agent and agent._user_turn_count == 9
            assert len(main_requests) == 9 and len(summary_requests) == 1
            assert (
                sum(
                    "boundary_reason" not in call.kwargs
                    for call in starts.call_args_list
                )
                == 1
            )
            return
        if write_state:
            assert (
                next(e for e in first if e["type"] == "STATE_SNAPSHOT")["snapshot"][
                    "compacted"
                ]
                is True
            )
        assert main_requests[-1].get("tools") == main_requests[0]["tools"], (
            "compaction dropped frozen client schemas"
        )
        call = next(e for e in first if e["type"] == "TOOL_CALL_START")
        resumed = result_request(request, call, "BROWSER_AFTER_COMPRESSION")
        events(client.post("/", json=resumed))
        compacted = copy.deepcopy(record.messages)
        assert "COMPRESSION_PROBE_SUMMARY" in str(compacted)
        assert len(str(compacted)) < len(str(before))
        assert record.agent is agent and agent._user_turn_count == 9
        assert (
            main_requests[-1]["messages"][: len(main_requests[-2]["messages"])]
            == main_requests[-2]["messages"]
        )
        request = next_user(
            resumed, "New user after compressed frontend result", user_id="user-10"
        )
        events(client.post("/", json=request))
        assert record.agent is agent and agent._user_turn_count == 10
        assert len(main_requests) == 11, "a turn or frontend effect was replayed"
        assert len(summary_requests) == 1
        assert "COMPRESSION_PROBE_SUMMARY" in str(main_requests[-1]["messages"])
        assert (
            main_requests[-1]["messages"][: len(main_requests[-2]["messages"])]
            == main_requests[-2]["messages"]
        )
        assert all(r["tools"] == main_requests[0]["tools"] for r in main_requests)
        assert (
            sum("boundary_reason" not in call.kwargs for call in starts.call_args_list)
            == 1
        )
        assert set(admitted) == agent.valid_tool_names
        assert {
            tool["function"]["name"]: tool["function"] for tool in agent.tools
        } == admitted
