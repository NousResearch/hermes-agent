import json
import re
from types import SimpleNamespace

from agent.agent_runtime_helpers import convert_to_trajectory_format
from agent.trajectory import save_trajectory


def test_save_trajectory_redacts_secrets_in_temp_file(tmp_path):
    secret = "sk-" + "proj-trajectory-secret-1234567890"
    path = tmp_path / "trajectory.jsonl"

    save_trajectory(
        [{"from": "human", "value": f"use api_key={secret}"}],
        "test-model",
        True,
        filename=str(path),
    )

    raw = path.read_text(encoding="utf-8")
    assert secret not in raw
    entry = json.loads(raw)
    assert entry["conversations"][0]["from"] == "human"
    assert secret not in entry["conversations"][0]["value"]
    assert entry["conversations"][0]["value"] != f"use api_key={secret}"


def test_save_trajectory_preserves_sharegpt_shape_while_redacting_tool_arguments(tmp_path):
    secret = "sk-" + "proj-trajectory-secret-1234567890"
    path = tmp_path / "trajectory.jsonl"
    trajectory = [
        {"from": "human", "value": "Run the tool"},
        {"from": "gpt", "value": '{"name":"terminal","arguments":"token=' + secret + '"}'},
        {"from": "tool", "value": "command completed"},
    ]

    save_trajectory(trajectory, "test-model", False, filename=str(path))

    entry = json.loads(path.read_text(encoding="utf-8"))
    assert entry["model"] == "test-model"
    assert entry["completed"] is False
    assert [item["from"] for item in entry["conversations"]] == ["human", "gpt", "tool"]
    assert secret not in json.dumps(entry)


def _call(call_id, name, **arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def test_save_trajectory_keeps_embedded_tool_call_and_response_json_parseable(tmp_path):
    """Tool calls/responses are JSON inside the turn text; redaction must not eat their quotes or braces."""
    secret = "sk-" + "proj-trajectory-secret-1234567890"
    messages = [
        {"role": "user", "content": "deploy"},
        {"role": "assistant", "content": f"Exporting {secret} first.", "tool_calls": [
            _call("c1", "terminal", command=f"export OPENAI_API_KEY={secret}"),
            _call("c2", "terminal", command="curl --password=hunter2pass"),
            _call("c3", "http_request", url="https://api.example", api_key="opaquevalue123",
                  headers={"x-api-key": "opaqueheader1234"}),
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": f"OPENAI_API_KEY={secret}\nDB_PASSWORD=opaque123"},
        {"role": "tool", "tool_call_id": "c2", "content": json.dumps({"lines": ["ok", "DB_PASSWORD=opaquelist123"]})},
        {"role": "tool", "tool_call_id": "c3", "content": "written"},
    ]
    agent = SimpleNamespace(_format_tools_for_system_message=lambda: "[]")
    path = tmp_path / "trajectory.jsonl"

    save_trajectory(convert_to_trajectory_format(agent, messages, "deploy", True), "m", True, filename=str(path))

    raw = path.read_text(encoding="utf-8")
    for leaked in (secret, "hunter2pass", "opaquevalue123", "opaqueheader1234", "opaque123", "opaquelist123"):
        assert leaked not in raw
    turns = json.loads(raw)["conversations"]
    blocks = [
        json.loads(body)
        for turn in turns
        if turn["from"] in ("gpt", "tool")  # the system turn holds a non-JSON format example
        for body in re.findall(r"<(?:tool_call|tool_response)>\n(.*?)\n</(?:tool_call|tool_response)>", turn["value"], re.DOTALL)
    ]
    assert len(blocks) == 6
    assert [b["name"] for b in blocks[:3]] == ["terminal", "terminal", "http_request"]
    assert blocks[2]["arguments"]["url"] == "https://api.example"
