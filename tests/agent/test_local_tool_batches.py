"""Local batches use the live agent pipeline, not registry fan-out."""

import copy
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _call(entries, call_id="batch"):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(
        name="tool_call", arguments=json.dumps({"calls": entries})))


def _response(calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="" if calls else "done", tool_calls=calls),
        finish_reason="tool_calls" if calls else "stop")], model="test/model", usage=None)


@pytest.fixture
def mcp_server(tmp_path):
    from tools.mcp_tool_discovery import register_mcp_servers
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers

    server = tmp_path / "server.py"
    server.write_text('''import json, pathlib, sys
root = pathlib.Path(__file__).parent
for line in sys.stdin:
    req = json.loads(line)
    if "id" not in req:
        continue
    result = {}
    if req["method"] == "initialize":
        result = {"protocolVersion": req["params"]["protocolVersion"],
                  "capabilities": {"tools": {}}, "serverInfo": {"name": "batch", "version": "1"}}
    elif req["method"] == "tools/list":
        result = {"tools": [{"name": "read_" + n, "description": "Read " + n,
                  "inputSchema": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
                  "annotations": {"readOnlyHint": True}} for n in ("alpha", "beta")]}
    elif req["method"] == "tools/call":
        name = req["params"]["name"]
        with (root / "calls.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(name) + "\\n")
        result = {"content": [{"type": "text", "text": name + "-result"}]}
    print(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result}), flush=True)
''', encoding="utf-8")
    names = sorted(register_mcp_servers({"batchfixture": {
        "command": sys.executable, "args": [str(server)], "timeout": 10}}))
    assert len(names) == 2
    try:
        yield names, tmp_path / "calls.jsonl"
    finally:
        shutdown_mcp_servers()


@pytest.mark.parametrize("mode", ["allow", "block", "scope", "schema", "interrupt"])
def test_local_batch_runs_once_per_entry_through_agent_and_persists_pairs(mcp_server, mode):
    from run_agent import AIAgent

    names, call_log = mcp_server
    with patch("agent.process_bootstrap.OpenAI"), patch("agent.model_metadata.fetch_model_metadata", return_value={}):
        agent = AIAgent(
            model="test/model", api_key="test-key", base_url="http://127.0.0.1:1/v1",
            enabled_toolsets=["mcp-batchfixture"], quiet_mode=True,
            skip_context_files=True, skip_memory=True, max_iterations=3)
    agent.client = MagicMock()
    agent._cached_system_prompt = "Use the available tools."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    entries = [{"name": name, "arguments": {"key": "value"}} for name in names]
    if mode == "schema":
        entries[1]["arguments"] = {}
    if mode == "scope":
        agent.disabled_toolsets = ["mcp-batchfixture"]
    agent.client.chat.completions.create.side_effect = [_response([_call(entries)]), _response()]
    if mode == "interrupt":
        def stop_after_first(event, name, *args, **kwargs):
            if event == "tool.completed" and name == names[0]:
                agent.interrupt("test stop after first result")
        agent.tool_progress_callback = stop_after_first

    hooks, snapshots = [], []

    def pre_hook(name, args, **kwargs):
        hooks.append(name)
        return ("blocked by test policy" if mode == "block" and name == names[1] else None), None

    flush = agent._flush_messages_to_session_db

    def capture_flush(messages, *args, **kwargs):
        snapshots.append(copy.deepcopy(messages))
        return flush(messages, *args, **kwargs)

    with patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", side_effect=pre_hook), patch.object(
        agent, "_flush_messages_to_session_db", side_effect=capture_flush
    ):
        result = agent.run_conversation("Read alpha and beta without changing anything.")

    if mode != "interrupt":
        assert result["final_response"] == "done"
    expected = {"allow": ["read_alpha", "read_beta"], "scope": []}.get(mode, ["read_alpha"])
    executed = [json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()] if call_log.exists() else []
    assert executed == expected
    assert hooks == ([] if mode == "scope" else names[:1] if mode in {"schema", "interrupt"} else names)
    messages = result["messages"]
    assistant = next(m for m in messages if m.get("tool_calls"))
    results = [m for m in messages if m["role"] == "tool"]
    ids = [c["id"] for c in assistant["tool_calls"]]
    assert len(set(ids)) == 2
    assert [m["tool_call_id"] for m in results] == ids
    assert ("not available" if mode == "scope" else "read_alpha-result") in results[0]["content"]
    expected_second = {"allow": "read_beta-result", "block": "blocked by test policy",
                       "scope": "not available", "schema": "key", "interrupt": "skipped"}[mode]
    assert expected_second in results[1]["content"]
    # Both calls are durable before the first result, and every result is flushed.
    before_results = next(s for s in snapshots if any(m.get("tool_calls") for m in s))
    assert not any(m["role"] == "tool" for m in before_results)
    assert next(m for m in before_results if m.get("tool_calls"))["tool_calls"] == assistant["tool_calls"]
    assert any(sum(m["role"] == "tool" for m in s) == 2 for s in snapshots)


def test_expansion_preserves_connector_batches_and_rejects_bad_envelopes():
    from agent.tool_call_batches import expand_local_tool_batches
    from tools.connectors.gateway.config import MAX_CALLS_PER_DISPATCH

    local = {"name": "mcp__fixture__read", "arguments": {"path": "alpha"}}
    remote = {"name": "connectors__drive__list", "arguments": {}}
    for entries in ([remote, remote], [local], [local] * (MAX_CALLS_PER_DISPATCH + 1), [local, {}]):
        call = _call(entries)
        original = call.function.arguments
        assert expand_local_tool_batches([call]) == [call]
        assert call.function.arguments == original
    call = _call([local, remote])
    original = call.function.arguments
    expanded = expand_local_tool_batches([call])
    assert [json.loads(c.function.arguments)["calls"] for c in expanded] == [[local], [remote]]
    assert call.function.arguments == original
    assert [c.id for c in expand_local_tool_batches([call])] == [c.id for c in expanded]
    thinking = {"type": "thinking", "thinking": "opaque", "signature": "signed"}
    bedrock_thinking = {"reasoningContent": {"reasoningText": {"text": "opaque", "signature": "signed"}}}
    provider_data = {
        "anthropic_content_blocks": [thinking, {"type": "tool_use", "id": "batch",
            "name": "tool_call", "input": json.loads(original)}],
        "bedrock_content_blocks": [bedrock_thinking, {"toolUse": {"toolUseId": "batch",
            "name": "tool_call", "input": json.loads(original)}}],
    }
    expanded = expand_local_tool_batches([call], provider_data=provider_data)
    assert provider_data["anthropic_content_blocks"][0] is thinking
    assert provider_data["bedrock_content_blocks"][0] is bedrock_thinking
    for index, child in enumerate(expanded, 1):
        anthropic = provider_data["anthropic_content_blocks"][index]
        bedrock = provider_data["bedrock_content_blocks"][index]["toolUse"]
        assert anthropic["id"] == bedrock["toolUseId"] == child.id
        assert anthropic["input"] == bedrock["input"] == json.loads(child.function.arguments)
