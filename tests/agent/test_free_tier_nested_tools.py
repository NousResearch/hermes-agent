"""Nested dispatch counts executed leaves, not connector-batch envelopes."""
import json
from types import SimpleNamespace


def test_connector_batch_counts_entries_without_counting_envelope(tmp_path, monkeypatch):
    from agent.free_tier import admit_turn
    from hermes_cli import auth, auth_nous, free_tier_usage as usage
    from model_tools import handle_function_call
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-auth"))
    guest = {"auth_method": "anonymous", "anon_token": "anon_nested_fixture"}
    auth._save_active_provider_state("nous", guest)
    auth_nous._write_shared_nous_state(guest)
    agent = SimpleNamespace(base_url="https://welcome-api.nousresearch.com/v1")
    from tools.tool_gateway.names import CONNECTOR_BATCH_SENTINEL
    calls = [{"name": "connectors__gmail__search", "arguments": {}},
             {"name": "connectors__gmail__read", "arguments": {}}]
    monkeypatch.setattr("model_tools._dispatch_bridge_tool", lambda name, *a: (
        (None, (CONNECTOR_BATCH_SENTINEL, {"calls": calls})) if name == "tool_call" else None))
    monkeypatch.setattr("model_tools._select_tool_names", lambda *a, **kw: {"manage_connections"})
    monkeypatch.setattr("model_tools_connectors.dispatch_connector_call", lambda *a: '{"response":{}}')
    with admit_turn(agent):
        result = handle_function_call("tool_call", {"calls": calls}, tool_call_id="batch")
    assert len(json.loads(result)["results"]) == 2
    assert usage.status()["tool_calls_used"] == 2
