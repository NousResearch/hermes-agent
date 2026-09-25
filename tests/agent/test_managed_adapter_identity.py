"""Reported identity on real managed Gemini HTTP and ACP subprocess paths.

The fictional temp-home policies exercise adapters, not operational roster admission.
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


def _receipt(home, endpoint, model, provider):
    from agent.model_selection import select
    from agent.model_selection_store import activate_policy, persist_receipt, publish_policy

    policy = dict(schema_version=1, policy_id="identity", revision=1, approval_ref="fixture",
                  routes=[dict(route_id="fixture", route_revision=1, provider=provider, model=model,
                               endpoint=endpoint, maker="fixture", model_family="fixture", status="approved",
                               allowed_roles=["builder"], capabilities=[], verified_input_budget=200000,
                               allowed_reasoning=["high"], qualifications=["deep"], assessment="fixture", evidence=[])],
                  rankings={"builder": {"deep": ["fixture"]}})
    requirements = dict(schema_version=1, execution_kind="delegation", execution_id="identity",
                        attempt_id="1", slot_id="", role="builder", task_class="cross-component",
                        required_capabilities=[], input_tokens=1000, reserve_tokens=8192, reasoning="high",
                        provenance=dict(frozen_sha="fixture", verified_by="fixture", complete=True, contributors=[]))
    publish_policy(home, policy, approval_ref="fixture")
    activate_policy(home, "identity", 1)
    return persist_receipt(home, select(requirements, policy, {}, now=1))


@pytest.fixture
def gemini_endpoint():
    class Handler(BaseHTTPRequestHandler):
        requests = []
        reported = None

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if "/models/" in self.path:
                self.requests.append((self.path, payload))
            response = dict(candidates=[dict(content=dict(parts=[dict(text="done")]), finishReason="STOP")])
            if self.reported is not None:
                response["modelVersion"] = self.reported
            stream = "streamGenerateContent" in self.path
            body = (("data: " + json.dumps(response) + "\n\n") if stream else json.dumps(response)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # The native-client resolver recognizes the Google API path; all traffic stays loopback.
        yield f"http://127.0.0.1:{server.server_port}/generativelanguage.googleapis.com/v1beta", Handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("reported", [None, "gemini-OTHER", "gemini-2.5-flash"])
@pytest.mark.parametrize("boundary", ["normal", "streaming", "auxiliary"])
def test_managed_gemini_records_wire_identity(tmp_path, monkeypatch, gemini_endpoint, reported, boundary):
    from agent.gemini_native_adapter import GeminiNativeClient
    from agent.model_selection_store import get_receipt, is_route_revoked, list_outcomes
    from run_agent import AIAgent

    url, handler = gemini_endpoint
    handler.reported = reported
    model = "gemini-2.5-flash"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    receipt_id = _receipt(tmp_path, url, model, "gemini")
    original = get_receipt(tmp_path, receipt_id)
    agent = AIAgent(api_key="fixture-key", provider="gemini", model=model, base_url=url,
                    max_iterations=1, enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
                    skip_memory=True, save_trajectories=False, reasoning_config={"enabled": True, "effort": "high"},
                    request_overrides={"reasoning_effort": "high"})
    agent._disable_streaming = boundary != "streaming"
    agent._managed_routing_home = tmp_path
    agent._managed_routing_receipt_id = receipt_id
    try:
        assert isinstance(agent.client, GeminiNativeClient)
        if boundary == "auxiliary":
            assert _auxiliary_result(agent).choices[0].message.content == "done"
        else:
            assert "done" in agent.run_conversation("private identity prompt")["final_response"]
        assert len(handler.requests) == 1
        assert ("streamGenerateContent" in handler.requests[0][0]) == (boundary != "normal")
        outcomes = list_outcomes(tmp_path, receipt_id)
        assert any(row["kind"] == "routing_wire_validated" for row in outcomes)
        health = [row["payload"] for row in outcomes if row["kind"] == "routing_health"]
        assert len(health) == 1
        assert health[0]["reported_model"] == reported
        assert health[0]["identity_status"] == ("missing" if reported is None else "matching" if reported == model else "changed")
        assert health[0]["status"] == "healthy" and health[0]["replay_safe"] is False
        assert get_receipt(tmp_path, receipt_id) == original
        assert is_route_revoked(tmp_path, "identity", "fixture") is None
        assert "private identity prompt" not in json.dumps(health)
    finally:
        agent.close()


def _auxiliary_result(agent):
    from agent.auxiliary_client import _create_with_progress
    from agent.managed_route_aux_wire import observe_moa_request
    from agent.managed_route_health import record_response_identity

    with observe_moa_request(agent._managed_routing_home, agent._managed_routing_receipt_id) as observation:
        return record_response_identity(observation, _create_with_progress(agent.client, dict(
            model=agent.model, messages=[dict(role="user", content="private identity prompt")],
            reasoning_effort="high", max_tokens=8192,
        ), force_stream=True))


@pytest.mark.parametrize("boundary", ["normal", "auxiliary"])
def test_managed_acp_does_not_report_requested_model(tmp_path, monkeypatch, boundary):
    from agent.copilot_acp_client import CopilotACPClient
    from agent.model_selection_store import list_outcomes
    from run_agent import AIAgent

    # An actual JSON-RPC peer, with no provider identity in its prompt response.
    peer = tmp_path / "peer.py"
    peer.write_text('''import json, sys
for line in sys.stdin:
    request = json.loads(line)
    method = request["method"]
    result = {"sessionId": "fixture"} if method == "session/new" else {}
    if method == "session/prompt":
        print(json.dumps({"jsonrpc": "2.0", "method": "session/update", "params": {
            "sessionId": "fixture", "update": {"sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "done"}}}}), flush=True)
        result = {"stopReason": "end_turn"}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
''')
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    receipt_id = _receipt(tmp_path, "acp://copilot", "fixture-model", "copilot-acp")
    agent = AIAgent(api_key="fixture-key", provider="copilot-acp", model="fixture-model",
                    base_url="acp://copilot", acp_command=sys.executable, acp_args=[str(peer)],
                    max_iterations=1, enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
                    skip_memory=True, save_trajectories=False, reasoning_config={"enabled": True, "effort": "high"},
                    request_overrides={"reasoning_effort": "high"})
    agent._managed_routing_home = tmp_path
    agent._managed_routing_receipt_id = receipt_id
    try:
        assert isinstance(agent.client, CopilotACPClient)
        if boundary == "auxiliary":
            assert _auxiliary_result(agent).choices[0].message.content == "done"
        else:
            assert "done" in agent.run_conversation("private ACP prompt")["final_response"]
        outcomes = list_outcomes(tmp_path, receipt_id)
        assert any(row["kind"] == "routing_wire_validated" for row in outcomes)
        health = [row["payload"] for row in outcomes if row["kind"] == "routing_health"]
        assert len(health) == 1
        assert health[0]["reported_model"] is None
        assert health[0]["identity_status"] == "missing"
        assert health[0]["status"] == "healthy" and health[0]["replay_safe"] is False
        assert "private ACP prompt" not in json.dumps(health)
    finally:
        agent.close()
