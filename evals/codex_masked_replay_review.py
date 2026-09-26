"""Local HTTP contract review of masked-error recovery; NOT live-provider proof.

Run in an isolated HOME/HERMES_HOME under the campaign test lock. Responses are
explicit synthetic fixtures; the client, transport, and AIAgent loop are real.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_agent
from agent import error_classifier
from openai import APIError
import httpx

ITEM = {"type": "reasoning", "id": "rs_review", "encrypted_content": "signed-control-opaque-do-not-alter", "summary": []}
ERROR = {"message": "Request blocked.", "type": "invalid_request_error", "param": None, "code": "invalid_prompt"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def scenario(name, provider="openai-codex", replay=True):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if "input" not in payload:
                raw = json.dumps({"id": "chat-local", "object": "chat.completion", "created": 1,
                                  "model": "gpt-5-codex", "choices": [{"index": 0, "finish_reason": "stop",
                                  "message": {"role": "assistant", "content": "local auxiliary fixture"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            requests.append(payload)
            has_replay = any(i.get("type") == "reasoning" for i in payload["input"])
            error = copy.deepcopy(ERROR)
            if name == "explicit_encrypted":
                error.update(code="invalid_encrypted_content", message="The encrypted content could not be verified.")
            reject = name in {"unrelated_block", "failed_frame"} or (name not in {"success", "failed_frame"} and has_replay)
            if reject and name == "failed_frame":
                # The shape openai/codex issues show for a blocked turn: response.in_progress, then a
                # ``response.failed`` terminal frame carrying the error (no HTTP status).
                failed = {"id": "resp_review", "object": "response", "created_at": 1, "status": "failed",
                          "model": "gpt-5-codex", "output": [], "error": error}
                events = [{"type": "response.created", "response": dict(failed, status="in_progress", error=None), "sequence_number": 0},
                          {"type": "response.in_progress", "response": dict(failed, status="in_progress", error=None), "sequence_number": 1},
                          {"type": "response.failed", "response": failed, "sequence_number": 2}]
                raw = "".join("data: " + json.dumps(e) + "\n\n" for e in events).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
            elif reject:
                raw = json.dumps({"error": error}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
            else:
                response = {"id": "resp_review", "object": "response", "created_at": 1,
                            "status": "completed", "model": "gpt-5-codex", "output": [
                                {"id": "msg_review", "type": "message", "role": "assistant", "status": "completed",
                                 "content": [{"type": "output_text", "text": "local fixture success", "annotations": []}]}],
                            "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}}
                created = dict(response, status="in_progress", output=[])
                events = [{"type": "response.created", "response": created, "sequence_number": 0},
                          {"type": "response.output_item.done", "item": response["output"][0], "output_index": 0, "sequence_number": 1},
                          {"type": "response.completed", "response": response, "sequence_number": 2}]
                raw = "".join("data: " + json.dumps(e) + "\n\n" for e in events).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("x-request-id", f"local-{name}-{len(requests)}")
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    agent = run_agent.AIAgent(
        model="gpt-5-codex", provider=provider, api_mode="codex_responses",
        base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="local-fixture-key",
        enabled_toolsets=["terminal"], max_iterations=2, quiet_mode=True,
        reasoning_config={"effort": "high"}, skip_context_files=True,
        skip_memory=True, skip_background_review=True, save_trajectories=False,
        session_id=f"local-review-{name}-{provider}-{replay}",
    )
    agent._api_max_retries = 3  # show whether a deterministic rejection burns identical retries
    history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hello"}]
    if replay:
        history[1]["codex_reasoning_items"] = [copy.deepcopy(ITEM)]
    try:
        result = agent.run_conversation("reply briefly", system_message="Local contract review.", conversation_history=history)
        reasoning_counts = [sum(i.get("type") == "reasoning" for i in p["input"]) for p in requests]
        stable = {k: len({digest(p.get(k)) for p in requests}) <= 1
                  for k in ["reasoning", "instructions", "tools"]}
        signed_before = [i for i in requests[0]["input"] if i.get("type") == "reasoning"] if requests else []
        return {"name": name, "provider": provider, "replay": replay, "completed": result.get("completed"),
                "requests": len(requests), "reasoning_counts": reasoning_counts,
                "stable_retry_fields": stable, "reasoning_parameters": [p.get("reasoning") for p in requests],
                "tool_count": len(requests[0].get("tools", [])) if requests else 0,
                "signed_payload_unchanged": [i.get("encrypted_content") for i in signed_before] == ([ITEM["encrypted_content"]] if replay else []),
                "replay_enabled_after": agent._codex_reasoning_replay_enabled,
                "canonical_replay_after": any(m.get("codex_reasoning_items") for m in result["messages"])}
    finally:
        agent.close()
        server.shutdown()
        server.server_close()
        thread.join()


def main():
    print(json.dumps({"run_agent": run_agent.__file__, "classifier": error_classifier.__file__,
                      "home": os.environ["HERMES_HOME"]}))
    results = [scenario("success"), scenario("masked_replay"), scenario("explicit_encrypted"),
               scenario("unrelated_block"), scenario("unrelated_block", replay=False),
               scenario("masked_replay", provider="custom"), scenario("failed_frame"), scenario("failed_frame", replay=False)]
    e = APIError("Request blocked.", request=httpx.Request("POST", "http://127.0.0.1"), body=ERROR)
    output = {"surface": "real AIAgent and SDK against synthetic local HTTP/SSE server; not provider proof",
              "statusless_reason": error_classifier.classify_api_error(e, provider="openai-codex").reason.value,
              "scenarios": results}
    Path(sys.argv[1]).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
