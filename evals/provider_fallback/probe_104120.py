import os, sys, tempfile, pathlib, json, threading, time, socket
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

ROOT = sys.argv[1]
home = tempfile.mkdtemp(prefix="hermes-104120-")
os.environ.clear()
os.environ.update(
    HOME=home,
    HERMES_HOME=home + "/.hermes",
    PATH="/usr/bin:/bin",
    PYTHONDONTWRITEBYTECODE="1",
    TOKENIZERS_PARALLELISM="false",
)
pathlib.Path(home + "/.hermes").mkdir()
pathlib.Path(home + "/.hermes/config.yaml").write_text(
    "model:\n  context_length: 131072\nagent:\n  api_max_retries: 0\ncompression:\n  enabled: false\n",
    encoding="utf-8",
)
os.chdir(home)
sys.path.insert(0, ROOT)
# Egress safety boundary, not a patched production predicate.
real_connect = socket.socket.connect


def local_connect(self, address):
    if isinstance(address, tuple) and address[0] not in (
        "127.0.0.1",
        "::1",
        "localhost",
    ):
        raise RuntimeError("Blocked non-loopback egress: " + str(address))
    return real_connect(self, address)


socket.socket.connect = local_connect
requests = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"data": []}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = body.get("model")
        status = (
            429
            if model == "primary-rate" or model == "fallback-rate"
            else 401
            if model == "primary-auth"
            else 200
        )
        requests.append({"path": self.path, "model": model, "status": status})
        response = (
            {
                "error": {
                    "message": "Rate limit exceeded"
                    if status == 429
                    else "Invalid API key",
                    "type": "rate_limit_error"
                    if status == 429
                    else "authentication_error",
                }
            }
            if status != 200
            else {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "LOCAL_OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            }
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_port}/v1"
from run_agent import AIAgent
from agent.error_classifier import classify_api_error
from agent.agent_runtime_helpers import restore_primary_runtime

results = []
for label, model, chain in [
    ("primary_rate", "primary-rate", ["fallback-ok"]),
    ("non_rate_auth", "primary-auth", ["fallback-ok"]),
    ("fallback_chain", "primary-rate", ["fallback-rate", "fallback-ok"]),
    ("skipped_first", "primary-rate", ["primary-rate", "fallback-ok"]),
    ("unconfigured_first", "primary-rate", ["unconfigured", "fallback-ok"]),
]:
    agent = AIAgent(
        base_url=url,
        api_key="local-not-a-secret",
        provider="openai",
        api_mode="chat_completions",
        model=model,
        enabled_toolsets=[],
        quiet_mode=True,
        skip_memory=True,
        skip_context_files=True,
        skip_background_review=True,
        max_iterations=4,
        fallback_model=[
            (
                {"provider": "unconfigured-fixture", "model": "missing"}
                if m == "unconfigured"
                else {
                    "provider": "custom",
                    "model": m,
                    "base_url": url,
                    "api_key": "local-not-a-secret",
                    "api_mode": "chat_completions",
                }
            )
            for m in chain
        ],
    )
    transitions = []
    for step in range(len(chain) + 1):
        agent.client.max_retries = 0
        try:
            response = agent.client.chat.completions.create(
                model=agent.model,
                messages=[{"role": "user", "content": "Reply LOCAL_OK"}],
            )
            answer = response.choices[0].message.content
            break
        except Exception as exc:
            classified = classify_api_error(exc)
            before = getattr(agent, "_rate_limited_until", 0)
            before_count = getattr(agent, "_rate_limit_backoff_count", 0)
            switched = agent._try_activate_fallback(classified.reason)
            after = getattr(agent, "_rate_limited_until", 0)
            transitions.append({
                "reason": str(classified.reason),
                "switched": switched,
                "before_deadline": before,
                "after_deadline": after,
                "remaining_seconds": max(0, after - time.monotonic()),
                "backoff_before": before_count,
                "backoff_after": getattr(agent, "_rate_limit_backoff_count", 0),
                "notice": getattr(agent, "_pending_fallback_notice", [])[:],
                "active_model": agent.model,
            })
            if not switched:
                raise
    agent._emit_pending_fallback_notice()
    restored = restore_primary_runtime(agent)
    results.append({
        "case": label,
        "transitions": transitions,
        "answer": answer,
        "restore_before_expiry": restored,
        "model_after_restore": agent.model,
    })
    agent.close()
server.shutdown()
print(
    json.dumps(
        {
            "repo": ROOT,
            "home": home,
            "scope": "Real AIAgent init, real OpenAI SDK HTTP errors and success, real classifier and fallback/restore; not full conversation loop",
            "requests": requests,
            "results": results,
        },
        indent=2,
    )
)
assert len(results) == 5 and all(x["answer"] == "LOCAL_OK" for x in results)
assert results[0]["transitions"][0]["remaining_seconds"] > 0
assert results[1]["transitions"][0]["after_deadline"] == 0
assert (
    results[2]["transitions"][1]["before_deadline"]
    == results[2]["transitions"][1]["after_deadline"]
)
assert "Primary retry eligible in ~" in results[0]["transitions"][0]["notice"][-1]
assert "Primary retry eligible" not in results[1]["transitions"][0]["notice"][-1]
assert "Primary retry eligible" not in results[2]["transitions"][1]["notice"][-1]
assert all(r["transitions"][0]["backoff_after"] == 1 for r in results[3:])
print(
    "PROBE_OK: armed cooldown notice only; skipped and unconfigured entries do not compound backoff"
)
