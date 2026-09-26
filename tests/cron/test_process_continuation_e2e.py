"""Real cron agent/tool dispatch across producer exit, using a local model server."""

import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


def test_agent_continues_durable_result_in_a_fresh_scheduler_process(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  provider: custom\n  api_mode: chat_completions\n"
        "terminal:\n  env: local\n"
        "approvals:\n  cron_mode: off\n"
        "memory:\n  memory_enabled: false\n  user_profile_enabled: false\n")
    observed_tools, observed_completions = [], []

    class Provider(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            messages = request.get("messages", [])
            continuation = any("Completed Process" in str(m.get("content", "")) for m in messages)
            results = [m for m in messages if m.get("role") == "tool"]
            tools = request.get("tools", [])
            has_terminal = any(t.get("function", {}).get("name") == "terminal" for t in tools)
            message = {"role": "assistant", "content": "CONTINUATION_INSPECTED" if continuation else "LAUNCHED"}
            if continuation:
                observed_completions.append(messages)
            elif has_terminal and not results:
                message.update(content=None, tool_calls=[{
                    "id": "call_build", "type": "function", "function": {
                        "name": "terminal", "arguments": json.dumps({
                            "command": "printf 'BUILD_RECEIPT'; exit 7", "background": True,
                            "continue_on_complete": True})}}])
            elif results:
                observed_tools.extend(json.loads(m["content"]) for m in results)
            response = {"id": "chatcmpl-local", "object": "chat.completion", "created": 1,
                "model": "test-model", "choices": [{"index": 0, "message": message,
                "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}
            content_type = "application/json"
            if request.get("stream"):
                response["object"] = "chat.completion.chunk"
                response["choices"][0]["delta"] = response["choices"][0].pop("message")
                for index, tool in enumerate(message.get("tool_calls", [])):
                    tool["index"] = index
                raw = ("data: " + json.dumps(response) + "\n\ndata: [DONE]\n\n").encode()
                content_type = "text/event-stream"
            else:
                raw = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self.send_error(404)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    env = {**os.environ, "HERMES_HOME": str(home), "OPENAI_API_KEY": "local-test-only",
           "OPENAI_BASE_URL": url, "PYTHONPATH": str(repo)}
    producer = """
import sys
from cron import jobs, scheduler
job = jobs.create_job(prompt='Run the build in the background and inspect its completed result.',
    schedule='every 1h', model='test-model', provider='custom', base_url=sys.argv[1],
    enabled_toolsets=['terminal'], workdir=sys.argv[2])
assert scheduler.run_one_job(job)
"""
    consumer = """
from cron import continuations, executions, jobs, scheduler
pending = continuations.pending_jobs()
assert len(pending) == 1, pending
job_id = pending[0]['id']
before = jobs.get_job(job_id)
scheduler.tick(verbose=False)
scheduler.tick(verbose=False)
rows = executions.list_executions(job_id=job_id)
assert len(rows) == 2 and all(r['status']=='completed' for r in rows), rows
assert sum(r['source']=='continuation' for r in rows) == 1
after = jobs.get_job(job_id)
assert before['next_run_at'] == after['next_run_at']
assert before['repeat'] == after['repeat']
assert not continuations.pending_jobs()
"""
    try:
        for program, args in [(producer, [url, str(tmp_path)]), (consumer, [])]:
            result = subprocess.run([sys.executable, "-c", program, *args], env=env, cwd=tmp_path,
                                    capture_output=True, text=True, timeout=90)
            assert result.returncode == 0, result.stdout + result.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
    assert any(r.get("continue_on_complete") and not r.get("notify_on_complete") for r in observed_tools)
    assert len(observed_completions) == 1
    assert "BUILD_RECEIPT" in str(observed_completions[0])
