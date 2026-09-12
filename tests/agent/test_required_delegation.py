"""Required delegation through real plugin discovery, agent loops and an HTTP peer."""

import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
import time

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_turn(tmp_path, *, mode="required", max_iterations=8):
    home = tmp_path / "profile"
    home.mkdir()
    plugin = home / "plugins" / "required-worker"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: required-worker\nversion: 1.0.0\n")
    (plugin / "__init__.py").write_text(textwrap.dedent('''\
        from agent.subagent_lifecycle import SubagentLaunchRequest

        def register(ctx):
            def route(user_message, parent_session_id, turn_id, **kwargs):
                if parent_session_id or not str(user_message).startswith("@worker "):
                    return
                for index in range(ctx.get_config("worker_count", 1)):
                    ctx.subagent_lifecycle.require(SubagentLaunchRequest(
                        goal=str(user_message)[8:],
                        model=ctx.get_config("worker_model"),
                        correlation_id=f"required:{turn_id}:{index}",
                        allowed_toolsets=("web",) if ctx.get_config("invalid", False) else None,
                    ))
            ctx.register_hook("pre_llm_call", route)
    '''))
    context = tmp_path / "context.txt"
    context.write_text("PARENT_CONTEXT_COLLECTED")
    requests, errors = [], []
    context_collected = threading.Event()

    class Provider(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_error(404)

        def do_POST(self):
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                messages = request.get("messages", [])
                if not messages:
                    self.send_error(404)
                    return
                requests.append(request)
                if request["model"] == "worker-model":
                    if mode == "interrupted":
                        (tmp_path / "cancel").touch()
                        deadline = time.monotonic() + 20
                        while not (tmp_path / "cancelled").exists() and time.monotonic() < deadline:
                            time.sleep(0.01)
                    if mode == "worker_failure":
                        raw = json.dumps({"error": {
                            "message": "Required worker fixture failure",
                            "type": "invalid_request_error", "code": "invalid_request_error",
                        }}).encode()
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(raw)))
                        self.end_headers()
                        self.wfile.write(raw)
                        return
                    if (max_iterations > 2 and mode not in {"no_tools", "interrupted"}
                            and not context_collected.wait(timeout=20)):
                        raise AssertionError("parent did not continue using its tools")
                    message = {"role": "assistant", "content": "REQUIRED_WORKER_RESULT"}
                else:
                    tool_text = "\n".join(str(m.get("content", "")) for m in messages if m["role"] == "tool")
                    if "REQUIRED_WORKER_RESULT" in tool_text:
                        message = {"role": "assistant", "content": "RESULT_REVIEWED"}
                    elif mode == "no_tools":
                        message = {"role": "assistant", "content": "PREMATURE_DONE"}
                    elif "PARENT_CONTEXT_COLLECTED" in tool_text:
                        context_collected.set()
                        message = {"role": "assistant", "content": "PREMATURE_DONE"}
                    else:
                        message = {"role": "assistant", "content": None, "tool_calls": [{
                            "id": "call_context", "type": "function", "function": {
                                "name": "read_file", "arguments": json.dumps({"path": str(context)}),
                            },
                        }]}
                response = {
                    "id": "chatcmpl-required", "object": "chat.completion", "created": 1,
                    "model": request["model"], "choices": [{
                        "index": 0, "message": message,
                        "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
                    }], "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                }
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
            except (BrokenPipeError, ConnectionResetError):
                if mode != "interrupted":
                    errors.append("unexpected disconnected model request")
            except Exception as exc:
                errors.append(repr(exc))
                self.send_error(500)

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    (home / "config.yaml").write_text(
        "memory:\n  memory_enabled: false\n  user_profile_enabled: false\n"
        "agent:\n  skip_background_review: true\n"
        "terminal:\n  env: local\n"
        f"delegation:\n  provider: custom\n  base_url: {url}\n"
        "  api_key: local-test-only\n  model: worker-model\n"
        "plugins:\n  enabled: [required-worker]\n  entries:\n    required-worker:\n"
        "      settings:\n        worker_model: worker-model\n"
        f"        worker_count: {2 if mode == 'two_workers' else 1}\n"
        f"        invalid: {'true' if mode == 'launch_failure' else 'false'}\n"
    )
    query = "ordinary task" if mode == "unmatched" else "@worker Implement the assigned change."
    output = tmp_path / "result.json"
    program = textwrap.dedent('''\
        import json, sys, threading, time
        from pathlib import Path
        from run_agent import AIAgent
        agent = AIAgent(
            model="parent-model", provider="custom", base_url=sys.argv[1], api_key="local-test-only",
            enabled_toolsets=["file"] if sys.argv[5] == "tool_unavailable" else ["file", "delegation"],
            skip_context_files=True, skip_memory=True,
            quiet_mode=True, save_trajectories=False, max_iterations=int(sys.argv[4]),
            run_budget_seconds=30,
        )
        streamed = []
        def cancel_on_request():
            deadline = time.monotonic() + 20
            while not Path("cancel").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            if Path("cancel").exists():
                agent.hard_interrupt("user cancelled required work")
                Path("cancelled").touch()
        if sys.argv[5] == "interrupted":
            threading.Thread(target=cancel_on_request, daemon=True).start()
        result = agent.run_conversation(sys.argv[2], stream_callback=streamed.append)
        result["observed_stream"] = streamed
        Path(sys.argv[3]).write_text(json.dumps(result, default=str))
        agent.close()
    ''')
    env = {key: os.environ[key] for key in (
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "LOCALAPPDATA", "APPDATA",
    ) if key in os.environ}
    env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path), HERMES_HOME=str(home),
               HERMES_MANAGED_DIR=str(tmp_path / "managed"), TERMINAL_CWD=str(tmp_path),
               OPENAI_BASE_URL=url, OPENAI_API_KEY="local-test-only", PYTHONPATH=str(REPO_ROOT),
               PYTHONDONTWRITEBYTECODE="1", LANG="C.UTF-8")
    try:
        process = subprocess.run(
            [sys.executable, "-c", program, url, query, str(output), str(max_iterations), mode],
            cwd=tmp_path, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
            encoding="utf-8", timeout=60,
        )
    finally:
        context_collected.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
    assert process.returncode == 0, (process.stdout, process.stderr, requests, errors)
    assert not errors, (errors, process.stderr)
    errors_log = home / "logs" / "errors.log"
    diagnostics = process.stderr + (errors_log.read_text() if errors_log.exists() else "")
    return json.loads(output.read_text()), requests, diagnostics


@pytest.mark.parametrize("mode", ["required", "no_tools", "two_workers", "unmatched"])
def test_required_worker_runs_without_a_model_delegation_call(tmp_path, mode):
    result, requests, stderr = _run_turn(tmp_path, mode=mode)
    parents = [r for r in requests if r.get("model") == "parent-model"]
    children = [r for r in requests if r.get("model") == "worker-model"]
    assert result["completed"] and not result["failed"], (result, stderr)
    if mode == "unmatched":
        assert not children
        assert "required_delegations" not in result
        assert result["final_response"] == "PREMATURE_DONE"
    else:
        count = 2 if mode == "two_workers" else 1
        assert len(children) == count, (result, stderr)
        assert result["final_response"] == "RESULT_REVIEWED", (result, stderr)
        assert [r["state"] for r in result["required_delegations"]] == ["SUCCEEDED"] * count
        assert all(r["result_consumed"] for r in result["required_delegations"])
        if mode != "no_tools":
            assert any("PARENT_CONTEXT_COLLECTED" in str(r["messages"]) for r in parents)
        assert "PREMATURE_DONE" not in str(result["messages"])
        assert "PREMATURE_DONE" not in "".join(result["observed_stream"])
    # Runtime receipts may append rows; neither schemas nor a sent prefix may change.
    for previous, following in zip(parents, parents[1:]):
        assert previous["tools"] == following["tools"]
        assert previous["messages"] == following["messages"][:len(previous["messages"])]


@pytest.mark.parametrize("mode,max_iterations", [
    ("launch_failure", 8), ("tool_unavailable", 8), ("worker_failure", 8),
    ("required", 2), ("interrupted", 8),
])
def test_missing_required_result_cannot_be_reported_as_success(tmp_path, mode, max_iterations):
    result, requests, stderr = _run_turn(tmp_path, mode=mode, max_iterations=max_iterations)
    assert result["failed"] and not result["completed"], (result, stderr)
    assert result["failure_reason"] == "required_delegation_incomplete", (result, stderr)
    assert result["required_delegations"], (result, stderr)
    assert result["final_response"] != "PREMATURE_DONE", (result, stderr)
    if mode in {"launch_failure", "tool_unavailable"}:
        assert not requests
    if mode == "interrupted":
        assert result["interrupted"]
