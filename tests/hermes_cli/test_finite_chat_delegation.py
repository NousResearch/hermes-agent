"""Finite chat must consume parallel delegated results before the CLI exits."""

import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", ["quiet", "oneshot", "redirected"])
def test_finite_chat_joins_parallel_children_before_final_response(tmp_path, mode):
    """Real parser, CLI, agent loops and delegate_task. Only inference is synthetic.

    Both child HTTP requests must reach the barrier before either can finish.
    The provider synthesizes a final answer only from returned tool results,
    not from child requests or transcripts that the parent has not consumed.
    """
    home = tmp_path / "profile"
    home.mkdir()
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    children, joined_results, errors = [], [], []
    workers = ("WORKER_ALPHA", "WORKER_BETA")

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
                users = [str(m.get("content", "")) for m in messages if m["role"] == "user"]
                worker = next((w for w in workers if w in users[-1]), None)
                results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
                message: dict[str, Any]
                if worker:
                    with lock:
                        children.append(worker)
                    barrier.wait(timeout=20)
                    message = {"role": "assistant", "content": worker + "_COMPLETE"}
                elif results:
                    with lock:
                        joined_results.extend(results)
                    summaries = json.dumps(results)
                    joined = all(w + "_COMPLETE" in summaries for w in workers)
                    message = {"role": "assistant", "content": (
                        "FANOUT_JOINED_AND_SYNTHESIZED" if joined else "PARENT_ENDED_BEFORE_JOIN"
                    )}
                else:
                    message = {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "call_fanout", "type": "function", "function": {
                            "name": "delegate_task", "arguments": json.dumps({
                                "tasks": [{"goal": f"Complete {w} and return its completion token."}
                                          for w in workers],
                            }),
                        },
                    }]}
                response = {
                    "id": "chatcmpl-local", "object": "chat.completion", "created": 1,
                    "model": "test-model", "choices": [{
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
            except Exception as exc:
                with lock:
                    errors.append(repr(exc))
                self.send_error(500)

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    (home / "config.yaml").write_text(
        f"model:\n  provider: custom\n  base_url: {url}\n  api_mode: chat_completions\n"
        "memory:\n  memory_enabled: false\n  user_profile_enabled: false\n"
        "terminal:\n  env: local\n  oneshot_completion_wait_seconds: 1\n"
        f"delegation:\n  max_concurrent_children: 2\n  base_url: {url}\n"
        "  api_key: local-test-only\n  model: test-model\n",
        encoding="utf-8",
    )
    query = tmp_path / "query.txt"
    query.write_text("Delegate two independent tasks, then synthesize their results.", encoding="utf-8")
    # Retain native Windows location variables, but never inherited credentials
    # or another session's finite/approval/runtime markers.
    env = {key: os.environ[key] for key in (
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "LOCALAPPDATA", "APPDATA",
    ) if key in os.environ}
    env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path), HERMES_HOME=str(home),
               HERMES_MANAGED_DIR=str(tmp_path / "managed"), TERMINAL_CWD=str(tmp_path),
               OPENAI_BASE_URL=url, OPENAI_API_KEY="local-test-only", PYTHONPATH=str(REPO_ROOT),
               PYTHONDONTWRITEBYTECODE="1", LANG="C.UTF-8")
    mode_flags = {"quiet": ["-Q"], "oneshot": ["--oneshot"], "redirected": []}
    command = [
        sys.executable, "-c", "from hermes_cli.main import main; main()", "chat",
        *mode_flags[mode], "--provider", "custom", "--model", "test-model",
        "--toolsets", "delegation", "--ignore-rules", "--query-file", str(query),
        "--reasoning", "high", "--max-turns", "10", "--run-budget", "60",
    ]
    try:
        result = subprocess.run(command, cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, encoding="utf-8", timeout=75)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    diagnostic = (result.stdout, result.stderr, joined_results, children, errors)
    assert result.returncode == 0, diagnostic
    assert "FANOUT_JOINED_AND_SYNTHESIZED" in result.stdout, diagnostic
    assert "PARENT_ENDED_BEFORE_JOIN" not in result.stdout, diagnostic
    assert sorted(children) == sorted(workers), diagnostic
    assert not errors, diagnostic
    assert len(joined_results) == 1, diagnostic
    assert [r["status"] for r in joined_results[0]["results"]] == ["completed", "completed"]
    assert [r["summary"] for r in joined_results[0]["results"]] == [w + "_COMPLETE" for w in workers]


@pytest.mark.parametrize("query,image", [("Delegate a task", None), (None, "image.png")])
def test_tty_seeded_chat_keeps_background_delegation(monkeypatch, query, image):
    """A TTY -q (or image seed) still owns a later-turn completion consumer."""
    import cli
    import tools.delegate_tool as dt
    from gateway.session_context import reset_session_vars
    from run_agent import AIAgent
    from tests.tools.test_delegate import _make_mock_parent

    reset_session_vars()
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(cli, "_collect_query_images", lambda q, i: (q, [i] if i else []))
    parent = _make_mock_parent()
    parent.session_id = "interactive-parent"
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: MagicMock())
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **kw: {
        "model": "test-model", "provider": "custom", "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    })
    dispatched = []

    def dispatch(unit, unit_id, slot_key, routing):
        dispatched.append(unit)
        return {"status": "dispatched", "delegation_id": "interactive-delegation"}

    monkeypatch.setattr("tools.delegate_tool_dispatch._dispatch_unit", dispatch)
    seeded = SimpleNamespace(run=lambda: AIAgent._dispatch_delegate_task(
        parent, {"tasks": [{"goal": "independent task"}]},
    ))
    try:
        result = cli._run_single_query_mode(seeded, query, image, False, False)
        assert isinstance(result, str)
        payload = json.loads(result)
        assert payload["status"] == "dispatched"
        assert len(dispatched) == 1
        assert "HERMES_SINGLE_QUERY_SESSION" not in os.environ
    finally:
        reset_session_vars()
