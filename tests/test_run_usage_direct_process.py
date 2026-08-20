from __future__ import annotations

import os
import signal
import subprocess
import sys
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent.run_usage_ledger import UsageLedger


def _isolated_cli_env(home: Path, run_id: str) -> dict[str, str]:
    """Launch a CLI child with an explicit, credential-free environment."""
    env = {key: os.environ[key] for key in ("PATH", "TZ", "LANG", "LC_ALL") if key in os.environ}
    env.update({
        "HOME": str(home / "home"),
        "HERMES_HOME": str(home),
        "HERMES_RUN_ID": run_id,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    })
    return env


def _write_cli_config(home: Path, *, base_url: str, provider: str = "custom",
                      model: str = "local-model", api_mode: str = "chat_completions",
                      extra: str = "", api_max_retries: int = 1) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"""model:
  provider: {provider}
  base_url: {base_url}
  api_mode: {api_mode}
  api_key: test-key
  default: {model}
agent:
  api_max_retries: {api_max_retries}
  max_turns: 2
  verify_on_stop: false
{extra}
""",
        encoding="utf-8",
    )


def _run_natural_cli(home: Path, run_id: str, *, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "-z", "hello"],
        cwd=Path(__file__).resolve().parents[1],
        env=_isolated_cli_env(home, run_id),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _spawn_natural_cli(home: Path, run_id: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "hermes_cli.main", "-z", "hello"],
        cwd=Path(__file__).resolve().parents[1],
        env=_isolated_cli_env(home, run_id),
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_direct_hermes_style_process_writes_receipt_without_card(tmp_path):
    code = """
from hermes_cli.lifecycle import invoke_hook
invoke_hook('on_session_start', session_id='direct-session', model='direct-model', provider='direct-provider', platform='cli')
invoke_hook('post_api_request', session_id='direct-session', turn_id='direct-turn', api_request_id='direct-api', model='direct-model', provider='direct-provider', usage={'input_tokens': 4, 'output_tokens': 3}, cost_usd=0.01)
invoke_hook('on_session_finalize', session_id='direct-session', completed=True, platform='cli')
"""
    env = {
        **os.environ,
        "HERMES_HOME": str(tmp_path),
        "HERMES_RUN_ID": "direct-process-run",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    subprocess.run([sys.executable, "-c", code], env=env, check=True)

    receipt = UsageLedger(tmp_path / "state.db").get_run("direct-process-run")
    assert receipt["task_id"] is None
    assert receipt["process_id"] != ""
    assert receipt["session_id"] == "direct-session"
    assert receipt["input_tokens"] == 4
    assert receipt["output_tokens"] == 3
    assert receipt["outcome"] == "completed"


def test_two_processes_resuming_one_session_keep_distinct_receipts(tmp_path):
    code = """
from hermes_cli.lifecycle import invoke_hook
invoke_hook('on_session_start', session_id='resumed', model='m', provider='p')
invoke_hook('post_api_request', session_id='resumed', turn_id='t', api_request_id='a', model='m', provider='p', usage={'input_tokens': 1, 'output_tokens': 1})
invoke_hook('on_session_finalize', session_id='resumed', completed=True)
"""
    env = {**os.environ, "HERMES_HOME": str(tmp_path), "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
    import sqlite3
    with sqlite3.connect(tmp_path / "state.db") as connection:
        rows = connection.execute("SELECT run_id, process_id FROM usage_runs WHERE session_id='resumed'").fetchall()
    assert len(rows) == 2
    assert rows[0][0] != rows[1][0]
    assert rows[0][1] != rows[1][1]


def test_real_aiagent_conversation_lifecycle_writes_direct_receipt(tmp_path, monkeypatch):
    from hermes_cli.lifecycle import finalize_session
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from run_agent import AIAgent

    token = set_hermes_home_override(tmp_path)
    monkeypatch.setenv("HERMES_RUN_ID", "e2e-direct-run")
    try:
        class Handler(BaseHTTPRequestHandler):
            requests = []

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                Handler.requests.append(json.loads(self.rfile.read(length)))
                chunks = [
                    {"id": "chatcmpl-local", "object": "chat.completion.chunk", "created": 1,
                     "model": "fake/local-model", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "done"}, "finish_reason": None}]},
                    {"id": "chatcmpl-local", "object": "chat.completion.chunk", "created": 1,
                     "model": "fake/local-model", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}},
                ]
                body = b"".join((b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks)) + b"data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()

            def log_message(self, *_args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            agent = AIAgent(
                api_key="test-key",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                provider="local",
                model="fake/local-model",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_id="e2e-session",
                platform="cli",
            )
            result = agent.run_conversation("hello")
            assert result["final_response"] == "done"
            finalize_session(session_id="e2e-session", platform="cli", completed=True)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        receipt = UsageLedger(tmp_path / "state.db").get_run("e2e-direct-run")
        assert receipt["task_run_id"] is None
        assert receipt["model"]
        assert receipt["provider"]
        assert receipt["input_tokens"] == 9
        assert receipt["output_tokens"] == 4
        assert receipt["cost_usd"] == 0.0
        assert receipt["outcome"] == "completed"
    finally:
        reset_hermes_home_override(token)


def test_real_aiagent_codex_responses_lifecycle_writes_usage_receipt(tmp_path, monkeypatch):
    from hermes_cli.lifecycle import finalize_session
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from run_agent import AIAgent

    token = set_hermes_home_override(tmp_path)
    monkeypatch.setenv("HERMES_RUN_ID", "e2e-codex-run")
    try:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                events = [
                    {"type": "response.created", "response": {"id": "resp-local", "model": "fake/codex-model"}},
                    {"type": "response.output_text.delta", "delta": "codex-done"},
                    {"type": "response.completed", "response": {"id": "resp-local", "model": "fake/codex-model", "usage": {"input_tokens": 11, "output_tokens": 5, "total_tokens": 16}}},
                ]
                body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()

            def log_message(self, format, *args):  # noqa: A002, ANN001
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            agent = AIAgent(
                api_key="test-key",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                provider="openai-codex",
                model="fake/codex-model",
                api_mode="codex_responses",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_id="e2e-codex-session",
                platform="cli",
            )
            result = agent.run_conversation("hello")
            assert result["final_response"] == "codex-done"
            finalize_session(session_id="e2e-codex-session", platform="cli", completed=True)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        receipt = UsageLedger(tmp_path / "state.db").get_run("e2e-codex-run")
        assert receipt["input_tokens"] == 11
        assert receipt["output_tokens"] == 5
        assert receipt["model"] == "fake/codex-model"
        assert receipt["provider"] == "openai-codex"
        assert receipt["outcome"] == "completed"
    finally:
        reset_hermes_home_override(token)


def test_natural_cli_chat_completion_writes_direct_receipt_without_manual_finalize(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            json.loads(self.rfile.read(length))
            chunks = [
                {"id": "natural-chat", "object": "chat.completion.chunk", "model": "local-chat",
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": "natural-ok"}, "finish_reason": None}]},
                {"id": "natural-chat", "object": "chat.completion.chunk", "model": "local-chat",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18}},
            ]
            body = b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks)
            body += b"data: [DONE]\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / "hermes-home"
    _write_cli_config(home, base_url=f"http://127.0.0.1:{server.server_port}/v1", model="local-chat")
    try:
        result = _run_natural_cli(home, "natural-cli-direct")
        assert result.returncode == 0, result.stderr
        assert "natural-ok" in result.stdout
        receipt = UsageLedger(home / "state.db").get_run("natural-cli-direct")
        assert receipt["task_run_id"] is None
        assert receipt["board"] is None
        assert receipt["input_tokens"] == 13
        assert receipt["output_tokens"] == 5
        assert "cost_usd" in receipt
        assert receipt["cost_usd"] == 0.0
        assert receipt["outcome"] == "completed"
        assert receipt["ended_at"] is not None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_natural_cli_sigint_finalizes_measured_usage_once(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            json.loads(self.rfile.read(length))
            if self.path.endswith("/api/show"):
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
                return
            requests.append(self.path)
            if len(requests) == 1:
                body = (b'data: {"id":"interrupt","object":"chat.completion.chunk","model":"interrupt-model",'
                        b'"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"todo-1",'
                        b'"type":"function","function":{"name":"todo","arguments":"{\\"todos\\":[]}"}}]},'
                        b'"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":23,"completion_tokens":4,"total_tokens":27}}\n\n'
                        b"data: [DONE]\n\n")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
                return
            entered.set()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.flush()
            release.wait(timeout=10)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / "hermes-home"
    _write_cli_config(home, base_url=f"http://127.0.0.1:{server.server_port}/v1", model="interrupt-model")
    process = _spawn_natural_cli(home, "natural-interrupt")
    try:
        assert entered.wait(timeout=5)
        os.killpg(process.pid, signal.SIGINT)
        process.communicate(timeout=10)
        deadline = time.monotonic() + 5
        receipt = None
        while time.monotonic() < deadline:
            try:
                receipt = UsageLedger(home / "state.db").get_run("natural-interrupt")
                if receipt["input_tokens"]:
                    break
            except KeyError:
                pass
            time.sleep(0.05)
        assert receipt is not None
        assert receipt["ended_at"] is not None
        assert receipt["outcome"] == "interrupted"
        assert receipt["failure_reason"]
        assert receipt["input_tokens"] == 23
        assert receipt["output_tokens"] == 4
        import sqlite3
        with sqlite3.connect(home / "state.db") as connection:
            assert connection.execute("SELECT COUNT(*) FROM usage_events WHERE run_id='natural-interrupt' AND event_type='model'").fetchone()[0] == 1
            assert connection.execute("SELECT COUNT(*) FROM usage_events WHERE run_id='natural-interrupt'").fetchone()[0] == 2
    finally:
        release.set()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_natural_cli_sigkill_leaves_open_receipt(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            json.loads(self.rfile.read(length))
            if self.path.endswith("/api/show"):
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
                return
            entered.set()
            release.wait(timeout=10)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / "hermes-home"
    _write_cli_config(home, base_url=f"http://127.0.0.1:{server.server_port}/v1", model="kill-model")
    process = _spawn_natural_cli(home, "natural-kill")
    try:
        assert entered.wait(timeout=5)
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
        deadline = time.monotonic() + 5
        receipt = None
        while time.monotonic() < deadline:
            try:
                receipt = UsageLedger(home / "state.db").get_run("natural-kill")
                break
            except KeyError:
                time.sleep(0.05)
        assert receipt is not None
        assert receipt["ended_at"] is None
        assert receipt["outcome"] in (None, "incomplete")
    finally:
        release.set()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_natural_cli_retry_records_one_retry_and_nonduplicated_usage(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            if "messages" not in request:
                body = b'{"data":[]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            requests.append(request)
            if len(requests) == 1:
                self.send_response(429)
                self.send_header("Retry-After", "0")
                self.end_headers()
                return
            body = (
                b'data: {"id":"retry-ok","object":"chat.completion.chunk","model":"retry-model",'
                b'"choices":[{"index":0,"delta":{"role":"assistant","content":"retry-ok"},"finish_reason":"stop"}],'
                b'"usage":{"prompt_tokens":17,"completion_tokens":6,"total_tokens":23}}\n\n'
                b"data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / "hermes-home"
    _write_cli_config(home, base_url=f"http://127.0.0.1:{server.server_port}/v1", model="retry-model", api_max_retries=2)
    try:
        result = _run_natural_cli(home, "natural-cli-retry")
        assert result.returncode == 0, result.stderr
        receipt = UsageLedger(home / "state.db").get_run("natural-cli-retry")
        assert len(requests) == 2, (result.stdout, result.stderr, receipt)
        assert receipt["input_tokens"] == 17
        assert receipt["output_tokens"] == 6
        assert receipt["cost_usd"] == 0.0
        assert receipt["retry_count"] == 1
        assert receipt["model_breakdown"] == [{
            "model": "retry-model", "provider": "custom", "input_tokens": 17,
            "output_tokens": 6, "cost_usd": 0.0, "event_count": 2,
        }]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_natural_cli_fallback_records_exact_mixed_model_breakdown(tmp_path):
    primary_requests = []
    fallback_requests = []

    class PrimaryHandler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            if "messages" not in request:
                body = b'{"data":[]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            primary_requests.append(request)
            if len(primary_requests) > 1:
                self.send_response(503)
                self.end_headers()
                return
            body = (
                b'data: {"id":"mixed-primary","object":"chat.completion.chunk","model":"primary-model",'
                b'"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"todo-1",'
                b'"type":"function","function":{"name":"todo","arguments":"{\\"todos\\":[]}"}}]},'
                b'"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
                b"data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    class FallbackHandler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            if "messages" not in request:
                body = b'{"data":[]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            fallback_requests.append(request)
            body = (
                b'data: {"id":"mixed-fallback","object":"chat.completion.chunk","model":"fallback-model",'
                b'"choices":[{"index":0,"delta":{"role":"assistant","content":"fallback-ok"},"finish_reason":"stop"}],'
                b'"usage":{"prompt_tokens":7,"completion_tokens":4,"total_tokens":11}}\n\n'
                b"data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    primary = ThreadingHTTPServer(("127.0.0.1", 0), PrimaryHandler)
    fallback = ThreadingHTTPServer(("127.0.0.1", 0), FallbackHandler)
    primary_thread = threading.Thread(target=primary.serve_forever, daemon=True)
    fallback_thread = threading.Thread(target=fallback.serve_forever, daemon=True)
    primary_thread.start()
    fallback_thread.start()
    home = tmp_path / "hermes-home"
    _write_cli_config(
        home,
        base_url=f"http://127.0.0.1:{primary.server_port}/v1",
        provider="local",
        model="primary-model",
        extra=(
            "fallback_providers:\n"
            "  - provider: custom\n"
            "    model: fallback-model\n"
            f"    base_url: http://127.0.0.1:{fallback.server_port}/v1\n"
            "    api_mode: chat_completions\n"
            "    api_key: test-key\n"
        ),
    )
    try:
        result = _run_natural_cli(home, "natural-cli-mixed")
        assert result.returncode == 0, result.stderr
        assert "fallback-ok" in result.stdout
        assert len(primary_requests) == 2
        assert len(fallback_requests) == 1
        receipt = UsageLedger(home / "state.db").get_run("natural-cli-mixed")
        assert receipt["input_tokens"] == 10
        assert receipt["output_tokens"] == 6
        assert receipt["cost_usd"] == 0.0
        assert receipt["model"] == "mixed"
        # The top-level provider is the successful fallback provider in the
        # current ledger contract.  The breakdown is the authoritative proof
        # that both real providers were attempted and accounted for.
        assert receipt["provider"] == "custom"
        assert receipt["model_breakdown"] == [
            {"model": "fallback-model", "provider": "custom", "input_tokens": 7,
             "output_tokens": 4, "cost_usd": 0.0, "event_count": 1},
            {"model": "primary-model", "provider": "custom", "input_tokens": 3,
             "output_tokens": 2, "cost_usd": 0.0, "event_count": 2},
        ]
    finally:
        primary.shutdown()
        fallback.shutdown()
        primary.server_close()
        fallback.server_close()
        primary_thread.join(timeout=2)
        fallback_thread.join(timeout=2)
