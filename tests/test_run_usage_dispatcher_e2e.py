from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hermes_cli import kanban_db


class _StreamingProvider:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                owner.requests.append(request)
                has_completion = any(
                    message.get("role") == "tool" for message in request.get("messages", [])
                )
                if not has_completion:
                    chunks = [
                        {
                            "id": "dispatcher-e2e-1",
                            "object": "chat.completion.chunk",
                            "created": 1,
                            "model": "local-worker-model",
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "tool_calls": [{
                                        "index": 0,
                                        "id": "complete-call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "kanban_complete",
                                            "arguments": json.dumps({
                                                "summary": "dispatcher usage e2e complete",
                                                "metadata": {"verified": True},
                                            }),
                                        },
                                    }],
                                },
                                "finish_reason": "tool_calls",
                            }],
                        },
                        {
                            "id": "dispatcher-e2e-1",
                            "object": "chat.completion.chunk",
                            "created": 1,
                            "model": "local-worker-model",
                            "choices": [],
                            "usage": {
                                "prompt_tokens": 17,
                                "completion_tokens": 6,
                                "total_tokens": 23,
                            },
                        },
                    ]
                else:
                    chunks = [{
                        "id": "dispatcher-e2e-2",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "local-worker-model",
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": "finished"},
                            "finish_reason": "stop",
                        }],
                        "usage": {
                            "prompt_tokens": 19,
                            "completion_tokens": 2,
                            "total_tokens": 21,
                        },
                    }]
                body = b"".join(
                    b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks
                ) + b"data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()

            def log_message(self, format, *args):  # noqa: A002, ANN001
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def test_real_dispatcher_runtime_subprocess_projects_authoritative_usage(tmp_path):
    provider = _StreamingProvider()
    home = tmp_path / "hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(
        f"""model:
  provider: local
  base_url: {provider.base_url}
  api_mode: chat_completions
  api_key: test-key
  default: local-worker-model
toolsets:
  - kanban
agent:
  max_turns: 3
  verify_on_stop: false
"""
    )
    board = tmp_path / "board.db"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kanban_db.init_db(board)
    with kanban_db.connect_closing(board) as connection:
        task_id = kanban_db.create_task(
            connection,
            title="real dispatcher usage e2e",
            assignee="worker",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )

    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "hermes"
    shim.write_text(f"#!/bin/sh\nexec {sys.executable} -m hermes_cli.main \"$@\"\n")
    shim.chmod(0o755)
    env = {
        **os.environ,
        "HERMES_HOME": str(home),
        "HOME": str(home),
        "HERMES_KANBAN_DB": str(board),
        "HERMES_KANBAN_BOARD": "default",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "PATH": f"{shim_dir}:{os.environ.get('PATH', '')}",
    }
    try:
        dispatch = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "kanban", "dispatch", "--json", "--max", "1"],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        result = json.loads(dispatch.stdout)
        assert [item["task_id"] for item in result["spawned"]] == [task_id]

        deadline = time.monotonic() + 20
        usage = None
        while time.monotonic() < deadline:
            with kanban_db.connect_closing(board) as connection:
                status = connection.execute(
                    "SELECT status FROM tasks WHERE id=?", (task_id,)
                ).fetchone()[0]
                run_row = connection.execute(
                    "SELECT id FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
                    (task_id,),
                ).fetchone()
                usage = (
                    connection.execute(
                        "SELECT task_run_id, usage_run_id, model, provider, input_tokens, output_tokens, cost_usd "
                        "FROM task_run_usage WHERE task_run_id=?",
                        (run_row[0],),
                    ).fetchone()
                    if run_row is not None else None
                )
            if status == "done" and usage is not None:
                break
            time.sleep(0.1)
        with kanban_db.connect_closing(board) as connection:
            task = connection.execute(
                "SELECT status, current_run_id FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            run = connection.execute(
                "SELECT id, status, outcome FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
                (task_id,),
            ).fetchone()
        assert tuple(task) == ("done", None)
        assert run[1:] == ("done", "completed")
        assert usage is not None
        assert tuple(usage) == (
            run[0], f"task-run:{run[0]}", "local-worker-model", "custom",
            36, 8, 0.0,
        )
        assert len(provider.requests) >= 2
    finally:
        provider.close()
