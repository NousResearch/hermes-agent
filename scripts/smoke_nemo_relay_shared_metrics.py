"""Run a real Rabbit CLI turn and validate the Relay shared-metrics output."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROMPT_CANARY = "relay-smoke-sensitive-prompt"
MODEL_CANARY = "gpt-relay-smoke-sensitive-model"
RESPONSE_CANARY = "relay-smoke-sensitive-response"
TOOL_CALL_CANARY = "relay-smoke-sensitive-tool-call"
TOOL_RESULT_CANARY = "relay-smoke-sensitive-tool-result"
TOOL_FILE = "relay-smoke-input.txt"
SKILL_CANARY = "relay-smoke-private-agent-skill"
INSTALLED_SKILL_CANARY = "relay-smoke-private-installed-skill"


def _resolve_rabbit_executable(rabbit_repo: Path) -> Path:
    for relative_path in (
        Path(".venv") / "bin" / "rabbit",
        Path(".venv") / "Scripts" / "rabbit.exe",
    ):
        candidate = rabbit_repo / relative_path
        if candidate.is_file():
            return candidate
    discovered = shutil.which("rabbit")
    if discovered:
        return Path(discovered)
    raise SystemExit(
        "Rabbit executable not found in the repository virtual environment "
        "or on PATH"
    )


class _ModelHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible model server for one deterministic turn."""

    protocol_version = "HTTP/1.1"
    requests: list[dict[str, Any]] = []

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/models":
            self.send_error(404)
            return
        self._write_json({
            "object": "list",
            "data": [
                {
                    "id": MODEL_CANARY,
                    "object": "model",
                    "created": 0,
                    "owned_by": "smoke-test",
                }
            ],
        })

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append(request)
        request_tool = not any(
            message.get("role") == "tool"
            for message in request.get("messages") or []
            if isinstance(message, dict)
        )
        if request.get("stream"):
            self._write_stream(request_tool=request_tool)
        else:
            self._write_json(self._completion(request_tool=request_tool))

    def _completion(self, *, request_tool: bool) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "" if request_tool else RESPONSE_CANARY,
        }
        finish_reason = "tool_calls" if request_tool else "stop"
        if request_tool:
            message["tool_calls"] = [
                {
                    "id": TOOL_CALL_CANARY,
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": TOOL_FILE}),
                    },
                }
            ]
        return {
            "id": "chatcmpl-relay-smoke",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_CANARY,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "total_tokens": 11,
            },
        }

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _write_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _write_stream(self, *, request_tool: bool) -> None:
        now = int(time.time())
        chunks: list[dict[str, Any]] = [
            {
                "id": "chatcmpl-relay-smoke",
                "object": "chat.completion.chunk",
                "created": now,
                "model": MODEL_CANARY,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "",
                        },
                        "finish_reason": None,
                    }
                ],
            }
        ]
        if request_tool:
            chunks.append({
                "id": "chatcmpl-relay-smoke",
                "object": "chat.completion.chunk",
                "created": now,
                "model": MODEL_CANARY,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": TOOL_CALL_CANARY,
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps({"path": TOOL_FILE}),
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            })
        else:
            chunks.append({
                "id": "chatcmpl-relay-smoke",
                "object": "chat.completion.chunk",
                "created": now,
                "model": MODEL_CANARY,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": RESPONSE_CANARY},
                        "finish_reason": None,
                    }
                ],
            })
        chunks.extend([
            {
                "id": "chatcmpl-relay-smoke",
                "object": "chat.completion.chunk",
                "created": now,
                "model": MODEL_CANARY,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if request_tool else "stop",
                    }
                ],
            },
            {
                "id": "chatcmpl-relay-smoke",
                "object": "chat.completion.chunk",
                "created": now,
                "model": MODEL_CANARY,
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
            },
        ])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rabbit-repo",
        type=Path,
        default=Path.cwd(),
        help="Rabbit source checkout containing .venv/bin/rabbit",
    )
    parser.add_argument(
        "--relay-python",
        type=Path,
        default=None,
        help="Optional NeMo Relay checkout's python directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the isolated RABBIT_HOME and captured output",
    )
    return parser.parse_args()


def _write_config(home: Path, port: int) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"""model:
  default: {MODEL_CANARY}
  provider: custom
  base_url: http://127.0.0.1:{port}/v1
  api_mode: chat_completions
  api_key: no-key-required
security:
  tirith_enabled: false
auxiliary:
  title_generation:
    enabled: false
telemetry:
  shared_metrics:
    enabled: true
""",
        encoding="utf-8",
    )


# One interactive turn (2 model calls, 1 read_file) on the canary custom model: the v5 per-turn,
# per-conversation rows it must produce, identical in SQLite and in the export. The agent-created
# skill is Rabbit' own work, so it records no feature_adoption row (the exact name sets assert that).
V5_EXPECTED_DIMENSIONS = {
    "rabbit.task_cost.count": {
        "api_calls_bucket": "2", "model": "custom", "outcome": "completed", "provider": "custom",
        "tokens_bucket": "lt_2k", "tool_calls_bucket": "1",
    },
    "rabbit.tool_output_truncation.count": {"original_size_bucket": "lt_1k", "tool": "read_file", "truncated": "no"},
    "rabbit.tool_enabled_unused.count": {"toolset": "file", "used": "yes"},
}


def _validate_v5_rows(rows: dict[str, list[dict[str, Any]]]) -> None:
    for name, dimensions in V5_EXPECTED_DIMENSIONS.items():
        if [row["dimensions"] for row in rows[name]] != [dimensions] or rows[name][0]["value"] != 1:
            raise AssertionError(f"Unexpected {name}: {rows[name]}")
    [overhead] = rows["rabbit.tool_overhead.count"]  # the file toolset's 4 tools, schema tokens estimated
    if (
        overhead["value"] != 1
        or overhead["dimensions"]["execution_surface"] != "cli"
        or overhead["dimensions"]["enabled_tool_count_bucket"] != "3_to_5"
        or overhead["dimensions"]["tool_schema_tokens_bucket"] in {"0", "unknown"}
    ):
        raise AssertionError(f"Unexpected tool overhead: {overhead}")


def _validate_model_and_tool_rows(by_name: dict[str, list[dict[str, Any]]], counters: list[dict[str, Any]]) -> None:
    # Two primary calls; each lands in whatever TTFT bucket the host's load put it in.
    models = by_name["rabbit.model_route.count"]
    expected_model = {
        "call_role": "primary", "error_class": "none", "model": "custom", "outcome": "success",
        "provider": "custom",
    }
    if (
        any({k: v for k, v in m["dimensions"].items() if k != "ttft_bucket"} != expected_model for m in models)
        or sum(m["value"] for m in models) != 2
        or sum(m["packaged_value"] for m in models) != 2
    ):
        raise AssertionError(
            f"Unexpected model counter: {by_name['rabbit.model_route.count']}"
        )
    expected_start = {
        "name": "rabbit.task_run.started",
        "dimensions": {
            "entrypoint": "one_shot",
            "execution_surface": "cli",
            "platform": "none",
        },
        "value": 1,
        "packaged_value": 1,
    }
    if by_name["rabbit.task_run.started"] != [expected_start]:
        raise AssertionError(
            f"Unexpected task start: {by_name['rabbit.task_run.started']}"
        )
    [terminal] = by_name["rabbit.task_run.finished"]
    expected_terminal_dimensions = {
        "end_reason": "completed",
        "entrypoint": "one_shot",
        "execution_surface": "cli",
        "failure_class": "none",
        "outcome": "success",
        "platform": "none",
        "termination": "none",
    }
    if (
        terminal["dimensions"] != expected_terminal_dimensions
        or terminal["value"] != 1
        or terminal["packaged_value"] != 1
    ):
        raise AssertionError(f"Unexpected task terminal counter: {terminal}")
    [duration] = by_name["rabbit.task_run.duration"]
    if duration["dimensions"] != {
        "duration_bucket": duration["dimensions"].get("duration_bucket"),
        "execution_surface": "cli", "outcome": "success", "retry_count_bucket": "0",
    } or duration["value"] != 1:
        raise AssertionError(f"Unexpected task duration counter: {duration}")
    [cost] = by_name["rabbit.task_cost.count"]
    if (cost["dimensions"]["api_calls_bucket"], cost["dimensions"]["tool_calls_bucket"]) != ("2", "1"):
        raise AssertionError(f"Unexpected task cost counter: {cost}")
    if [c["dimensions"] for c in by_name["rabbit.tool.usage.count"]] != [
        {"error_class": "none", "outcome": "success", "tool_name": "read_file"}
    ]:
        raise AssertionError(f"Unexpected tool usage: {by_name['rabbit.tool.usage.count']}")
    [snapshot] = by_name["rabbit.install.snapshot"]
    if snapshot["value"] != 1 or snapshot["dimensions"]["memory_provider"] != "builtin":
        raise AssertionError(f"Unexpected install snapshot: {snapshot}")
    [session] = by_name["rabbit.session.count"]
    if (session["dimensions"]["turn_count_bucket"], session["dimensions"]["last_outcome"]) != (
        "1", "success",
    ):
        raise AssertionError(f"Unexpected session summary: {session}")
    tokens = {c["dimensions"]["token_type"]: c["value"] for c in by_name["rabbit.model_tokens.sum"]}
    if tokens != {"input": 20, "output": 2}:
        raise AssertionError(f"Unexpected token sums: {by_name['rabbit.model_tokens.sum']}")
    [quality] = by_name["rabbit.model_tool_quality.count"]
    if quality["dimensions"] != {"call_role": "primary", "issue": "none", "model": "custom", "provider": "custom"}:
        raise AssertionError(f"Unexpected tool-call quality: {quality}")
    [reply] = by_name["rabbit.model_reply_issue.count"]  # both scripted replies are usable
    if (reply["dimensions"], reply["value"]) != ({"issue": "none", "model": "custom", "provider": "custom"}, 2):
        raise AssertionError(f"Unexpected model reply issues: {reply}")
    [peak] = by_name["rabbit.context_peak.count"]
    if (peak["dimensions"]["provider"], peak["dimensions"]["model"], peak["dimensions"]["limit_hit"]) != (
        "custom", "custom", "no",
    ):
        raise AssertionError(f"Unexpected context peak: {peak}")
    [startup] = by_name["rabbit.startup.latency"]
    if startup["dimensions"]["surface"] != "cli":
        raise AssertionError(f"Unexpected startup latency: {startup}")
    milestones = {c["dimensions"]["milestone"] for c in by_name["rabbit.install.milestone"]}
    if not {"first_task_started", "first_task_success", "first_tool_success"} <= milestones:
        raise AssertionError(f"Missing install milestones: {sorted(milestones)}")
    [tool] = by_name["rabbit.tool_call.count"]
    expected_tool_dimensions = {
        "approval_outcome": "not_required", "outcome": "success", "tool_category": "file",
    }
    [latency] = by_name["rabbit.tool_call.latency"]
    if (
        tool["dimensions"] != expected_tool_dimensions
        or latency["dimensions"]["latency_bucket"] == "unknown"
        or (latency["dimensions"]["retry_count_bucket"], latency["dimensions"]["tool_category"]) != ("unknown", "file")
        or tool["value"] != 1
        or tool["packaged_value"] != 1
    ):
        raise AssertionError(f"Unexpected tool counter: {tool}")
    lifecycle = by_name["rabbit.skill.lifecycle.count"]
    expected_actions = {
        "archived",
        "created",
        "edited",
        "installed",
        "patched",
        "restored",
        "stale",
    }
    if (
        {counter["dimensions"]["action"] for counter in lifecycle} != expected_actions
        or any(counter["value"] != 1 for counter in lifecycle)
        or any(counter["packaged_value"] != 1 for counter in lifecycle)
    ):
        raise AssertionError(f"Unexpected skill lifecycle counters: {lifecycle}")
    loads = by_name["rabbit.skill.load.count"]
    expected_load_states = {
        ("first_use", "not_applicable", "1"),
        ("reused", "no_new_patch", "2"),
        ("reused", "reused_after_patch", "3_to_5"),
    }
    observed_load_states = {
        (
            counter["dimensions"]["reuse_state"],
            counter["dimensions"]["post_patch_state"],
            counter["dimensions"]["use_count_bucket"],
        )
        for counter in loads
    }
    if (
        observed_load_states != expected_load_states
        or any(counter["value"] != 1 for counter in loads)
        or any(counter["packaged_value"] != 1 for counter in loads)
    ):
        raise AssertionError(f"Unexpected skill load counters: {loads}")
    _validate_v5_rows(by_name)
    serialized = json.dumps(counters)
    for prohibited in (
        MODEL_CANARY,
        PROMPT_CANARY,
        RESPONSE_CANARY,
        TOOL_CALL_CANARY,
        TOOL_RESULT_CANARY,
        SKILL_CANARY,
        INSTALLED_SKILL_CANARY,
    ):
        if prohibited in serialized:
            raise AssertionError(
                f"Stored metrics leaked prohibited value: {prohibited!r}"
            )


def _validate_store(database_path: Path) -> list[dict[str, Any]]:
    if not database_path.is_file():
        raise AssertionError(f"Metrics database was not created: {database_path}")
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT metric_name, dimensions_json, value, packaged_value
            FROM counter_aggregates
            ORDER BY metric_name, dimensions_json
            """
        ).fetchall()
    counters = [
        {
            "name": name,
            "dimensions": json.loads(dimensions),
            "value": value,
            "packaged_value": packaged_value,
        }
        for name, dimensions, value, packaged_value in rows
    ]
    by_name: dict[str, list[dict[str, Any]]] = {}
    for counter in counters:
        by_name.setdefault(counter["name"], []).append(counter)
    if set(by_name) != {
        "rabbit.client.active",
        "rabbit.context_peak.count",
        "rabbit.install.milestone",
        "rabbit.install.snapshot",
        "rabbit.model_reply_issue.count",
        "rabbit.model_route.count",
        "rabbit.model_tokens.sum",
        "rabbit.model_tool_quality.count",
        "rabbit.session.count",
        "rabbit.skill.lifecycle.count",
        "rabbit.skill.load.count",
        "rabbit.startup.latency",
        "rabbit.task_cost.count",
        "rabbit.task_run.duration",
        "rabbit.task_run.finished",
        "rabbit.task_run.started",
        "rabbit.tool.usage.count",
        "rabbit.tool_call.count",
        "rabbit.tool_call.latency",
        "rabbit.tool_enabled_unused.count",
        "rabbit.tool_output_truncation.count",
        "rabbit.tool_overhead.count",
    }:
        raise AssertionError(
            f"Unexpected SQLite counters:\n{json.dumps(counters, indent=2)}"
        )
    if by_name["rabbit.client.active"] != [
        {
            "name": "rabbit.client.active",
            "dimensions": {},
            "value": 1,
            "packaged_value": 1,
        }
    ]:
        raise AssertionError(
            f"Unexpected client-active counter: {by_name['rabbit.client.active']}"
        )
    _validate_model_and_tool_rows(by_name, counters)
    return counters


def main() -> int:
    args = _arguments()
    rabbit_repo = args.rabbit_repo.resolve()
    relay_python = args.relay_python.resolve() if args.relay_python else None
    rabbit = _resolve_rabbit_executable(rabbit_repo)
    if relay_python is not None and not any(
        (relay_python / "nemo_relay").glob("_native.*")
    ):
        raise SystemExit(
            "Built NeMo Relay Python binding not found under "
            f"{relay_python}; run the Relay Python build first"
        )

    if args.output_dir:
        root = args.output_dir.resolve()
        if root.exists():
            raise SystemExit(f"Refusing to replace existing output directory: {root}")
        root.mkdir(parents=True)
    else:
        root = Path(tempfile.mkdtemp(prefix="rabbit-relay-shared-metrics-"))
    home = root / "rabbit-home"
    workdir = root / "workspace"
    workdir.mkdir()
    (workdir / TOOL_FILE).write_text(TOOL_RESULT_CANARY, encoding="utf-8")
    home.mkdir()
    (home / ".no-bundled-skills").touch()
    agent_skill = home / "skills" / SKILL_CANARY
    agent_skill.mkdir(parents=True)
    (agent_skill / "SKILL.md").write_text(
        f"---\nname: {SKILL_CANARY}\ndescription: private smoke skill\n---\n",
        encoding="utf-8",
    )
    installed_skill = home / "skills" / INSTALLED_SKILL_CANARY
    installed_skill.mkdir(parents=True)
    (installed_skill / "SKILL.md").write_text(
        f"---\nname: {INSTALLED_SKILL_CANARY}\ndescription: installed smoke skill\n---\n",
        encoding="utf-8",
    )
    hub_state = home / "skills" / ".hub"
    hub_state.mkdir()
    (hub_state / "lock.json").write_text(
        json.dumps({
            "version": 1,
            "installed": {
                INSTALLED_SKILL_CANARY: {"source": "smoke/local"},
            },
        }),
        encoding="utf-8",
    )

    _ModelHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _write_config(home, server.server_port)
        env = os.environ.copy()
        env["RABBIT_HOME"] = str(home)
        python_paths = [str(rabbit_repo)]
        if relay_python is not None:
            python_paths.append(str(relay_python))
        python_paths.append(env.get("PYTHONPATH", ""))
        env["PYTHONPATH"] = os.pathsep.join(python_paths).rstrip(os.pathsep)
        result = subprocess.run(
            [
                str(rabbit),
                "chat",
                "--query",
                PROMPT_CANARY,
                "--provider",
                "custom",
                "--model",
                MODEL_CANARY,
                "--quiet",
                "--ignore-rules",
                "--toolsets",
                "file",
                "--max-turns",
                "2",
            ],
            cwd=workdir,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    (root / "rabbit.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (root / "rabbit.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise AssertionError(
            f"Rabbit exited with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if len(_ModelHandler.requests) != 2:
        raise AssertionError(
            f"Expected two model requests, got {len(_ModelHandler.requests)}"
        )
    request = _ModelHandler.requests[0]
    if request.get("model") != MODEL_CANARY:
        raise AssertionError(f"Unexpected model request: {request.get('model')!r}")
    if PROMPT_CANARY not in json.dumps(request.get("messages", [])):
        raise AssertionError("Rabbit model request did not contain the prompt canary")
    follow_up = json.dumps(_ModelHandler.requests[1].get("messages", []))
    if TOOL_CALL_CANARY not in follow_up or TOOL_RESULT_CANARY not in follow_up:
        raise AssertionError("Rabbit did not return the tool result to the model")
    if RESPONSE_CANARY not in result.stdout:
        raise AssertionError("Rabbit did not print the mock model response")

    skill_result = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join([
                "from rabbit_cli.observability import relay_shared_metrics",
                "from tools.skill_usage import (",
                "    STATE_ACTIVE, STATE_ARCHIVED, STATE_STALE, bump_patch,",
                "    bump_use, record_created, record_installed, set_state,",
                ")",
                f"skill = {SKILL_CANARY!r}",
                f"installed = {INSTALLED_SKILL_CANARY!r}",
                "record_created(skill, agent_created=True)",
                "bump_use(skill)",
                "bump_use(skill)",
                "bump_patch(skill)",
                "bump_use(skill)",
                "bump_patch(skill, action='edit')",
                "set_state(skill, STATE_STALE)",
                "set_state(skill, STATE_ACTIVE)",
                "set_state(skill, STATE_ARCHIVED)",
                "set_state(skill, STATE_ACTIVE)",
                "record_installed(installed)",
                "runtime = relay_shared_metrics._get_runtime()",
                "assert runtime is not None",
                "runtime.shutdown()",
            ]),
        ],
        cwd=workdir,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    (root / "skills.stdout.txt").write_text(
        skill_result.stdout,
        encoding="utf-8",
    )
    (root / "skills.stderr.txt").write_text(
        skill_result.stderr,
        encoding="utf-8",
    )
    if skill_result.returncode != 0:
        raise AssertionError(
            f"Skill lifecycle probe exited with {skill_result.returncode}\n"
            f"stdout:\n{skill_result.stdout}\nstderr:\n{skill_result.stderr}"
        )

    telemetry = home / "telemetry" / "shared_metrics"
    # Local-only collection: nothing is packaged or uploaded, so the store itself is
    # what this smoke validates (counters and the no-canary privacy check).
    counters = _validate_store(telemetry / "metrics.sqlite3")

    print("Rabbit -> NeMo Relay shared-metrics smoke test passed")
    print(f"Artifact directory: {root}")
    print(f"Model requests: {len(_ModelHandler.requests)}")
    print(f"SQLite counters: {json.dumps(counters, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
