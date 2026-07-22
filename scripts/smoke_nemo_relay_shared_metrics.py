"""Run a real Hermes CLI turn and validate the Relay shared-metrics output."""

from __future__ import annotations

import argparse
import json
import os
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
        "--hermes-repo",
        type=Path,
        default=Path.cwd(),
        help="Hermes source checkout containing .venv/bin/hermes",
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
        help="Directory for the isolated HERMES_HOME and captured output",
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
telemetry:
  shared_metrics:
    enabled: true
""",
        encoding="utf-8",
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
        "hermes.client.active",
        "hermes.client.first_successful_task",
        "hermes.client.first_usable",
        "hermes.model_call.count",
        "hermes.setup.finished",
        "hermes.setup.started",
        "hermes.skill.lifecycle.count",
        "hermes.skill.load.count",
        "hermes.task_run.finished",
        "hermes.task_run.started",
        "hermes.tool_call.count",
    }:
        raise AssertionError(
            f"Unexpected SQLite counters:\n{json.dumps(counters, indent=2)}"
        )
    if by_name["hermes.client.active"] != [
        {
            "name": "hermes.client.active",
            "dimensions": {},
            "value": 1,
            "packaged_value": 1,
        }
    ]:
        raise AssertionError(
            f"Unexpected client-active counter: {by_name['hermes.client.active']}"
        )
    for metric_name in (
        "hermes.client.first_usable",
        "hermes.client.first_successful_task",
    ):
        if by_name[metric_name] != [
            {
                "name": metric_name,
                "dimensions": {},
                "value": 1,
                "packaged_value": 1,
            }
        ]:
            raise AssertionError(f"Unexpected one-time metric: {by_name[metric_name]}")
    if by_name["hermes.setup.started"] != [
        {
            "name": "hermes.setup.started",
            "dimensions": {"mode": "quick"},
            "value": 1,
            "packaged_value": 1,
        }
    ]:
        raise AssertionError(
            f"Unexpected setup start: {by_name['hermes.setup.started']}"
        )
    if by_name["hermes.setup.finished"] != [
        {
            "name": "hermes.setup.finished",
            "dimensions": {
                "failure_stage": "none",
                "mode": "quick",
                "outcome": "success",
            },
            "value": 1,
            "packaged_value": 1,
        }
    ]:
        raise AssertionError(
            f"Unexpected setup finish: {by_name['hermes.setup.finished']}"
        )
    models = by_name["hermes.model_call.count"]
    if sum(counter["value"] for counter in models) != 2:
        raise AssertionError(f"Unexpected model counters: {models}")
    for counter in models:
        dimensions = counter["dimensions"]
        expected_dimensions = {
            "call_role": "primary",
            "cost_bucket": "unknown",
            "input_token_bucket": "1_to_1k",
            "latency_bucket": dimensions.get("latency_bucket"),
            "locality": "local",
            "model_family": "gpt",
            "outcome": "success",
            "output_token_bucket": "1_to_1k",
            "provider_family": "custom",
            "retry_count_bucket": "0",
        }
        if (
            dimensions != expected_dimensions
            or dimensions["latency_bucket"] == "unknown"
            or counter["packaged_value"] != counter["value"]
        ):
            raise AssertionError(f"Unexpected model counter: {counter}")
    expected_start = {
        "name": "hermes.task_run.started",
        "dimensions": {
            "entrypoint": "interactive",
            "execution_surface": "cli",
        },
        "value": 1,
        "packaged_value": 1,
    }
    if by_name["hermes.task_run.started"] != [expected_start]:
        raise AssertionError(
            f"Unexpected task start: {by_name['hermes.task_run.started']}"
        )
    [terminal] = by_name["hermes.task_run.finished"]
    expected_terminal_dimensions = {
        "duration_bucket": terminal["dimensions"].get("duration_bucket"),
        "end_reason": "completed",
        "entrypoint": "interactive",
        "execution_surface": "cli",
        "model_call_count_bucket": "2",
        "outcome": "success",
        "retry_count_bucket": "0",
        "termination": "none",
        "tool_call_count_bucket": "1",
    }
    if (
        terminal["dimensions"] != expected_terminal_dimensions
        or terminal["value"] != 1
        or terminal["packaged_value"] != 1
    ):
        raise AssertionError(f"Unexpected task terminal counter: {terminal}")
    [tool] = by_name["hermes.tool_call.count"]
    expected_tool_dimensions = {
        "approval_outcome": "not_required",
        "latency_bucket": tool["dimensions"].get("latency_bucket"),
        "outcome": "success",
        "retry_count_bucket": "unknown",
        "tool_category": "file",
    }
    if (
        tool["dimensions"] != expected_tool_dimensions
        or tool["dimensions"]["latency_bucket"] == "unknown"
        or tool["value"] != 1
        or tool["packaged_value"] != 1
    ):
        raise AssertionError(f"Unexpected tool counter: {tool}")
    lifecycle = by_name["hermes.skill.lifecycle.count"]
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
    loads = by_name["hermes.skill.load.count"]
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
    return counters


def _validate_packages(
    outbox: Path,
    schema_path: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    package_paths = sorted(outbox.glob("*.json"))
    if len(package_paths) != 3:
        raise AssertionError(
            f"Expected three delta packages in {outbox}, found {len(package_paths)}"
        )
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError(
            "The Hermes development environment requires jsonschema"
        ) from exc
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    packages = [
        json.loads(package_path.read_text(encoding="utf-8"))
        for package_path in package_paths
    ]
    for package in packages:
        jsonschema.validate(package, schema)
        if set(package["resource"]) != {
            "architecture",
            "hermes_version",
            "install_method",
            "os_family",
        }:
            raise AssertionError(f"Unexpected client resource: {package['resource']}")

    serialized = json.dumps(packages)
    for prohibited in (
        PROMPT_CANARY,
        MODEL_CANARY,
        RESPONSE_CANARY,
        TOOL_CALL_CANARY,
        TOOL_RESULT_CANARY,
        SKILL_CANARY,
        INSTALLED_SKILL_CANARY,
    ):
        if prohibited in serialized:
            raise AssertionError(
                f"Exported package leaked prohibited value: {prohibited!r}"
            )
    metrics: dict[str, list[dict[str, Any]]] = {}
    for package in packages:
        for metric in package.get("metrics", []):
            metrics.setdefault(metric["name"], []).append(metric)
    if set(metrics) != {
        "hermes.client.active",
        "hermes.client.first_successful_task",
        "hermes.client.first_usable",
        "hermes.model_call.count",
        "hermes.setup.finished",
        "hermes.setup.started",
        "hermes.skill.lifecycle.count",
        "hermes.skill.load.count",
        "hermes.task_run.finished",
        "hermes.task_run.started",
        "hermes.tool_call.count",
    }:
        raise AssertionError(
            f"Unexpected package metrics:\n{json.dumps(metrics, indent=2)}"
        )
    if metrics["hermes.client.active"] != [
        {
            "name": "hermes.client.active",
            "type": "counter",
            "dimensions": {},
            "value": 1,
        }
    ]:
        raise AssertionError(
            f"Unexpected client-active metric: {metrics['hermes.client.active']}"
        )
    for metric_name in (
        "hermes.client.first_usable",
        "hermes.client.first_successful_task",
    ):
        if metrics[metric_name] != [
            {
                "name": metric_name,
                "type": "counter",
                "dimensions": {},
                "value": 1,
            }
        ]:
            raise AssertionError(
                f"Unexpected one-time package metric: {metrics[metric_name]}"
            )
    if metrics["hermes.setup.started"][0]["dimensions"] != {"mode": "quick"}:
        raise AssertionError(
            f"Unexpected setup-start metric: {metrics['hermes.setup.started']}"
        )
    if metrics["hermes.setup.finished"][0]["dimensions"] != {
        "failure_stage": "none",
        "mode": "quick",
        "outcome": "success",
    }:
        raise AssertionError(
            f"Unexpected setup-finish metric: {metrics['hermes.setup.finished']}"
        )
    models = metrics["hermes.model_call.count"]
    if sum(metric["value"] for metric in models) != 2:
        raise AssertionError(f"Unexpected model metrics: {models}")
    for model in models:
        dimensions = model["dimensions"]
        if (
            dimensions
            != {
                "call_role": "primary",
                "cost_bucket": "unknown",
                "input_token_bucket": "1_to_1k",
                "latency_bucket": dimensions.get("latency_bucket"),
                "locality": "local",
                "model_family": "gpt",
                "outcome": "success",
                "output_token_bucket": "1_to_1k",
                "provider_family": "custom",
                "retry_count_bucket": "0",
            }
            or dimensions["latency_bucket"] == "unknown"
        ):
            raise AssertionError(f"Unexpected model metric: {model}")
    [terminal] = metrics["hermes.task_run.finished"]
    if terminal["dimensions"] != {
        "duration_bucket": terminal["dimensions"].get("duration_bucket"),
        "end_reason": "completed",
        "entrypoint": "interactive",
        "execution_surface": "cli",
        "model_call_count_bucket": "2",
        "outcome": "success",
        "retry_count_bucket": "0",
        "termination": "none",
        "tool_call_count_bucket": "1",
    }:
        raise AssertionError(f"Unexpected task terminal metric: {terminal}")
    [tool] = metrics["hermes.tool_call.count"]
    if (
        tool["dimensions"]
        != {
            "approval_outcome": "not_required",
            "latency_bucket": tool["dimensions"].get("latency_bucket"),
            "outcome": "success",
            "retry_count_bucket": "unknown",
            "tool_category": "file",
        }
        or tool["dimensions"]["latency_bucket"] == "unknown"
    ):
        raise AssertionError(f"Unexpected tool metric: {tool}")
    lifecycle = metrics["hermes.skill.lifecycle.count"]
    if {metric["dimensions"]["action"] for metric in lifecycle} != {
        "archived",
        "created",
        "edited",
        "installed",
        "patched",
        "restored",
        "stale",
    }:
        raise AssertionError(f"Unexpected skill lifecycle metrics: {lifecycle}")
    loads = metrics["hermes.skill.load.count"]
    if {
        (
            metric["dimensions"]["reuse_state"],
            metric["dimensions"]["post_patch_state"],
            metric["dimensions"]["use_count_bucket"],
        )
        for metric in loads
    } != {
        ("first_use", "not_applicable", "1"),
        ("reused", "no_new_patch", "2"),
        ("reused", "reused_after_patch", "3_to_5"),
    }:
        raise AssertionError(f"Unexpected skill load metrics: {loads}")
    return package_paths, packages


def main() -> int:
    args = _arguments()
    hermes_repo = args.hermes_repo.resolve()
    relay_python = args.relay_python.resolve() if args.relay_python else None
    hermes = hermes_repo / ".venv" / "bin" / "hermes"
    if not hermes.is_file():
        raise SystemExit(f"Hermes executable not found: {hermes}")
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
        root = Path(tempfile.mkdtemp(prefix="hermes-relay-shared-metrics-"))
    home = root / "hermes-home"
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
        env["HERMES_HOME"] = str(home)
        if relay_python is not None:
            env["PYTHONPATH"] = os.pathsep.join([
                str(relay_python),
                env.get("PYTHONPATH", ""),
            ]).rstrip(os.pathsep)
        setup_result = subprocess.run(
            [
                str(hermes.parent / "python"),
                "-c",
                "\n".join([
                    "from types import SimpleNamespace",
                    "from hermes_cli import main, setup",
                    "setup.run_setup_wizard = lambda _: True",
                    "main.cmd_setup(SimpleNamespace(portal=False, quick=True, reset=False, section=None))",
                ]),
            ],
            cwd=workdir,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )
        if setup_result.returncode != 0:
            raise AssertionError(
                f"Setup lifecycle probe exited with {setup_result.returncode}\n"
                f"stdout:\n{setup_result.stdout}\nstderr:\n{setup_result.stderr}"
            )
        result = subprocess.run(
            [
                str(hermes),
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

    (root / "hermes.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (root / "hermes.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise AssertionError(
            f"Hermes exited with {result.returncode}\n"
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
        raise AssertionError("Hermes model request did not contain the prompt canary")
    follow_up = json.dumps(_ModelHandler.requests[1].get("messages", []))
    if TOOL_CALL_CANARY not in follow_up or TOOL_RESULT_CANARY not in follow_up:
        raise AssertionError("Hermes did not return the tool result to the model")
    if RESPONSE_CANARY not in result.stdout:
        raise AssertionError("Hermes did not print the mock model response")

    skill_result = subprocess.run(
        [
            str(hermes.parent / "python"),
            "-c",
            "\n".join([
                "from hermes_cli.observability import relay_shared_metrics",
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
    counters = _validate_store(telemetry / "metrics.sqlite3")
    package_paths, packages = _validate_packages(
        telemetry / "outbox",
        hermes_repo
        / "hermes_cli"
        / "observability"
        / "schemas"
        / "hermes.shared_metrics.v1.schema.json",
    )

    print("Hermes -> NeMo Relay shared-metrics smoke test passed")
    print(f"Artifact directory: {root}")
    print(f"Model requests: {len(_ModelHandler.requests)}")
    print(f"SQLite counters: {json.dumps(counters, indent=2)}")
    print(f"Export packages: {', '.join(str(path) for path in package_paths)}")
    print(json.dumps(packages, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
