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
        if request.get("stream"):
            self._write_stream()
        else:
            self._write_json({
                "id": "chatcmpl-relay-smoke",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_CANARY,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": RESPONSE_CANARY,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
            })

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

    def _write_stream(self) -> None:
        now = int(time.time())
        chunks = [
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
                            "content": RESPONSE_CANARY,
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-relay-smoke",
                "object": "chat.completion.chunk",
                "created": now,
                "model": MODEL_CANARY,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
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
        ]
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
    expected = [
        {
            "name": "hermes.model_call.count",
            "dimensions": {
                "call_role": "primary",
                "locality": "local",
                "model_family": "gpt",
                "outcome": "success",
                "provider_family": "custom",
            },
            "value": 1,
            "packaged_value": 1,
        }
    ]
    if counters != expected:
        raise AssertionError(
            f"Unexpected SQLite counters:\n{json.dumps(counters, indent=2)}"
        )
    return counters


def _validate_package(outbox: Path, schema_path: Path) -> tuple[Path, dict[str, Any]]:
    packages = sorted(outbox.glob("*.json"))
    if len(packages) != 1:
        raise AssertionError(f"Expected one package in {outbox}, found {len(packages)}")
    package_path = packages[0]
    package = json.loads(package_path.read_text(encoding="utf-8"))
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError(
            "The Hermes development environment requires jsonschema"
        ) from exc
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(package, schema)

    serialized = json.dumps(package)
    for prohibited in (PROMPT_CANARY, MODEL_CANARY, RESPONSE_CANARY):
        if prohibited in serialized:
            raise AssertionError(
                f"Exported package leaked prohibited value: {prohibited!r}"
            )
    expected_metric = {
        "name": "hermes.model_call.count",
        "type": "counter",
        "dimensions": {
            "call_role": "primary",
            "locality": "local",
            "model_family": "gpt",
            "outcome": "success",
            "provider_family": "custom",
        },
        "value": 1,
    }
    if package.get("metrics") != [expected_metric]:
        raise AssertionError(
            f"Unexpected package metrics:\n{json.dumps(package.get('metrics'), indent=2)}"
        )
    return package_path, package


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
    home.mkdir()
    (home / ".no-bundled-skills").touch()

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
                "search",
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
    if not _ModelHandler.requests:
        raise AssertionError("Hermes did not call the local model endpoint")
    request = _ModelHandler.requests[0]
    if request.get("model") != MODEL_CANARY:
        raise AssertionError(f"Unexpected model request: {request.get('model')!r}")
    if PROMPT_CANARY not in json.dumps(request.get("messages", [])):
        raise AssertionError("Hermes model request did not contain the prompt canary")
    if RESPONSE_CANARY not in result.stdout:
        raise AssertionError("Hermes did not print the mock model response")

    telemetry = home / "telemetry" / "shared_metrics"
    counters = _validate_store(telemetry / "metrics.sqlite3")
    package_path, package = _validate_package(
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
    print(f"Export package: {package_path}")
    print(json.dumps(package, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
