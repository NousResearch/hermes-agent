# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the complete shared-metrics smoke test plus one live NVIDIA NIM turn."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any


DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3-super-120b-a12b"
LIVE_PROMPT_CANARY = "relay-live-nvidia-sensitive-prompt"
LIVE_RESPONSE_CANARY = "RELAY_LIVE_NVIDIA_OK"


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
    parser.add_argument(
        "--nvidia-model",
        default=DEFAULT_NVIDIA_MODEL,
        help="NVIDIA NIM model used for the live turn",
    )
    return parser.parse_args()


def _new_output_directory(output_dir: Path | None) -> Path:
    if output_dir is not None:
        root = output_dir.resolve()
    else:
        root = Path(tempfile.gettempdir()) / (
            f"hermes-relay-shared-metrics-nvidia-{uuid.uuid4().hex[:10]}"
        )
    if root.exists():
        raise SystemExit(f"Refusing to replace existing output directory: {root}")
    return root


def _counter_rows(database_path: Path) -> dict[tuple[str, str], int]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT metric_name, dimensions_json, value
            FROM counter_aggregates
            """
        ).fetchall()
    return {(name, dimensions): value for name, dimensions, value in rows}


def _counter_totals(rows: dict[tuple[str, str], int]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for (name, _), value in rows.items():
        totals[name] = totals.get(name, 0) + value
    return totals


def _row_deltas(
    before: dict[tuple[str, str], int],
    after: dict[tuple[str, str], int],
) -> list[tuple[str, dict[str, Any], int]]:
    deltas = []
    for key in sorted(set(before) | set(after)):
        delta = after.get(key, 0) - before.get(key, 0)
        if delta:
            deltas.append((key[0], json.loads(key[1]), delta))
    return deltas


def _write_nvidia_config(home: Path, model: str) -> None:
    (home / "config.yaml").write_text(
        f"""model:
  default: {model}
  provider: nvidia
security:
  tirith_enabled: false
telemetry:
  shared_metrics:
    enabled: true
""",
        encoding="utf-8",
    )


def _run_deterministic_smoke(
    *,
    hermes_repo: Path,
    relay_python: Path | None,
    root: Path,
) -> None:
    command = [
        str(hermes_repo / ".venv" / "bin" / "python"),
        str(hermes_repo / "scripts" / "smoke_nemo_relay_shared_metrics.py"),
        "--hermes-repo",
        str(hermes_repo),
        "--output-dir",
        str(root),
    ]
    if relay_python is not None:
        command.extend(["--relay-python", str(relay_python)])
    result = subprocess.run(
        command,
        cwd=hermes_repo,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        timeout=240,
    )
    if root.exists():
        (root / "deterministic-smoke.stdout.txt").write_text(
            result.stdout,
            encoding="utf-8",
        )
        (root / "deterministic-smoke.stderr.txt").write_text(
            result.stderr,
            encoding="utf-8",
        )
    if result.returncode != 0:
        raise AssertionError(
            f"Deterministic smoke exited with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _assert_live_deltas(
    before: dict[tuple[str, str], int],
    after: dict[tuple[str, str], int],
) -> list[tuple[str, dict[str, Any], int]]:
    before_totals = _counter_totals(before)
    after_totals = _counter_totals(after)
    changed_totals = {
        name: after_totals.get(name, 0) - before_totals.get(name, 0)
        for name in set(before_totals) | set(after_totals)
        if after_totals.get(name, 0) != before_totals.get(name, 0)
    }
    expected = {
        "hermes.model_call.count": 1,
        "hermes.task_run.finished": 1,
        "hermes.task_run.started": 1,
    }
    if changed_totals != expected:
        raise AssertionError(
            f"Unexpected live counter changes: {json.dumps(changed_totals, indent=2)}"
        )

    deltas = _row_deltas(before, after)
    [model] = [item for item in deltas if item[0] == "hermes.model_call.count"]
    model_dimensions = model[1]
    if (
        model[2] != 1
        or model_dimensions.get("provider_family") != "direct"
        or model_dimensions.get("locality") != "remote"
        or model_dimensions.get("outcome") != "success"
        or model_dimensions.get("retry_count_bucket") != "0"
        or model_dimensions.get("latency_bucket") == "unknown"
        or model_dimensions.get("input_token_bucket") == "unknown"
        or model_dimensions.get("output_token_bucket") == "unknown"
    ):
        raise AssertionError(f"Unexpected live model delta: {model}")

    [terminal] = [item for item in deltas if item[0] == "hermes.task_run.finished"]
    terminal_dimensions = terminal[1]
    if (
        terminal[2] != 1
        or terminal_dimensions.get("outcome") != "success"
        or terminal_dimensions.get("end_reason") != "completed"
        or terminal_dimensions.get("model_call_count_bucket") != "1"
        or terminal_dimensions.get("tool_call_count_bucket") != "0"
    ):
        raise AssertionError(f"Unexpected live task delta: {terminal}")
    return deltas


def _validate_live_package(
    *,
    package_path: Path,
    schema_path: Path,
    model: str,
    api_key: str,
    database_path: Path,
) -> dict[str, Any]:
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError(
            "The Hermes development environment requires jsonschema"
        ) from exc

    package = json.loads(package_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(package, schema)
    metric_names = sorted(metric["name"] for metric in package["metrics"])
    if metric_names != [
        "hermes.model_call.count",
        "hermes.task_run.finished",
        "hermes.task_run.started",
    ]:
        raise AssertionError(f"Unexpected live package metrics: {metric_names}")

    serialized_package = json.dumps(package)
    with sqlite3.connect(database_path) as connection:
        serialized_store = "\n".join(connection.iterdump())
    for prohibited in (LIVE_PROMPT_CANARY, model, api_key):
        if prohibited in serialized_package or prohibited in serialized_store:
            raise AssertionError(
                f"Live metrics persisted prohibited value: {prohibited!r}"
            )
    return package


def main() -> int:
    args = _arguments()
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY must be set for the live smoke test")

    hermes_repo = args.hermes_repo.resolve()
    relay_python = args.relay_python.resolve() if args.relay_python else None
    hermes = hermes_repo / ".venv" / "bin" / "hermes"
    if not hermes.is_file():
        raise SystemExit(f"Hermes executable not found: {hermes}")
    root = _new_output_directory(args.output_dir)

    _run_deterministic_smoke(
        hermes_repo=hermes_repo,
        relay_python=relay_python,
        root=root,
    )
    home = root / "hermes-home"
    workdir = root / "workspace"
    telemetry = home / "telemetry" / "shared_metrics"
    database_path = telemetry / "metrics.sqlite3"
    before_rows = _counter_rows(database_path)
    before_packages = set((telemetry / "outbox").glob("*.json"))

    _write_nvidia_config(home, args.nvidia_model)
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
            (
                f"{LIVE_PROMPT_CANARY}: Reply with exactly "
                f"{LIVE_RESPONSE_CANARY} and do not call any tools."
            ),
            "--provider",
            "nvidia",
            "--model",
            args.nvidia_model,
            "--quiet",
            "--ignore-rules",
            "--max-turns",
            "1",
        ],
        cwd=workdir,
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
    )
    (root / "nvidia.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (root / "nvidia.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise AssertionError(
            f"Live NVIDIA turn exited with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if LIVE_RESPONSE_CANARY not in result.stdout:
        raise AssertionError(
            f"Live NVIDIA response did not contain {LIVE_RESPONSE_CANARY!r}:\n"
            f"{result.stdout}"
        )

    after_rows = _counter_rows(database_path)
    deltas = _assert_live_deltas(before_rows, after_rows)
    new_packages = set((telemetry / "outbox").glob("*.json")) - before_packages
    if len(new_packages) != 1:
        raise AssertionError(
            f"Expected one live delta package, found {len(new_packages)}"
        )
    [package_path] = list(new_packages)
    package = _validate_live_package(
        package_path=package_path,
        schema_path=(
            hermes_repo
            / "hermes_cli"
            / "observability"
            / "schemas"
            / "hermes.shared_metrics.v1.schema.json"
        ),
        model=args.nvidia_model,
        api_key=api_key,
        database_path=database_path,
    )

    print("Hermes -> NeMo Relay complete live smoke test passed")
    print(f"Artifact directory: {root}")
    print(f"NVIDIA model: {args.nvidia_model}")
    print(f"Live counter deltas: {json.dumps(deltas, indent=2)}")
    print(f"Live package: {package_path}")
    print(f"Live package metrics: {len(package['metrics'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
