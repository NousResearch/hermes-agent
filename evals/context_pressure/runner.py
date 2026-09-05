"""Run bounded Hermes configurations against deterministic context-pressure tasks.

The runner deliberately executes user-supplied commands instead of importing an
agent implementation.  That makes it useful for comparing normal Hermes
configuration, feature branches, or provider modes without coupling the eval
to any particular reasoning backend.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from .tasks import (
        create_distributed_evidence_workspace,
        validate_distributed_evidence,
    )
except ImportError:  # Direct ``python evals/context_pressure/runner.py`` invocation.
    from tasks import (
        create_distributed_evidence_workspace,
        validate_distributed_evidence,
    )

DEFAULT_COMMAND = (
    "{python} -m hermes_cli.main -z {prompt} --yolo "
    "--usage-file {usage_file} {model_flags}"
)


def _parse_arm(value: str) -> tuple[str, str]:
    name, separator, command = value.partition("=")
    if not separator or not name.strip() or not command.strip():
        raise argparse.ArgumentTypeError("arm must be NAME=COMMAND")
    return name.strip(), command.strip()


def _format_command(template: str, values: dict[str, str]) -> list[str]:
    try:
        rendered = template.format_map(values)
    except KeyError as exc:
        raise ValueError(f"unknown command placeholder: {{{exc.args[0]}}}") from exc
    return shlex.split(rendered)


def _terminate(process: subprocess.Popen[str]) -> None:
    if os.name != "posix":
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _load_usage(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def run_one(
    *,
    arm: str,
    command_template: str,
    repetition: int,
    timeout: float,
    model: str | None,
    provider: str | None,
    config: Path | None,
    root_output: Path,
) -> dict[str, Any]:
    """Run and validate one isolated cell, preserving usage on timeout."""

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="hermes-context-pressure-") as temp_dir:
        root = Path(temp_dir)
        workspace = root / "workspace"
        hermes_home = root / "hermes-home"
        usage_file = root / "usage.json"
        workspace.mkdir()
        hermes_home.mkdir()
        task = create_distributed_evidence_workspace(workspace)
        if config is not None:
            target_config = hermes_home / "config.yaml"
            target_config.write_bytes(config.read_bytes())

        values = {
            "python": sys.executable,
            "prompt": task.prompt,
            "workspace": str(workspace),
            "hermes_home": str(hermes_home),
            "usage_file": str(usage_file),
            "model": model or "",
            "provider": provider or "",
            "model_flags": " ".join(
                part
                for part in (
                    f"--model {shlex.quote(model)}" if model else "",
                    f"--provider {shlex.quote(provider)}" if provider else "",
                )
                if part
            ),
        }
        try:
            command = _format_command(command_template, values)
        except ValueError as exc:
            return {
                "task": task.task_id,
                "arm": arm,
                "repetition": repetition,
                "error": str(exc),
                "timed_out": False,
                "validated": False,
                "wall_seconds": time.monotonic() - started,
            }

        env = os.environ.copy()
        env["HERMES_HOME"] = str(hermes_home)
        process = subprocess.Popen(
            command,
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            _terminate(process)
            remaining_stdout, remaining_stderr = process.communicate()
            stdout += remaining_stdout or ""
            stderr += remaining_stderr or ""

        validation = validate_distributed_evidence(workspace)
        usage = _load_usage(usage_file)
        result = {
            "task": task.task_id,
            "arm": arm,
            "repetition": repetition,
            "return_code": 124 if timed_out else process.returncode,
            "timed_out": timed_out,
            "validated": validation.passed and not timed_out,
            "validation": validation.as_dict(),
            "usage": usage,
            "wall_seconds": time.monotonic() - started,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }
        output_path = root_output / f"{task.task_id}-{arm}-r{repetition:02d}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["result_file"] = str(output_path)
        return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("distributed_evidence",), default="distributed_evidence"
    )
    parser.add_argument(
        "--arm", action="append", type=_parse_arm, metavar="NAME=COMMAND"
    )
    parser.add_argument("--model")
    parser.add_argument("--provider")
    parser.add_argument(
        "--config",
        type=Path,
        help="Copy this config.yaml into each isolated HERMES_HOME",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Per-cell wall-clock limit in seconds",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("/tmp/hermes-context-pressure")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.repetitions < 1 or args.timeout <= 0:
        parser.error(
            "--repetitions must be positive and --timeout must be greater than zero"
        )
    if args.provider and not args.model:
        parser.error("--provider requires --model")
    arms = args.arm or [("default", DEFAULT_COMMAND)]
    output_dir = args.out.expanduser().resolve()
    config = args.config.expanduser().resolve() if args.config else None
    if config is not None and not config.is_file():
        parser.error(f"config file does not exist: {config}")

    records: list[dict[str, Any]] = []
    for arm, command in arms:
        for repetition in range(1, args.repetitions + 1):
            record = run_one(
                arm=arm,
                command_template=command,
                repetition=repetition,
                timeout=args.timeout,
                model=args.model,
                provider=args.provider,
                config=config,
                root_output=output_dir,
            )
            records.append(record)
            status = "PASS" if record.get("validated") else "FAIL"
            print(
                f"{arm} r{repetition:02d}: {status} ({record.get('wall_seconds', 0):.1f}s)"
            )

    index = output_dir / "index.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps({"runs": records}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {index}")
    return 0 if records and all(record.get("validated") for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
