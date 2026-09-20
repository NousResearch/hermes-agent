#!/usr/bin/env python3
"""Drive a coding-agent CLI in a bounded loop with the decision gate as the judge.

One attempt is: run your harness command, then ask the gate one typed question about the
exit code and the output tail.

    complete    -> stop, report success
    retry       -> run again with the failure tail appended to the prompt
    abort       -> stop: environment/credential failure, another pass cannot help
    unavailable -> stop without declaring anything (a missing verdict is not a verdict)

The harness command is a template where `{prompt}` is replaced by the shell-quoted prompt, so
this is harness-agnostic: DeepSeek Harness, OpenCode, Codex, a make target, a test script.

Usage
    gated_harness_loop.py --task "make the failing cart test pass" \
        --cmd 'node /path/to/checkout/apps/cli/lib/bin.js --profile headless {prompt}' \
        --workdir /path/to/repo --max-loops 2 --log /tmp/run.json

Exit codes
    0 complete       2 retries exhausted       3 abort or gate unavailable
    4 harness command not found (launch failure, no gate call is made)
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_gate import PRESETS, decide  # noqa: E402  (sibling script, not a package)

TAIL_CHARS = 1500
LAUNCH_FAILURE_CODE = 127


def load_hermes_env() -> list[str]:
    """Pull keys from ~/.hermes/.env into the process env (variables already set win)."""
    loaded: list[str] = []
    path = Path.home() / ".hermes" / ".env"
    if not path.is_file():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def run_harness(cmd_template: str, prompt: str, workdir: str, timeout: float) -> dict:
    """Render the template with a shell-quoted prompt, run it, and capture the result."""
    rendered = cmd_template.replace("{prompt}", shlex.quote(prompt))
    started = time.monotonic()
    try:
        proc = subprocess.run(
            rendered, shell=True, cwd=workdir, capture_output=True, text=True, timeout=timeout
        )
        code = proc.returncode
        out = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        code = 124
        partial = exc.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        out = f"timeout after {timeout}s\n{partial}"
    return {"exit_code": code, "tail": out[-TAIL_CHARS:],
            "duration_ms": round((time.monotonic() - started) * 1000),
            "command": rendered}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="the prompt handed to the harness")
    p.add_argument("--cmd", required=True,
                   help="harness command template; must contain {prompt}")
    p.add_argument("--workdir", default=os.getcwd())
    p.add_argument("--max-loops", type=int, default=2)
    p.add_argument("--task-timeout", type=float, default=1800.0)
    p.add_argument("--gate-backend", default="auto",
                   choices=["auto", "typesafe", "openrouter", "ollama"])
    p.add_argument("--gate-model", default="")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="extra environment variable for the harness (repeatable)")
    p.add_argument("--log", default="gated_harness_loop.json")
    a = p.parse_args(argv)

    if "{prompt}" not in a.cmd:
        p.error("--cmd must contain the {prompt} placeholder")
    for item in a.env:
        if "=" not in item:
            p.error(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        os.environ[key.strip()] = value

    loaded = load_hermes_env()
    preset = PRESETS["harness"]
    attempts: list[dict] = []
    prompt = a.task
    print(f"cmd={a.cmd}\nworkdir={a.workdir}\n"
          f"env keys loaded from ~/.hermes/.env: {', '.join(loaded) if loaded else 'none'}")

    for loop in range(1, a.max_loops + 1):
        print(f"\n--- attempt {loop}/{a.max_loops} ---", flush=True)
        run = run_harness(a.cmd, prompt, a.workdir, a.task_timeout)
        print(f"harness exit={run['exit_code']} in {run['duration_ms']}ms")
        if run["exit_code"] == LAUNCH_FAILURE_CODE and "not found" in run["tail"].lower():
            print(f"harness command not found - fix --cmd before gating anything.\n"
                  f"tail:\n{run['tail']}")
            return 4

        verdict = decide(
            f"Exit code: {run['exit_code']}\nTerminal output tail:\n{run['tail']}",
            preset["question"], preset["choices"], preset["describe"],
            backend=a.gate_backend, model=a.gate_model,
            unavailable_default=preset["unavailable_default"],
        )
        attempts.append({"loop": loop, "harness": run, "gate": verdict})
        Path(a.log).write_text(json.dumps({"task": a.task, "attempts": attempts}, indent=2),
                               encoding="utf-8")

        if verdict.get("status") != "ok":
            print(f"gate UNAVAILABLE ({verdict.get('reason')}) -> stop; policy default "
                  f"'{verdict.get('fallback')}' is recorded, not claimed as a verdict. "
                  f"Log: {a.log}")
            return 3
        confidence = verdict.get("confidence")
        suffix = f" confidence={confidence:.2f}" if isinstance(confidence, (int, float)) else ""
        print(f"gate: {verdict['choice']}{suffix} via {verdict['backend']}")

        if verdict["choice"] == "complete":
            print(f"COMPLETE after {loop} attempt(s). Log: {a.log}")
            return 0
        if verdict["choice"] == "abort":
            print(f"ABORTED by the gate after {loop} attempt(s). Log: {a.log}")
            return 3
        prompt = (f"{a.task}\n\nThe previous attempt exited {run['exit_code']} with:\n"
                  f"{run['tail']}\n\nFix the cause of that failure, then re-run the checks.")

    print(f"retries exhausted ({a.max_loops} attempt(s), last gate verdict 'retry'). "
          f"Log: {a.log}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
