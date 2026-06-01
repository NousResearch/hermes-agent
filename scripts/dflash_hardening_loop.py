#!/usr/bin/env python3
"""Run the dflash stability canary as a supervised hardening loop.

The base canary is intentionally finite: it proves a build at one point in
time.  This wrapper is the ownership layer for iterative hardening.  It keeps
running bounded canary cycles until a failure appears, records local JSONL
evidence, and can update a MeshBoard repair task so a worker can pick up the
next root cause without waiting for a human screenshot.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from dflash_stability_canary import (
    DEFAULT_LOG_DIR,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_TOOLSETS,
    iter_cases,
    marker_suffix,
    record_path,
    run_case,
    write_record,
)


TASK_ID_RE = re.compile(r"[^a-z0-9-]+")


def slug(value: object) -> str:
    text = str(value or "unknown").strip().lower().replace("_", "-")
    text = TASK_ID_RE.sub("-", text)
    return text.strip("-") or "unknown"


def failure_task_id(record: dict, *, prefix: str) -> str:
    return "-".join(
        part
        for part in [
            slug(prefix),
            slug(record.get("case")),
            slug(record.get("failure")),
        ]
        if part
    )


def build_meshboard_failure_command(
    *,
    meshboard_root: Path,
    task_id: str,
    record: dict,
    log_path: Path,
    actor: str,
    parent_task: str,
) -> list[str]:
    meshctl = meshboard_root / ".mesh" / "tools" / "meshctl.py"
    details = (
        "Automated dflash hardening loop detected a canary failure. "
        f"case={record.get('case')} failure={record.get('failure')} "
        f"returncode={record.get('returncode')} elapsed_s={record.get('elapsed_s')}. "
        "Raw stdout/stderr stay in the local evidence log."
    )
    next_step = (
        "Inspect the evidence log, reproduce the canary failure from the same "
        "Hermes checkout, fix the root cause, verify with the canary, then "
        "restart the dflash hardening loop."
    )
    cmd = [
        sys.executable,
        str(meshctl),
        "task",
        "register",
        task_id,
        "--update",
        "--apply",
        "--state",
        "open",
        "--owner",
        actor,
        "--priority",
        "High",
        "--size",
        "M",
        "--risk",
        "medium",
        "--requires-human",
        "false",
        "--data-policy",
        "workspace-private",
        "--title",
        f"Repair dflash canary failure: {record.get('case')} {record.get('failure')}",
        "--next",
        next_step,
        "--details",
        details,
        "--link",
        str(log_path),
        "--required-tool",
        "hermes",
        "--verification",
        "Run scripts/dflash_stability_canary.py with the failing case until it passes.",
    ]
    if parent_task:
        cmd.extend(["--parent-task", parent_task])
    return cmd


def register_meshboard_failure(
    *,
    meshboard_root: Path,
    task_id: str,
    record: dict,
    log_path: Path,
    actor: str,
    parent_task: str,
) -> dict:
    cmd = build_meshboard_failure_command(
        meshboard_root=meshboard_root,
        task_id=task_id,
        record=record,
        log_path=log_path,
        actor=actor,
        parent_task=parent_task,
    )
    result = subprocess.run(
        cmd,
        cwd=str(meshboard_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "cmd": [*cmd[:4], "<task-register-args>"],
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-2000:],
        "task_id": task_id,
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means run until failure or interrupt.")
    parser.add_argument("--cycle-sleep", type=float, default=60.0, help="Seconds between successful cycles.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-case timeout in seconds.")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Working directory for hermes -z.")
    parser.add_argument("--hermes-bin", default="hermes", help="Hermes executable.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="Hermes provider override.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hermes model override.")
    parser.add_argument("--toolsets", default=DEFAULT_TOOLSETS, help="Comma-separated toolsets for canary turns.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory for JSONL evidence logs.")
    parser.add_argument("--case", action="append", dest="cases", help="Run only this case name; repeatable.")
    parser.add_argument("--allow-extra-output", action="store_true", help="Allow marker to appear inside extra text.")
    parser.add_argument("--meshboard-root", type=Path, help="If set, register/update a MeshBoard task on failure.")
    parser.add_argument("--meshboard-actor", default="dflash-hardening-loop", help="Owner for failure tasks.")
    parser.add_argument("--meshboard-parent-task", default="", help="Parent MeshBoard task id for failures.")
    parser.add_argument("--task-prefix", default="hermes-dflash-hardening-loop", help="Failure task id prefix.")
    parser.add_argument("--continue-after-failure", action="store_true", help="Keep looping after filing a failure.")
    parser.add_argument("--json", action="store_true", help="Print JSON records to stdout.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    cases = iter_cases(args.cases)
    strict_marker = not args.allow_extra_output
    log_path = record_path(args.log_dir)
    cycle = 0

    while args.max_cycles <= 0 or cycle < args.max_cycles:
        cycle += 1
        for case in cases:
            record = run_case(
                case,
                cwd=args.cwd,
                hermes_bin=args.hermes_bin,
                model=args.model,
                provider=args.provider,
                toolsets=args.toolsets,
                timeout_s=args.timeout,
                strict_marker=strict_marker,
                marker_nonce=marker_suffix(log_path.stem, f"c{cycle}", case.name),
            )
            record["cycle"] = cycle
            write_record(log_path, record)

            if args.json:
                print(json.dumps(record, sort_keys=True), flush=True)
            else:
                status = "ok" if record["ok"] else f"fail:{record['failure']}"
                print(f"cycle={cycle} case={case.name} {status} evidence={log_path}", flush=True)

            if not record["ok"]:
                if args.meshboard_root:
                    task_id = failure_task_id(record, prefix=args.task_prefix)
                    mesh_record = register_meshboard_failure(
                        meshboard_root=args.meshboard_root,
                        task_id=task_id,
                        record=record,
                        log_path=log_path,
                        actor=args.meshboard_actor,
                        parent_task=args.meshboard_parent_task,
                    )
                    write_record(log_path, {"cycle": cycle, "meshboard": mesh_record})
                    if not args.json:
                        print(
                            f"registered MeshBoard failure task {task_id} "
                            f"rc={mesh_record['returncode']}",
                            flush=True,
                        )
                if not args.continue_after_failure:
                    return 1

        if args.max_cycles > 0 and cycle >= args.max_cycles:
            break
        if args.cycle_sleep > 0:
            time.sleep(args.cycle_sleep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
