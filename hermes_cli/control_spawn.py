from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

from hermes_cli import control_db as cp


def spawn_statute_worker(
    dispatch_id: str,
    payload: dict[str, Any],
    root: Path | None,
    parent_dispatch_id: str,
    *,
    live: bool = False,
    handler: str = "agent",
    timeout_s: float | None = None,
    soft_timeout_s: float | None = None,
    hard_timeout_s: float | None = None,
) -> int:
    instance_id = f"statute-worker:{uuid.uuid4().hex}"
    control_root = cp.control_db_path(root).parent.parent if live else Path(root or os.environ.get("HERMES_CONTROL_ROOT") or ".").resolve()
    log_dir = control_root / "control-plane" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{dispatch_id}.log"
    cmd = [
        "hermes",
        "-p",
        "statute-worker",
        "control",
        "worker",
        "run",
        dispatch_id,
    ]
    if live:
        cmd.append("--live")
    else:
        cmd.extend(["--root", str(control_root)])
    cmd.extend(["--profile-id", "statute-worker", "--instance-id", instance_id, "--handler", handler])
    if timeout_s is not None and hard_timeout_s is None:
        hard_timeout_s = timeout_s
    if soft_timeout_s is not None:
        cmd.extend(["--soft-timeout-s", str(float(soft_timeout_s))])
    if hard_timeout_s is not None:
        cmd.extend(["--hard-timeout-s", str(float(hard_timeout_s))])
    env = os.environ.copy()
    env.update(
        {
            "HERMES_PROFILE": "statute-worker",
            "HERMES_PROFILE_ID": "statute-worker",
            "HERMES_CONTROL_ROOT": str(control_root),
            "HERMES_CONTROL_INSTANCE_ID": instance_id,
            "HERMES_CONTROL_DISPATCH_ID": dispatch_id,
            "HERMES_CONTROL_PARENT_DISPATCH_ID": parent_dispatch_id,
            "HERMES_CONTROL_REPO_ROOT": str(payload.get("repo_root") or ""),
            "HERMES_CONTROL_ALLOWED_PATHS": json.dumps(payload.get("allowed_paths") or []),
            "TERMINAL_TIMEOUT": env.get("TERMINAL_TIMEOUT", "300"),
            "TERMINAL_MAX_FOREGROUND_TIMEOUT": env.get("TERMINAL_MAX_FOREGROUND_TIMEOUT", "300"),
        }
    )
    if not live:
        env["HERMES_HOME"] = str(control_root / "profiles" / "statute-worker")
    with log_path.open("ab") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return int(proc.pid)
