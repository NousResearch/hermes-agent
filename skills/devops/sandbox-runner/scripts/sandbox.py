#!/usr/bin/env python3
"""sandbox.py — Ephemeral container and environment execution runner for Hermes Agent.

Usage:
  sandbox.py run "<command>" [--image IMAGE] [--mount PATH] [--timeout SECONDS] [--network none|bridge] [--json]
  sandbox.py exec-file <script_path> [--image IMAGE] [--timeout SECONDS] [--json]
  sandbox.py check [--json]
  sandbox.py prune [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_IMAGE = "python:3.11-slim"
DEFAULT_TIMEOUT = 120


def is_docker_available() -> bool:
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return False
    try:
        res = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return res.returncode == 0
    except Exception:
        return False


def run_in_docker(
    command: str,
    *,
    image: str = DEFAULT_IMAGE,
    mount_dir: Optional[Path] = None,
    timeout: int = DEFAULT_TIMEOUT,
    network: str = "none",
    env_vars: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Execute command inside an ephemeral throwaway Docker container."""
    container_name = f"hermes_sandbox_{int(time.time())}_{os.getpid()}"
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        network,
        "--memory",
        "512m",
        "--cpus",
        "1.0",
    ]

    if env_vars:
        for k, v in env_vars.items():
            docker_cmd.extend(["-e", f"{k}={v}"])

    if mount_dir:
        abs_mount = Path(mount_dir).resolve()
        docker_cmd.extend(["-v", f"{abs_mount}:/workspace", "-w", "/workspace"])
    else:
        docker_cmd.extend(["-w", "/tmp"])

    docker_cmd.extend([image, "sh", "-c", command])

    start_time = time.time()
    try:
        proc = subprocess.run(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        elapsed = round(time.time() - start_time, 2)
        return {
            "engine": "docker",
            "image": image,
            "container": container_name,
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_seconds": elapsed,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as e:
        # Force clean container on timeout
        subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {
            "engine": "docker",
            "image": image,
            "container": container_name,
            "exit_code": 124,
            "stdout": e.stdout or "",
            "stderr": (e.stderr or "") + f"\nProcess timed out after {timeout} seconds.",
            "elapsed_seconds": timeout,
            "timed_out": True,
        }


def run_in_local_sandbox(
    command: str,
    *,
    mount_dir: Optional[Path] = None,
    timeout: int = DEFAULT_TIMEOUT,
    env_vars: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Fallback execution inside a temporary isolated directory."""
    temp_dir = Path(tempfile.mkdtemp(prefix="hermes_isolated_"))
    cwd = mount_dir if mount_dir else temp_dir

    env = dict(os.environ)
    if env_vars:
        env.update(env_vars)

    start_time = time.time()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        elapsed = round(time.time() - start_time, 2)
        return {
            "engine": "local_isolated",
            "workspace": str(cwd),
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_seconds": elapsed,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "engine": "local_isolated",
            "workspace": str(cwd),
            "exit_code": 124,
            "stdout": e.stdout or "",
            "stderr": (e.stderr or "") + f"\nProcess timed out after {timeout} seconds.",
            "elapsed_seconds": timeout,
            "timed_out": True,
        }
    finally:
        if not mount_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def execute_sandbox(
    command: str,
    *,
    image: str = DEFAULT_IMAGE,
    mount_dir: Optional[Path] = None,
    timeout: int = DEFAULT_TIMEOUT,
    network: str = "none",
    env_vars: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    if is_docker_available():
        return run_in_docker(
            command,
            image=image,
            mount_dir=mount_dir,
            timeout=timeout,
            network=network,
            env_vars=env_vars,
        )
    return run_in_local_sandbox(
        command,
        mount_dir=mount_dir,
        timeout=timeout,
        env_vars=env_vars,
    )


def execute_script_file(
    script_path: Path,
    *,
    image: str = DEFAULT_IMAGE,
    timeout: int = DEFAULT_TIMEOUT,
    args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    ext = script_path.suffix.lower()
    parent_dir = script_path.parent
    file_name = script_path.name

    arg_str = " ".join(args) if args else ""
    if ext == ".py":
        cmd = f"python3 {file_name} {arg_str}".strip()
    elif ext == ".sh":
        cmd = f"sh {file_name} {arg_str}".strip()
    elif ext in (".js", ".mjs"):
        cmd = f"node {file_name} {arg_str}".strip()
    else:
        cmd = f"./{file_name} {arg_str}".strip()

    return execute_sandbox(
        cmd,
        image=image,
        mount_dir=parent_dir,
        timeout=timeout,
    )


def prune_sandboxes() -> Dict[str, Any]:
    if not is_docker_available():
        return {"engine": "none", "pruned_containers": 0, "status": "Docker not available"}

    res = subprocess.run(
        ["docker", "container", "prune", "-f", "--filter", "label=hermes_sandbox"],
        capture_output=True,
        text=True,
    )
    return {"engine": "docker", "status": "Pruned idle sandbox containers", "output": res.stdout.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes Sandbox Runner CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = subparsers.add_parser("run", help="Run a shell command inside an isolated sandbox.")
    p_run.add_argument("cmd", type=str, help="Command string to execute.")
    p_run.add_argument("--image", type=str, default=DEFAULT_IMAGE, help="Docker image to use.")
    p_run.add_argument("--mount", type=Path, default=None, help="Directory to mount as workspace.")
    p_run.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Execution timeout in seconds.")
    p_run.add_argument("--network", type=str, default="none", choices=["none", "bridge", "host"], help="Network mode.")
    p_run.add_argument("--json", action="store_true", help="Output as JSON.")

    # exec-file
    p_file = subparsers.add_parser("exec-file", help="Execute a script file inside an isolated sandbox.")
    p_file.add_argument("script", type=Path, help="Path to script file.")
    p_file.add_argument("--image", type=str, default=DEFAULT_IMAGE, help="Docker image to use.")
    p_file.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Execution timeout in seconds.")
    p_file.add_argument("--json", action="store_true", help="Output as JSON.")

    # check
    p_chk = subparsers.add_parser("check", help="Check container and sandbox runtime status.")
    p_chk.add_argument("--json", action="store_true", help="Output as JSON.")

    # prune
    p_pru = subparsers.add_parser("prune", help="Clean up stopped sandbox containers.")
    p_pru.add_argument("--json", action="store_true", help="Output as JSON.")

    args = parser.parse_args()

    if args.command == "run":
        res = execute_sandbox(
            args.cmd,
            image=args.image,
            mount_dir=args.mount,
            timeout=args.timeout,
            network=args.network,
        )
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"[{res['engine']}] Exit Code: {res['exit_code']} (Time: {res['elapsed_seconds']}s)")
            if res["stdout"]:
                print(f"STDOUT:\n{res['stdout']}")
            if res["stderr"]:
                print(f"STDERR:\n{res['stderr']}")

    elif args.command == "exec-file":
        res = execute_script_file(args.script, image=args.image, timeout=args.timeout)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"[{res['engine']}] Executed {args.script.name} (Exit Code: {res['exit_code']})")
            if res["stdout"]:
                print(f"STDOUT:\n{res['stdout']}")
            if res["stderr"]:
                print(f"STDERR:\n{res['stderr']}")

    elif args.command == "check":
        docker_ok = is_docker_available()
        status = {
            "docker_available": docker_ok,
            "engine": "docker" if docker_ok else "local_isolated",
            "default_image": DEFAULT_IMAGE,
            "default_timeout": DEFAULT_TIMEOUT,
        }
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Sandbox Engine: {status['engine']}")
            print(f"Docker Available: {docker_ok}")

    elif args.command == "prune":
        res = prune_sandboxes()
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(res.get("status", "Prune complete"))


if __name__ == "__main__":
    main()
